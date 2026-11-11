"""Select independent coarse calibration for the locked fine-detail predictor."""
import argparse
import fcntl
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from v15_study import (RUN,CODE,ROOT,P,Path,base_camera,cache,case_path,split_rows,render,
                       evaluate_output,aggregate,contrasts,load_heads,save_png,save_tensor,image,write_json,hashlib)
from texture import proposal

Q=json.loads((CODE/"refinement.json").read_text())


class ColorCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(12,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),
                               nn.SiLU(),nn.Conv2d(32,3,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,lr,base,raw):
        a,b=F.avg_pool2d(base,4),F.avg_pool2d(raw,4)
        return .12*self.net(torch.cat((lr,a,8*(lr-a),6*(b-a)),1)).tanh()


def train_color(rows):
    path=RUN/"color.pt"
    torch.manual_seed(1521)
    net=ColorCorrection().cuda()
    if path.exists():
        net.load_state_dict(torch.load(path,weights_only=True))
        return net.eval()
    train=[torch.load(case_path("train",r,6),map_location="cpu",weights_only=True) for r in rows]
    opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.0001)
    for step in range(Q["train_updates"]):
        rng=np.random.default_rng(15000+step)
        data={k:[] for k in ("raw","v14","hr","lr")}
        for _ in range(10):
            row=train[rng.integers(len(train))]
            y,x=4*rng.integers(0,97,2)
            for k in data:
                q=4 if k=="lr" else 1
                data[k].append(row[k][...,y//q:(y+128)//q,x//q:(x+128)//q])
        data={k:torch.cat(batch).cuda() for k,batch in data.items()}
        offset=net(data["lr"],data["v14"],data["raw"])
        diff=(F.avg_pool2d(data["v14"]-data["hr"],4)+offset)[...,2:-2,2:-2]
        loss=diff.square().mean()+.035*diff.abs().mean()+.006*(offset[...,1:]-offset[...,:-1]).abs().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    save_tensor(path,net.state_dict())
    return net.eval()


def specification(fine,coarse):
    return f"fine{fine:g}_coarse{coarse:g}"


def calibrated(data,head,color,method,fine,coarse):
    y,b,r=data["lr"],data["v14"],data["raw"]
    native_detail=proposal(method,head,r,b,y) if method!="identity" else b
    delta=F.interpolate(color(y,b,r),size=b.shape[-2:],mode="bilinear",align_corners=False)
    return torch.lerp(b,native_detail,fine)+delta*coarse


def caldev_rows():
    manifest=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    out=[]
    for cam in ("Canon","Nikon"):
        unique={}
        for r in sorted(manifest,key=lambda z:z["id"]):
            if (r["dataset"],r["role"],r.get("camera"))==("RealSR","V",cam):
                unique.setdefault(r["scene_id"],r)
        out+=list(unique.values())[23:30]
    assert len({r["scene_id"] for r in out})==14
    return out


@torch.no_grad()
def panel(group,rows,seed,camera,net,head,initial,settings):
    result=[]
    for r in rows:
        path=case_path(group,r,seed)
        record=RUN/"calibrated_scores"/group/(path.stem+".json")
        if not record.exists():
            data=torch.load(path,map_location="cpu",weights_only=True)
            s={k:data[k].cuda() for k in ("lr","v14","raw","hr")}
            scores={k:evaluate_output(s[k],s["hr"],camera.metric) for k in ("v14","raw")}
            donor=proposal(initial["method"],head,s["raw"],s["v14"],s["lr"]) if initial["method"]!="identity" else s["v14"]
            offset=F.interpolate(net(s["lr"],s["v14"],s["raw"]),size=s["v14"].shape[-2:],mode="bilinear",align_corners=False)
            for f,c in settings:
                rgb=torch.lerp(s["v14"],donor,f)+c*offset
                scores[specification(f,c)]=evaluate_output(rgb,s["hr"],camera.metric)
            write_json(record,dict(scene=r["scene_id"],seed=seed,scores=scores))
        result.append(json.loads(record.read_text()))
    return result,aggregate(result)


def select(means,initial,settings):
    previous=specification(initial["strength"],0)
    eligible=[(f,c) for f,c in settings if
              means[specification(f,c)]["psnr"]>=means["v14"]["psnr"] and
              means[specification(f,c)]["lpips"]<=means[previous]["lpips"] and
              means[specification(f,c)]["texerr"]<=means[previous]["texerr"]]
    return min(eligible,key=lambda fc:means[specification(*fc)]["lpips"]) if eligible else (initial["strength"],0)


def study():
    if not (RUN/"release.json").is_file():
        raise RuntimeError("Primary confirmation must finish first")
    base,camera=base_camera()
    initial=json.loads((RUN/"choice.json").read_text())["choice"]
    head=load_heads(initial["method"]).get(initial["method"])
    splits=split_rows()
    dev=caldev_rows()
    assert not ({r["scene_id"] for r in dev}&{r["scene_id"] for r in splits["dev"]})
    color=train_color(splits["train"])
    cache("caldev",dev,6,base,camera)
    options=[(f,c) for f in Q["detail_strengths"] for c in Q["coarse_strengths"]]
    previous=(initial["strength"],0)
    if previous not in options:options.append(previous)
    _,means=panel("caldev",dev,6,camera,color,head,initial,options)
    fine,coarse=select(means,initial,options)
    key=specification(fine,coarse)
    choice=dict(key=key,method=initial["method"],fine=fine,coarse=coarse)
    lock=RUN/"calibration_choice.json"
    if lock.exists() and choice!=json.loads(lock.read_text())["choice"]:
        raise RuntimeError("Calibration selection changed")
    write_json(lock,dict(choice=choice,means=means))
    print("locked",choice,flush=True)
    options=sorted(set([previous,(fine,coarse),(fine,0),(0,coarse)]))
    for group in ("real","dreal","repeat","external_repeat"):
        seed=9 if "repeat" in group else 6
        records,means=panel(group,splits[group],seed,camera,color,head,initial,options)
        effects={k:contrasts(records,key,k) for k in means if k!=key}
        write_json(RUN/(group+"_final.json"),dict(selected=key,cases=len(records),means=means,effects=effects))
        print(group,len(records),flush=True)
    y=image(ROOT/dev[0]["lr"])[None].cuda().float()/255
    out=restore(base,camera,y,6,choice,color,head)
    save_png(out,RUN/"v15_check.png")
    write_json(RUN/"v15_release.json",dict(choice=choice,shape=list(out.shape),upstream="visual_trajectory_mix_261108_v1",
               texture_sha=hashlib.sha256((RUN/(initial["method"]+".pt")).read_bytes()).hexdigest() if head else None,
               color_sha=hashlib.sha256((RUN/"color.pt").read_bytes()).hexdigest() if coarse else None))


@torch.no_grad()
def restore(base,camera,y,seed,choice,color,head):
    data=render(base,camera,y,seed)
    data["lr"]=y
    if not choice["coarse"]:
        donor=proposal(choice["method"],head,data["raw"],data["v14"],y) if choice["method"]!="identity" else data["v14"]
        return torch.lerp(data["v14"],donor,choice["fine"])
    return calibrated(data,head,color,choice["method"],choice["fine"],choice["coarse"])


def load_release():
    choice=json.loads((RUN/"v15_release.json").read_text())["choice"]
    head=load_heads(choice["method"]).get(choice["method"])
    color=None
    if choice["coarse"]:
        color=ColorCorrection().cuda().eval()
        color.load_state_dict(torch.load(RUN/"color.pt",map_location="cuda",weights_only=True))
    return choice,color,head


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=("study","restore"))
    ap.add_argument("--input");ap.add_argument("--output");ap.add_argument("--seed",type=int,default=6)
    args=ap.parse_args()
    torch.set_num_threads(4)
    if args.action=="study":
        with (RUN/"calibration.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Join the active primary study before acquiring the GPU.
            with (RUN/"study.lock").open("a+") as primary:
                fcntl.flock(primary,fcntl.LOCK_EX)
            study()
    else:
        if not args.input or not args.output:ap.error("Provide --input and --output")
        base,camera=base_camera()
        y=image(args.input)[None].cuda().float()/255
        if y.shape!=(1,3,128,128):raise ValueError("Expected native 128 × 128 RGB")
        save_png(restore(base,camera,y,args.seed,*load_release()),args.output)
