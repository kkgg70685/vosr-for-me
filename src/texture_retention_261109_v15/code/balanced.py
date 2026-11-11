"""Fit an MSE-oriented band predictor and validate independent transfer gains."""
import argparse
import fcntl
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from release_study import (CODE,RUN,ROOT,base_camera,cache,case_path,split_rows,render,save_png,
                           evaluate_output,aggregate,contrasts,write_json,image,load_heads,save_tensor)
from texture import features,smooth,luminance
Q=json.loads((CODE/"balanced.json").read_text())


def components(data,head):
    r,b,y=(data[k] for k in ("raw","v14","lr"))
    db,bb,desc=features(r,b,y,True)
    gates=F.interpolate(head.net(F.avg_pool2d(desc,2)).sigmoid(),size=b.shape[-2:],mode="bilinear",align_corners=False)
    learned=[1.25*gates[:,j:j+1]*(db[j]-bb[j])+gates[:,j+2:j+3]*bb[j] for j in range(2)]
    # Middle-scale components, neighboring detail and the discrepancy to the observable LR.
    up=F.interpolate(y,size=b.shape[-2:],mode="bicubic",align_corners=False)
    fields=[bb[0],bb[1],db[0],db[1],b-up]
    bank=[]
    for field in fields:
        h,w=field.shape[-2:]
        padded=F.pad(field,(2,2,2,2),mode="replicate")
        bank.extend(padded[...,2+dy:2+dy+h,2+dx:2+dx+w] for dy,dx in ((0,0),(0,2),(0,-2),(2,0),(-2,0)))
    bank=torch.stack(bank,2)
    activity=(12*smooth(luminance(bb[1]).abs(),9)).clamp(0,1).unsqueeze(2)
    return learned,torch.cat((bank,bank*activity),2)


@torch.no_grad()
def train_ridge(rows,head):
    p=RUN/"ridge.pt"
    if p.exists():return torch.load(p,weights_only=True).cuda()
    xx=xy=None
    count=0
    for i,r in enumerate(rows):
        data=torch.load(case_path("train",r,6),map_location="cpu",weights_only=True)
        data={k:data[k].cuda() for k in ("raw","v14","lr","hr")}
        _,features=components(data,head)
        offset=i%4
        # Identical fitted linear filter across color channels.
        x=features[:,:,:,offset::4,(i//4)%4::4].permute(0,1,3,4,2).reshape(-1,features.shape[2]).double()
        y=(data["hr"]-data["v14"])[...,offset::4,(i//4)%4::4].reshape(-1).double()
        moment=x.T@x;cross=x.T@y
        xx=moment if xx is None else xx+moment
        xy=cross if xy is None else xy+cross
        count+=len(y)
    coef=torch.linalg.solve(xx/count+1e-4*torch.eye(xx.shape[0],device=xx.device),xy/count).float()
    save_tensor(p,coef.cpu())
    return coef


def corrections(data,head,coef):
    read,bank=components(data,head)
    linear=torch.einsum("bcfhw,f->bchw",bank,coef)
    return read[0],read[1],linear


def key(spec):
    return "_".join(f"{x:g}" for x in spec)


def candidates():
    return sorted(set([(0.,0.,0.),(.75,.75,0.)]+[(f,m,r) for f in Q["fine"] for m in Q["mid"] for r in Q["reg"]]))


def pools():
    manifest=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    def unique(ds,cam=None):
        by={}
        for r in sorted(manifest,key=lambda a:a["id"]):
            if r["dataset"]==ds and r["role"]=="V" and (cam is None or r.get("camera")==cam):
                by.setdefault(r["scene_id"],r)
        return list(by.values())
    dev=sum((unique("RealSR",cam)[:15] for cam in ("Canon","Nikon")),[])
    assert len(dev)==30
    return dev,unique("DRealSR")


@torch.no_grad()
def panel(group,rows,seed,head,coef,specs,metric):
    results=[]
    for i,r in enumerate(rows):
        path=case_path(group,r,seed)
        file=RUN/"balanced_scores"/group/(path.stem+".json")
        if not file.exists():
            s=torch.load(path,map_location="cpu",weights_only=True)
            data={k:s[k].cuda() for k in ("lr","raw","v14","hr")}
            fine,mid,linear=corrections(data,head,coef)
            scores={"raw":evaluate_output(data["raw"],data["hr"],metric)}
            for spec in specs:
                f,m,c=spec
                rgb=data["v14"]+f*fine+m*mid+c*linear
                scores[key(spec)]=evaluate_output(rgb,data["hr"],metric)
            write_json(file,dict(scene=r["scene_id"],camera=r.get("camera","unknown"),seed=seed,scores=scores))
        results.append(json.loads(file.read_text()))
    return results,aggregate(results)


def select(records,means,specs):
    valid={}
    for cam in ("Canon","Nikon"):
        table=aggregate([r for r in records if r["camera"]==cam])
        ref=table["0_0_0"]
        valid[cam]={k for k in map(key,specs) if table[k]["psnr"]>=ref["psnr"]-.03 and
                    table[k]["lpips"]<=ref["lpips"] and table[k]["texerr"]<=ref["texerr"]}
    options=valid["Canon"]&valid["Nikon"]
    chosen=min(options,key=lambda k:(means[k]["lpips"],means[k]["texerr"],k))
    return next(s for s in specs if key(s)==chosen)


@torch.no_grad()
def predict(data,head,coef,spec):
    f,m,c=spec
    a,b,d=corrections(data,head,coef)
    return data["v14"]+f*a+m*b+c*d


def restore(base,camera,y,seed,head,coef,spec):
    data=render(base,camera,y,seed)
    data["lr"]=y
    return predict(data,head,coef,spec)


def study():
    base,camera=base_camera()
    head=load_heads("learned")["learned"]
    splits=split_rows()
    coef=train_ridge(splits["train"],head)
    dev,new_dreal=pools()
    assert not ({r["scene_id"] for r in dev}&{r["scene_id"] for r in splits["dev"]})
    cache("baldev",dev,6,base,camera)
    records,means=panel("baldev",dev,6,head,coef,candidates(),camera.metric)
    spec=select(records,means,candidates())
    lock=RUN/"balanced_choice.json"
    if lock.exists() and list(spec)!=json.loads(lock.read_text())["spec"]:raise RuntimeError("Lock changed")
    write_json(lock,dict(selected=key(spec),spec=spec,means=means))
    print("locked",spec,flush=True)
    f,m,c=spec
    controls=sorted(set([tuple(spec),(0,0,0),(.75,.75,0),(f,m,0),(0,0,c),(f,0,c),(0,m,c)]))
    if new_dreal:splits["unseen_dreal"]=new_dreal
    for group in ("real","dreal","repeat","external_repeat","unseen_dreal"):
        if group not in splits:continue
        seed=23 if group=="unseen_dreal" else (9 if "repeat" in group else 6)
        if group=="unseen_dreal":
            assert not ({r["scene_id"] for r in new_dreal}&{r["scene_id"] for r in splits["dreal"]})
            cache(group,new_dreal,seed,base,camera)
        rec,table=panel(group,splits[group],seed,head,coef,controls,camera.metric)
        effects={k:contrasts(rec,key(spec),k) for k in table if k!=key(spec)}
        write_json(RUN/(group+"_balanced.json"),dict(selected=key(spec),cases=len(rec),means=table,effects=effects))
        print(group,len(rec),flush=True)
    y=image(ROOT/dev[0]["lr"])[None].cuda().float()/255
    out=restore(base,camera,y,6,head,coef,spec)
    save_png(out,RUN/"balanced_check.png")
    write_json(RUN/"balanced_release.json",dict(spec=spec,selected=key(spec),shape=list(out.shape),
               texture_sha=hashlib.sha256((RUN/"learned.pt").read_bytes()).hexdigest(),
               ridge_sha=hashlib.sha256((RUN/"ridge.pt").read_bytes()).hexdigest()))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=("study","restore"))
    ap.add_argument("--input");ap.add_argument("--output");ap.add_argument("--seed",type=int,default=6)
    args=ap.parse_args()
    torch.set_num_threads(4)
    if args.action=="study":
        with (RUN/"balanced.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            study()
    else:
        if not args.input or not args.output:ap.error("Provide --input and --output")
        base,camera=base_camera()
        y=image(ROOT/args.input)[None].cuda().float()/255
        if y.shape!=(1,3,128,128):raise ValueError("Expected native 128 × 128 RGB")
        head=load_heads("learned")["learned"];coef=torch.load(RUN/"ridge.pt",weights_only=True).cuda()
        spec=json.loads((RUN/"balanced_release.json").read_text())["spec"]
        save_png(restore(base,camera,y,args.seed,head,coef,spec),args.output)
