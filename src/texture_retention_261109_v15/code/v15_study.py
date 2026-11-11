"""Train the v1.5 readout, select on validation, then run locked paired confirmations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from upstream import CODE,P,ROOT,RUN,base_camera,cache,case_path,image,render,split_rows,write_json
from texture import TextureHead,bands,proposal,texture_loss,texture_stats

METHODS=("sharpen","direct","aligned","unregistered","learned")
STAGES=("raw","bridge","reader","camera","native_end","guided_camera","guided_end","pre_mix","v14")


def save_tensor(path,state):
    tmp=path.with_suffix(".partial")
    torch.save(state,tmp)
    os.replace(tmp,path)


def train(head_name,rows,perceptual):
    path=RUN/(head_name+".pt")
    head=TextureHead().cuda()
    if path.exists():
        head.load_state_dict(torch.load(path,weights_only=True))
        return head.eval()
    torch.manual_seed(1501)
    head=TextureHead().cuda()
    opt=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.0001)
    samples=[torch.load(case_path("train",r,P["primary_seed"]),map_location="cpu",weights_only=True) for r in rows]
    for step in range(P["train_updates"]):
        rng=np.random.default_rng(18000+step)
        batch={k:[] for k in ("raw","v14","hr","lr")}
        for _ in range(P["train_batch"]):
            sample=samples[rng.integers(len(samples))]
            side=P["crop"]
            y,x=4*rng.integers(0,(512-side)//4+1,2)
            for k in batch:
                q=4 if k=="lr" else 1
                batch[k].append(sample[k][...,y//q:(y+side)//q,x//q:(x+side)//q])
        batch={k:torch.cat(v).cuda() for k,v in batch.items()}
        out=head(batch["raw"],batch["v14"],batch["lr"],aligned=head_name=="learned").clamp(0,1)
        hr=batch["hr"]
        gt=hr[...,8:-8,8:-8]
        pred=out[...,8:-8,8:-8]
        loss=4*(pred-gt).abs().mean()+perceptual.net(pred,gt).mean()+.08*texture_loss(pred,gt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True)
        opt.step()
        if (step+1)%300==0:print(head_name,step+1,float(loss),flush=True)
    save_tensor(path,head.state_dict())
    return head.eval()


@torch.no_grad()
def evaluate_output(x,hr,metric):
    x=x.clamp(0,1)
    return dict(psnr=float(-10*(x-hr).square().mean().log10()),lpips=float(metric(x,hr).mean()),**texture_stats(x,hr))


def aggregate(records):
    return {k:{metric:float(np.mean([r["scores"][k][metric] for r in records]))
               for metric in records[0]["scores"][k]} for k in records[0]["scores"]}


@torch.no_grad()
def evaluate(group,rows,seed,heads,metric,lock=None):
    records=[]
    folder=RUN/"scores"/group
    for row in rows:
        path=case_path(group,row,seed)
        target=folder/(path.stem+".json")
        if not target.exists():
            sample=torch.load(path,map_location="cpu",weights_only=True)
            s={k:v.cuda() for k,v in sample.items() if isinstance(v,torch.Tensor)}
            scores={k:evaluate_output(s[k],s["hr"],metric) for k in STAGES}
            for method in METHODS:
                pred=proposal(method,heads.get(method),s["raw"],s["v14"],s["lr"])
                strengths=P["validation_strength"][1:] if lock is None else [lock["families"][method]["strength"]]
                for strength in strengths:
                    scores[f"{method}/{strength:g}"]=evaluate_output(torch.lerp(s["v14"],pred,strength),s["hr"],metric)
            write_json(target,dict(scene=row["scene_id"],seed=seed,scores=scores))
        records.append(json.loads(target.read_text()))
    return records,aggregate(records)


def choose(dev):
    ref=dev["v14"]
    def eligible(m):
        return (m["psnr"]>=ref["psnr"]-.1 and m["lpips"]<=ref["lpips"] and m["texerr"]<ref["texerr"])
    families={}
    for method in METHODS:
        candidates=[k for k in dev if k.startswith(method+"/") and eligible(dev[k])]
        if candidates:
            key=min(candidates,key=lambda k:(dev[k]["lpips"],dev[k]["texerr"]))
            strength=float(key.split("/")[1])
        else:
            key="v14";strength=0.
        families[method]=dict(key=key,strength=strength)
    key=min(["v14"]+[r["key"] for r in families.values()],key=lambda k:(dev[k]["lpips"],dev[k]["texerr"]))
    method,strength=(key.split("/") if key!="v14" else ("identity","0"))
    return dict(selected=key,method=method,strength=float(strength),families=families)


def contrasts(records,key,comp):
    values={}
    for field in ("psnr","lpips","texerr","fine_ratio","mid_ratio","correlation"):
        ds=np.array([r["scores"][key][field]-r["scores"][comp][field] for r in records])
        ci=np.quantile(np.random.default_rng(261109).choice(ds,(5000,len(ds))).mean(1),[.025,.975])
        values[field]=[float(ds.mean()),*ci.tolist()]
    return values


def prediction(sample,choice,heads):
    if choice["method"]=="identity":return sample["v14"]
    prop=proposal(choice["method"],heads.get(choice["method"]),sample["raw"],sample["v14"],sample["lr"])
    return torch.lerp(sample["v14"],prop,choice["strength"])


def load_heads(method):
    if method not in ("unregistered","learned"):return {}
    head=TextureHead().cuda().eval()
    head.load_state_dict(torch.load(RUN/(method+".pt"),map_location="cuda",weights_only=True))
    return {method:head}


@torch.no_grad()
def restore(base,camera,lr,seed,choice,heads):
    stages=render(base,camera,lr,seed)
    return prediction(dict(raw=stages["raw"],v14=stages["v14"],lr=lr),choice,heads)


def save_png(x,path):
    path=Path(path)
    path.parent.mkdir(exist_ok=True,parents=True)
    pixels=x[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
    Image.fromarray(pixels).save(path)


def study():
    splits=split_rows()
    base,camera=base_camera()
    with (RUN/"cache_dev.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for g in ("dev","train"):cache(g,splits[g],P["primary_seed"],base,camera)
    heads={k:train(k,splits["train"],camera.metric) for k in ("unregistered","learned")}
    _,dev=evaluate("dev",splits["dev"],6,heads,camera.metric)
    choice=choose(dev)
    if (RUN/"choice.json").exists() and choice!=json.loads((RUN/"choice.json").read_text())["choice"]:
        raise ValueError("Locked choice differs")
    write_json(RUN/"choice.json",dict(selected=choice["selected"],choice=choice,means=dev))
    print("locked",choice,flush=True)
    for g in ("real","dreal","repeat","external_repeat"):
        seed=P["repeat_seed"] if "repeat" in g else P["primary_seed"]
        cache(g,splits[g],seed,base,camera)
        records,means=evaluate(g,splits[g],seed,heads,camera.metric,choice)
        key=choice["selected"]
        comparisons=[k for k in means if k!=key]
        write_json(RUN/(g+"_summary.json"),dict(selected=key,cases=len(records),means=means,
                     effects={k:contrasts(records,key,k) for k in comparisons}))
    y=image(ROOT/splits["dev"][0]["lr"])[None].cuda().float()/255
    out=restore(base,camera,y,6,choice,heads)
    save_png(out,RUN/"release_check.png")
    write_json(RUN/"release.json",dict(selected=choice["selected"],choice=choice,shape=list(out.shape),
               parent="visual_trajectory_mix_261108_v1",head_sha=hashlib.sha256((RUN/(choice["method"]+".pt")).read_bytes()).hexdigest()
               if choice["method"] in heads else None))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("study","restore"))
    p.add_argument("--input");p.add_argument("--output");p.add_argument("--seed",type=int,default=6)
    args=p.parse_args()
    torch.set_num_threads(4)
    RUN.mkdir(parents=True,exist_ok=True)
    if args.action=="study":
        with (RUN/"study.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            study()
    else:
        if not args.input or not args.output:p.error("Provide --input and --output")
        choice=json.loads((RUN/"release.json").read_text())["choice"]
        y=image(args.input)[None].cuda().float()/255
        if y.shape!=(1,3,128,128):raise ValueError("Use native 128 x 128 RGB LR")
        base,camera=base_camera()
        save_png(restore(base,camera,y,args.seed,choice,load_heads(choice["method"])),args.output)
