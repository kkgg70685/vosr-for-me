"""Use LR evidence for local phase and retain an independently controlled contrast target."""
import argparse
import fcntl
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F
from visual_study import (CODE,RUN,OLD,ROOT,P,FAMILIES,ReferenceGate,LocalSR,load_net,load_heads,base_camera,
                          rows,obtain,retained,refpath,cache_refs,image,evaluate_output,aggregate,write_json,effects,combine)
from evidence import references,bands,smooth,luminance,render,save_png
from balanced import corrections
from texture import align

Q=json.loads((CODE/"phase_protocol.json").read_text())
DEST=RUN/"phase"


def phase_terms(anchor,source,raw):
    result=[]
    for a,b,r in zip(bands(anchor),bands(source),bands(raw)):
        x,y,z=map(luminance,(a,b,r))
        ea=(smooth(x.square(),9)+1e-6).sqrt()
        eb=(smooth(y.square(),9)+1e-6).sqrt()
        er=(smooth(z.square(),9)+1e-6).sqrt()
        cov=smooth(x*y,9)/(ea*eb)
        support=(eb/(eb+.004))*(.35+.65*cov.clamp(0,1))
        normalized=b*(ea/eb).clamp(.35,2.5)
        target=(er-ea).clamp_min(0).minimum(.8*ea)
        # Raw evidence may vote for extra contrast only where the local LR prediction also has power.
        result.append((a,normalized,support,ea,target))
    low=(source-anchor)-sum(bands(source-anchor))
    return result,low


def generate(anchor,terms,low,spec):
    h,g,l=spec
    result=anchor+l*low
    for a,b,support,ea,target in terms:
        aligned=torch.lerp(a,b,h*support)
        energy=(smooth(luminance(aligned).square(),9)+1e-6).sqrt()
        gain=((ea+g*support*target)/energy).clamp(.5,1.8)
        result=result+aligned*gain-a
    return result


def version(spec):
    return "_".join(f"{z:g}" for z in spec)


def options():
    return [(h,g,l) for h in Q["phase_weights"] for g in Q["amplitude_gains"] for l in Q["coarse_weights"]]


def source_bank(y,nets,bank,controls):
    local=nets["localnet"](y)
    out=dict(localnet=local)
    modes=[("patch","patch","local"),("semantic","semantic","semantic")]
    if controls:modes += [("shift","semantic","misplaced"),("wrong","semantic","wrong_value")]
    for name,net,ref in modes:
        out[name]=nets[net](y,local,bank[ref].cuda().float())
    return out


def dev_pool(old):
    used={r["scene_id"] for r in old}
    manifest=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    unique={}
    for r in sorted(manifest,key=lambda row:row["id"]):
        if r["dataset"]=="RealSR" and r["role"]=="V" and r["scene_id"] not in used:unique.setdefault(r["scene_id"],r)
    assert len(unique)==30
    return list(unique.values())


@torch.no_grad()
def panel(base,camera,records,group,seed,nets,head,coef,lock):
    results=[]
    for idx,row in enumerate(records):
        stem=hashlib.sha256(row["scene_id"].encode()).hexdigest()[:20]+f"_{seed}.json"
        path=DEST/"scores"/group/stem
        if not path.exists():
            data=obtain(row,"dev" if group=="phase_dev" else group,seed,base,camera)
            if not refpath(row["scene_id"]).exists():cache_refs(base,[row])
            bank=torch.load(refpath(row["scene_id"]),map_location="cpu",weights_only=True)
            sources=source_bank(data["lr"],nets,bank,lock is not None)
            anchor=retained(data,head,coef)
            hr=data["hr"];raw=align(data["raw"],data["v14"])
            def measure(x):return evaluate_output(x,hr,camera.metric)
            scores={k:measure(x) for k,x in (("reference",anchor),("v14",data["v14"]),("raw",data["raw"]))}
            for mode,src in sources.items():
                terms,low=phase_terms(anchor,src,raw)
                if lock is None:specs=options()
                else:
                    own=mode if mode in FAMILIES else "semantic"
                    specs=sorted(set((tuple(lock["families"][own]),tuple(lock["matched"]))))
                for spec in specs:
                    scores[mode+"/"+version(spec)]=measure(generate(anchor,terms,low,spec))
                if lock is not None:
                    h,_,l=lock["matched"]
                    scores[mode+"/direct"]=measure(combine(anchor,src,h,l))
            write_json(path,dict(scene=row["scene_id"],camera=row.get("camera","unknown"),scores=scores))
        results.append(json.loads(path.read_text()))
        if (idx+1)%12==0:print(group,idx+1,len(records),flush=True)
    return results,aggregate(results)


def choose(records,means):
    eligible=set(means)
    for cam in ("Canon","Nikon"):
        t=aggregate([row for row in records if row["camera"]==cam]);r=t["reference"]
        eligible &= {k for k,m in t.items() if m["psnr"]>=r["psnr"]-.03 and m["lpips"]<=r["lpips"] and m["texerr"]<=r["texerr"]}
    families={};opts=[]
    for mode in FAMILIES:
        pool=sorted(k for k in eligible if k.startswith(mode+"/"))+["reference"]
        key=min(pool,key=lambda k:(means[k]["lpips"],means[k]["texerr"],k))
        families[mode]=([0,0,0] if key=="reference" else [float(v) for v in key.split("/")[1].split("_")])
        opts.append(key)
    best=min(opts,key=lambda k:(means[k]["lpips"],means[k]["texerr"],k))
    match=([.65,.45,.25] if best=="reference" else families[best.split("/")[0]])
    return dict(selected=best,families=families,matched=match)


def model_parts():
    nets={m:load_net(m) for m in FAMILIES}
    return nets,load_heads("learned")["learned"],torch.load(OLD/"ridge.pt",weights_only=True,map_location="cuda")


@torch.no_grad()
def apply_recipe(data,sem,nets,head,coef,choice):
    anchor=retained(data,head,coef)
    key=choice["selected"]
    if key=="reference":return anchor
    mode=key.split("/")[0];spec=choice["families"][mode];y=data["lr"]
    source=nets["localnet"](y)
    if mode!="localnet":
        bank=references(y,sem)
        source=nets[mode](y,source,bank["local" if mode=="patch" else "semantic"])
    terms,low=phase_terms(anchor,source,align(data["raw"],data["v14"]))
    return generate(anchor,terms,low,spec)


def restore(base,camera,lr,seed,nets,head,coef,choice):
    data=render(base,camera,lr,seed);data["lr"]=lr
    with torch.no_grad():
        mode=choice["selected"].split("/")[0]
        sem=base.conditions(lr)[1] if mode in ("patch","semantic") else None
        return apply_recipe(data,sem,nets,head,coef,choice)


def study():
    splits=rows();base,camera=base_camera();nets,head,coef=model_parts()
    dev=dev_pool(splits["dev"])
    records,m=panel(base,camera,dev,"phase_dev",6,nets,head,coef,None)
    lock=choose(records,m)
    f=DEST/"lock.json"
    if f.exists() and json.loads(f.read_text())["choice"]!=lock:raise ValueError("Lock changed")
    write_json(f,dict(selected=lock["selected"],choice=lock,means=m))
    print("locked",lock,flush=True)
    from balanced import pools
    splits["additional_dreal"]=pools()[1]
    for group in ("real","dreal","new_real","new_dreal","additional_dreal"):
        seed=37 if group=="additional_dreal" else (29 if group.startswith("new_") else 6)
        rec,table=panel(base,camera,splits[group],group,seed,nets,head,coef,lock)
        write_json(DEST/(group+".json"),dict(cases=len(rec),selected=lock["selected"],means=table,effects=effects(rec,lock["selected"])))
    y=image(ROOT/dev[0]["lr"])[None].cuda().float()/255
    rgb=restore(base,camera,y,6,nets,head,coef,lock)
    save_png(rgb,DEST/"release_check.png")
    write_json(DEST/"release.json",dict(selected=lock["selected"],choice=lock,shape=list(rgb.shape)))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=("study","restore"));ap.add_argument("--input");ap.add_argument("--output")
    ap.add_argument("--seed",type=int,default=6);args=ap.parse_args()
    torch.set_num_threads(4);DEST.mkdir(exist_ok=True,parents=True)
    if args.action=="study":
        with (DEST/"study.lock").open("a+") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (RUN/"study.lock").open("a+") as prior:fcntl.flock(prior,fcntl.LOCK_EX)
            study()
    else:
        if not args.input or not args.output:ap.error("Provide --input and --output")
        y=image(args.input)[None].cuda().float()/255
        if y.shape!=(1,3,128,128):raise ValueError("Use native 128 × 128 RGB")
        base,camera=base_camera();nets,head,coef=model_parts()
        lock=json.loads((DEST/"release.json").read_text())["choice"]
        save_png(restore(base,camera,y,args.seed,nets,head,coef,lock),args.output)
