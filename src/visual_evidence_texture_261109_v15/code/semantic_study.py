"""Material-supported phase synthesis, matched interventions and locked release."""
import argparse
import fcntl
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F
from semantic_model import (PUB,Q,MODES,basis,store_tokens,tokenpath,inputs,fit,load_heads,projected)
from phase_study import (RUN,OLD,ROOT,CODE,rows,base_camera,model_parts,dev_pool,obtain,refpath,retained,
                         phase_terms,generate,align,evaluate_output,aggregate,effects,render,image,save_png,
                         references,version,write_json)


def sources(data,row,base,nets,lr_net,params,controls):
    if not tokenpath(row["scene_id"]).exists():store_tokens(base,[row],params)
    bank={k:v.cuda().float() for k,v in torch.load(refpath(row["scene_id"]),weights_only=True).items()}
    t=torch.load(tokenpath(row["scene_id"]),weights_only=True).cuda().float()
    return predict_sources(data["lr"],lr_net,nets,bank,t,controls)


def predict_sources(y,lr_net,nets,bank,t,controls):
    t=F.interpolate(t,size=y.shape[-2:],mode="bilinear",align_corners=False)
    pred=lr_net(y);out={}
    modes=list(MODES)+(["shift","wrong","nosem","novalue"] if controls else [])
    for mode in modes:
        b,s=inputs(y,bank,t,mode)
        out[mode]=nets[mode if mode in MODES else "vision"](y,pred,b,s)
    return out


def options():
    return [(h,g,l) for h in Q["phase"] for g in Q["gain"] for l in Q["low"]]


@torch.no_grad()
def panel(base,camera,records,group,seed,lr_net,nets,params,head,coef,lock):
    result=[]
    for i,r in enumerate(records):
        path=PUB/"scores"/group/(hashlib.sha256(r["scene_id"].encode()).hexdigest()[:20]+f"_{seed}.json")
        if not path.exists():
            data=obtain(r,"dev" if group=="dev" else group,seed,base,camera)
            y,hr=data["lr"],data["hr"];anchor=retained(data,head,coef)
            raw=align(data["raw"],data["v14"]);preds=sources(data,r,base,nets,lr_net,params,lock is not None)
            scores={k:evaluate_output(v,hr,camera.metric) for k,v in (("reference",anchor),("v14",data["v14"]),("raw",data["raw"]))}
            for mode,source in preds.items():
                terms,low=phase_terms(anchor,source,raw)
                specs=options() if lock is None else sorted(set([tuple(lock["matched"]),tuple(lock["families"][mode if mode in MODES else "vision"])]))
                for spec in specs:scores[mode+"/"+version(spec)]=evaluate_output(generate(anchor,terms,low,spec),hr,camera.metric)
            write_json(path,dict(scene=r["scene_id"],camera=r.get("camera","unknown"),scores=scores))
        result.append(json.loads(path.read_text()))
        if (i+1)%10==0:print(group,i+1,len(records),flush=True)
    return result,aggregate(result)


def choose(records,means):
    eligible=set(means)
    for cam in ("Canon","Nikon"):
        t=aggregate([r for r in records if r["camera"]==cam]);ref=t["reference"]
        eligible&={k for k,m in t.items() if m["psnr"]>=ref["psnr"]-.03 and m["lpips"]<=ref["lpips"] and m["texerr"]<=ref["texerr"]}
    families={};winners=[]
    for mode in MODES:
        pool=sorted(k for k in eligible if k.startswith(mode+"/"))+["reference"]
        best=min(pool,key=lambda k:(means[k]["lpips"],means[k]["texerr"],k))
        families[mode]=[0,0,0] if best=="reference" else [float(x) for x in best.split("/")[1].split("_")]
        winners.append(best)
    top=min(winners,key=lambda k:(means[k]["lpips"],means[k]["texerr"],k))
    spec=[.35,.2,.25] if top=="reference" else families[top.split("/")[0]]
    return dict(selected=top,families=families,matched=spec)


def compose(data,anchor,sources,lock):
    mode=lock["selected"].split("/")[0]
    if mode=="reference":return anchor
    terms,low=phase_terms(anchor,sources[mode],align(data["raw"],data["v14"]))
    return generate(anchor,terms,low,lock["families"][mode])


@torch.no_grad()
def predict(base,camera,y,seed,lrnet,nets,params,head,coef,choice):
    data=render(base,camera,y,seed);data["lr"]=y
    anchor=retained(data,head,coef)
    if choice["selected"]=="reference":return anchor
    mode=choice["selected"].split("/")[0]
    lrpred=lrnet(y)
    if mode=="local":
        b=y.new_zeros((1,49,*y.shape[-2:]));t=y.new_zeros((1,16,*y.shape[-2:]))
    else:
        sem=base.conditions(y)[1];bank=references(y,sem)
        t=F.interpolate(projected(sem,params),size=y.shape[-2:],mode="bilinear",align_corners=False)
        b,t=inputs(y,bank,t,mode)
    return compose(data,anchor,{mode:nets[mode](y,lrpred,b,t)},choice)


def main():
    splits=rows();base,camera=base_camera()
    lrnets,head,coef=model_parts();lrnet=lrnets["localnet"]
    params=basis(base,splits["train"]);dev=dev_pool(splits["dev"])
    store_tokens(base,splits["train"]+dev,params)
    nets={m:fit(m,splits["train"],lrnet,camera.metric) for m in MODES}
    rec,table=panel(base,camera,dev,"dev",6,lrnet,nets,params,head,coef,None)
    choice=choose(rec,table)
    if (PUB/"lock.json").exists() and choice!=json.loads((PUB/"lock.json").read_text())["choice"]:raise RuntimeError("Lock changed")
    write_json(PUB/"lock.json",dict(selected=choice["selected"],choice=choice,means=table))
    print("locked",choice,flush=True)
    from balanced import pools
    splits["additional_dreal"]=pools()[1]
    for group in ("real","dreal","new_real","new_dreal","additional_dreal"):
        seed=37 if group=="additional_dreal" else (29 if group.startswith("new_") else 6)
        rec,t=panel(base,camera,splits[group],group,seed,lrnet,nets,params,head,coef,choice)
        write_json(PUB/(group+".json"),dict(selected=choice["selected"],cases=len(rec),means=t,effects=effects(rec,choice["selected"])))
    y=image(ROOT/dev[0]["lr"])[None].cuda().float()/255
    out=predict(base,camera,y,6,lrnet,nets,params,head,coef,choice)
    save_png(out,PUB/"release_check.png")
    write_json(PUB/"release.json",dict(selected=choice["selected"],choice=choice,shape=list(out.shape)))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=("study","restore"))
    ap.add_argument("--input");ap.add_argument("--output");ap.add_argument("--seed",type=int,default=6)
    args=ap.parse_args();torch.set_num_threads(4);PUB.mkdir(exist_ok=True,parents=True)
    if args.action=="study":
        with (PUB/"study.lock").open("a+") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main()
    else:
        if not args.input or not args.output:ap.error("Provide --input and --output")
        y=image(args.input)[None].cuda().float()/255
        if y.shape!=(1,3,128,128):raise ValueError("Expected 128 x 128 RGB LR")
        base,camera=base_camera();lrnets,head,coef=model_parts()
        choice=json.loads((PUB/"release.json").read_text())["choice"]
        params=torch.load(PUB/"pca.pt",weights_only=True,map_location="cuda")
        save_png(predict(base,camera,y,args.seed,lrnets["localnet"],load_heads(),params,head,coef,choice),args.output)
