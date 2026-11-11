"""Use the published estimate as an optional observation correction of the native branch."""
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cache import ROOT,RUN,base_camera,intervene,rows,save_tensor,write_json,image
from native_model import NativeCorrection
from bridge_trial import (get_case,legacy_images,source,candidate,eligible,summarize,gallery,read)
from bridge_calibrate import fresh_rows
from evidence import evaluate_output
from structure import metrics,object_scores,BOXES
from texture import bands,smooth,luminance
from upstream import render,split_rows
from infer_v15 import dependencies

OUT=RUN/"routed"
SPECS=[(c,f,act) for c in (.6,1.) for f in (.25,.5,.75) for act in ("uniform","evidence")]

def name(spec):return "_".join(str(v) for v in spec)

def routed(native,proposal,source,spec,control=None):
    coarse,fine,act=spec
    nf,nm=bands(native);pf,pm=bands(proposal);sf,sm=bands(source)
    low=proposal-native-(pf-nf)-(pm-nm)
    out=native+coarse*low
    maps=[]
    for a,b,s,w in ((nf,pf,sf,fine),(nm,pm,sm,(coarse+fine)*.5)):
        if act=="uniform":gate=torch.full_like(s[:,:1],w)
        else:
            # Relative signed error against the LR-predicted local phase, normalized by joint contrast.
            er=smooth((luminance(a-s)).square(),9)
            ep=smooth((luminance(b-s)).square(),9)
            vote=(er-ep)/(er+ep+.000064)
            gate=(w+.6*vote).clamp(0,1)
        if control=="shift":gate=torch.roll(gate,(native.shape[-2]//2,native.shape[-1]//2),(-2,-1))
        if control=="mean":gate=gate.mean((-1,-2),keepdim=True)
        maps.append(gate)
        out=out+gate*(b-a)
    return out,maps

def select_rows():
    data=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    used={r["scene_id"] for r in split_rows()["dev"]}
    rs=[]
    for c in ("Canon","Nikon"):
        u={}
        for r in sorted(data,key=lambda a:a["id"]):
            if r["dataset"]=="RealSR" and r["role"]=="V" and r["camera"]==c and r["scene_id"] not in used:u.setdefault(r["scene_id"],r)
        rs+=sorted(u.values(),key=lambda r:hashlib.sha256(("A9 route/"+r["scene_id"]).encode()).digest())[:12]
    assert len(rs)==24
    return rs

@torch.no_grad()
def stage(row,group,base,camera,assets,nets,method):
    path=OUT/"cache"/group/(row["scene_id"]+".pt")
    if path.exists():return {k:v.cuda() for k,v in read(path).items()}
    if group in ("real","dreal"):d=get_case(row,group,base,camera,assets)
    else:
        seed=43 if group.startswith("fresh") else 6
        lr=image(ROOT/row["lr"])[None].cuda().float()/255
        d=render(base,camera,lr,seed)|intervene(base,camera,lr,seed)
        d.update(lr=lr,source=source(lr,d["sem"],assets),hr=image(ROOT/row["HR"])[None].cuda().float()/255)
    old,_=legacy_images(d,assets)
    x,res=candidate(d,nets,method,"geometry",0.)
    absent=candidate(d,nets,method,"geometry",0.,"native")[0]
    full=candidate(d,nets,method,"geometry",0.,"full")[0]
    out={k:d[k] for k in ("hr","lr","source","full","native","v14")}
    out.update(x=x,v15=old,without=x-res,absent=absent,post=full)
    path.parent.mkdir(exist_ok=True,parents=True);save_tensor(path,{k:v.cpu() for k,v in out.items()})
    return out

@torch.no_grad()
def panel(group,rs,base,camera,assets,net,method,dists,spec,picked):
    rec=[]
    for i,r in enumerate(rs):
        f=OUT/"scores"/group/(r["scene_id"]+".json");f.parent.mkdir(parents=True,exist_ok=True)
        if f.exists():rec.append(json.loads(f.read_text()));continue
        d=stage(r,group,base,camera,assets,net,method);old=d["v15"];x=d["x"];src=d["source"]
        ims={k:d[k] for k in ("native","full","v14","v15","x")}
        for opt in (SPECS if spec is None else [spec]):
            ims[name(opt)],gs=routed(x,old,src,opt)
        if spec:
            ims["no_bridge"]=routed(d["absent"],old,src,spec)[0]
            ims["full_bridge"]=routed(d["post"],old,src,spec)[0]
            ims["no_regeneration"]=routed(d["without"],old,src,spec)[0]
            ims["shift"]=routed(x,old,src,spec,"shift")[0]
            ims["mean"]=routed(x,old,src,spec,"mean")[0]
            ims["equal"]=.5*x+.5*old
        scores={}
        for k,out in ims.items():
            v=evaluate_output(out,d["hr"],camera.metric)
            v.update(metrics(out,d["hr"]),dists=float(dists(out.clamp(0,1),d["hr"]).mean()),
                     obs=float((camera.sensor(out.clamp(0,1))-d["lr"]).abs().mean()))
            scores[k]=v
        row=dict(scene=r["scene_id"],camera=r.get("camera"),scores=scores,diagnosis={})
        if spec:
            row["diagnosis"]={"fine_correction_weight":float(gs[0].mean()),"mid_correction_weight":float(gs[1].mean())}
        meta=picked.get(r["scene_id"])
        if meta and spec and meta["id"][:2] in BOXES:
            label=name(spec);pref=meta["id"][:2]
            row["object"]={k:object_scores(ims[k],d["hr"],pref) for k in ("native","v15","x",label,"no_regeneration")}
            gallery(meta,ims,d,OUT/"gallery",label)
        write_json(f,row);rec.append(row)
        if (i+1)%12==0:print(group,i+1,len(rs),flush=True)
    return rec

def strict(table,cand):
    a,b=table[cand],table["v15"]
    return eligible(table,cand) and a["psnr"]>=b["psnr"]-.03 and a["dists"]<=b["dists"] and a["fine_ratio"]>=min(1,b["fine_ratio"])-.01

def main():
    import pyiqa
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    dev=select_rows();fresh=fresh_rows()
    write_json(OUT/"manifest.json",{"validation":[r["scene_id"] for r in dev]}|{g:[r["scene_id"] for r in rs] for g,rs in fresh.items()})
    base,camera=base_camera();assets=dependencies()
    net=NativeCorrection().cuda().eval();net.load_state_dict(read(RUN/"geometry.pt"));nets={"geometry":net}
    method=json.loads((RUN/"lock.json").read_text())["candidate"].split("_")[0]
    dists=pyiqa.create_metric("dists",device="cuda").eval().requires_grad_(False)
    rec=panel("dev",dev,base,camera,assets,nets,method,dists,None,{})
    table=summarize(rec)["means"]
    cams=[summarize([r for r in rec if r["camera"]==c])["means"] for c in ("Canon","Nikon")]
    pool=[s for s in SPECS if all(strict(t,name(s)) for t in cams)]
    spec=min(pool or SPECS,key=lambda s:(table[name(s)]["lpips"],table[name(s)]["texerr"]))
    valid=bool(pool)
    write_json(OUT/"lock.json",dict(spec=spec,validation=valid,method=method,means=table))
    print("locked",spec,valid,flush=True)
    groups=split_rows();groups={"real":groups["real"],"dreal":groups["dreal"]}|fresh
    picked={r["scene"]:r for r in json.loads((ROOT/"vosr_dual_visuals_261109_v1/display_manifest.json").read_text())["records"]}
    passes={}
    for g,rs in groups.items():
        rec=panel(g,rs,base,camera,assets,nets,method,dists,spec,picked)
        result=summarize(rec,name(spec))
        result["objects"]={r["scene"]:r["object"] for r in rec if "object" in r}
        passes[g]=eligible(result["means"],name(spec))
        result["eligible"]=passes[g]
        write_json(OUT/(g+".json"),result)
    write_json(OUT/"result.json",dict(spec=spec,validation=valid,passes=passes,promote=valid and all(passes.values())))
if __name__=="__main__":
    with (RUN/"route.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
