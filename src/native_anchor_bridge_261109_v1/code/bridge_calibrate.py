"""Lock only coarse and midband observation correction on V; confirm at seed 6 and on new T parents at seed 43."""
import fcntl
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F
from cache import ROOT,RUN,intervene,base_camera,lift,rows,image,write_json
from native_model import NativeCorrection
from bridge_trial import get_case,legacy_images,candidate,source,key,eligible,summarize,gallery
from texture import smooth,bands
from upstream import split_rows,render
from infer_v15 import dependencies
from evidence import evaluate_output
from structure import metrics,object_scores,BOXES
from balanced import case_path

OUT=RUN/"calibrated"
SPECS=[(target,m,w) for target in ("v14","source","camera") for m in (0.,.5,1.) for w in ("fixed","obs")]
def name(spec):return spec[0]+"_"+str(spec[1])+"_"+spec[2]

def correction(x,target,y,sensor,m,mode):
    delta=target-x
    fine,mid=bands(delta)
    direction=delta-fine-(1-m)*mid
    alpha=1.
    if mode=="obs":
        measured=sensor(x)
        v=sensor(x+direction)-measured
        num=smooth((v*(y-measured)).mean(1,keepdim=True),5)
        den=smooth(v.square().mean(1,keepdim=True),5)
        alpha=lift((num/(den+1e-5)).clamp(0,1),x)
    return x+alpha*direction

def fresh_rows():
    r=rows();used={v["scene_id"] for rs in r.values() for v in rs}
    for dom in ("RealSR","DRealSR"):
        p=ROOT/"texture_retention_261109_v15/transfer_scores"/("transfer17_"+dom)
        used.update(json.loads(f.read_text())["scene"] for f in p.glob("*.json"))
    manifest=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    result={}
    for domain,cams in (("RealSR",("Canon","Nikon")),("DRealSR",(None,))):
        pool=[]
        for cam in cams:
            unique={}
            for row in sorted(manifest,key=lambda r:r["id"]):
                if row["dataset"]==domain and row["role"]=="T" and row["scene_id"] not in used and (not cam or row["camera"]==cam):
                    unique.setdefault(row["scene_id"],row)
            seq=sorted(unique.values(),key=lambda row:hashlib.sha256(("A9-seed43/"+row["scene_id"]).encode()).digest())
            pool+=seq[:12//len(cams)]
        if len(pool)!=12:raise RuntimeError((domain,len(pool)))
        result["fresh_"+domain]=pool
    return result

@torch.no_grad()
def build(r,g,base,camera,assets):
    if not g.startswith("fresh"):
        d=get_case(r,g,base,camera,assets)
        d["camera"]=torch.load(case_path(g,r,6),weights_only=True,map_location="cpu")["camera"].cuda()
        return d
    y=image(ROOT/r["lr"])[None].cuda().float()/255
    p=intervene(base,camera,y,43)
    hist=render(base,camera,y,43)
    d=hist|p
    d.update(lr=y,hr=image(ROOT/r["HR"])[None].cuda().float()/255,source=source(y,p["sem"],assets))
    return d

@torch.no_grad()
def panel(g,rs,base,camera,assets,net,dists,spec,method,picked):
    rec=[]
    for i,r in enumerate(rs):
        f=OUT/"scores"/g/(r["scene_id"]+".json");f.parent.mkdir(parents=True,exist_ok=True)
        if f.exists():rec.append(json.loads(f.read_text()));continue
        d=build(r,g,base,camera,assets);y=d["lr"];hr=d["hr"]
        published,_=legacy_images(d,assets)
        x,res=candidate(d,net,method,"geometry",0.)
        ims={"native":d["native"],"full":d["full"],"v14":d["v14"],"v15":published,"stage1":x}
        for t,m,w in (SPECS if spec is None else [spec]):
            ims[name((t,m,w))]=correction(x,d[t],y,camera.sensor,m,w)
        if spec:
            t,m,w=spec
            for ctrl,start in (("no_bridge",candidate(d,net,method,"geometry",0.,"native")[0]),
                               ("no_regeneration",x-res),
                               ("full_bridge",candidate(d,net,method,"geometry",0.,"full")[0])):
                ims[ctrl]=correction(start,d[t],y,camera.sensor,m,w)
        scores={}
        for k,out in ims.items():
            v=evaluate_output(out,hr,camera.metric)
            v.update(metrics(out,hr),dists=float(dists(out.clamp(0,1),hr).mean()),obs=float((camera.sensor(out.clamp(0,1))-y).abs().mean()))
            scores[k]=v
        row={"scene":r["scene_id"],"camera":r.get("camera"),"scores":scores,"diagnosis":{}}
        if spec and r["scene_id"] in picked:
            meta=picked[r["scene_id"]];pref=meta["id"][:2]
            if pref in BOXES:
                row["object"]={k:object_scores(ims[k],hr,pref) for k in ("native","v15","stage1",name(spec),"no_regeneration")}
                gallery(meta,ims,d,OUT/"gallery",name(spec))
        write_json(f,row);rec.append(row)
        if (i+1)%12==0:print(g,i+1,len(rs),flush=True)
    return rec

def main():
    import pyiqa
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    fresh=fresh_rows()
    write_json(OUT/"fresh_manifest.json",{g:[r["scene_id"] for r in rs] for g,rs in fresh.items()})
    base,camera=base_camera();assets=dependencies()
    net=NativeCorrection().cuda().eval();net.load_state_dict(torch.load(RUN/"geometry.pt",weights_only=True))
    nets={"geometry":net};dists=pyiqa.create_metric("dists",device="cuda").eval().requires_grad_(False)
    splits=split_rows()
    picked={r["scene"]:r for r in json.loads((ROOT/"vosr_dual_visuals_261109_v1/display_manifest.json").read_text())["records"]}
    method=json.loads((RUN/"lock.json").read_text())["candidate"].split("_")[0]
    dev=panel("dev",splits["dev"],base,camera,assets,nets,dists,None,method,{})
    tab=summarize(dev)["means"]
    cams=[summarize([r for r in dev if r["camera"]==c])["means"] for c in ("Canon","Nikon")]
    candidates=[s for s in SPECS if all(eligible(t,name(s)) for t in cams)]
    valid=bool(candidates)
    spec=min(candidates or SPECS,key=lambda s:(tab[name(s)]["lpips"],tab[name(s)]["texerr"]))
    write_json(OUT/"lock.json",dict(spec=spec,eligible=valid,means=tab,method=method))
    print("locked",spec,valid,flush=True)
    all_groups={"dreal":splits["dreal"],"real":splits["real"]}|fresh
    passes=[]
    for g,rs in all_groups.items():
        rec=panel(g,rs,base,camera,assets,nets,dists,spec,method,picked)
        result=summarize(rec,name(spec))
        result["eligible"]=eligible(result["means"],name(spec))
        result["objects"]={r["scene"]:r["object"] for r in rec if "object" in r}
        write_json(OUT/(g+".json"),result);passes.append(result["eligible"])
    write_json(OUT/"result.json",dict(spec=spec,validation=valid,passes=passes,promote=valid and all(passes)))

if __name__=="__main__":
    with (RUN/"calibrate.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
