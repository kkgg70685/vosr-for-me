"""Match training registration to inference and fit a fidelity-weighted native correction."""
import fcntl
import hashlib
import json
import time
import numpy as np
import torch
from cache import ROOT,RUN,base_camera,write_json,save_tensor,image,intervene
from native_model import NativeCorrection,foundation,structure_loss
from bridge_trial import get_case,source,legacy_images,read
from route import routed,name,strict
from bridge_calibrate import fresh_rows
from upstream import split_rows,render
from evidence import rows,evaluate_output
from infer_v15 import dependencies
from texture import texture_loss
from structure import metrics,BOXES,object_scores
OUT=RUN/"refit"
def train(samples,camera,weight):
    torch.manual_seed(29061)
    net=NativeCorrection().cuda()
    net.load_state_dict(read(RUN/"geometry.pt"))
    path=OUT/(weight+".pt")
    if path.exists():net.load_state_dict(read(path));return net.eval()
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
    for step in range(1200):
        rng=np.random.default_rng(102101+step);batch={k:[] for k in ("x","lr","source","hr")}
        for _ in range(6):
            d=samples[rng.integers(len(samples))]
            y,x=4*rng.integers(0,107,2)
            for k,v in batch.items():
                q=4 if k=="lr" else 1
                v.append(d[k][...,y//q:(y+88)//q,x//q:(x+88)//q])
        d={k:torch.cat(v).cuda() for k,v in batch.items()}
        out,_=net(d["x"],d["lr"],d["source"])
        p=out[...,8:-8,8:-8].clamp(0,1);hr=d["hr"][...,8:-8,8:-8]
        loss=4*(p-hr).abs().mean()+.08*texture_loss(p,hr)+.08*structure_loss(p,hr)
        loss=loss+(.6 if weight=="fidelity" else 1.)*camera.metric.net(p,hr).mean()
        if weight=="fidelity":loss=loss+16*(p-hr).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),2,error_if_nonfinite=True);opt.step()
        if (step+1)%400==0:print(weight,step+1,float(loss.detach()),flush=True)
    save_tensor(path,net.state_dict());return net.eval()

def new_parents():
    old=fresh_rows();used={r["scene_id"] for rs in rows().values() for r in rs}
    used.update(r["scene_id"] for rs in old.values() for r in rs)
    manifest=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    result={}
    for domain,cams in (("RealSR",("Canon","Nikon")),("DRealSR",(None,))):
        pool=[]
        for c in cams:
            u={}
            for r in sorted(manifest,key=lambda r:r["id"]):
                if r["dataset"]==domain and r["role"]=="T" and r["scene_id"] not in used and (c is None or r["camera"]==c):u.setdefault(r["scene_id"],r)
            seq=sorted(u.values(),key=lambda r:hashlib.sha256(("A9 new61/"+r["scene_id"]).encode()).digest())
            pool+=seq[:12//len(cams)]
        if len(pool)!=12:raise RuntimeError((domain,len(pool)))
        result["new_"+domain]=pool
    return result

@torch.no_grad()
def data_for(row,group,base,camera,assets):
    if group in ("dev","real","dreal","train"):return get_case(row,group,base,camera,assets)
    path=OUT/"cache"/group/(row["scene_id"]+".pt")
    if path.exists():return {k:v.cuda() for k,v in read(path).items()}
    y=image(ROOT/row["lr"])[None].cuda().float()/255
    d=render(base,camera,y,61)|intervene(base,camera,y,61)
    d.update(lr=y,hr=image(ROOT/row["HR"])[None].cuda().float()/255)
    d["source"]=source(y,d["sem"],assets)
    path.parent.mkdir(parents=True,exist_ok=True);save_tensor(path,{k:v.cpu() for k,v in d.items()})
    return d

@torch.no_grad()
def predictions(d,nets,mode):
    x=foundation(d["native"],d["pre"],d["source"],mode)
    ims={}
    for k,net in nets.items():ims[k]=net(x,d["lr"],d["source"])[0]
    return ims

@torch.no_grad()
def panel(group,rs,base,camera,assets,nets,mode,dists,lock=None):
    rec=[]
    for i,r in enumerate(rs):
        f=OUT/"scores"/group/(r["scene_id"]+".json");f.parent.mkdir(parents=True,exist_ok=True)
        if f.exists():rec.append(json.loads(f.read_text()));continue
        d=data_for(r,group,base,camera,assets);old,_=legacy_images(d,assets)
        ims={"native":d["native"],"v15":old}|predictions(d,nets,mode)
        # Independent pure model choice, plus an optional evidence correction path.
        specs=[(k,(1.,.25,"evidence")) for k in nets] if lock is None else [(lock["head"],tuple(lock["spec"]))]
        for k,spec in specs:ims[k+"_routed"]=routed(ims[k],old,d["source"],spec)[0]
        scores={}
        for k,out in ims.items():
            v=evaluate_output(out,d["hr"],camera.metric)
            v.update(metrics(out,d["hr"]),dists=float(dists(out.clamp(0,1),d["hr"]).mean()))
            scores[k]=v
        write_json(f,dict(scene=r["scene_id"],camera=r.get("camera"),scores=scores))
        rec.append(dict(scene=r["scene_id"],camera=r.get("camera"),scores=scores))
        if (i+1)%12==0:print(group,i+1,len(rs),flush=True)
    return rec

def table(rec):
    return {k:{m:float(np.mean([r["scores"][k][m] for r in rec])) for m in rec[0]["scores"][k]} for k in rec[0]["scores"]}

def main():
    import pyiqa
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    while not (RUN/"routed/result.json").exists():time.sleep(10)
    base,camera=base_camera();assets=dependencies()
    mode=json.loads((RUN/"lock.json").read_text())["candidate"].split("_")[0]
    splits=split_rows();samples=[]
    for r in splits["train"]:
        d=get_case(r,"train",base,camera,assets)
        x=foundation(d["native"],d["pre"],d["source"],mode)
        samples.append({k:d[k].cpu() for k in ("lr","hr","source")}|{"x":x.cpu()})
    nets={w:train(samples,camera,w) for w in ("matched","fidelity")}
    dists=pyiqa.create_metric("dists",device="cuda").eval().requires_grad_(False)
    rec=panel("dev",splits["dev"],base,camera,assets,nets,mode,dists)
    tab=table(rec);cams=[table([r for r in rec if r["camera"]==c]) for c in ("Canon","Nikon")]
    pool=[k for k in ("matched","fidelity","matched_routed","fidelity_routed") if all(strict(t,k) for t in cams)]
    chosen=min(pool or ("matched","fidelity","matched_routed","fidelity_routed"),key=lambda k:tab[k]["lpips"])
    lock=dict(head=chosen.split("_")[0],spec=[1.,.25,"evidence"],selected=chosen,validation=bool(pool),mode=mode)
    write_json(OUT/"lock.json",dict(choice=lock,means=tab));print("locked",lock,flush=True)
    fresh=new_parents()
    write_json(OUT/"manifest.json",{g:[r["scene_id"] for r in rs] for g,rs in fresh.items()})
    for g,rs in ({"real":splits["real"],"dreal":splits["dreal"]}|fresh).items():
        rec=panel(g,rs,base,camera,assets,nets,mode,dists,lock)
        write_json(OUT/(g+".json"),dict(n=len(rec),means=table(rec)))
    write_json(OUT/"result.json",lock)
if __name__=="__main__":
    with (RUN/"refit.lock").open("a+") as l:
        fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
