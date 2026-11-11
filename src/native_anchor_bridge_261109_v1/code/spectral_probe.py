"""Test an orthogonal low-frequency bridge proposal with actual sensor acceptance."""
import fcntl,json,time
import numpy as np
import torch
from cache import ROOT,RUN,base_camera,write_json
from native_model import NativeCorrection,foundation
from bridge_trial import get_case,read
from texture import bands
from upstream import split_rows
from infer_v15 import dependencies
from evidence import evaluate_output
from structure import metrics
OUT=RUN/"spectral_probe"

def orthogonal_low(x,cutoff=.125):
    h,w=x.shape[-2:]
    fy=torch.fft.fftfreq(h,device=x.device)[:,None]
    fx=torch.fft.rfftfreq(w,device=x.device)[None,:]
    mask=(fy.abs()<=cutoff)&(fx<=cutoff)
    return torch.fft.irfft2(torch.fft.rfft2(x,norm="ortho")*mask,s=(h,w),norm="ortho")

@torch.no_grad()
def accepted(x,direction,y,sensor):
    opts=[x+t*direction for t in (0.,.25,.5,.75,1.)]
    errors=torch.stack([(sensor(v.clamp(0,1))-y).square().mean() for v in opts])
    k=int(errors.argmin())
    return opts[k],k*.25,errors[0],errors[k]

@torch.no_grad()
def main():
    import pyiqa
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    while not (RUN/"refit/result.json").exists():time.sleep(10)
    base,camera=base_camera();assets=dependencies()
    net=NativeCorrection().cuda().eval();net.load_state_dict(read(RUN/"geometry.pt"))
    dists=pyiqa.create_metric("dists",device="cuda").eval().requires_grad_(False)
    groups=split_rows()
    for g in ("dev","real","dreal"):
        rec=[]
        for r in groups[g]:
            f=OUT/"scores"/g/(r["scene_id"]+".json");f.parent.mkdir(parents=True,exist_ok=True)
            if f.exists():rec.append(json.loads(f.read_text()));continue
            d=get_case(r,g,base,camera,assets);raw=d["native"];src=d["source"];y=d["lr"]
            x=foundation(raw,raw,src,"conf")
            original=foundation(raw,d["pre"],src,"conf")
            direction=orthogonal_low(original-x)
            moved,alpha,before,after=accepted(x,direction,y,camera.sensor)
            high_delta=(moved-x)-orthogonal_low(moved-x)
            clipped=moved.clamp(0,1)-x.clamp(0,1)
            row=dict(scene=r["scene_id"],alpha=alpha,obs_before=float(before),obs_after=float(after),
                     high_delta=float(high_delta.abs().max()),
                     clipped_high=float((clipped-orthogonal_low(clipped)).square().mean().sqrt()))
            ims={"no_bridge":net(x,y,src)[0],"previous_bridge":net(original,y,src)[0],"accepted_bridge":net(moved,y,src)[0]}
            row["scores"]={}
            for k,out in ims.items():
                m=evaluate_output(out,d["hr"],camera.metric)
                m.update(metrics(out,d["hr"]),dists=float(dists(out.clamp(0,1),d["hr"]).mean()))
                row["scores"][k]=m
            write_json(f,row);rec.append(row)
        means={k:{m:float(np.mean([r["scores"][k][m] for r in rec])) for m in rec[0]["scores"][k]} for k in rec[0]["scores"]}
        write_json(OUT/(g+".json"),dict(n=len(rec),means=means,
           alpha=float(np.mean([r["alpha"] for r in rec])),
           rejected=sum(r["alpha"]==0 for r in rec),
           increased=sum(r["obs_after"]>r["obs_before"]+1e-12 for r in rec),
           high_delta=max(r["high_delta"] for r in rec),
           clipped_high=float(np.mean([r["clipped_high"] for r in rec]))))
    write_json(OUT/"result.json",{"complete":True})
if __name__=="__main__":
    with (RUN/"spectral_probe.lock").open("a+") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
