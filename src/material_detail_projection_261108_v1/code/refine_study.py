"""Optimize detail along the observed material direction and lock the release on validation."""
import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path

UP=Path("/home/zhangxu/hx/HX001/material_direction_flow_261108_v1")
sys.path.append(str(UP/"code"))
from study import (ROOT,CAM,Camera,FrozenVOSR,F,finish,group,image,measure,np,sha,start,synth,torch,write_json)
from detail import coefficients, refine

CODE=Path(__file__).resolve().parent
Q=json.loads((CODE/"refinement.json").read_text())
RUN=ROOT/Q["run"]
SOURCE=("native_camera","visual_camera","visual")


def choose_rows(domain,stage):
    images=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    if domain=="dreal":
        rows=[r for r in images if r["dataset"]=="DRealSR" and r["role"]=="E"]
        assert len({r["scene_id"] for r in rows})==len(rows)==15
        return rows
    excluded={p.stem for p in (CAM/"new20").glob("*.pt")}
    assert len(excluded)==20
    excluded_scenes={r["scene_id"] for r in images if r["id"] in excluded}
    assert len(excluded_scenes)==20
    selected=[]
    for camera in ("Canon","Nikon"):
        unique={}
        for r in sorted(images,key=lambda r:r["id"]):
            if r["dataset"]=="RealSR" and r.get("camera")==camera and r["role"]==("V" if stage=="dev" else "E"):
                unique.setdefault(r["scene_id"],r)
        if stage=="dev":
            chosen=list(unique.values())[3:9]
        else:
            ranked=sorted(set(unique)-excluded_scenes,key=lambda k:hashlib.sha256(k.encode()).hexdigest())
            chosen=[unique[k] for k in ranked[:12]]
        assert len(chosen)==(6 if stage=="dev" else 12)
        selected.extend(chosen)
    return selected


def cases(domain,stage):
    if domain=="synth":
        for n in Q[domain+"_"+stage]:
            lr,hr,curve,turn=synth(n)
            for seed in (Q["seeds"][:1] if stage=="dev" else Q["seeds"]):
                yield str(n),seed,lr,hr,(curve,turn)
    else:
        for r in choose_rows(domain,stage):
            lr,hr=(image(ROOT/r[k])[None].cuda().float()/255 for k in ("lr","HR"))
            for seed in ([6] if stage=="dev" or domain=="dreal" else Q["real_seeds"]):
                yield r["scene_id"],seed,lr,hr,None


@torch.no_grad()
def texture_score(rgb,lr,hr,curve,turn):
    scores=measure(rgb,lr,hr,curve,turn)
    def gradients(img):
        g=img.rot90(-turn,(-2,-1)).mean(1)[0]
        return ((g.roll(-1,1)-g.roll(1,1))*.5,(g.roll(-1,0)-g.roll(1,0))*.5)
    x=torch.arange(512,device=rgb.device)[None,:]-curve[:,None]
    band=(x.abs()>=4)&(x.abs()<=24)
    band[:4]=band[-4:]=False
    def normal(img):
        gx,gy=gradients(img)
        return torch.where(x<0,gx,gy)[band]
    a,b=normal(rgb),normal(hr)
    scores.update(amplitude_ratio=float(a.norm()/b.norm()), signed_correlation=float(F.cosine_similarity(a,b,dim=0)))
    return scores


@torch.no_grad()
def sources(base,camera,lr,seed,domain):
    real=domain!="synth"
    key="real" if real else "synth"
    sel=json.loads((UP/key/"choice.json").read_text())["selected"]
    cfg=-.5 if real else .5
    state=start(base,lr,seed,cfg)
    sensor=camera.sensor if real else lambda x:F.avg_pool2d(x,4)
    native=finish(base,state,lr,cfg,"native",0,0,sensor)
    guided=finish(base,state,lr,cfg,"visual",sel["scale"],sel["weight"],sensor)
    images=dict(native=base.decode(native),visual=base.decode(guided))
    if real:
        images["native_camera"]=camera.pipeline(base,native,lr)
        images["visual_camera"]=camera.pipeline(base,guided,lr)
    return state[1],images


def rows_for(base,camera,domain,stage,choice=None):
    records=[]
    folder=RUN/domain/stage
    for name,seed,lr,hr,truth in cases(domain,stage):
        path=folder/(hashlib.sha256(name.encode()).hexdigest()[:20]+f"_{seed}.json")
        if not path.exists():
            sem,src=sources(base,camera,lr,seed,domain)
            scores={}
            def score(rgb):
                return (texture_score(rgb,lr,hr,*truth) if truth else camera.scores(rgb,hr,lr))
            for kind,x in src.items():
                scores[kind]=score(x)
            if stage=="dev":
                jobs=[(s,"visual",d) for s in (SOURCE if truth is None else ("visual",)) for d in Q["depths"]]
            else:
                jobs=[(choice["source"],m,d) for m,d in
                      (("visual",0),("visual",choice["steps"]),("local",choice["steps"]),("misplaced",choice["steps"]))]
            coeff={}
            for source,mode,depth in jobs:
                key=f"{source}/{mode}/{depth}"
                if key in scores:continue
                if mode not in coeff:coeff[mode]=coefficients(lr,sem,mode)
                restored=refine(src[source],lr,coeff[mode],depth,truth is not None)
                scores[key]=score(restored)
                if truth is None and depth>0:
                    from detail import dc
                    scores[key]["dc_drift"]=float((dc(restored)-dc(src[source])).abs().max())
            write_json(path,dict(scene=name,seed=seed,scores=scores))
        records.append(json.loads(path.read_text()))
        print(domain,stage,len(records),flush=True)
    means={k:group([r["scores"][k] for r in records]) for k in records[0]["scores"]}
    return records,means


def select(domain,means):
    candidates=[k for k in means if "/" in k]
    if domain=="synth":
        eligible=[k for k in candidates if means[k]["psnr"]>=means["native"]["psnr"] and
                  means[k]["boundary"]<=means["native"]["boundary"]+.25]
        best=min(eligible,key=lambda k:means[k]["leakage"]) if eligible else None
        if best is None:raise ValueError("No eligible synthetic candidate")
    else:
        eligible=[k for k in candidates if means[k]["lpips"]<=means["native_camera"]["lpips"]+.002]
        best=max(eligible,key=lambda k:means[k]["psnr"])
    source,_,depth=best.split("/")
    return dict(source=source,steps=int(depth),method="visual",key=best)


def delta(rows,selected,reference,field):
    groups={}
    for row in rows:
        scores=row["scores"]
        groups.setdefault(row["scene"],[]).append(scores[selected][field]-scores[reference][field])
    values=np.array([np.mean(v) for v in groups.values()])
    ci=np.quantile(np.random.default_rng(261108).choice(values,(5000,len(values))).mean(1),[.025,.975])
    return [float(values.mean()),*ci.tolist()]


def confirm(base,camera,domain,choice):
    rows,means=rows_for(base,camera,domain,"confirm",choice)
    key=choice["key"]
    effects={other:{f:delta(rows,key,other,f) for f in means[key] if f!="dc_drift"}
             for other in means if other!=key}
    write_json(RUN/domain/"summary.json",dict(selected=choice,parents=len({r["scene"] for r in rows}),
               cases=len(rows),means=means,effects=effects))


def restore(base,camera,domain,lr,seed):
    key="real" if domain!="synth" else domain
    selected=json.loads((RUN/key/"choice.json").read_text())["selected"]
    if lr.shape!=(1,3,128,128):raise ValueError("Use native 128 × 128 RGB LR")
    # Only run the chosen trajectory and requested terminal decoder.
    cfg=-.5 if key=="real" else .5
    flow=json.loads((UP/key/"choice.json").read_text())["selected"]
    st=start(base,lr,seed,cfg)
    method="visual" if selected["source"].startswith("visual") else "native"
    sensor=camera.sensor if camera else lambda x:F.avg_pool2d(x,4)
    z=finish(base,st,lr,cfg,method,flow["scale"],flow["weight"],sensor)
    with torch.no_grad():
        source=camera.pipeline(base,z,lr) if selected["source"].endswith("camera") else base.decode(z)
        return refine(source,lr,coefficients(lr,st[1],"visual"),selected["steps"],key=="synth")


def save_image(output,path):
    from PIL import Image
    pixels=output[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(pixels).save(path)


def main():
    base=FrozenVOSR()
    base.use_checkpoint=False
    camera=None
    for domain in ("synth","real","dreal"):
        if domain=="real":camera=Camera()
        if domain!="dreal":
            _,means=rows_for(base,camera,domain,"dev")
            choice=select(domain,means)
            path=RUN/domain/"choice.json"
            if path.exists() and json.loads(path.read_text())["selected"]!=choice:raise ValueError("Lock changed")
            write_json(path,dict(selected=choice,dev=means))
        confirm(base,camera,domain,choice)
    row=choose_rows("real","dev")[0]
    y=image(ROOT/row["lr"])[None].cuda().float()/255
    test=restore(base,camera,"real",y,6)
    path=RUN/"release_check.png"
    save_image(test,path)
    release=dict(base_sha=sha(base.weight_path), code_sha={k:sha(CODE/k) for k in ("detail.py","refine_study.py","refinement.json")},
                 camera_sha=sha(CAM/"real_camera_adapter.pt"),flow_run=str(UP),
                 presets={d:json.loads((RUN/d/"choice.json").read_text())["selected"] for d in ("real","synth")},
                 smoke_shape=list(test.shape),smoke_output_sha=sha(path),
                 smoke_metrics=camera.scores(test,image(ROOT/row["HR"])[None].cuda().float()/255,y))
    write_json(RUN/"release.json",release)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("study","restore"))
    parser.add_argument("--input");parser.add_argument("--output")
    parser.add_argument("--domain",choices=("real","synth"),default="real")
    parser.add_argument("--seed",type=int,default=6)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.action=="study":
        RUN.mkdir(parents=True,exist_ok=True)
        with (RUN/"lock").open("a+") as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main()
    else:
        if not args.input or not args.output:parser.error("Provide --input and --output")
        base=FrozenVOSR();base.use_checkpoint=False
        camera=Camera() if args.domain=="real" else None
        y=image(args.input)[None].cuda().float()/255
        save_image(restore(base,camera,args.domain,y,args.seed),args.output)
