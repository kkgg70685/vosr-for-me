"""Same-seed native/guided mixtures. All image and scene records stay on the server."""
import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path("/home/zhangxu/hx/HX001")
PREV=ROOT/"material_detail_projection_261108_v1"
sys.path.insert(0,str(PREV/"code"))
from refine_study import Camera,FrozenVOSR,F,choose_rows,coefficients,finish,group,image,np,refine,save_image,start,torch,write_json
CODE=Path(__file__).resolve().parent
P=json.loads((CODE/"mix_protocol.json").read_text())
RUN=ROOT/P["run"]


@torch.no_grad()
def endpoints(model,camera,y,seed):
    state=start(model,y,seed,-.5)
    coeff=coefficients(y,state[1],"visual")
    result={}
    for method in ("native","visual"):
        z=finish(model,state,y,-.5,method,.04,8,camera.sensor)
        rgb=camera.pipeline(model,z,y)
        if method=="native": result["v13"]=rgb
        result[method]=refine(rgb,y,coeff,24,False)
    return result,state[1]


def pooled(x):
    return F.avg_pool2d(F.pad(x,(2,)*4,mode="replicate"),5,stride=1)


@torch.no_grad()
def gates(y,sem,sensor,ends):
    a=sensor(ends["native"])
    diff=sensor(ends["visual"])-a
    num=pooled(((y-a)*diff).mean(1,keepdim=True))
    denom=pooled(diff.square().mean(1,keepdim=True))
    desc=F.normalize(sem-sem.mean(1,keepdim=True),dim=-1)
    scores,inds=(desc@desc.transpose(1,2)).topk(32,dim=2)
    def vote(x):
        tokens=F.avg_pool2d(x,4).flatten(2).transpose(1,2)
        votes=(tokens[0][inds]*(10*scores).softmax(2)[...,None]).sum(2)
        return F.interpolate(votes.transpose(1,2).reshape(1,1,32,32),size=y.shape[-2:],mode="bilinear",align_corners=False)
    return (num,denom,vote(num),vote(denom))


def mix(ends,fields,kind,center):
    if kind=="fixed":
        return torch.lerp(ends["native"],ends["visual"],center)
    num,denom,vnum,vdenom=fields
    if kind in ("visual","shifted"):
        if kind=="shifted":
            vnum=vnum.roll((64,64),(-2,-1))
            vdenom=vdenom.roll((64,64),(-2,-1))
        num=(num+vnum)*.5
        denom=(denom+vdenom)*.5
    alpha=((num+P["ridge"]*center)/(denom+P["ridge"])).clamp(0,1)
    alpha=F.interpolate(alpha,size=ends["native"].shape[-2:],mode="bilinear",align_corners=False)
    return torch.lerp(ends["native"],ends["visual"],alpha)


def selection_rows(stage):
    allrows=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    if stage=="dreal":return choose_rows("dreal","confirm")
    old_ids={p.stem for p in (ROOT/"camera_feedback_261108_v1/new20").glob("*.pt")}
    excluded={r["scene_id"] for r in allrows if r["id"] in old_ids}
    excluded.update(r["scene_id"] for r in choose_rows("real","confirm"))
    assert len(excluded)==44
    rows=[]
    for camera in ("Canon","Nikon"):
        unique={}
        for r in sorted(allrows,key=lambda row:row["id"]):
            if r["dataset"]=="RealSR" and r["role"]==("V" if stage=="dev" else "E") and r["camera"]==camera:
                unique.setdefault(r["scene_id"],r)
        rows.extend(list(unique.values())[9:15] if stage=="dev" else [r for k,r in unique.items() if k not in excluded])
    assert len(rows)==(12 if stage=="dev" else 56)
    return rows


def get_candidates(locked):
    if locked is None:
        return [(k,c) for k in ("fixed","local","visual") for c in (P["scale_grid"] if k=="fixed" else P["adaptive_centers"])]
    return [(k,float(label.split("/")[1])) for k,label in locked["families"].items()]+[
        ("shifted",float(locked["families"]["visual"].split("/")[1]))]


def panel(model,camera,stage,locked=None):
    rows=[]
    for r in selection_rows(stage):
        file=RUN/stage/(hashlib.sha256(r["scene_id"].encode()).hexdigest()[:20]+".json")
        if not file.exists():
            y,hr=(image(ROOT/r[k])[None].cuda().float()/255 for k in ("lr","HR"))
            ends,sem=endpoints(model,camera,y,P["seed"])
            fields=gates(y,sem,camera.sensor,ends)
            metrics={k:camera.scores(v,hr,y) for k,v in ends.items()}
            for kind,center in get_candidates(locked):
                metrics[f"{kind}/{center:g}"]=camera.scores(mix(ends,fields,kind,center),hr,y)
            write_json(file,dict(scene=r["scene_id"],camera=r.get("camera"),scores=metrics))
        rows.append(json.loads(file.read_text()))
        print(stage,len(rows),flush=True)
    means={k:group([row["scores"][k] for row in rows]) for k in rows[0]["scores"]}
    return rows,means


def choose(means):
    limit=min(means["v13"]["lpips"]+.001,min(means[k]["lpips"] for k in ("native","visual"))+.002)
    def best(options):
        feasible=[k for k in options if means[k]["lpips"]<=limit]
        return max(feasible,key=lambda k:means[k]["psnr"]) if feasible else min(options,key=lambda k:means[k]["lpips"])
    families={k:best([m for m in means if m.startswith(k+"/")]) for k in ("fixed","local","visual")}
    return dict(families=families,winner=best(["v13",*families.values()]),lpips_limit=limit)


def summarize(rows,means,lock):
    winner=lock["winner"]
    effects={}
    for key in means:
        diffs={}
        for f in ("psnr","lpips"):
            ds=np.array([row["scores"][winner][f]-row["scores"][key][f] for row in rows])
            ci=np.quantile(np.random.default_rng(261108).choice(ds,(5000,len(ds))).mean(1),[.025,.975])
            diffs[f]=[float(ds.mean()),*ci.tolist()]
        effects[key]=diffs
    return dict(scenes=len(rows),winner=winner,means=means,effects=effects)


def restored(model,camera,y,seed,winner):
    if winner=="v13":
        state=start(model,y,seed,-.5)
        z=finish(model,state,y,-.5,"native",0,0,camera.sensor)
        return camera.pipeline(model,z,y)
    ends,sem=endpoints(model,camera,y,seed)
    k,c=winner.split("/")
    fields=None if k=="fixed" else gates(y,sem,camera.sensor,ends)
    return mix(ends,fields,k,float(c))


def study(model,camera):
    _,dev=panel(model,camera,"dev")
    lock=choose(dev)
    if (RUN/"choice.json").exists() and json.loads((RUN/"choice.json").read_text())["choice"]!=lock:
        raise RuntimeError("Validation lock changed")
    write_json(RUN/"choice.json",dict(choice=lock,means=dev))
    for stage in ("confirm","dreal"):
        rows,means=panel(model,camera,stage,lock)
        write_json(RUN/(stage+"_summary.json"),summarize(rows,means,lock))
    r=selection_rows("dev")[0]
    y=image(ROOT/r["lr"])[None].cuda().float()/255
    out=restored(model,camera,y,P["seed"],lock["winner"])
    save_image(out,RUN/"release_check.png")
    write_json(RUN/"release.json",dict(winner=lock["winner"],source_run=str(PREV),seed=P["seed"],
              code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),smoke_shape=list(out.shape)))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=("study","restore"));ap.add_argument("--input");ap.add_argument("--output");ap.add_argument("--seed",type=int,default=6)
    args=ap.parse_args()
    torch.set_num_threads(4)
    model=FrozenVOSR();model.use_checkpoint=False
    camera=Camera()
    if args.action=="study":
        RUN.mkdir(parents=True,exist_ok=True)
        with (RUN/"lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            study(model,camera)
    else:
        if not args.input or not args.output:ap.error("Provide --input and --output")
        y=image(args.input)[None].cuda().float()/255
        if y.shape!=(1,3,128,128):raise ValueError("Expected native 128 × 128 RGB")
        save_image(restored(model,camera,y,args.seed,json.loads((RUN/"release.json").read_text())["winner"]),args.output)
