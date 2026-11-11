"""Same-seed staged V1.4 renderer. Shared computation with the deployed real preset."""
import json
import os
import sys
from pathlib import Path
import hashlib
import torch

ROOT=Path("/home/zhangxu/hx/HX001")
CODE=Path(__file__).resolve().parent
P=json.loads((CODE/"protocol.json").read_text())
RUN=ROOT/P["run"]
MIX=ROOT/"visual_trajectory_mix_261108_v1"
sys.path.insert(0,str(MIX/"code"))
from mix_study import Camera,FrozenVOSR,coefficients,finish,image,refine,start,write_json


@torch.no_grad()
def camera_stages(camera,base,z,lr):
    f=base.decoder_prefix(z)
    raw=base.decoder_suffix(f)
    calibrated=base.decoder_suffix(torch.lerp(f,camera.bridge(f,lr),camera.alpha))
    pair=torch.stack((raw,calibrated),dim=1)[:,None]
    read=torch.lerp(pair[:,:,1],camera.reader(pair,lr),camera.pack["readout_strength"])[:,0]
    adjusted=camera.head(read,lr,camera.pack["kind"],camera.sensor)
    return raw,calibrated,read,torch.lerp(read,adjusted,camera.pack["strength"])


@torch.no_grad()
def render(base,camera,lr,seed):
    state=start(base,lr,seed,-.5)
    coeff=coefficients(lr,state[1],"visual")
    z=finish(base,state,lr,-.5,"native",.04,8,camera.sensor)
    raw,bridge,reader,a=camera_stages(camera,base,z,lr)
    z=finish(base,state,lr,-.5,"visual",.04,8,camera.sensor)
    b=camera.pipeline(base,z,lr)
    ra=refine(a,lr,coeff,24,False)
    rb=refine(b,lr,coeff,24,False)
    return dict(raw=raw,bridge=bridge,reader=reader,camera=a,native_end=ra,guided_camera=b,
                guided_end=rb,pre_mix=(a+b)*.5,v14=torch.lerp(ra,rb,.5))


def ranked(rows):
    return sorted(rows,key=lambda r:hashlib.sha256(("v15/"+r["scene_id"]).encode()).digest())


def split_rows():
    data=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    real={}
    for role in ("T","V","E"):
        chosen=[]
        for cam in ("Canon","Nikon"):
            uniq={}
            for r in sorted(data,key=lambda r:r["id"]):
                if (r["dataset"],r["role"],r.get("camera"))==("RealSR",role,cam):
                    uniq.setdefault(r["scene_id"],r)
            rows=list(uniq.values())
            if role=="T":rows=ranked(rows)[:P["train_per_camera"]]
            if role=="V":rows=rows[P["validation_offset"]:P["validation_offset"]+P["validation_per_camera"]]
            chosen+=rows
        real[role]=chosen
    ids=[{r["scene_id"] for r in real[k]} for k in ("T","V","E")]
    assert list(map(len,ids))==[2*P["train_per_camera"],2*P["validation_per_camera"],100]
    assert sum(map(len,ids))==len(set.union(*ids))
    foreign=[r for r in data if r["dataset"]=="DRealSR" and r["role"]=="E"]
    assert len({r["scene_id"] for r in foreign})==15
    repeat=sum((ranked([r for r in real["E"] if r["camera"]==cam])[:P["repeat_real_per_camera"]]
                for cam in ("Canon","Nikon")),[])
    return dict(train=real["T"],dev=real["V"],real=real["E"],dreal=foreign,repeat=repeat,external_repeat=foreign)


def case_path(group,row,seed):
    key=hashlib.sha256(row["scene_id"].encode()).hexdigest()[:20]
    return RUN/"cache"/group/f"{key}_{seed}.pt"


def cache(group,rows,seed,base,camera):
    for idx,row in enumerate(rows):
        path=case_path(group,row,seed)
        if path.exists():continue
        lr,hr=(image(ROOT/row[k])[None].cuda().float()/255 for k in ("lr","HR"))
        assert lr.shape==(1,3,128,128) and hr.shape==(1,3,512,512)
        stages=render(base,camera,lr,seed)
        if group=="train":
            stages={k:stages[k] for k in ("raw","v14")}
        item={k:x.cpu() for k,x in stages.items()}
        item.update(lr=lr.cpu(),hr=hr.cpu(),scene=row["scene_id"],seed=seed)
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(".part")
        torch.save(item,tmp)
        os.replace(tmp,path)
        print(group,idx+1,len(rows),flush=True)


def base_camera():
    release=json.loads((MIX/"release.json").read_text())
    if release["winner"]!="fixed/0.5":raise ValueError("Unexpected v1.4 release")
    base=FrozenVOSR()
    base.use_checkpoint=False
    return base,Camera()


if __name__=="__main__":
    import fcntl
    torch.set_num_threads(4)
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/"cache_dev.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        splits=split_rows()
        base,camera=base_camera()
        for group in ("dev","train"):
            cache(group,splits[group],P["primary_seed"],base,camera)
        write_json(RUN/"dev_cache_ready.json",dict(train=len(splits["train"]),dev=len(splits["dev"])))
