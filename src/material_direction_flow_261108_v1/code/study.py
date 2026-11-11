"""Lock development choices, compare matched generators, then evaluate real transfer."""
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from flow import ROOT, FrozenVOSR, finish, measure, sha, start, write_json

CODE = Path(__file__).resolve().parent
P = json.loads((CODE / "protocol.json").read_text())
RUN = ROOT / P["run"]
CAM = ROOT / "camera_feedback_261108_v1"
os.environ.setdefault("TORCH_HOME", str(ROOT / "runtime_cache/torch"))
sys.path[:0] = [str(CAM/"code"), str(ROOT/"paired_detail_readout_261108_v1/code")]
from prepare import image, teacher
from camera import perceptual_metric
from feedback import CameraProjection, CoarseFeedback
from readout import DetailReadout


def synth(parent):
    from probe import scene
    lr, hr, curve, turn = scene(parent)
    # Add a fine harmonic, and change the subpixel phase and line spacing by parent.
    # The dominant stripe orientation stays horizontal or vertical within each material.
    upright = hr.rot90(-turn,(-2,-1))
    y,x = torch.meshgrid(torch.arange(512,device=lr.device),torch.arange(512,device=lr.device),indexing="ij")
    fine = torch.where(x < curve[:,None], torch.cos(x*(.75+.06*(parent%3))+.7*(parent%7)),
                       torch.cos(y*(.82+.04*(parent%4))+.5*(parent%5)))
    hr = (upright + .029*fine).rot90(turn,(-2,-1)).contiguous()
    return (F.avg_pool2d(hr,4)*255).round()/255, hr, curve, turn


class Camera:
    def __init__(self):
        path = CAM/"real_camera_adapter.pt"
        if sha(path) != P["real_camera_sha"]:
            raise ValueError("Unexpected published camera model")
        pack = torch.load(path,map_location="cpu",weights_only=True)
        self.sensor = CameraProjection().cuda().eval().requires_grad_(False)
        self.sensor.load_state_dict(pack["projection"])
        self.head = CoarseFeedback().cuda().eval()
        self.head.load_state_dict(pack["head"])
        self.reader = DetailReadout("single_student").cuda().eval()
        self.reader.load_state_dict(pack["readout"])
        self.bridge, self.alpha = teacher()
        self.pack = pack
        self.metric = perceptual_metric()

    @torch.no_grad()
    def pipeline(self, base, z, lr):
        f = base.decoder_prefix(z)
        raw = base.decoder_suffix(f)
        paired = base.decoder_suffix(torch.lerp(f,self.bridge(f,lr),self.alpha))
        inputs = torch.stack((raw,paired),dim=1)[:,None]
        read = torch.lerp(inputs[:,:,1],self.reader(inputs,lr),self.pack["readout_strength"])[:,0]
        adjusted = self.head(read,lr,self.pack["kind"],self.sensor)
        return torch.lerp(read,adjusted,self.pack["strength"])

    @torch.no_grad()
    def scores(self, rgb, hr, lr):
        rgb = rgb.clamp(0,1)
        return dict(psnr=float(-10*(rgb-hr).square().mean().log10()),lpips=float(self.metric(rgb,hr).mean()),
                    projection_mse=float((self.sensor(rgb)-lr).square().mean()))


def real_dev():
    images=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    records=[]
    for camera in ("Canon","Nikon"):
        chosen = {}
        for r in sorted(images,key=lambda z:z["id"]):
            if (r["dataset"],r["role"],r.get("camera")) == ("RealSR","V",camera):
                chosen.setdefault(r["scene_id"],r)
        rows=list(chosen.values())[:P["real_dev_per_camera"]]
        assert len(rows)==P["real_dev_per_camera"]
        records.extend(rows)
    return records


def cases(domain, stage):
    if domain=="synth":
        for parent in P["synth_"+stage]:
            lr,hr,curve,turn=synth(parent)
            for seed in (P["synth_seeds"] if stage=="confirm" else [P["synth_seeds"][0]]):
                yield str(parent),seed,lr,hr,(curve,turn)
    elif stage=="dev":
        for r in real_dev():
            yield r["scene_id"],6,image(ROOT/r["lr"])[None].cuda().float()/255,image(ROOT/r["HR"])[None].cuda().float()/255,None
    else:
        paths=sorted((CAM/"new20").glob("*.pt"))
        assert len(paths)==20
        seen=set()
        for path in paths:
            data=torch.load(path,map_location="cpu",weights_only=True)
            scene_id=data["scene"]
            if scene_id in seen: raise ValueError("Repeated parent")
            seen.add(scene_id)
            lr,hr=(data[k][None].cuda().float()/255 for k in ("lr","hr"))
            for seed in (6,7):
                yield scene_id,seed,lr,hr,None


def configurations(stage, selected):
    if stage=="dev":
        return [("native",0,0)]+[("visual",s,w) for s in P["strengths"] for w in P["direction_weights"]]
    return [(method, 0 if method=="native" else selected["scale"],selected["weight"]) for method in P["methods"]]


def group(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}


def study(base, camera, domain, stage, selected=None):
    cfg = P["cfg_"+domain]
    variants=configurations(stage,selected)
    projector=camera.sensor if domain=="real" else lambda x:F.avg_pool2d(x,4)
    output=RUN/domain/stage
    records=[]
    for name,seed,lr,hr,truth in cases(domain,stage):
        # Stable IDs, including scene IDs that are not safe path components.
        import hashlib
        tag=hashlib.sha256(name.encode()).hexdigest()[:20]
        file=output/f"{tag}_{seed}.json"
        cached=json.loads(file.read_text()) if file.exists() else dict(name=name,seed=seed,scores={})
        state=None
        for method,scale,weight in variants:
            key=f"{method}_{scale}_{weight}" if stage=="dev" else method
            if key in cached["scores"]:continue
            if state is None:
                state=start(base,lr,seed,cfg)
            z=finish(base,state,lr,cfg,method,scale,weight,projector)
            with torch.no_grad():
                rgb=base.decode(z)
                scores=(measure(rgb,lr,hr,*truth) if domain=="synth" else camera.scores(rgb,hr,lr))
                if domain=="real" and stage=="confirm" and method in ("native","visual"):
                    legacy=camera.scores(camera.pipeline(base,z,lr),hr,lr)
                    cached["scores"][method+"_camera"] = legacy
                cached["scores"][key]=scores
            write_json(file,cached)
        records.append(cached)
        print(domain,stage,len(records),flush=True)
    labels=records[0]["scores"]
    return records,{k:group([r["scores"][k] for r in records]) for k in labels}


def lock(domain, stats):
    native=stats["native_0_0"]
    good=[(k,v) for k,v in stats.items() if k!="native_0_0" and
          ((v["psnr"]>=native["psnr"] and v["boundary"]<=native["boundary"]+.25) if domain=="synth"
           else v["lpips"]<=native["lpips"]+.005)]
    # Include native as a selectable reference. Resolve ties in favor of lower cost.
    good.insert(0,("native_0_0",native))
    key,_=min(good,key=lambda kv:kv[1]["leakage"] if domain=="synth" else -kv[1]["psnr"])
    _,scale,weight=key.split("_")
    return dict(key=key,scale=float(scale),weight=float(weight))


def effects(records, domain):
    by_scene={}
    for r in records:
        by_scene.setdefault(r["name"],[]).append(r["scores"])
    refs=["native","observation","local","misplaced"]
    pairs=[("visual",ref) for ref in refs]
    if domain=="real":pairs += [("visual_camera","native_camera"),("visual_camera","visual")]
    out={}
    for new,old in pairs:
        metrics={}
        for f in records[0]["scores"][new]:
            ds=np.array([np.mean([m[new][f]-m[old][f] for m in repeats]) for repeats in by_scene.values()])
            ci=np.quantile(np.random.default_rng(261108).choice(ds,(5000,len(ds))).mean(1),[.025,.975])
            metrics[f]=[float(ds.mean()),*ci.tolist()]
        out[new+"_vs_"+old]=metrics
    return out


def main():
    base=FrozenVOSR()
    base.use_checkpoint=False
    identity={k:sha(CODE/k) for k in ("study.py","flow.py","protocol.json")}
    identity["frozen_dit"]=sha(base.weight_path)
    if (RUN/"identity.json").exists() and json.loads((RUN/"identity.json").read_text())!=identity:
        raise ValueError("Run identity changed")
    write_json(RUN/"identity.json",identity)
    camera=None
    for domain in ("synth","real"):
        if domain=="real":camera=Camera()
        _,dev=study(base,camera,domain,"dev")
        choice=lock(domain,dev)
        file=RUN/domain/"choice.json"
        if file.exists() and choice!=json.loads(file.read_text())["selected"]:
            raise ValueError("Development choice changed")
        write_json(file,dict(selected=choice,dev=dev))
        rows,means=study(base,camera,domain,"confirm",choice)
        write_json(RUN/domain/"summary.json",dict(selected=choice,parents=len({r["name"] for r in rows}),
                   cases=len(rows),means=means,effects=effects(rows,domain)))
    write_json(RUN/"complete.json",dict(domains=["synth","real"]))


if __name__=="__main__":
    torch.set_num_threads(4)
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/"run.lock").open("a+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
