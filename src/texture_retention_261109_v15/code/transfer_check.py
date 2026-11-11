"""Fresh scenes and seed for the already fixed zero-midband control. No fitting or selection."""
import hashlib
import json
import torch
from balanced import (ROOT,RUN,cache,case_path,base_camera,split_rows,load_heads,corrections,key,
                      evaluate_output,aggregate,write_json)
from v15_restore import preset
from check_release import scores


def heldout():
    trained={r["scene_id"] for r in split_rows()["train"]}
    manifest=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    groups={}
    for dataset,n in (("RealSR",10),("DRealSR",20)):
        pool={}
        for r in sorted(manifest,key=lambda a:a["id"]):
            if r["role"]=="T" and r["dataset"]==dataset and r["scene_id"] not in trained:
                pool.setdefault(r["scene_id"],r)
        ordered=sorted(pool.values(),key=lambda r:hashlib.sha256(("v15-fidelity-transfer/"+r["scene_id"]).encode()).hexdigest())
        groups[dataset]=sum(([r for r in ordered if r["camera"]==cam][:n] for cam in ("Canon","Nikon")),[]) if dataset=="RealSR" else ordered[:n]
        assert len(groups[dataset])==20
    return groups


@torch.no_grad()
def study():
    base,camera=base_camera()
    import pyiqa
    torch.hub.set_dir(str(ROOT/"runtime_cache/torch/hub"))
    head=load_heads("learned")["learned"]
    coef=torch.load(RUN/"ridge.pt",weights_only=True,map_location="cuda")
    dists=pyiqa.create_metric("dists",device="cuda").eval().requires_grad_(False)
    spec1,spec2=preset("balanced"),preset("fidelity")
    for dataset,rows in heldout().items():
        group="transfer17_"+dataset
        cache(group,rows,17,base,camera)
        records=[]
        for row in rows:
            path=case_path(group,row,17)
            dest=RUN/"transfer_scores"/group/(path.stem+".json")
            if not dest.exists():
                s=torch.load(path,weights_only=True,map_location="cpu")
                data={k:s[k].cuda() for k in ("raw","v14","hr","lr")}
                fine,mid,linear=corrections(data,head,coef)
                results={}
                for label,spec in (("v14",(0,0,0)),("balanced",spec1),("fidelity",spec2)):
                    f,m,c=spec
                    rgb=data["v14"]+f*fine+m*mid+c*linear
                    results[label]=evaluate_output(rgb,data["hr"],camera.metric)
                    results[label]["dists"]=float(dists(rgb.clamp(0,1),data["hr"]).mean())
                write_json(dest,dict(scene=row["scene_id"],scores=results))
            records.append(json.loads(dest.read_text()))
        means=aggregate(records)
        effects={k:{f:scores([r["scores"][k][f]-r["scores"]["v14"][f] for r in records]) for f in means[k]}
                 for k in ("balanced","fidelity")}
        write_json(RUN/(group+".json"),dict(cases=len(records),means=means,effects=effects))
    write_json(RUN/"transfer_complete.json",dict(seed=17,real=20,dreal=20))


if __name__=="__main__":
    torch.set_num_threads(4)
    study()
