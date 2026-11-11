"""Select whether the native branch needs an extra bridge proposal, using the same validation scenes."""
import fcntl,json
import numpy as np
import torch
from cache import ROOT,RUN,write_json
from bridge_trial import read,legacy_images
from native_model import NativeCorrection,foundation
from route import routed,name,strict
from infer_v15 import dependencies
from evidence import evaluate_output
from structure import metrics
OUT=RUN/"pruned"

@torch.no_grad()
def main():
    import pyiqa
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    lp=pyiqa.create_metric("lpips",device="cuda").eval().requires_grad_(False)
    di=pyiqa.create_metric("dists",device="cuda").eval().requires_grad_(False)
    spec=json.loads((RUN/"routed/lock.json").read_text())["spec"]
    rec=[]
    for p in sorted((RUN/"routed/cache/dev").glob("*.pt")):
        d={k:v.cuda() for k,v in read(p).items()}
        rs={"v15":d["v15"],"pre":routed(d["x"],d["v15"],d["source"],spec)[0],
            "none":routed(d["absent"],d["v15"],d["source"],spec)[0]}
        scores={}
        for k,x in rs.items():
            v=evaluate_output(x,d["hr"],lp);v.update(metrics(x,d["hr"]),dists=float(di(x.clamp(0,1),d["hr"]).mean()))
            scores[k]=v
        # Camera identities come from the already fixed route validation manifest scores.
        original=json.loads((RUN/"routed/scores/dev"/(p.stem+".json")).read_text())
        rec.append(dict(scene=p.stem,camera=original["camera"],scores=scores))
    def table(rows):
        return {k:{m:float(np.mean([r["scores"][k][m] for r in rows])) for m in rows[0]["scores"][k]} for k in rows[0]["scores"]}
    tab=table(rec);bycam=[table([r for r in rec if r["camera"]==c]) for c in ("Canon","Nikon")]
    good=all(strict(t,"none") and t["none"]["psnr"]>=t["pre"]["psnr"]-.03 and
             t["none"]["lpips"]<=t["pre"]["lpips"]+.001 and t["none"]["dists"]<=t["pre"]["dists"]+.001 for t in bycam)
    chosen="none" if good else "pre"
    write_json(OUT/"lock.json",dict(bridge=chosen,spec=spec,method="conf",means=tab))
    print("bridge selection",chosen,tab,flush=True)
    # Fixed old confirmation groups can be reconstructed from their existing per-scene records.
    for g in ("real","dreal","fresh_RealSR","fresh_DRealSR"):
        old=json.loads((RUN/"routed"/(g+".json")).read_text())
        old["selected"]="no_bridge" if chosen=="none" else name(spec)
        write_json(OUT/(g+".json"),old)
    assets=dependencies()
    net=NativeCorrection().cuda().eval();net.load_state_dict(read(RUN/"geometry.pt"))
    for g in ("new_RealSR","new_DRealSR"):
        rec=[]
        for p in sorted((RUN/"refit/cache"/g).glob("*.pt")):
            d={k:v.cuda() for k,v in read(p).items()}
            prior,_=legacy_images(d,assets)
            basis=foundation(d["native"],d["native"] if chosen=="none" else d["pre"],d["source"],"conf")
            x=net(basis,d["lr"],d["source"])[0]
            output=routed(x,prior,d["source"],spec)[0]
            scores={}
            for k,rgb in (("v15",prior),("selected",output)):
                v=evaluate_output(rgb,d["hr"],lp);v.update(metrics(rgb,d["hr"]),dists=float(di(rgb.clamp(0,1),d["hr"]).mean()))
                scores[k]=v
            rec.append(dict(scene=p.stem,scores=scores))
        if len(rec)!=12:raise ValueError((g,len(rec)))
        write_json(OUT/(g+".json"),dict(n=12,means=table(rec)))
    write_json(OUT/"result.json",dict(bridge=chosen,complete=True))
if __name__=="__main__":
    with (RUN/"prune.lock").open("a+") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
