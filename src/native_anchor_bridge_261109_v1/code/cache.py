"""Fork native decoder features before their trained bridge, keeping the same seed."""
from pathlib import Path
import fcntl
import sys
import torch
from torch.nn import functional as F
ROOT=Path("/home/zhangxu/hx/HX001")
RUN=ROOT/"native_anchor_bridge_261109_v1"
sys.path[:0]=[str(ROOT/"visual_evidence_texture_261109_v15/code"),str(ROOT/"material_direction_flow_261108_v1/code")]
from evidence import rows, case_path, base_camera, image, save_tensor, write_json
from upstream import start,finish
from texture import smooth

def bridge_path(row):
    return RUN/"bridge"/case_path("dev",row,6).name

def lift(x,target):
    return F.interpolate(x,size=target.shape[-2:],mode="bilinear",align_corners=False)

@torch.no_grad()
def intervene(base,camera,y,seed):
    state=start(base,y,seed,-.5)
    z=finish(base,state,y,-.5,"native",.04,8,camera.sensor)
    f=base.decoder_prefix(z)
    delta=camera.alpha*(camera.bridge(f,y)-f)
    native=base.decoder_suffix(f)
    full=base.decoder_suffix(f+delta)
    raw_lr=camera.sensor(native)
    direction=camera.sensor(full)-raw_lr
    num=smooth((direction*(y-raw_lr)).mean(1,keepdim=True),5)
    den=smooth(direction.square().mean(1,keepdim=True),5)
    gate=(num/(den+.000016)).clamp(0,1)
    proposal=base.decoder_suffix(f+lift(gate,f)*smooth(delta,5))
    return dict(native=native,full=full,pre=proposal,obs_gate=gate,sem=state[1])

def main():
    (RUN/"bridge").mkdir(parents=True,exist_ok=True)
    table=rows()
    from upstream import split_rows
    table["dev"]=split_rows()["dev"]
    base,camera=base_camera()
    for group in ("dev","dreal","real"):
        for i,r in enumerate(table[group]):
            path=bridge_path(r)
            if not path.exists():
                y=image(ROOT/r["lr"])[None].cuda().float()/255
                save_tensor(path,{k:v.cpu() for k,v in intervene(base,camera,y,6).items()})
            if (i+1)%5==0:print(group,i+1,len(table[group]),flush=True)
    write_json(RUN/"cache_done.json",{g:len(table[g]) for g in ("dev","real","dreal")})

if __name__=="__main__":
    torch.set_num_threads(4)
    with (RUN/"cache.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
