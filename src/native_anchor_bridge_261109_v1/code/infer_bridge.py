"""Run the locked bridge candidate directly from a 128x128 RGB observation."""
import argparse
import json
import torch
from cache import RUN,base_camera,intervene,image
from native_model import NativeCorrection,foundation
from bridge_trial import source
from infer_v15 import dependencies
from evidence import save_png
from texture import bands

@torch.no_grad()
def restore(lr,seed,base,camera,assets,net,choice):
    stage=intervene(base,camera,lr,seed)
    src=source(lr,stage["sem"],assets)
    mode,keep=choice.split("_")
    native=stage["native"]
    x=foundation(native,stage["pre"],src,mode)
    result,_=net(x,lr,src)
    return result+float(keep)*(bands(native)[0]-bands(result)[0])

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=6);args=p.parse_args()
    torch.set_num_threads(4)
    lr=image(args.input)[None].cuda().float()/255
    if tuple(lr.shape)!=(1,3,128,128):raise ValueError("Expected native 128 x 128 RGB input")
    base,camera=base_camera();net=NativeCorrection().cuda().eval()
    net.load_state_dict(torch.load(RUN/"geometry.pt",weights_only=True,map_location="cuda"))
    selected=json.loads((RUN/"lock.json").read_text())["candidate"]
    save_png(restore(lr,args.seed,base,camera,dependencies(),net,selected),args.output)
