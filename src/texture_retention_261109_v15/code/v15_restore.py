"""V1.5 single-image inference using the validated texture head and ridge predictor."""
import argparse
import json
from pathlib import Path
import torch
from balanced import ROOT,RUN,base_camera,image,load_heads,restore,save_png


def preset(name):
    spec=json.loads((RUN/"balanced_release.json").read_text())["spec"]
    return (spec[0],0,spec[2]) if name=="fidelity" else tuple(spec)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=6)
    p.add_argument("--preset",choices=("balanced","fidelity"),default="balanced")
    args=p.parse_args()
    torch.set_num_threads(4)
    y=image(Path(args.input))[None].cuda().float()/255
    if y.shape!=(1,3,128,128):raise ValueError("Expected native 128 × 128 RGB LR")
    base,camera=base_camera()
    out=restore(base,camera,y,args.seed,load_heads("learned")["learned"],
                torch.load(RUN/"ridge.pt",weights_only=True,map_location="cuda"),preset(args.preset))
    save_png(out,args.output)
