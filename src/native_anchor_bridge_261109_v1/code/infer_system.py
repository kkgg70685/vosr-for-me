"""Shared-trajectory inference for the validation-locked native/visual correction model."""
import argparse,json
import torch
from cache import ROOT,RUN,base_camera,lift
from native_model import NativeCorrection,foundation
from bridge_trial import source
from upstream import start,finish,coefficients,refine
from balanced import corrections
from infer_v15 import dependencies
from phase_study import phase_terms,generate
from texture import smooth,align
from route import routed
from evidence import image,save_png

@torch.no_grad()
def restore(y,seed,base,camera,assets,net,recipe):
    # Native and guided trajectories share their initial 16 steps.
    state=start(base,y,seed,-.5);sem=state[1]
    z=finish(base,state,y,-.5,"native",.04,8,camera.sensor)
    f=base.decoder_prefix(z);delta=camera.alpha*(camera.bridge(f,y)-f)
    raw=base.decoder_suffix(f);full=base.decoder_suffix(f+delta)
    pre=raw
    if recipe.get("bridge","pre")!="none":
        raw_lr=camera.sensor(raw);direction=camera.sensor(full)-raw_lr
        gate=(smooth((direction*(y-raw_lr)).mean(1,keepdim=True),5)/
              (smooth(direction.square().mean(1,keepdim=True),5)+.000016)).clamp(0,1)
        pre=base.decoder_suffix(f+lift(gate,f)*smooth(delta,5))
    paired=torch.stack((raw,full),dim=1)[:,None]
    read=torch.lerp(paired[:,:,1],camera.reader(paired,y),camera.pack["readout_strength"])[:,0]
    adjusted=camera.head(read,y,camera.pack["kind"],camera.sensor)
    calibrated=torch.lerp(read,adjusted,camera.pack["strength"])
    zg=finish(base,state,y,-.5,"visual",.04,8,camera.sensor)
    guided=camera.pipeline(base,zg,y)
    coef=coefficients(y,sem,"visual")
    v14=torch.lerp(refine(calibrated,y,coef,24,False),refine(guided,y,coef,24,False),.5)
    data={"raw":raw,"v14":v14,"lr":y}
    fine,mid,reg=corrections(data,assets[3],assets[4])
    anchor=v14+fine+.25*mid+.5*reg
    src=source(y,sem,assets)
    terms,low=phase_terms(anchor,src,align(raw,v14))
    old=generate(anchor,terms,low,assets[5]["matched"])
    x=foundation(raw,pre,src,recipe["method"])
    native,_=net(x,y,src)
    result,maps=routed(native,old,src,recipe["spec"])
    return result,{"raw":raw,"v14":v14,"previous":old,"native_corrected":native,"fine_gate":maps[0],"middle_gate":maps[1]}

def load_model():
    net=NativeCorrection().cuda().eval()
    net.load_state_dict(torch.load(RUN/"geometry.pt",weights_only=True,map_location="cuda"))
    recipe=json.loads((RUN/"pruned/lock.json").read_text())
    return net,recipe

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=6);a=p.parse_args();torch.set_num_threads(4)
    y=image(a.input)[None].cuda().float()/255
    if tuple(y.shape)!=(1,3,128,128):raise ValueError("Expected native 128x128 RGB")
    base,camera=base_camera();net,recipe=load_model()
    out,_=restore(y,a.seed,base,camera,dependencies(),net,recipe)
    save_png(out,a.output)
