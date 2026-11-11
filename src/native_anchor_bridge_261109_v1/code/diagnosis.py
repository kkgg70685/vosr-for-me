"""Exact measurements of published gates, their scaled RGB actions, and registration losses."""
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from cache import RUN
from texture import features,bands,luminance,warp,align
from native_model import field,cubic

def rms(x):
    return float(x.square().mean().sqrt())

def diagnostic(data,head,scene,save=True):
    raw,base,lr=(data[k] for k in ("raw","v14","lr"))
    donor,bb,desc=features(raw,base,lr,True)
    g=F.interpolate(head.net(F.avg_pool2d(desc,2)).sigmoid(),size=base.shape[-2:],mode="bilinear",align_corners=False)
    off,_=field(raw,base)
    rw=warp(raw,off)
    if (rw-align(raw,base)).abs().max()>1e-5:raise ValueError("Published field mismatch")
    fine=bands(raw)[0]
    out={"warp_bilinear":rms(bands(rw)[0])/rms(fine),"warp_cubic":rms(bands(cubic(raw,off))[0])/rms(fine),
         "shift_px":float(off.square().sum(1).sqrt().mean())}
    actions=[]
    for j in range(4):
        b=j%2; gain=(1.,.25)[b]
        action=gain*g[:,j:j+1]*(1.25*(donor[b]-bb[b]) if j<2 else bb[b])
        actions.append(action)
        key=("T_f","T_m","S_f","S_m")[j]
        for name,val in zip(("q10","median","q90"),torch.quantile(g[:,j].flatten()[::4],g.new_tensor([.1,.5,.9]))):
            out[key+"_"+name]=float(val)
        out[key+"_open"]=float((g[:,j]>.5).float().mean())
        out[key+"_rms255"]=255*rms(action)
        out[key+"_over1"]=float((action.abs().mean(1)>1/255).float().mean())
    for j,label in enumerate(("fine","mid")):
        total=actions[j]+actions[j+2]
        out[label+"_rms255"]=255*rms(total)
        out[label+"_cancel"]=1-rms(total)/(rms(actions[j])+rms(actions[j+2])+1e-12)
    if save:
        folder=RUN/"gates"/scene;folder.mkdir(parents=True,exist_ok=True)
        torch.save({"gates":g.cpu(),"actions":torch.stack(actions).cpu(),"offset_hr":off.cpu()},folder/"measured.pt")
        for j,label in enumerate(("T_f","T_m","S_f","S_m")):
            Image.fromarray((g[0,j]*255).round().byte().cpu().numpy()).save(folder/(label+".png"))
    return out

def sampling():
    x=torch.arange(256,device="cuda").float()
    result={}
    for f in (.125,.25,.375,.4375,.5):
        u=(2*torch.pi*f*x).cos()[None,None,None].expand(1,1,32,-1)
        off=u.new_zeros(1,2,32,256);off[:,0]=.5
        def norm(v):return rms(v[...,8:-8])/rms(u[...,8:-8])
        result[str(f)]=dict(identity=1.,bilinear=norm(warp(u,off)),cubic=norm(cubic(u,off)))
    return result
