"""Native-grid SR correction using the measured LR, a frozen visual source and a feature-side bridge."""
import torch
from torch import nn
from torch.nn import functional as F
from cache import lift
from texture import smooth,bands,luminance,warp

def field(raw,source):
    a,b=(luminance(bands(im)[1]) for im in (raw,source))
    offsets=[(dx,dy) for dy in (-2,0,2) for dx in (-2,0,2)]
    h,w=a.shape[-2:]
    padded=F.pad(a,(2,2,2,2),mode="replicate")
    pool=lambda v:F.avg_pool2d(v,8)
    scores=[]
    for dx,dy in offsets:
        moved=padded[...,2+dy:2+dy+h,2+dx:2+dx+w]
        denom=(pool(moved.square())*pool(b.square())).clamp_min(1e-12).sqrt()
        scores.append(pool(moved*b)/(denom+.00005)-.012*(abs(dx)+abs(dy)))
    scores=torch.cat(scores,1)
    delta=torch.einsum("ac,bchw->bahw",raw.new_tensor(offsets).T,(16*scores).softmax(1))
    better=scores.max(1,keepdim=True).values-scores[:,4:5]
    reliable=(better>.10)&(delta.square().sum(1,keepdim=True)>.75**2)
    return lift(delta,raw),lift(reliable.float(),raw)

def cubic(x,off):
    h,w=x.shape[-2:]
    y,u=torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing="ij")
    grid=torch.stack((2*(u+.5+off[:,0])/w-1,2*(y+.5+off[:,1])/h-1),-1)
    return F.grid_sample(x,grid,mode="bicubic",padding_mode="border",align_corners=False)

def foundation(raw,pre,source,mode):
    # Keep raw RGB on an identity path and add a smoothed correction. The smoothing bands overlap.
    delta=pre-raw
    base=raw+delta-bands(delta)[0]
    if mode=="direct":return base
    off,mask=field(raw,source)
    registered=warp(raw,off)
    rf,rm=bands(raw)
    wf,wm=bands(registered)
    base=base+(wm-rm)
    if mode=="mid":return base
    if mode=="all":return base+wf-rf
    if mode=="conf":return base+mask*(bands(cubic(raw,off))[0]-rf)
    raise ValueError(mode)

class NativeCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(105,40,3,padding=1),nn.SiLU(),
                               nn.Conv2d(40,40,3,padding=1),nn.SiLU(),nn.Conv2d(40,50,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,lr,s):
        diff=s-x
        fine,mid=bands(diff)
        low=diff-fine-mid
        desc=torch.cat((lr-.5,8*(lr-smooth(lr,3)),F.avg_pool2d(diff,4)*5,
                        6*F.pixel_unshuffle(fine,4),6*F.pixel_unshuffle(bands(x)[0],4)),1)
        p=self.net(desc)
        gf,gm=lift(p[:,:2].sigmoid(),x).chunk(2,1)
        # A signed subpixel residual can create a stroke missing in both available estimates.
        res=.16*F.pixel_shuffle(p[:,2:].tanh(),4)
        return x+.8*low+gf*fine+gm*mid+res, (gf,gm,res)

def structure_loss(rgb,truth):
    a,b=luminance(rgb),luminance(truth)
    terms=[]
    for dx,dy in ((1,0),(0,1),(1,1),(2,0),(0,2),(3,3)):
        def diff(z):
            h,w=z.shape[-2:]
            return z[...,dy:,dx:]-z[...,:h-dy,:w-dx]
        u,v=diff(a),diff(b)
        # Compare signed contrasts and gradient orientation around strokes, with a floor for flat areas.
        local=(smooth(v.square(),5)+.0004).sqrt()
        terms.append(((u-v).abs()/local).clamp_max(4).mean())
    return torch.stack(terms).mean()
