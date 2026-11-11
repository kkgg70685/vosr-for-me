"""Fuse two scales of same-seed native detail while retaining V1.4's low frequencies."""
import torch
from torch import nn
from torch.nn import functional as F


def smooth(x,k):
    pad=k//2
    x=F.avg_pool2d(F.pad(x,(pad,pad,0,0),mode="replicate"),(1,k),stride=1)
    return F.avg_pool2d(F.pad(x,(0,0,pad,pad),mode="replicate"),(k,1),stride=1)


def bands(x):
    fine=smooth(x,5)
    return x-fine,fine-smooth(fine,13)


def luminance(x):
    return .299*x[:,:1]+.587*x[:,1:2]+.114*x[:,2:3]


def warp(source,offset):
    h,w=source.shape[-2:]
    y,x=torch.meshgrid(torch.arange(h,device=source.device),torch.arange(w,device=source.device),indexing="ij")
    gx=(2*(x+.5+offset[:,0])/w)-1
    gy=(2*(y+.5+offset[:,1])/h)-1
    return F.grid_sample(source,torch.stack((gx,gy),-1),align_corners=False,padding_mode="border")


def align(raw,base):
    # Estimate a small displacement from the visible midband; resample the donor once.
    a=luminance(bands(raw)[1])
    b=luminance(bands(base)[1])
    offsets=[(dx,dy) for dy in (-2,0,2) for dx in (-2,0,2)]
    scores=[]
    padded=F.pad(a,(2,2,2,2),mode="replicate")
    h,w=a.shape[-2:]
    for dx,dy in offsets:
        shifted=padded[...,2+dy:2+dy+h,2+dx:2+dx+w]
        num=F.avg_pool2d(shifted*b,8)
        den=(F.avg_pool2d(shifted.square(),8)*F.avg_pool2d(b.square(),8)).clamp_min(1e-12).sqrt()
        scores.append(num/(den+.00005)-.012*(abs(dx)+abs(dy)))
    probs=(16*torch.cat(scores,1)).softmax(1)
    coords=raw.new_tensor(offsets).T
    offset=torch.einsum("ac,bchw->bahw",coords,probs)
    offset=F.interpolate(offset,size=(h,w),mode="bilinear",align_corners=False)
    return warp(raw,offset)


def features(raw,base,lr,aligned):
    donor=align(raw,base) if aligned else raw
    db,bb=bands(donor),bands(base)
    grid=F.interpolate(lr,size=base.shape[-2:],mode="bilinear",align_corners=False)
    desc=[luminance(base),luminance(grid),smooth((luminance(base)-luminance(grid)).abs(),9)*10]
    for a,b in zip(db,bb):
        x,y=luminance(a),luminance(b)
        ex=(smooth(x.square(),9)+.000001).sqrt()
        ey=(smooth(y.square(),9)+.000001).sqrt()
        desc.extend([torch.log1p(50*ex),torch.log1p(50*ey),smooth(x*y,9)/(ex*ey)])
    return db,bb,torch.cat(desc,1)


class TextureHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(9,24,3,padding=1),nn.SiLU(),
                               nn.Conv2d(24,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,4,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias,-1.)

    def forward(self,raw,base,lr,aligned=True):
        db,bb,desc=features(raw,base,lr,aligned)
        gates=self.net(F.avg_pool2d(desc,2)).sigmoid()
        gates=F.interpolate(gates,size=base.shape[-2:],mode="bilinear",align_corners=False)
        return base+sum(1.25*gates[:,j:j+1]*(db[j]-bb[j])+gates[:,j+2:j+3]*bb[j] for j in range(2))


def proposal(method,model,raw,base,lr):
    if method in ("learned","unregistered"):
        return model(raw,base,lr,aligned=method=="learned")
    if method=="sharpen":return base+sum(bands(base))
    donor=align(raw,base) if method=="aligned" else raw
    return base+sum(bands(donor-base))


def tile_energy(x):
    return tuple((F.avg_pool2d(luminance(b).square(),16)+.000009).sqrt() for b in bands(x))


def texture_loss(out,hr):
    return sum((a.log()-b.log()).abs().mean() for a,b in zip(tile_energy(out),tile_energy(hr)))*.5


def texture_stats(x,hr):
    y=luminance(hr)
    dx=F.pad(y[...,1:]-y[...,:-1],(0,1,0,0))
    dy=F.pad(y[...,1:,:]-y[...,:-1,:],(0,0,0,1))
    pool=lambda t:F.avg_pool2d(t,16)
    xx,yy,xy=pool(dx.square()),pool(dy.square()),pool(dx*dy)
    interest=(xx+yy-((xx-yy).square()+4*xy.square()).clamp_min(0).sqrt()).clamp_min(0)
    n=interest.numel()//4
    # Fixed HR-selected texture tiles across all methods; model never sees the mask.
    idx=interest.flatten().topk(n).indices
    ex,eh=tile_energy(x),tile_energy(hr)
    errors=[];ratios=[];corr=[]
    for a,b,xb,hb in zip(ex,eh,bands(x),bands(hr)):
        ratio=(a.flatten()[idx]/b.flatten()[idx]).log()
        errors.append(ratio.abs().mean());ratios.append(ratio.mean().exp())
        u,v=luminance(xb),luminance(hb)
        cc=pool(u*v)/(pool(u.square())*pool(v.square())).clamp_min(1e-12).sqrt()
        corr.append(cc.flatten()[idx].mean())
    return dict(texerr=float(torch.stack(errors).mean()),fine_ratio=float(ratios[0]),
                mid_ratio=float(ratios[1]),correlation=float(torch.stack(corr).mean()))
