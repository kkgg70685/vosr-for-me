"""Anisotropic detail refinement in the block-mean-preserving subspace."""
import torch
from torch.nn import functional as F
from flow import direction_field


def coefficients(lr, sem, mode):
    field=direction_field(lr,sem,mode)
    guide=F.interpolate(lr,size=field.shape[-2:],mode="bilinear",align_corners=False)
    offsets=((0,1),(1,0),(1,1),(1,-1))
    coeff=[]
    for y,x in offsets:
        dist=x*x+y*y
        # A tangent is orthogonal to the estimated gradient normal.
        preference=-(field[:,0:1]*(x*x-y*y)+field[:,1:2]*(2*x*y))/dist
        other=guide.roll((-y,-x),(-2,-1))
        weight=preference.clamp_min(0).square()*torch.exp(-40*(guide-other).abs().mean(1,keepdim=True))/dist
        if y:weight[...,-y:,:]=0
        if x>0:weight[...,-x:]=0
        if x<0:weight[...,:-x]=0
        coeff.append((y,x,weight))
    return coeff


def dc(image):
    return F.avg_pool2d(image,4).repeat_interleave(4,-2).repeat_interleave(4,-1)


@torch.no_grad()
def refine(source,lr,coeff,steps,known_box):
    # Synthetic degradation is exactly block averaging up to input quantization.
    # For a real camera retain the input estimate's coarse coefficients.
    coarse = lr.repeat_interleave(4,-2).repeat_interleave(4,-1) if known_box else dc(source)
    result=source+coarse-dc(source)
    for _ in range(steps):
        flux=torch.zeros_like(result)
        for y,x,w in coeff:
            delta=w*(result.roll((-y,-x),(-2,-1))-result)
            flux+=delta-delta.roll((y,x),(-2,-1))
        result=result+.12*(flux-dc(flux))
    return result
