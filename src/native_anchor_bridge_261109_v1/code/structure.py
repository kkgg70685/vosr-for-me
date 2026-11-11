"""Position-sensitive structural readouts and fixed object crops, declared before evaluation."""
import numpy as np
from scipy import ndimage as nd
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

BOXES={"02":(145,106,460,265),"05":(305,100,410,315),"06":(85,63,218,177)}
def gray(x):
    a=x[0].clamp(0,1).cpu().numpy()
    return .299*a[0]+.587*a[1]+.114*a[2]

def fscore(a,b,d=1.5):
    if not a.any() or not b.any():return float(not(a.any() or b.any()))
    p=(nd.distance_transform_edt(~b)[a]<=d).mean()
    r=(nd.distance_transform_edt(~a)[b]<=d).mean()
    return float(2*p*r/(p+r+1e-12))

def edges(a):
    return canny(a,sigma=.9,low_threshold=.025,high_threshold=.065)

def metrics(x,hr):
    a,b=gray(x),gray(hr)
    return {"edge":fscore(edges(a),edges(b))}

def object_scores(x,hr,prefix):
    l,t,r,b=BOXES[prefix]
    a=x[...,t:b,l:r];ref=hr[...,t:b,l:r]
    result=metrics(a,ref)
    def contrast(im):
        rgb=im[0].clamp(0,1).cpu().numpy()
        if prefix=="02": return rgb[0]-rgb[1]
        if prefix=="05": return -gray(im)
        return gray(im)
    refc=contrast(ref);q=threshold_otsu(refc)
    pred,truth=contrast(a)>q,refc>q
    result["skeleton"]=fscore(skeletonize(pred),skeletonize(truth))
    def holes(mask):
        inner=nd.binary_fill_holes(mask)&~mask
        labels,n=nd.label(inner)
        return int(np.sum(np.bincount(labels.ravel(),minlength=n+1)[1:]>=8))
    result["holes"]=holes(pred);result["hr_holes"]=holes(truth)
    return result
