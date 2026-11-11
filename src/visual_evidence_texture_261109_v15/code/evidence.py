"""Find subpixel examples in the observed LR and train a local vision-only SR estimate."""
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path("/home/zhangxu/hx/HX001")
OLD=ROOT/"texture_retention_261109_v15"
sys.path.insert(0,str(OLD/"code"))
from balanced import (RUN as OLD_RUN,base_camera,case_path,cache,split_rows,load_heads,predict,
                      evaluate_output,aggregate,contrasts,save_tensor,save_png,write_json,image,render)
from texture import smooth,bands,luminance
CODE=Path(__file__).resolve().parent
P=json.loads((CODE/"protocol.json").read_text());RUN=ROOT/P["run"]


def rows():
    data=json.loads((ROOT/"data/manifest.json").read_text())["images"]
    split=split_rows()
    ours=split["train"]
    used={r["scene_id"] for r in ours}
    last_check={}
    for domain in ("RealSR","DRealSR"):
        # Reconstruct the transfer17 split using its actual saved scene identities.
        folder=OLD_RUN/"transfer_scores"/("transfer17_"+domain)
        last_check[domain]={json.loads(f.read_text())["scene"] for f in folder.glob("*.json")}
        if len(last_check[domain])!=20:raise ValueError("Complete the frozen upstream transfer study first")
    def subset(ds,cam,role,exclude):
        unique={}
        for row in sorted(data,key=lambda x:x["id"]):
            if row["dataset"]==ds and row["role"]==role and (not cam or row["camera"]==cam) and row["scene_id"] not in exclude:
                unique.setdefault(row["scene_id"],row)
        return sorted(unique.values(),key=lambda r:hashlib.sha256(("A8-v15/"+r["scene_id"]).encode()).digest())
    train=[];new_real=[];dev=[]
    for cam in ("Canon","Nikon"):
        pool=subset("RealSR",cam,"T",last_check["RealSR"]|used)
        new_real+=pool[:12]
        train+=pool[12:12+P["pretrain_scenes_per_camera"]]
        dev+=subset("RealSR",cam,"V",set())[:15]
    new_dreal=subset("DRealSR",None,"T",last_check["DRealSR"])[:24]
    assert len(train)==192 and len(dev)==30 and len(new_real)==len(new_dreal)==24
    return dict(train=train,dev=dev,fitanchors=ours,real=split["real"],dreal=split["dreal"],
                new_real=new_real,new_dreal=new_dreal)


def descriptor(y):
    g=luminance(y)
    patch=F.unfold(F.pad(g,(1,1,1,1),mode="replicate"),3)[0].T
    centered=patch-patch.mean(1,keepdim=True)
    std=centered.square().mean(1).clamp_min(1e-6).sqrt()
    return F.normalize(centered,dim=1),std,y.flatten(2)[0].T


@torch.no_grad()
def references(lr,sem):
    # Each key has an actual observed 4x4 RGB residual, rather than a generated embedding as its value.
    pooled=F.avg_pool2d(lr,4)
    q,qs,qcolor=descriptor(lr);k,ks,kcolor=descriptor(pooled)
    high=lr-F.interpolate(pooled,size=lr.shape[-2:],mode="nearest")
    values=F.pixel_unshuffle(high,4).flatten(2)[0].T
    tokens=F.normalize(sem[0]-sem[0].mean(0,keepdim=True),dim=1)
    affinity=tokens@tokens.T
    idx=torch.arange(128*128,device=lr.device)
    block=(idx//128//4)*32+(idx%128//4)
    banks={k:[] for k in ("local","semantic","misplaced","wrong_value")}
    for start in range(0,len(q),1024):
        sl=slice(start,start+1024)
        sim=q[sl]@k.T
        color_dist=torch.cdist(qcolor[sl],kcolor,p=1)/3
        rank=sim-1.5*color_dist-.14*(qs[sl,None].log()-ks[None,:].log()).abs()
        score,keys=rank.topk(24,dim=1)
        ids=block[sl]
        shifted=((ids//32+16)%32)*32+(ids%32+16)%32
        for mode in banks:
            affinity_score=score if mode=="local" else score+.45*affinity[shifted if mode=="misplaced" else ids].gather(1,keys)
            top,order=affinity_score.topk(4,dim=1)
            pos=keys.gather(1,order)
            val_idx=(pos+511)%len(k) if mode=="wrong_value" else pos
            weights=(top*14).softmax(1)
            gain=(qs[sl,None]/ks[pos].clamp_min(.001)).clamp(.4,2)
            detail=(values[val_idx]*(weights*gain)[:,:,None]).sum(1)
            trust=(sim.gather(1,pos)*weights).sum(1,keepdim=True)
            banks[mode].append(torch.cat((detail,trust),1))
    return {mode:torch.cat(chunks,0).T.reshape(1,49,128,128) for mode,chunks in banks.items()}


class LocalSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(6,40,3,padding=1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(40,40,3,padding=1),nn.SiLU(),
                                                  nn.Conv2d(40,40,3,padding=1)) for _ in range(4)])
        self.out=nn.Conv2d(40,48,3,padding=1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,lr):
        high=lr-smooth(lr,3)
        x=self.stem(torch.cat((lr-.5,high*8),1))
        for block in self.blocks:x=x+.25*block(x)
        up=F.interpolate(lr,scale_factor=4,mode="bicubic",align_corners=False)
        return up+.25*F.pixel_shuffle(self.out(x),4)


class ReferenceGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(6,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,2,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias,-1.)

    def forward(self,lr,pred,reference):
        donor=F.pixel_shuffle(reference[:,:48],4)
        band=bands(pred)
        refband=bands(donor)
        a,b=luminance(band[0]),luminance(refband[0])
        pool=lambda x:F.avg_pool2d(x,4)
        ea=(pool(a.square())+1e-6).sqrt();eb=(pool(b.square())+1e-6).sqrt()
        feats=torch.cat((luminance(lr),reference[:,48:],torch.log1p(40*ea),torch.log1p(40*eb),
                         pool(a*b)/(ea*eb),8*(lr-smooth(lr,3)).abs().mean(1,keepdim=True)),1)
        gate=F.interpolate(self.net(feats).sigmoid(),scale_factor=4,mode="bilinear",align_corners=False)
        return pred+sum(gate[:,j:j+1]*refband[j] for j in (0,1))


def refpath(scene):
    return RUN/"references"/(hashlib.sha256(scene.encode()).hexdigest()[:20]+".pt")


def cache_refs(base,records):
    for i,row in enumerate(records):
        path=refpath(row["scene_id"])
        if path.exists():continue
        y=image(ROOT/row["lr"])[None].cuda().float()/255
        with torch.no_grad():
            _,sem,_=base.conditions(y)
            bank=references(y,sem)
        path.parent.mkdir(exist_ok=True,parents=True)
        save_tensor(path,{k:v.half().cpu() for k,v in bank.items()})
        if (i+1)%32==0:print("reference images",i+1,flush=True)


if __name__=="__main__":
    torch.set_num_threads(4)
    base,_=base_camera();groups=rows()
    cache_refs(base,groups["train"]+groups["dev"])
