"""Predict signed subpixel detail from LR evidence with optional visual conditioning."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from phase_study import (CODE,RUN,ROOT,OLD,DEST,rows,dev_pool,base_camera,refpath,cache_refs,
                         image,source_bank,model_parts,phase_terms,generate,align,torch,np)
from evidence import save_tensor,bands,luminance,smooth

PUB=RUN/"semantic_readout"
Q=json.loads((CODE/"semantic_protocol.json").read_text())
MODES=("local","internal","vision")


def tokenpath(scene):
    return PUB/"tokens"/refpath(scene).name


@torch.no_grad()
def basis(base,train):
    target=PUB/"pca.pt"
    if target.exists():return torch.load(target,weights_only=True,map_location="cuda")
    mean=torch.zeros(768,device="cuda",dtype=torch.float64);cov=torch.zeros(768,768,device="cuda",dtype=torch.float64)
    count=0
    for row in train:
        y=image(ROOT/row["lr"])[None].cuda().float()/255
        t=F.normalize(base.conditions(y)[1][0].float(),dim=-1).double()
        mean+=t.sum(0);cov+=t.T@t;count+=t.shape[0]
    mean/=count
    eig,axes=torch.linalg.eigh(cov/count-mean[:,None]*mean[None,:])
    params=dict(mean=mean.float(),axes=(axes[:,-16:]/eig[-16:].sqrt().clamp_min(1e-6)).float())
    save_tensor(target,{k:v.cpu() for k,v in params.items()})
    return params


def projected(sem,params):
    t=(F.normalize(sem,dim=-1)-params["mean"])@params["axes"]
    return t.transpose(1,2).reshape(1,16,32,32).clamp(-4,4)


@torch.no_grad()
def store_tokens(base,records,params):
    for row in records:
        p=tokenpath(row["scene_id"])
        if p.exists():continue
        y=image(ROOT/row["lr"])[None].cuda().float()/255
        sem=base.conditions(y)[1];p.parent.mkdir(exist_ok=True,parents=True)
        save_tensor(p,projected(sem,params).half().cpu())
        if not refpath(row["scene_id"]).exists():cache_refs(base,[row])


class EvidenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(6+48+49,32,3,padding=1)
        self.context=nn.Conv2d(16,64,1)
        self.body=nn.Sequential(nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,48,1))
        nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)

    def forward(self,y,pred,reference,semantic):
        hp=pred-smooth(pred,7)
        x=self.stem(torch.cat((y-.5,8*(y-smooth(y,3)),5*F.pixel_unshuffle(hp,4),
                               reference[:,:48]*5,reference[:,48:]),1))
        a,b=self.context(semantic).chunk(2,1)
        res=.15*F.pixel_shuffle(self.body(x*(1+.5*a.tanh())+.3*b.tanh()).tanh(),4)
        return pred+res-smooth(res,13)


def inputs(y,bank,tokens,mode):
    if mode=="local":
        return torch.zeros_like(bank["local"]),torch.zeros_like(tokens)
    if mode=="internal":return bank["local"],torch.zeros_like(tokens)
    if mode=="wrong":return bank["wrong_value"],tokens
    if mode=="shift":return bank["misplaced"],tokens.roll((y.shape[-2]//2,y.shape[-1]//2),(-2,-1))
    if mode=="nosem":return bank["local"],torch.zeros_like(tokens)
    if mode=="novalue":
        mute=torch.zeros_like(bank["semantic"]);return mute,tokens
    return bank["semantic"],tokens


def fit(mode,train,lr_net,metric):
    from texture import texture_loss
    path=PUB/(mode+".pt")
    torch.manual_seed(6185);net=EvidenceHead().cuda()
    if path.exists():
        net.load_state_dict(torch.load(path,weights_only=True,map_location="cuda"));return net.eval()
    examples=[]
    for r in train:
        y=image(ROOT/r["lr"]).float()/255;hr=image(ROOT/r["HR"]).float()/255
        b=torch.load(refpath(r["scene_id"]),weights_only=True)
        t=torch.load(tokenpath(r["scene_id"]),weights_only=True).float()
        t=F.interpolate(t,size=y.shape[-2:],mode="bilinear",align_corners=False)
        bank,sem=inputs(y[None],b,t,mode)
        examples.append((y,hr,bank[0].half(),sem[0].half()))
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    for i in range(900):
        rng=np.random.default_rng(2611095+i);data=[[],[],[],[]]
        for _ in range(8):
            row=examples[rng.integers(len(examples))]
            oy,ox=rng.integers(0,97,2)
            for j,val in enumerate(row):
                scale=4 if j==1 else 1
                data[j].append(val[...,oy*scale:(oy+32)*scale,ox*scale:(ox+32)*scale])
        y,hr,bank,sem=(torch.stack(t).cuda().float() for t in data)
        with torch.no_grad():pred=lr_net(y)
        out=net(y,pred,bank,sem)[...,8:-8,8:-8].clamp(0,1)
        hr=hr[...,8:-8,8:-8]
        loss=3*(out-hr).abs().mean()+10*(out-hr).square().mean()+.5*metric.net(out,hr).mean()+.06*texture_loss(out,hr)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),2,error_if_nonfinite=True);opt.step()
        if (i+1)%300==0:print(mode,i+1,float(loss),flush=True)
    save_tensor(path,net.state_dict())
    return net.eval()


def load_heads():
    nets={}
    for k in MODES:
        model=EvidenceHead().cuda().eval()
        model.load_state_dict(torch.load(PUB/(k+".pt"),weights_only=True,map_location="cuda"))
        nets[k]=model
    return nets
