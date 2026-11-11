"""Paired decoder bridge study. Only validation selects a recipe. Published gates retain their original weights."""
import fcntl
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from cache import ROOT,RUN,base_camera,bridge_path,intervene,rows,save_tensor,write_json
from native_model import NativeCorrection,foundation,structure_loss,field,cubic
from diagnosis import diagnostic,sampling,rms
from structure import BOXES,metrics,object_scores
from evidence import LocalSR, references, image, evaluate_output
from infer_v15 import dependencies
from semantic_model import projected,inputs
from phase_study import generate,phase_terms
from balanced import corrections,case_path
from upstream import split_rows
from texture import align,bands,features,texture_loss
from gallery import pixels,font
from torch.nn import functional as F

def read(path):
    return torch.load(path,map_location="cpu",weights_only=True)

@torch.no_grad()
def source(lr,sem,assets):
    local,visual,pca,*_=assets
    bank=references(lr,sem)
    t=F.interpolate(projected(sem,pca),size=lr.shape[-2:],mode="bilinear",align_corners=False)
    b,t=inputs(lr,bank,t,"vision")
    return visual(lr,local(lr),b,t)

@torch.no_grad()
def get_case(row,group,base,camera,assets):
    path=bridge_path(row)
    if not path.exists():
        lr=image(ROOT/row["lr"])[None].cuda().float()/255
        save_tensor(path,{k:v.cpu() for k,v in intervene(base,camera,lr,6).items()})
    stage={k:v.cuda() for k,v in read(path).items()}
    hist=read(case_path(group,row,6))
    data={k:hist[k].cuda() for k in ("raw","v14","hr","lr")}
    data.update(stage)
    data["source"]=source(data["lr"],stage["sem"],assets)
    return data

def fit(samples,metric,structural):
    torch.manual_seed(29061)
    name="geometry" if structural else "control"
    path=RUN/(name+".pt")
    net=NativeCorrection().cuda()
    if path.exists():
        net.load_state_dict(read(path));return net.eval()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    for step in range(800):
        rng=np.random.default_rng(291100+step)
        batch={k:[] for k in ("x","source","hr","lr")}
        for _ in range(6):
            d=samples[rng.integers(len(samples))]
            y,x=4*rng.integers(0,107,2)
            for k,v in batch.items():
                q=4 if k=="lr" else 1
                v.append(d[k][...,y//q:(y+88)//q,x//q:(x+88)//q])
        d={k:torch.cat(v).cuda() for k,v in batch.items()}
        pred,_=net(d["x"],d["lr"],d["source"])
        pred=pred[...,8:-8,8:-8].clamp(0,1);hr=d["hr"][...,8:-8,8:-8]
        loss=4*(pred-hr).abs().mean()+metric.net(pred,hr).mean()+.09*texture_loss(pred,hr)
        if structural:loss=loss+.16*structure_loss(pred,hr)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),2,error_if_nonfinite=True)
        opt.step()
        if (step+1)%400==0:print(name,step+1,float(loss),flush=True)
    save_tensor(path,net.state_dict());return net.eval()

def legacy_images(data,assets):
    raw,anchor,y=data["raw"],data["v14"],data["lr"]
    _,_,_,head,ridge,choice=assets
    fine,mid,linear=corrections(data,head,ridge)
    reg=align(raw,anchor)
    spec=choice["matched"]
    def final(x):
        terms,low=phase_terms(x,data["source"],reg)
        return generate(x,terms,low,spec)
    before=anchor+fine+.25*mid+.5*linear
    published=final(before)
    donor,bb,desc=features(raw,anchor,y,True)
    g=F.interpolate(head.net(F.avg_pool2d(desc,2)).sigmoid(),size=anchor.shape[-2:],mode="bilinear",align_corners=False)
    rec={}
    for j,k in enumerate(("T_f","T_m","S_f","S_m")):
        b=j%2
        action=(1 if b==0 else .25)*g[:,j:j+1]*(1.25*(donor[b]-bb[b]) if j<2 else bb[b])
        rec[k+"_final_rms255"]=255*rms(published-final(before-action))
    return published,rec

def candidate(data,nets,mode,head,k,pre="pre"):
    raw,src=data["native"],data["source"]
    x=foundation(raw,data[pre],src,mode)
    out,(_,_,res)=nets[head](x,data["lr"],src)
    if k:out=out+k*(bands(raw)[0]-bands(out)[0])
    return out,res

def key(mode,k):return mode+"_"+str(k)

def gallery(meta,ims,data,outdir,selected):
    outdir.mkdir(exist_ok=True,parents=True);name=meta["id"];prefix=name[:2]
    box=BOXES[prefix]
    labels=["Native VOSR","Legacy full bridge","V1.5 published","Native correction","HR target"]
    frames=[pixels(ims[k]) for k in ("native","full","v15",selected)]+[pixels(data["hr"])]
    out=Image.new("RGB",(1180,536),"#fafaf8");d=ImageDraw.Draw(out)
    d.text((20,12),name+"    |    same LR / seed 6",font=font(21,True),fill="#193b3e")
    for i,(im,label) in enumerate(zip(frames,labels)):
        x=12+i*234;d.text((x,49),label,font=font(15,True),fill="#193b3e")
        thumb=im.resize((212,212),Image.Resampling.LANCZOS)
        ImageDraw.Draw(thumb).rectangle(tuple(round(z*212/512) for z in box),outline="#e93f1f",width=2)
        out.paste(thumb,(x,78))
        patch=im.crop(box);scale=min(218/patch.width,190/patch.height)
        sz=tuple(round(v*scale) for v in patch.size)
        out.paste(patch.resize(sz,Image.Resampling.NEAREST),(x,314))
    out.save(outdir/(name+"_structure.png"));frames[-2].save(outdir/(name+"_output.png"))
    vals=read(RUN/"gates"/meta["scene"]/"measured.pt")
    g=vals["gates"][0];act=vals["actions"][:,0].abs().mean(1)
    out=Image.new("RGB",(920,520),"#fafaf8");d=ImageDraw.Draw(out)
    d.text((14,9),name+"    |    published V1.5 gates and actual residuals",font=font(19,True),fill="#193b3e")
    for i,lab in enumerate(("Transfer fine","Transfer middle","Self fine","Self middle")):
        x=14+i*228;d.text((x,43),lab,font=font(16),fill="#193b3e")
        for j,v in enumerate((g[i],act[i]*255/12)):
            heat=Image.fromarray((v.clamp(0,1)*255).round().byte().numpy())
            out.paste(heat.resize((190,190)),(x,70+j*211))
    d.text((14,478),"Upper: gate 0 (black) to 1 (white). Lower: mean absolute RGB action, white = 12/255.",font=font(15),fill="#193b3e")
    out.save(outdir/(name+"_gates.png"))

def summarize(records,picked=None):
    def avg(items):
        return {k:float(np.mean([v[k] for v in items])) for k in items[0]}
    means={k:avg([r["scores"][k] for r in records]) for k in records[0]["scores"]}
    res={"n":len(records),"means":means,"diagnosis":avg([r["diagnosis"] for r in records])}
    if picked:
        contrasts={}
        for ref in ("v15","native","no_bridge","control","direct","mid","all","conf"):
            if ref not in means:continue
            contrasts[ref]={}
            for metric in means[ref]:
                v=np.array([r["scores"][picked][metric]-r["scores"][ref][metric] for r in records])
                ran=np.random.default_rng(261109).choice(v,(3000,len(v))).mean(1)
                contrasts[ref][metric]=[float(v.mean()),*np.quantile(ran,[.025,.975]).tolist()]
        res["effects"]=contrasts
    return res

def eligible(tab,k):
    a,b=tab[k],tab["v15"]
    return a["psnr"]>=b["psnr"]-.08 and a["lpips"]<=b["lpips"] and a["dists"]<=b["dists"]+.001 and a["edge"]>=b["edge"]-.003 and a["texerr"]<=b["texerr"]

@torch.no_grad()
def panel(group,items,base,camera,assets,nets,dists,choice,picked):
    records=[]
    for i,r in enumerate(items):
        file=RUN/"scores"/group/(r["scene_id"]+".json");file.parent.mkdir(exist_ok=True,parents=True)
        if file.exists():
            records.append(json.loads(file.read_text()));continue
        data=get_case(r,group,base,camera,assets)
        published,extras=legacy_images(data,assets)
        diag=diagnostic(data,assets[3],r["scene_id"])
        diag.update(extras,native_recompute_mae=float((data["raw"]-data["native"]).abs().mean()),obs_gate=float(data["obs_gate"].mean()))
        ims={k:data[k] for k in ("native","pre","full","v14")};ims["v15"]=published
        cfg=[(m,k) for m in ("direct","mid","all","conf") for k in (0.,.25,.5)]
        if choice:
            mode,k=choice
            cfg=[(m,k) for m in ("direct","mid","all","conf")]
        for m,k in cfg:ims[key(m,k)]=candidate(data,nets,m,"geometry",k)[0]
        if choice:
            sel=key(mode,k)
            ims["no_bridge"]=candidate(data,nets,mode,"geometry",k,"native")[0]
            ims["control"]=candidate(data,nets,mode,"control",k)[0]
            ims["post_full"]=candidate(data,nets,mode,"geometry",k,"full")[0]
            ims["without_new_res"]=ims[sel]-(1-k)*candidate(data,nets,mode,"geometry",k)[1]-k*(candidate(data,nets,mode,"geometry",k)[1]-bands(candidate(data,nets,mode,"geometry",k)[1])[0])
            for m in ("direct","mid","all","conf"):ims[m]=ims[key(m,k)]
        scores={}
        for n,x in ims.items():
            vals=evaluate_output(x,data["hr"],camera.metric)
            vals.update(metrics(x,data["hr"]),dists=float(dists(x.clamp(0,1),data["hr"]).mean()),
                        obs=float((camera.sensor(x.clamp(0,1))-data["lr"]).abs().mean()))
            scores[n]=vals
        row=dict(scene=r["scene_id"],camera=r.get("camera","unknown"),scores=scores,diagnosis=diag)
        if r["scene_id"] in picked and choice:
            meta=picked[r["scene_id"]];idx=meta["id"][:2]
            if idx in BOXES:
                row["object"]={m:object_scores(ims[m],data["hr"],idx) for m in ("native","full","v15",sel,"control","without_new_res")}
                gallery(meta,ims,data,RUN/"gallery",sel)
        write_json(file,row);records.append(row)
        if (i+1)%10==0:print(group,i+1,len(items),flush=True)
    return records

def main():
    torch.set_num_threads(4)
    # Avoid keeping two copies of the backbone on a shared GPU.
    while not (RUN/"cache_done.json").exists():time.sleep(10)
    base,camera=base_camera();assets=dependencies()
    import pyiqa
    dists=pyiqa.create_metric("dists",device="cuda").eval().requires_grad_(False)
    write_json(RUN/"sampling.json",sampling())
    groups=split_rows()
    samples=[]
    for i,r in enumerate(groups["train"]):
        d=get_case(r,"train",base,camera,assets)
        x=foundation(d["native"],d["pre"],d["source"],"direct")
        samples.append({k:d[k].cpu() for k in ("lr","source","hr")}|{"x":x.cpu()})
        if (i+1)%12==0:print("training inputs",i+1,flush=True)
    nets={name:fit(samples,camera.metric,name=="geometry") for name in ("control","geometry")}
    picked={r["scene"]:r for r in json.loads((ROOT/"vosr_dual_visuals_261109_v1/display_manifest.json").read_text())["records"]}
    dev=panel("dev",groups["dev"],base,camera,assets,nets,dists,None,{})
    v=summarize(dev);tab=v["means"]
    cams=[summarize([r for r in dev if r["camera"]==c])["means"] for c in ("Canon","Nikon")]
    pool=[k for k in tab if "_" in k and all(eligible(t,k) for t in cams)]
    valid=bool(pool)
    if not pool:pool=[k for k in tab if "_" in k]
    winner=min(pool,key=lambda k:(tab[k]["lpips"],tab[k]["texerr"]))
    mode,gain=winner.split("_");choice=(mode,float(gain))
    write_json(RUN/"lock.json",{"candidate":winner,"both_camera_eligible":valid,"means":tab});print("locked",winner,valid,flush=True)
    for group in ("dreal","real"):
        rec=panel(group,groups[group],base,camera,assets,nets,dists,choice,picked)
        result=summarize(rec,winner);result["eligible"]=eligible(result["means"],winner)
        result["objects"]={r["scene"]:r["object"] for r in rec if "object" in r}
        write_json(RUN/(group+".json"),result)
    good=valid and all(json.loads((RUN/(g+".json")).read_text())["eligible"] for g in ("real","dreal"))
    write_json(RUN/"result.json",dict(candidate=winner,promote=good))
    print("complete",winner,good,flush=True)

if __name__=="__main__":
    with (RUN/"study.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
