"""Intervene on registration and gate actions inside the locked model; render matched examples."""
import argparse
import fcntl
import hashlib
import json
import random
import subprocess
import sys
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
from semantic_study import (PUB,RUN,OLD,ROOT,CODE,rows,base_camera,model_parts,obtain,retained,phase_terms,generate,
                            align,sources,predict,load_heads as load_sem_heads,MODES)
from semantic_model import load_heads as sem_heads
from balanced import load_heads,corrections,case_path
from texture import features,luminance,bands
from evidence import save_png,write_json
from gallery import board,texture_box,pixels,font

OUT=PUB/"final"
MOD=["v14","reference","published","fidelity","no_align","retrained_no_align","transfer","self","neither"]


def action_terms(data,head,registered):
    donor,base,desc=features(data["raw"],data["v14"],data["lr"],registered)
    g=head.net(F.avg_pool2d(desc,2)).sigmoid()
    g=F.interpolate(g,size=data["v14"].shape[-2:],mode="bilinear",align_corners=False)
    transfer=[1.25*g[:,j:j+1]*(donor[j]-base[j]) for j in range(2)]
    self_detail=[g[:,j+2:j+3]*base[j] for j in range(2)]
    return transfer[0]+.25*transfer[1],self_detail[0]+.25*self_detail[1]


def method_images(data,head,no_reg,coef,source,spec):
    base=data["v14"]
    fine,mid,linear=corrections(data,head,coef)
    transfer,self_detail=action_terms(data,head,True)
    ut,us=action_terms(data,head,False);rt,rs=action_terms(data,no_reg,False)
    fixed=base+.5*linear
    bases={"published":fixed+transfer+self_detail,"fidelity":fixed+fine,"transfer":fixed+transfer,
           "self":fixed+self_detail,"neither":fixed,"no_align":fixed+ut+us,"retrained_no_align":fixed+rt+rs}
    result={"v14":base,"reference":base+fine+.25*mid+.5*linear}
    reg=align(data["raw"],base)
    for method,z in bases.items():
        native=data["raw"] if method in ("no_align","retrained_no_align") else reg
        terms,low=phase_terms(z,source,native)
        result[method]=generate(z,terms,low,spec)
    return result


def tiles(data,hr):
    y=luminance(hr)
    dx=F.pad(y[...,1:]-y[...,:-1],(0,1,0,0));dy=F.pad(y[...,1:,:]-y[...,:-1,:],(0,0,0,1))
    pool=lambda z:F.avg_pool2d(z,16)
    xx,yy,xy=pool(dx.square()),pool(dy.square()),pool(dx*dy)
    val=(xx+yy-((xx-yy).square()+4*xy.square()).clamp_min(0).sqrt()).flatten()
    idx=val.topk(val.numel()//4).indices
    target=[(pool(luminance(x).square())+.000009).sqrt().flatten()[idx] for x in bands(hr)]
    err={}
    for k in ("neither","transfer","self","published"):
        band=bands(data[k].clamp(0,1))
        err[k]=sum(((pool(luminance(x).square())+.000009).sqrt().flatten()[idx]/t).log().abs()
                   for x,t in zip(band,target))*.5
    a=err["transfer"]<err["neither"];b=err["self"]<err["neither"]
    return {k:float(v.float().mean()) for k,v in {"transfer_only":a&~b,"self_only":b&~a,"both":a&b,"neither":~(a|b),
                    "combined_beats_each":(err["published"]<err["self"])&(err["published"]<err["transfer"])}.items()}


def ci(x):
    x=np.array(x);lo,hi=np.quantile(np.random.default_rng(261109).choice(x,(5000,len(x))).mean(1),[.025,.975])
    return [float(x.mean()),float(lo),float(hi)]


@torch.no_grad()
def run():
    import pyiqa
    from semantic_study import evaluate_output
    from balanced import pools
    torch.hub.set_dir(str(ROOT/"runtime_cache/torch/hub"))
    metric=pyiqa.create_metric("lpips",device="cuda").eval().requires_grad_(False)
    dists=pyiqa.create_metric("dists",device="cuda").eval().requires_grad_(False)
    release=json.loads((PUB/"release.json").read_text())
    lock=release["choice"];mode=lock["selected"].split("/")[0];spec=lock["matched"]
    assert mode in MODES
    oldnets,head,coef=model_parts();lrnet=oldnets["localnet"];nets=sem_heads()
    no_reg=load_heads("unregistered")["unregistered"]
    params=torch.load(PUB/"pca.pt",weights_only=True,map_location="cuda")
    groups=rows();groups["additional_dreal"]=pools()[1]
    original=json.loads((ROOT/"vosr_dual_visuals_261109_v1/display_manifest.json").read_text())["records"]
    picked={r["scene"]:r for r in original}
    snapshots={}
    for group in ("real","dreal","new_real","new_dreal","additional_dreal"):
        records=[]
        for idx,r in enumerate(groups[group]):
            seed=37 if group=="additional_dreal" else (29 if group.startswith("new_") else 6)
            file=OUT/"records"/group/(case_path(group,r,seed).stem+".json")
            selected_sample=group in ("real","dreal") and r["scene_id"] in picked
            if not file.exists() or selected_sample:
                data=obtain(r,group,seed,None,None)
                proof=sources(data,r,None,nets,lrnet,params,False)
                ims=method_images(data,head,no_reg,coef,proof[mode],spec)
                scores={}
                for k,x in ims.items():
                    m=evaluate_output(x,data["hr"],metric)
                    m["dists"]=float(dists(x.clamp(0,1),data["hr"]).mean())
                    scores[k]=m
                result=dict(scene=r["scene_id"],scores=scores,tiles=tiles(ims,data["hr"]))
                write_json(file,result)
                if selected_sample:
                    snapshots[r["scene_id"]]=gallery_item(picked[r["scene_id"]],ims,data,scores)
            records.append(json.loads(file.read_text()))
            if (idx+1)%25==0:print(group,idx+1,len(groups[group]),flush=True)
        means={k:{f:float(np.mean([r["scores"][k][f] for r in records])) for f in records[0]["scores"][k]} for k in MOD}
        contrasts={k:{f:ci([r["scores"]["published"][f]-r["scores"][k][f] for r in records]) for f in means[k]}
                   for k in MOD if k!="published"}
        fidelity={f:ci([r["scores"]["fidelity"][f]-r["scores"]["v14"][f] for r in records]) for f in means["v14"]}
        nbetter={m:sum(r["scores"]["published"][m]<r["scores"]["v14"][m] for r in records) for m in ("lpips","texerr","dists")}
        tile={k:ci([r["tiles"][k] for r in records]) for k in records[0]["tiles"]}
        write_json(OUT/(group+".json"),dict(cases=len(records),means=means,contrasts=contrasts,fidelity=fidelity,nbetter=nbetter,tiles=tile))
    blind_and_overview(original,snapshots)
    write_json(OUT/"gallery.json",dict(records=[snapshots[r["scene"]][0] for r in original]))
    print("confirmed",flush=True)


def gallery_item(row,ims,data,scores):
    folder=OUT/"gallery";folder.mkdir(parents=True,exist_ok=True)
    name=row["id"];crop=texture_box(data["lr"]);final=ims["published"]
    frames=[pixels(x) for x in (data["raw"],data["v14"],final,data["hr"])]
    metric_file=PUB/"scores"/("dreal" if row["group"]=="DRealSR" else "real")/(hashlib.sha256(row["scene"].encode()).hexdigest()[:20]+"_6.json")
    prior=json.loads(metric_file.read_text())["scores"]
    metrics=[prior["raw"],scores["v14"],scores["published"]]
    image=board(row,frames,row["crop_hr"],crop,metrics)
    image.save(folder/(name+"_compare.png"))
    save_png(final,folder/(name+"_v15.png"))
    save_png(ims["fidelity"],folder/(name+"_fidelity.png"))
    info=dict(id=name,scene=row["scene"],output=name+"_v15.png",fidelity=name+"_fidelity.png",comparison=name+"_compare.png",
              metrics=[{k:m[k] for k in ("psnr","lpips","texerr")} for m in metrics],texture_crop=crop)
    return info,[im.crop(crop) for im in frames]


def blind_and_overview(original,snapshots):
    folder=OUT/"gallery";bg="#f6f7f5";ink="#192e34"
    blind=Image.new("RGB",(1056,1406),bg);d=ImageDraw.Draw(blind)
    d.text((20,14),"True texture or invented detail?   |   A / B",font=font(28,True),fill=ink)
    d.text((20,59),"Match A and B to the HR crop. Their order changes between scenes.",font=font(16),fill=ink)
    overview=Image.new("RGB",(1190,1084),bg);od=ImageDraw.Draw(overview)
    od.text((24,15),"Matched texture crops  /  V1.4, V1.5 and HR",font=font(28,True),fill=ink)
    rng=random.Random(15110961);flips=[0]*3+[1]*3;rng.shuffle(flips);key={}
    for i,r in enumerate(original):
        info,ims=snapshots[r["scene"]]
        seq=[ims[2],ims[1]] if flips[i] else [ims[1],ims[2]]
        key[info["id"]]="A" if flips[i] else "B"
        y=100+i*216;d.text((20,y+12),info["id"].replace("_"," "),font=font(14,True),fill=ink)
        for j,(im,label) in enumerate(zip(seq+[ims[3]],("A","B","HR"))):
            x=238+j*267;d.text((x,y+3),label,font=font(15,True),fill=ink)
            blind.paste(im.resize((182,182),Image.Resampling.NEAREST),(x,y+26))
        x=24+(i%2)*583;y=83+(i//2)*327
        od.text((x,y),info["id"].replace("_"," "),font=font(17,True),fill=ink)
        for j,(im,label) in enumerate(zip(ims[1:],("V1.4","V1.5","HR"))):
            xx=x+j*185;od.text((xx,y+34),label,font=font(16),fill=ink)
            overview.paste(im.resize((175,175),Image.Resampling.NEAREST),(xx,y+64))
    blind.save(folder/"blind.png");overview.save(folder/"overview.png")
    write_json(folder/"key.json",key)


def pack():
    dest=RUN/"v15_vision_release.zip"
    folders={OLD:["learned.pt","ridge.pt","balanced_release.json"],
             RUN:["localnet.pt","local.pt","semantic.pt"],
             PUB:["local.pt","internal.pt","vision.pt","pca.pt","release.json"]}
    with zipfile.ZipFile(dest,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path,files in folders.items():
            for f in files:z.write(path/f,(path/f).relative_to(ROOT))
        for path in (OLD/"code",CODE):
            for f in path.iterdir():
                if f.suffix in (".py",".json"):z.write(f,f.relative_to(ROOT))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def cli():
    dev=dev_pool_for_cli()
    subprocess.run([sys.executable,str(CODE/"semantic_study.py"),"restore","--input",str(ROOT/dev["lr"]),
                    "--output",str(OUT/"cli.png"),"--seed","6"],cwd=RUN,check=True)
    a=np.asarray(Image.open(OUT/"cli.png")).astype("float32");b=np.asarray(Image.open(PUB/"release_check.png")).astype("float32")
    assert a.shape==b.shape==(512,512,3)
    diff=np.abs(a-b)
    if diff.max()>3 or diff.mean()>.2:raise ValueError("CLI did not reproduce selected model")
    return dict(shape=list(a.shape),maximum=float(diff.max()),mean=float(diff.mean()))


def dev_pool_for_cli():
    from phase_study import dev_pool
    return dev_pool(rows()["dev"])[0]


if __name__=="__main__":
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True,parents=True)
    if "--evaluate" in sys.argv:
        run()
    else:
        with (OUT/"lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not (OUT/"gallery.json").is_file():
                subprocess.run([sys.executable,str(Path(__file__)),"--evaluate"],cwd=RUN,check=True)
            smoke=cli()
            write_json(RUN/"final.json",dict(selected=json.loads((PUB/"release.json").read_text())["selected"],
                       cli=smoke,package_sha=pack(),gallery=str(OUT/"gallery")))
