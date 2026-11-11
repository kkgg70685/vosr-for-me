"""Train the local predictor and image-internal readout, then lock fusion on validation."""
import fcntl
import json
import numpy as np
import torch
from torch.nn import functional as F

from evidence import (ROOT,OLD,RUN,CODE,P,rows,base_camera,refpath,cache_refs,image,save_tensor,write_json,
                      LocalSR,ReferenceGate,bands,case_path,render,evaluate_output,aggregate,load_heads)
from balanced import corrections
from texture import texture_loss

FAMILIES=("localnet","patch","semantic")
HIGH=(0,.25,.5,.8)
LOW=(0,.25,.5)


def load_train(record,bank):
    result={k:image(ROOT/record[k]).float()/255 for k in ("lr","HR")}
    if bank:
        result["ref"]=torch.load(refpath(record["scene_id"]),weights_only=True)[bank][0]
    return result


def train_source(train,metric,local,mode):
    is_local=mode=="localnet"
    path=RUN/(mode+".pt")
    torch.manual_seed(P["seed"])
    model=(LocalSR() if is_local else ReferenceGate()).cuda()
    if path.exists():
        model.load_state_dict(torch.load(path,map_location="cuda",weights_only=True));return model.eval()
    records=[load_train(r,None if is_local else mode) for r in train]
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    steps=P["training_updates"] if is_local else 600
    for step in range(steps):
        rng=np.random.default_rng(P["pretrain_rng"]+step)
        xs,ts,refs=[],[],[]
        for _ in range(P["batch"]):
            row=records[rng.integers(len(records))]
            n=P["crop_lr"];y,x=rng.integers(0,129-n,2)
            xs.append(row["lr"][:,y:y+n,x:x+n])
            ts.append(row["HR"][:,4*y:4*(y+n),4*x:4*(x+n)])
            if not is_local:refs.append(row["ref"][:,y:y+n,x:x+n])
        y,hr=torch.stack(xs).cuda(),torch.stack(ts).cuda()
        if is_local:rgb=model(y)
        else:
            with torch.no_grad():pred=local(y)
            rgb=model(y,pred,torch.stack(refs).cuda().float())
        pred=rgb[...,8:-8,8:-8].clamp(0,1);truth=hr[...,8:-8,8:-8]
        loss=3*(pred-truth).abs().mean()+8*(pred-truth).square().mean()+.5*metric.net(pred,truth).mean()+.06*texture_loss(pred,truth)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True)
        opt.step()
        if (step+1)%400==0:print(mode,step+1,float(loss),flush=True)
    save_tensor(path,model.state_dict())
    return model.eval()


def load_net(mode):
    net=(LocalSR() if mode=="localnet" else ReferenceGate()).cuda().eval()
    filename="local" if mode=="patch" else mode
    net.load_state_dict(torch.load(RUN/(filename+".pt"),map_location="cuda",weights_only=True))
    return net


def retained(data,head,coef):
    f,m,offset=corrections(data,head,coef)
    spec=json.loads((OLD/"balanced_release.json").read_text())["spec"]
    return data["v14"]+spec[0]*f+spec[1]*m+spec[2]*offset


def combine(base,source,h,l):
    diff=source-base
    f,m=bands(diff)
    return base+h*(f+m)+l*(diff-f-m)


def key(mode,h,l):
    return f"{mode}/{h:g}/{l:g}"


def obtain(row,group,seed,base,camera):
    # Reuse exact cached V1.5 tensors when available. Never use HR in either generator.
    folders={"dev":["dev","caldev","baldev"],"real":["real"],"dreal":["dreal"],"fitanchors":["train"]}
    for folder in folders.get(group,[]):
        path=case_path(folder,row,seed)
        if path.is_file():
            s=torch.load(path,map_location="cpu",weights_only=True)
            return {k:s[k].cuda() for k in ("raw","v14","hr","lr")}
    path=RUN/"cache"/group/(case_path(group,row,seed).name)
    if not path.exists():
        y=image(ROOT/row["lr"])[None].cuda().float()/255
        stages=render(base,camera,y,seed)
        path.parent.mkdir(parents=True,exist_ok=True)
        save_tensor(path,dict(lr=y.cpu(),raw=stages["raw"].cpu(),v14=stages["v14"].cpu(),
                              hr=image(ROOT/row["HR"])[None].float()/255))
    s=torch.load(path,map_location="cpu",weights_only=True)
    return {k:x.cuda() for k,x in s.items()}


def configurations(lock):
    if lock is None:return [(mode,h,l) for mode in FAMILIES for h in HIGH for l in LOW if h or l]
    jobs=[(mode,*lock["families"][mode]) for mode in FAMILIES]
    hs,ls=lock["families"]["semantic"]
    jobs.extend((mode,hs,ls) for mode in ("shift","wrong","empty"))
    return jobs


@torch.no_grad()
def panel(base,camera,rows,group,seed,networks,head,coef,lock):
    records=[]
    for index,r in enumerate(rows):
        path=RUN/"scores"/group/(case_path(group,r,seed).stem+".json")
        if not path.exists():
            data=obtain(r,group,seed,base,camera)
            if not refpath(r["scene_id"]).exists():cache_refs(base,[r])
            bank=torch.load(refpath(r["scene_id"]),weights_only=True)
            y=data["lr"];hr=data["hr"];anchor=retained(data,head,coef)
            pred=networks["localnet"](y)
            sources={"localnet":pred}
            for mode,reference,gate in [("patch","local","patch"),("semantic","semantic","semantic"),
                                       ("shift","misplaced","semantic"),("wrong","wrong_value","semantic")]:
                if lock is not None or mode in FAMILIES:
                    sources[mode]=networks[gate](y,pred,bank[reference].cuda().float())
            sources["empty"]=pred
            measured={k:evaluate_output(x,hr,camera.metric) for k,x in (("raw",data["raw"]),("v14",data["v14"]),("reference",anchor))}
            for mode,h,l in configurations(lock):
                measured[key(mode,h,l)]=evaluate_output(combine(anchor,sources[mode],h,l),hr,camera.metric)
            write_json(path,dict(scene=r["scene_id"],camera=r.get("camera","unknown"),scores=measured))
        records.append(json.loads(path.read_text()))
        if (index+1)%10==0:print(group,index+1,len(rows),flush=True)
    return records,aggregate(records)


def choose(records,means):
    eligible=set(means)
    for camera in ("Canon","Nikon"):
        m=aggregate([r for r in records if r["camera"]==camera])
        ref=m["reference"]
        eligible&={k for k,v in m.items() if v["psnr"]>=ref["psnr"]-.03 and v["lpips"]<=ref["lpips"] and v["texerr"]<=ref["texerr"]}
    families={}
    winners=[]
    for mode in FAMILIES:
        pool=[k for k in eligible if k.startswith(mode+"/")]+["reference"]
        best=min(pool,key=lambda k:(means[k]["lpips"],means[k]["texerr"],k))
        families[mode]=[0.,0.] if best=="reference" else list(map(float,best.split("/")[1:]))
        winners.append(best)
    return dict(selected=min(winners,key=lambda k:(means[k]["lpips"],means[k]["texerr"],k)),families=families)


def effects(records,winner):
    output={}
    for k in records[0]["scores"]:
        if k==winner:continue
        output[k]={}
        for metric in ("psnr","lpips","texerr","fine_ratio","mid_ratio","correlation"):
            d=np.array([r["scores"][winner][metric]-r["scores"][k][metric] for r in records])
            lo,hi=np.quantile(np.random.default_rng(615).choice(d,(5000,len(d))).mean(1),[.025,.975])
            output[k][metric]=[float(d.mean()),float(lo),float(hi)]
    return output


def main():
    splits=rows()
    base,camera=base_camera()
    cache_refs(base,splits["train"]+splits["dev"])
    local=train_source(splits["train"],camera.metric,None,"localnet")
    nets={"localnet":local,"patch":train_source(splits["train"],camera.metric,local,"local"),
          "semantic":train_source(splits["train"],camera.metric,local,"semantic")}
    head=load_heads("learned")["learned"]
    coef=torch.load(OLD/"ridge.pt",weights_only=True,map_location="cuda")
    dev,means=panel(base,camera,splits["dev"],"dev",6,nets,head,coef,None)
    lock=choose(dev,means)
    if (RUN/"lock.json").exists() and json.loads((RUN/"lock.json").read_text())["choice"]!=lock:raise ValueError("Selection changed")
    write_json(RUN/"lock.json",dict(selected=lock["selected"],choice=lock,means=means))
    print("locked",lock,flush=True)
    for group in ("real","dreal","new_real","new_dreal"):
        seed=29 if group.startswith("new_") else 6
        rec,m=panel(base,camera,splits[group],group,seed,nets,head,coef,lock)
        write_json(RUN/(group+".json"),dict(cases=len(rec),selected=lock["selected"],means=m,effects=effects(rec,lock["selected"])))
    write_json(RUN/"release.json",dict(selected=lock["selected"],choice=lock,base_run=str(OLD)))


if __name__=="__main__":
    torch.set_num_threads(4)
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/"study.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
