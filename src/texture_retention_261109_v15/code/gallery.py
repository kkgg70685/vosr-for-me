"""Render the original six display scenes and anonymized V1.4/V1.5 crops."""
import fcntl
import hashlib
import json
import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from release_study import RUN,ROOT,Path,load_release,calibrated,proposal,split_rows,case_path,write_json,save_png


def font(size,bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",size)


def pixels(x):
    return Image.fromarray(x[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy())


def texture_box(lr):
    gray=np.asarray(pixels(lr).convert("L"),dtype=np.float32)
    gy,gx=np.gradient(gray)
    candidates=[]
    for y in range(0,97,8):
        for x in range(0,97,8):
            dx,dy=gx[y:y+32,x:x+32],gy[y:y+32,x:x+32]
            a,b,c=(dx*dx).mean(),(dy*dy).mean(),(dx*dy).mean()
            candidates.append((a+b-np.sqrt((a-b)**2+4*c*c),x,y))
    _,x,y=max(candidates,key=lambda e:e[0])
    return (4*x,4*y,4*(x+32),4*(y+32))


def board(row,frames,old_crop,tex_crop,metrics):
    bg="#f6f7f5";ink="#192e34";accent="#08786a"
    out=Image.new("RGB",(1492,932),bg);draw=ImageDraw.Draw(out)
    draw.text((24,16),row["id"]+"   /   "+row["scene"],font=font(25,True),fill=ink)
    draw.text((24,57),"Same input and seed 6   |   RGB PSNR / LPIPS   |   Cyan: previous crop     Gold: LR texture crop",font=font(16),fill=ink)
    for i,(im,label) in enumerate(zip(frames,("Native VOSR","V1.4 dual trajectory","V1.5 texture recovery","HR reference"))):
        x=24+i*364
        draw.text((x,94),label,font=font(18,True),fill=accent if i==2 else ink)
        resized=im.resize((352,352),Image.Resampling.LANCZOS)
        for box,color in ((old_crop,"#00d7c4"),(tex_crop,"#fbb535")):
            ImageDraw.Draw(resized).rectangle(tuple(round(c*352/512) for c in box),outline=color,width=2)
        out.paste(resized,(x,124))
        if i<3:
            m=metrics[i]
            draw.text((x,487),f"{m['psnr']:.2f} dB   |   {m['lpips']:.4f}",font=font(15),fill=ink)
        for j,box in enumerate((old_crop,tex_crop)):
            crop=im.crop(box).resize((192,192),Image.Resampling.NEAREST)
            out.paste(crop,(x+80,526+j*200))
    return out


@torch.no_grad()
def main():
    choice,color,head=load_release()
    prior=ROOT/"vosr_dual_visuals_261109_v1/display_manifest.json"
    rows=json.loads(prior.read_text())["records"]
    folder=RUN/"gallery";folder.mkdir(exist_ok=True)
    blind=Image.new("RGB",(1056,1406),"#f6f7f5")
    draw=ImageDraw.Draw(blind);ink="#192e34"
    draw.text((20,14),"Texture comparison | anonymized A / B",font=font(28,True),fill=ink)
    draw.text((20,59),"Compare A and B with HR. Labels are shuffled within each row; see key after scoring.",font=font(16),fill=ink)
    rng=random.Random(151109)
    flips=[False]*3+[True]*3;rng.shuffle(flips)
    records=[];key={}
    splits=split_rows()
    for i,row in enumerate(rows):
        group="real" if row["group"].startswith("RealSR") else "dreal"
        source=next(r for r in splits[group] if r["scene_id"]==row["scene"])
        path=case_path(group,source,6)
        raw=torch.load(path,map_location="cpu",weights_only=True)
        data={k:raw[k].cuda() for k in ("raw","v14","lr","hr")}
        if choice["coarse"]:
            out=calibrated(data,head,color,choice["method"],choice["fine"],choice["coarse"])
        else:
            donor=proposal(choice["method"],head,data["raw"],data["v14"],data["lr"]) if head or choice["method"]!="identity" else data["v14"]
            out=torch.lerp(data["v14"],donor,choice["fine"])
        scores=json.loads((RUN/"calibrated_scores"/group/(path.stem+".json")).read_text())["scores"]
        metrics=[scores[k] for k in ("raw","v14",choice["key"])]
        frames=[pixels(x) for x in (data["raw"],data["v14"],out,data["hr"])]
        crop=texture_box(data["lr"])
        name=row["id"];out_name=name+"_v15.png";board_name=name+"_compare.png"
        frames[2].save(folder/out_name)
        board(row,frames,row["crop_hr"],crop,metrics).save(folder/board_name)
        seq=[frames[1],frames[2]] if not flips[i] else [frames[2],frames[1]]
        key[name]="B" if not flips[i] else "A"
        y=100+i*216
        draw.text((20,y+12),name.replace("_"," "),font=font(14,True),fill=ink)
        for j,(im,lab) in enumerate(zip(seq+[frames[3]],("A","B","HR"))):
            x=238+j*267
            draw.text((x,y+3),lab,font=font(15,True),fill=ink)
            blind.paste(im.crop(crop).resize((182,182),Image.Resampling.NEAREST),(x,y+26))
        records.append(dict(id=name,scene=row["scene"],output=out_name,comparison=board_name,old_crop=row["crop_hr"],texture_crop=crop,
                       metrics=[{k:m[k] for k in ("psnr","lpips","texerr")} for m in metrics]))
    blind.save(folder/"blind.png")
    write_json(folder/"key.json",key)
    write_json(RUN/"gallery.json",dict(completed=len(records),records=records))


if __name__=="__main__":
    torch.set_num_threads(4)
    with (RUN/"calibration.lock").open("a+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
    main()
