"""Server-side float artifacts, fixed-scale comparison panels, strict JSON."""
import json
import csv
import html
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F


def json_ready(value):
    if isinstance(value, dict): return {k: json_ready(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'Infinity' if value == math.inf else '-Infinity' if value == -math.inf else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def highpass(x):
    """RGB minus separable [1,4,6,4,1]/16 lowpass; HR-pixel domain."""
    k=x.new_tensor([1,4,6,4,1])/16
    c=x.shape[1]
    low=F.conv2d(F.pad(x,(2,2,0,0),mode='replicate'),k.view(1,1,1,5).expand(c,1,1,5),groups=c)
    low=F.conv2d(F.pad(low,(0,0,2,2),mode='replicate'),k.view(1,1,5,1).expand(c,1,5,1),groups=c)
    return x-low


def texture_metrics(output, reference):
    hp=highpass(output)[...,4:-4,4:-4]
    scores={'HF_RMS':float(hp.square().mean().sqrt())}
    scores['HF_RMSE']=float((hp-highpass(reference)[...,4:-4,4:-4]).square().mean().sqrt()) if reference is not None else None
    return scores


def image_array(x):
    return (x[0].detach().clamp(0,1).permute(1,2,0)*255).round().byte().cpu().numpy()


def save_rgb(x, path):
    Image.fromarray(image_array(x)).save(path)


def choose_roi(native, size=128, score_map=None):
    """Reference-free deterministic base-only crop; never pick v2.5's best gain."""
    h,w=native.shape[-2:];size=min(size,h,w)
    energy=highpass(native).square().mean(1,keepdim=True) if score_map is None else score_map
    best=(-1.,0,0)
    for y in sorted(set([0,(h-size)//2,h-size])):
        for x in sorted(set([0,(w-size)//2,w-size])):
            score=float(energy[...,y:y+size,x:x+size].mean())
            if score>best[0]:best=(score,x,y)
    return [best[1],best[2],size,size]


def panels(lr, outputs, reference, directory, roi=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    native=outputs['native'];h,w=native.shape[-2:]
    roi=choose_roi(native) if roi is None else roi
    if len(roi)!=4 or any(type(v) is not int for v in roi): raise ValueError('ROI must be integer [x,y,w,h]')
    x,y,rw,rh=roi
    if min(x,y)<0 or min(rw,rh)<1 or x+rw>w or y+rh>h:raise ValueError('ROI outside image')
    bicubic=F.interpolate(lr,(h,w),mode='bicubic',align_corners=False)
    views={'LR bicubic':bicubic,'Base native':native,'v2.5 final':outputs['observable_transport']}
    for key in ('translation_only','smooth_unfiltered'):
        if key in outputs: views[key]=outputs[key]
    if reference is not None:views['HR reference']=reference
    views['abs(final-base) x 8']=(outputs['observable_transport']-native).abs()*8
    full=Image.new('RGB',(w*len(views),h+32),'white');draw=ImageDraw.Draw(full)
    for i,(name,tensor) in enumerate(views.items()):
        full.paste(Image.fromarray(image_array(tensor)),(i*w,32));draw.text((i*w+8,8),name,fill='black')
        draw.rectangle((i*w+x,32+y,i*w+x+rw-1,32+y+rh-1),outline='yellow',width=2)
    full.save(directory/'base_vs_v25.png')
    texture_views={k:v for k,v in views.items() if not k.startswith('abs(')}
    zoom=2;tw,th=rw*zoom,rh*zoom
    strip=Image.new('RGB',(tw*len(texture_views),2*th+64),'white');draw=ImageDraw.Draw(strip)
    clipped={}
    for i,(name,tensor) in enumerate(texture_views.items()):
        crop=tensor[...,y:y+rh,x:x+rw]
        band=highpass(tensor)[...,y:y+rh,x:x+rw]
        encoded=.5+4*band
        clipped[name]=float(((encoded<0)|(encoded>1)).float().mean())
        strip.paste(Image.fromarray(image_array(crop)).resize((tw,th),Image.Resampling.NEAREST),(i*tw,32))
        strip.paste(Image.fromarray(image_array(encoded)).resize((tw,th),Image.Resampling.NEAREST),(i*tw,th+64))
        draw.text((i*tw+8,8),name+' / same ROI',fill='black')
        draw.text((i*tw+8,th+40),'signed HF: 0.5 + 4*band',fill='black')
    strip.save(directory/'high_frequency_comparison.png')
    return {'roi_xywh':roi,'roi_rule':'manifest_or_max_base_HF_3x3_grid','HF_display_gain':4,'HF_display_clipped_fraction':clipped,
            'comparison':'base_vs_v25.png','texture':'high_frequency_comparison.png'}


def write_report(directory, result, matching=None):
    """Static, server-only report; retains all cases in fixed manifest order."""
    directory=Path(directory)
    columns=['method','RGB_PSNR','LPIPS','DISTS','HF_RMSE','HF_RMS','seconds','accepted_images','images']
    table=[]
    for method, summary in result['summary'].items():
        table.append({'method':method,**{k:v['mean_parent_value'] for k,v in summary['metrics'].items()},
                      'accepted_images':summary['accepted_images'],'images':summary['images']})
    with (directory/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(table)
    def esc(x):return html.escape(str(x))
    rows=''.join('<tr>'+''.join('<td>'+esc(r.get(c))+'</td>' for c in columns)+'</tr>' for r in table)
    body='<h1>VOSR v2.5 / '+esc(result['identity']['phase'])+'</h1><p>All cases; one seed; parent macro means. HF energy alone is not quality.</p>'
    body+='<table border="1"><tr>'+''.join('<th>'+c+'</th>' for c in columns)+'</tr>'+rows+'</table>'
    if matching:body+='<h2>Homography (HR pixels)</h2><pre>'+esc(json.dumps(matching['summary'],indent=2))+'</pre>'
    for r in result['records']:
        if r['method']!='observable_transport':continue
        folder=Path(r['image']).parent.as_posix()
        body+='<h2>'+esc(r['id'])+'</h2><p>Accepted steps: '+esc(r['accepted_steps'])+'</p>'
        for name in ('base_vs_v25.png','high_frequency_comparison.png','edited_region/high_frequency_comparison.png'):
            if (directory/folder/name).is_file():body+='<p>'+esc(name)+'</p><a href="'+esc(folder+'/'+name)+'"><img loading="lazy" style="max-width:100%" src="'+esc(folder+'/'+name)+'"></a>'
        if not r['accepted_steps']:body+='<p>No accepted motion: final equals base. No edited-region preservation claim.</p>'
    (directory/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>VOSR v2.5 evaluation</title><style>body{font-family:system-ui;margin:24px}td,th{padding:6px}table{border-collapse:collapse}</style>'+body,encoding='utf-8')
