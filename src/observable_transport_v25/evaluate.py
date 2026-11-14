"""Dataset-agnostic paired evaluation on server-side cached native RGB.

Input rows: id, parent, lr, native, optional hr; paths relative to manifest.
Native is generated once at the protocol seed. This CLI does not regenerate it.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from .infer import read_image, source_identity
from .metrics import rgb_psnr
from .model import AreaCamera, Config, ObservableTransport
from .runtime import load_camera
from .artifacts import write_json, texture_metrics, panels, choose_roi


@torch.inference_mode()
def evaluate_rows(rows, root, camera, config, device="cpu", perceptual=None, output_root=None, native_provider=None):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty manifest with unique ids required")
    records = []
    variants = {"translation_only": replace(config, modes=1),
                "smooth_unfiltered": replace(config, spectral_filter=False),
                "observable_transport": config}
    for index, row in enumerate(rows):
        if not row.get("parent"):
            raise ValueError("A physical parent id is required")
        y = read_image(root/row['lr'], device)
        generation_seconds = None
        if native_provider is not None:
            if device.startswith('cuda'): torch.cuda.synchronize()
            started=time.perf_counter();native=native_provider(y)
            if device.startswith('cuda'): torch.cuda.synchronize()
            generation_seconds=time.perf_counter()-started
        elif Path(row['native']).suffix == '.pt':
            native=torch.load(root/row['native'],weights_only=True,map_location=device)['native']
        else:
            native=read_image(root/row['native'], device)
        hr = read_image(root/row["hr"], device) if row.get("hr") else None
        outputs = {"native": native}
        full_result = None
        for name, cfg in variants.items():
            if device.startswith("cuda"): torch.cuda.synchronize()
            start = time.perf_counter()
            result = ObservableTransport(camera, cfg).restore(y, native)
            if device.startswith("cuda"): torch.cuda.synchronize()
            outputs[name] = result["final"]
            if name == 'observable_transport': full_result = result
            records.append(dict(id=row["id"], parent=row["parent"], method=name,
                                seconds=time.perf_counter()-start, **result["diagnostics"]))
        records.append(dict(id=row["id"], parent=row["parent"], method="native", seconds=0., accepted_steps=0))
        panel_metadata = None
        if output_root is not None:
            case_dir=output_root/f'{index:06d}';case_dir.mkdir(parents=True,exist_ok=True)
            torch.save({**{k:v.cpu() for k,v in outputs.items()},
                        'revised':full_result['revised'].cpu(),
                        'backward_displacement':full_result['backward_displacement'].cpu(),
                        'velocity':full_result['velocity'].cpu()},case_dir/'stages.pt')
            panel_metadata=panels(y,outputs,hr,case_dir,row.get('roi'))
            motion=full_result['backward_displacement'].square().sum(1,keepdim=True)
            panel_metadata['edited_region']=None
            if float(motion.max())>1e-12:
                edit_roi=choose_roi(native,score_map=motion)
                edit=panels(y,outputs,hr,case_dir/'edited_region',edit_roi)
                edit['roi_rule']='maximum_mean_displacement_3x3_grid; diagnostic, not HR-selected'
                panel_metadata['edited_region']=edit
            write_json(case_dir/'diagnostics.json',{'model':full_result['diagnostics'],'panels':panel_metadata})
        for record in records[-4:]:
            output = outputs[record["method"]]
            if output_root is not None:
                relative = Path(f"{index:06d}") / (record["method"] + ".png")
                destination = output_root/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                arr = (output[0].clamp(0,1).permute(1,2,0)*255).round().byte().cpu().numpy()
                Image.fromarray(arr).save(destination)
                record["image"] = relative.as_posix()
            record['generation_seconds']=generation_seconds
            record['panels']=panel_metadata
            record['input_out_of_range_fraction']=float(((output<0)|(output>1)).float().mean())
            record['raw_RGB_PSNR']=rgb_psnr(output, hr) if hr is not None else None
            record["RGB_PSNR"] = rgb_psnr(output.clamp(0,1), hr) if hr is not None else None
            record.update(texture_metrics(output.clamp(0,1),hr))
            record["reference_available"] = hr is not None
            for name, metric in (perceptual or {}).items():
                # Fixed [0,1] clipping, including native, for pretrained IQA only.
                record[name] = float(metric(output.clamp(0,1), hr.clamp(0,1))) if hr is not None else None
    return records


def parent_summary(records):
    summary = {}
    for method in sorted({r["method"] for r in records}):
        group = [r for r in records if r["method"] == method]
        parents = sorted({r["parent"] for r in group})
        metrics = {}
        for metric in ("RGB_PSNR", "LPIPS", "DISTS", "HF_RMS", "HF_RMSE", "seconds"):
            means = []
            for parent in parents:
                values = [r[metric] for r in group if r["parent"] == parent and r.get(metric) is not None]
                if values: means.append(sum(values)/len(values))
            metrics[metric] = {"mean_parent_value": sum(means)/len(means) if means else None,
                               "parents_available": len(means), "parents_total": len(parents)}
        summary[method] = {"images": len(group), "metrics": metrics}
        summary[method]['accepted_images']=sum(r.get('accepted_steps',0)>0 for r in group)
        paired={}
        for metric in ('RGB_PSNR','LPIPS','DISTS','HF_RMSE'):
            deltas=[]
            for parent in parents:
                differences=[]
                for r in group:
                    if r['parent']!=parent: continue
                    base=next(b for b in records if b['id']==r['id'] and b['method']=='native')
                    a,b=r.get(metric),base.get(metric)
                    if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):differences.append(a-b)
                if differences:deltas.append(sum(differences)/len(differences))
            paired[metric]={'mean_parent_delta':sum(deltas)/len(deltas) if deltas else None,
                            'finite_parents':len(deltas),'parents_total':len(parents)}
        summary[method]['paired_vs_base']=paired
    return summary


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--runtime-root", default=None, help="Asset directory; defaults to VOSR_RUNTIME_ROOT")
    p.add_argument("--camera", choices=("deployed", "synthetic-area"), default="deployed")
    p.add_argument("--config", default=str(Path(__file__).parents[2]/"configs/observable_transport_v25.json"))
    p.add_argument("--device", default="cuda"); p.add_argument("--perceptual", action="store_true")
    a = p.parse_args(); manifest = Path(a.manifest)
    protocol = json.loads(Path(a.config).read_text(encoding="utf-8"))
    cfg = Config(**protocol["model"])
    camera = AreaCamera(cfg.scale) if a.camera == "synthetic-area" else load_camera(a.runtime_root, a.device)
    metrics = {}
    if a.perceptual:
        import pyiqa
        metrics = {n: pyiqa.create_metric(n.lower(), device=a.device) for n in ("LPIPS", "DISTS")}
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    dest = Path(a.output_dir); dest.mkdir(parents=True, exist_ok=False)
    records = evaluate_rows(rows, manifest.parent, camera, cfg, a.device, metrics, dest)
    pack = {"summary": parent_summary(records), "records": records,
            "camera": a.camera, "native_source": "precomputed; .pt=float tensor, image=quantized RGB; identity not verified by this legacy CLI",
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
            "source_sha256": source_identity()}
    # IEEE infinity (perfect PSNR) is explicitly serialized as a string.
    write_json(dest/'evaluation.json',pack)


if __name__ == "__main__": main()
