"""Dataset-agnostic paired evaluation on server-side cached native RGB.

Input rows: id, parent, lr, native, optional hr; paths relative to manifest.
Native is generated once at the protocol seed. This CLI does not regenerate it.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from .infer import read_image, source_identity
from .metrics import rgb_psnr
from .model import AreaCamera, Config, ObservableTransport
from .runtime import load_camera


def evaluate_rows(rows, root, camera, config, device="cpu", perceptual=None, output_root=None):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty manifest with unique ids required")
    records = []
    variants = {"translation_only": replace(config, modes=1),
                "smooth_unfiltered": replace(config, spectral_filter=False),
                "observable_transport": config}
    for index, row in enumerate(rows):
        if not row.get("parent"):
            raise ValueError("A physical parent id is required")
        y, native = (read_image(root/row[k], device) for k in ("lr", "native"))
        hr = read_image(root/row["hr"], device) if row.get("hr") else None
        outputs = {"native": native}
        for name, cfg in variants.items():
            if device.startswith("cuda"): torch.cuda.synchronize()
            start = time.perf_counter()
            result = ObservableTransport(camera, cfg).restore(y, native)
            if device.startswith("cuda"): torch.cuda.synchronize()
            outputs[name] = result["final"]
            records.append(dict(id=row["id"], parent=row["parent"], method=name,
                                seconds=time.perf_counter()-start, **result["diagnostics"]))
        records.append(dict(id=row["id"], parent=row["parent"], method="native", seconds=0.))
        for record in records[-4:]:
            output = outputs[record["method"]]
            if output_root is not None:
                relative = Path(f"{index:06d}") / (record["method"] + ".png")
                destination = output_root/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                arr = (output[0].clamp(0,1).permute(1,2,0)*255).round().byte().cpu().numpy()
                Image.fromarray(arr).save(destination)
                record["image"] = relative.as_posix()
            record["RGB_PSNR"] = rgb_psnr(output, hr) if hr is not None else None
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
        for metric in ("RGB_PSNR", "LPIPS", "DISTS", "seconds"):
            means = []
            for parent in parents:
                values = [r[metric] for r in group if r["parent"] == parent and r.get(metric) is not None]
                if values: means.append(sum(values)/len(values))
            metrics[metric] = {"mean_parent_value": sum(means)/len(means) if means else None,
                               "parents_available": len(means), "parents_total": len(parents)}
        summary[method] = {"images": len(group), "metrics": metrics}
    return summary


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--runtime-root", default="/home/zhangxu/hx/HX001")
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
            "camera": a.camera, "native_source": "precomputed_quantized_rgb",
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
            "source_sha256": source_identity()}
    # IEEE infinity (perfect PSNR) is explicitly serialized as a string.
    text = json.dumps(pack, indent=2).replace("Infinity", '"Infinity"')
    (dest/"evaluation.json").write_text(text, encoding="utf-8")


if __name__ == "__main__": main()
