"""Matched-seed stage evaluation and parent-level bootstrap, no test-time tuning."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import ndimage
import torch
from .data import load_row, read_manifest
from .model import restore
from .runtime import load_checkpoint, load_file, load_runtime
from .structure import thin


def f1_tolerance(pred, truth, tolerance=2):
    if not pred.any() or not truth.any():
        return float(not pred.any() and not truth.any())
    precision = (ndimage.distance_transform_edt(~truth)[pred] <= tolerance).mean()
    recall = (ndimage.distance_transform_edt(~pred)[truth] <= tolerance).mean()
    return float(2 * precision * recall / max(1e-12, precision + recall))


def read_connections(mask, relations, radius=2):
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    h, w = mask.shape

    def endpoint(x, y):
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < w and 0 <= y < h):
            return set()
        yy, xx = np.mgrid[max(0, y-radius):min(h, y+radius+1), max(0, x-radius):min(w, x+radius+1)]
        local = labels[yy, xx][(xx-x)**2 + (yy-y)**2 <= radius**2]
        return set(local[local > 0].tolist())
    result = []
    for relation in relations:
        a = endpoint(*relation["a"]); b = endpoint(*relation["b"])
        result.append(-1 if not a or not b else int(bool(a & b)))
    return result


def metrics(rgb, hr, target=None, relations=(), period=(0., 0.)):
    result = {"psnr": float(-10 * (rgb - hr).square().mean().clamp_min(1e-12).log10())}
    if target is None:
        return result
    gt = target[0, 0].cpu().numpy() > .5
    if not gt.any() or gt.all():
        return result
    truth = hr[0].detach().cpu().numpy(); image = rgb[0].detach().cpu().numpy()
    # Fixed reference-colour readout; reject low contrast rather than calling a
    # missing endpoint a disconnected stroke. HR is used only for evaluation.
    foreground = np.median(truth[:, gt], axis=1)
    ring = ndimage.binary_dilation(gt, iterations=5) & ~gt
    background = np.median(truth[:, ring], axis=1)
    axis = foreground - background
    contrast = float(np.linalg.norm(axis))
    result["readout_contrast"] = contrast
    if contrast < .03:
        result.update(connection_total=len(relations), connection_missing=len(relations), connection_wrong=0,
                      connection_unstable=0)
        if relations:
            result.update(connection_missing_rate=1., connection_unstable_rate=0.,
                          connection_error_lower=0., connection_error_upper=1.)
        return result
    score = ((image - background[:, None, None]) * axis[:, None, None]).sum(0) / np.dot(axis, axis)
    pred = score > .5
    result["skeleton_f1"] = f1_tolerance(thin(pred), thin(gt))
    boundary = lambda m: m & ~ndimage.binary_erosion(m)
    result["boundary_f1"] = f1_tolerance(boundary(pred), boundary(gt))
    reads = [read_connections(score > threshold, relations) for threshold in (.45, .5, .55)]
    missing = wrong = unstable = 0
    for i, relation in enumerate(relations):
        labels = [r[i] for r in reads]
        if -1 in labels:
            missing += 1
        elif len(set(labels)) != 1:
            unstable += 1
        else:
            wrong += int(labels[1] != int(relation["connected"]))
    result.update(connection_total=len(relations), connection_missing=missing,
                  connection_unstable=unstable, connection_wrong=wrong)
    if relations:
        n = len(relations)
        result.update(connection_missing_rate=missing/n, connection_unstable_rate=unstable/n,
                      connection_error_lower=wrong/n, connection_error_upper=(wrong+missing+unstable)/n)
    period = np.asarray(period, dtype=float)
    if np.linalg.norm(period) >= 3:
        yy, xx = np.mgrid[:gt.shape[0], :gt.shape[1]]
        carrier = np.exp(-2j * np.pi * (xx*period[0] + yy*period[1]) / np.dot(period, period))
        region = ndimage.binary_dilation(gt, iterations=5)
        def coefficient(im):
            gray = im.mean(0); gray = gray - gray[region].mean()
            return (gray[region] * carrier[region]).mean()
        reference, predicted = coefficient(truth), coefficient(image)
        result["phase_readout_valid"] = bool(abs(reference) >= .001 and abs(predicted) >= .001)
        if result["phase_readout_valid"]:
            result["phase_error_radians"] = float(abs(np.angle(predicted * reference.conjugate())))
    return result


def parent_bootstrap(records, method="full", baseline="native", metric="psnr", count=2000):
    groups = {}
    for row in records:
        a, b = row["scores"].get(method, {}), row["scores"].get(baseline, {})
        if metric in a and metric in b:
            groups.setdefault((row["dataset"], row["parent"]), []).append(a[metric] - b[metric])
    values = np.array([np.mean(v) for v in groups.values()])
    if not len(values):
        return {"parents": 0, "mean": None, "ci95": None}
    result = {"parents": len(values), "mean": float(values.mean()), "ci95": None}
    if len(values) >= 2:
        rng = np.random.default_rng(261111)
        means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(count)])
        result["ci95"] = np.quantile(means, [.025, .975]).tolist()
    return result


def load_v15(root):
    root = Path(root)
    legacy = load_file(root / "native_anchor_bridge_261109_v1/code/infer_system.py", "v2_legacy_system")
    base, camera = legacy.base_camera()
    net, recipe = legacy.load_model()
    assets = legacy.dependencies()
    return lambda lr, seed: legacy.restore(lr, seed, base, camera, assets, net, recipe)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--runtime-root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--seeds", type=int, nargs="+", default=[6, 43, 61])
    p.add_argument("--controls", nargs="+", default=["full", "no_feedback", "uniform", "no_acceptance", "wrong_graph"])
    p.add_argument("--shared-steps", type=int, choices=(8, 12, 16, 20))
    p.add_argument("--include-v15", action="store_true")
    p.add_argument("--perceptual", action="store_true", help="Also run installed pyiqa LPIPS/DISTS")
    args = p.parse_args(); torch.set_num_threads(4)
    rows = [r for r in read_manifest(args.manifest) if r["split"] == args.split]
    if not rows:
        p.error("Selected split is empty")
    if "full" not in args.controls or len(args.controls) != len(set(args.controls)):
        p.error("controls must contain full exactly once and have no duplicates")
    destination = Path(args.output)
    if destination.exists():
        p.error("output already exists; use a fresh run directory to preserve protocol identity")
    head, adapter, cfg, meta = load_checkpoint(args.checkpoint, "cuda")
    if args.shared_steps is not None:
        cfg = replace(cfg, shared_steps=args.shared_steps)
    base, camera = load_runtime(args.runtime_root)
    legacy = load_v15(args.runtime_root) if args.include_v15 else None
    perceptual = {}
    if args.perceptual:
        import pyiqa
        perceptual = {key: pyiqa.create_metric(key, device="cuda").eval() for key in ("lpips", "dists")}
    destination.mkdir(parents=True)
    identity = {"checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "config": cfg.to_dict(), "arguments": vars(args), "training": meta}
    (destination / "identity.json").write_text(json.dumps(identity, indent=2))
    records = []
    for row in rows:
        lr, hr, target = load_row(row, cfg)
        lr, hr = lr[None].cuda(), hr[None].cuda()
        target = target[None] if target is not None else None
        for seed in args.seeds:
            outputs, decisions, timings = {}, {}, {}
            for control in args.controls:
                torch.cuda.synchronize(); start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                out, stages = restore(base, head, adapter, camera, lr, seed, cfg, control=control)
                torch.cuda.synchronize()
                timings[control] = {"seconds": time.perf_counter()-start,
                                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                outputs[control] = out
                decisions[control] = {k: stages[k] for k in ("decisions", "feedback_step")}
                if control == "full":
                    outputs.update({k: stages[k] for k in ("native", "revised", "composite")})
            if legacy:
                outputs["v15"] = legacy(lr, seed)
            scores = {}
            with torch.no_grad():
                for key, out in outputs.items():
                    scores[key] = metrics(out, hr, target, row.get("relations", []), row.get("period", (0., 0.)))
                    scores[key]["lr_mse"] = float((camera.nominal(out)-lr).square().mean())
                    scores[key].update({name: float(fn(out.clamp(0, 1), hr).mean()) for name, fn in perceptual.items()})
            record = {k: row[k] for k in ("id", "parent", "dataset", "camera")}
            record.update(seed=seed, scores=scores, decisions=decisions, timings=timings,
                          reference_readout=metrics(hr, hr, target, row.get("relations", []), row.get("period", (0., 0.))))
            records.append(record)
            with (destination / "records.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            tag = hashlib.sha256((row["id"] + f"/{seed}").encode()).hexdigest()[:20]
            torch.save({k: v.cpu() for k, v in outputs.items()}, destination / f"{tag}.pt")
    summaries = {}
    for dataset, camera_name in sorted({(r["dataset"], r["camera"]) for r in records}):
        subset = [r for r in records if (r["dataset"], r["camera"]) == (dataset, camera_name)]
        summaries[f"{dataset}/{camera_name}"] = {
            f"full-{baseline}/{metric}": parent_bootstrap(subset, baseline=baseline, metric=metric)
            for baseline in (["native", "v15"] if legacy else ["native"])
            for metric in ("psnr", "skeleton_f1", "boundary_f1", "lpips", "dists",
                           "connection_missing_rate", "connection_unstable_rate",
                           "connection_error_lower", "connection_error_upper", "phase_error_radians")}
    (destination / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
