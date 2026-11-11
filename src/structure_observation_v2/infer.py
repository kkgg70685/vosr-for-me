"""Single-image v2 inference using explicit weights and the existing CUDA runtime."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .data import read_rgb
from .model import restore
from .runtime import load_checkpoint, load_runtime


def save_rgb(rgb, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pixels = rgb[0].detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(pixels).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--runtime-root", required=True)
    p.add_argument("--seed", type=int, default=6); p.add_argument("--diagnostics")
    p.add_argument("--camera-mode", choices=("deployed", "synthetic"), default="deployed")
    p.add_argument("--control", default="full", choices=("full", "native", "uniform", "no_acceptance", "no_feedback", "wrong_graph"))
    args = p.parse_args(); torch.set_num_threads(4)
    head, adapter, cfg, metadata = load_checkpoint(args.checkpoint, "cuda")
    base, camera = load_runtime(args.runtime_root, camera_mode=args.camera_mode)
    lr = read_rgb(args.input)[None].cuda()
    output, stages = restore(base, head, adapter, camera, lr, args.seed, cfg, control=args.control)
    save_rgb(output, args.output)
    if args.diagnostics:
        folder = Path(args.diagnostics); folder.mkdir(parents=True, exist_ok=True)
        for key in ("native", "revised", "composite"):
            save_rgb(stages.pop(key), folder / f"{key}.png")
        (folder / "decisions.json").write_text(json.dumps(stages, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
