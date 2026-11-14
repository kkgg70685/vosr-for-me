"""Server-side CLI. Full floating outputs stay on the server."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import torch
from . import VERSION
from .model import Config, AreaCamera, ObservableTransport
from .runtime import load_base, load_camera, generate_native


def read_image(path, device):
    with Image.open(path) as image:
        a = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)/255


def source_identity():
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--runtime-root", default="/home/zhangxu/hx/HX001")
    p.add_argument("--native", help="Optional precomputed native RGB; never HR reference")
    p.add_argument("--camera", choices=("deployed", "synthetic-area"), default="deployed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--config", default=str(Path(__file__).parents[2]/"configs/observable_transport_v25.json"))
    a = p.parse_args()
    protocol = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.seed != protocol["seed"]:
        raise ValueError("Seed differs from frozen single-seed protocol")
    cfg = Config(**protocol["model"])
    lr = read_image(a.input, a.device)
    if a.native:
        native = read_image(a.native, a.device)
    else:
        if lr.shape != (1, 3, 128, 128):
            raise ValueError("Existing native runtime requires 128x128 LR")
        native = generate_native(load_base(a.runtime_root), lr, a.seed)
    camera = AreaCamera(cfg.scale) if a.camera == "synthetic-area" else load_camera(a.runtime_root, a.device)
    result = ObservableTransport(camera, cfg).restore(lr, native)
    dest = Path(a.output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    torch.save({k: v.cpu() for k, v in result.items() if isinstance(v, torch.Tensor)}, dest/"stages.pt")
    for name in ("native", "revised", "final"):
        arr = (result[name][0].clamp(0, 1).permute(1, 2, 0)*255).round().byte().cpu().numpy()
        Image.fromarray(arr).save(dest/f"{name}.png")
    diagnostics = result["diagnostics"]
    diagnostics.update(version=VERSION, seed=a.seed, camera=a.camera,
                       protocol_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
                       input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
                       native_source="precomputed_quantized_rgb" if a.native else "frozen_vosr_float",
                       source_sha256=source_identity(), torch_version=torch.__version__)
    (dest/"diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
