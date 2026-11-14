"""Explicit connection to the retained server assets; no v2 checkpoint needed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch


def load_file(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing server runtime source: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_camera(root, device):
    root = Path(root)
    asset = root / "camera_feedback_261108_v1/real_camera_adapter.pt"
    protocol = Path(__file__).parents[1] / "material_direction_flow_261108_v1/code/protocol.json"
    expected = json.loads(protocol.read_text())["real_camera_sha"]
    if hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
        raise ValueError("Frozen camera identity differs from retained protocol")
    module = load_file(root / "camera_feedback_261108_v1/code/feedback.py", "v25_camera")
    sensor = module.CameraProjection().to(device).eval().requires_grad_(False)
    sensor.load_state_dict(torch.load(asset, map_location=device, weights_only=True)["projection"])
    return sensor


def load_base(root):
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen VOSR requires the existing CUDA server environment")
    module = load_file(Path(root) / "visual_region_condition_261108_v1/code/probe.py", "v25_probe")
    base = module.FrozenVOSR()
    base.use_checkpoint = False
    for value in vars(base).values():
        if isinstance(value, torch.nn.Module):
            value.eval().requires_grad_(False)
    for name in ("conditions", "velocity", "decoder_prefix", "decoder_suffix"):
        if not callable(getattr(base, name, None)):
            raise TypeError(f"Frozen runtime lacks {name}")
    return base


@torch.no_grad()
def generate_native(base, lr, seed=6, steps=25, guidance=-.5):
    """Same Euler/CFG convention as v2 native, one trajectory and one decode."""
    cond, sem, _ = base.conditions(lr)
    generator = torch.Generator(device=lr.device).manual_seed(seed)
    z = torch.randn(cond.shape, device=lr.device, generator=generator)
    times = torch.linspace(1, 0, steps + 1, device=lr.device)
    for t, end in zip(times[:-1], times[1:]):
        full = base.velocity(cond, z, t[None], sem)
        weak = base.velocity(.1*cond, z, t[None], torch.zeros_like(sem))
        z = z - (t-end) * (weak + guidance*(full-weak))
    return base.decoder_suffix(base.decoder_prefix(z))
