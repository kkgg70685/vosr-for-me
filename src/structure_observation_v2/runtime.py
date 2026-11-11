"""Thin, asset-explicit connection to the existing server's FrozenVOSR API."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from . import VERSION
from .config import Config
from .model import SpatialVelocityAdapter
from .observation import SharedCamera
from .structure import StructureHead


def freeze(obj):
    """FrozenVOSR may be a wrapper rather than nn.Module."""
    visited = set()

    def visit(value, depth=0):
        if id(value) in visited or depth > 4:
            return
        visited.add(id(value))
        if isinstance(value, torch.nn.Module):
            value.eval().requires_grad_(False)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, depth + 1)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item, depth + 1)
        elif depth == 0 and hasattr(value, "__dict__"):
            for item in vars(value).values():
                visit(item, depth + 1)
    visit(obj)
    return obj


def load_file(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen runtime source: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime(root, device="cuda", camera_mode="deployed"):
    root = Path(root).expanduser().resolve()
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The existing FrozenVOSR adapter requires a CUDA runtime")
    probe = load_file(root / "visual_region_condition_261108_v1/code/probe.py", "v2_frozen_probe")
    base = freeze(probe.FrozenVOSR())
    base.use_checkpoint = False
    for name in ("conditions", "velocity", "decoder_prefix", "decoder_suffix"):
        if not callable(getattr(base, name, None)):
            raise TypeError(f"FrozenVOSR adapter lacks {name}")
    if camera_mode == "synthetic":
        return base, SharedCamera()
    if camera_mode != "deployed":
        raise ValueError("Unknown camera mode")
    asset = root / "camera_feedback_261108_v1/real_camera_adapter.pt"
    protocol = Path(__file__).parents[1] / "material_direction_flow_261108_v1/code/protocol.json"
    expected = json.loads(protocol.read_text())["real_camera_sha"]
    if hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
        raise ValueError("Deployed camera differs from the retained v1.5 asset identity")
    module = load_file(root / "camera_feedback_261108_v1/code/feedback.py", "v2_camera_feedback")
    sensor = module.CameraProjection().to(device).eval().requires_grad_(False)
    sensor.load_state_dict(torch.load(asset, map_location=device, weights_only=True)["projection"])
    return base, SharedCamera(sensor)


def save_checkpoint(path, head, cfg, *, adapter=None, metadata=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pack = {"version": VERSION, "config": cfg.to_dict(), "structure": head.state_dict(),
            "metadata": metadata or {}}
    if adapter is not None:
        pack.update(adapter=adapter.state_dict(), latent_channels=adapter.latent_channels)
    temporary = path.with_suffix(".part")
    torch.save(pack, temporary); temporary.replace(path)


def load_checkpoint(path, device="cpu", require_adapter=True):
    pack = torch.load(path, weights_only=True, map_location=device)
    if pack.get("version") != VERSION:
        raise ValueError("Checkpoint architecture/version mismatch")
    cfg = Config(**pack["config"])
    head = StructureHead(cfg.width).to(device)
    head.load_state_dict(pack["structure"], strict=True)
    adapter = None
    if "adapter" in pack:
        adapter = SpatialVelocityAdapter(pack["latent_channels"], cfg.width, cfg.adapter_bound).to(device)
        adapter.load_state_dict(pack["adapter"], strict=True)
    elif require_adapter:
        raise ValueError("This checkpoint has no trained velocity adapter; run the velocity stage")
    return head.eval(), adapter.eval() if adapter is not None else None, cfg, pack.get("metadata", {})
