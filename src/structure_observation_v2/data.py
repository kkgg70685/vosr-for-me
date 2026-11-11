"""Explicit parent splits and synthetic stroke truth; no evaluation-HR inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from .observation import SharedCamera
from .structure import soft_targets


def read_rgb(path):
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).float() / 255


def read_manifest(path):
    path = Path(path)
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty list")
    parents, ids = {}, set()
    for row in rows:
        if row.get("split") not in ("train", "dev", "test"):
            raise ValueError("Every row needs train/dev/test split")
        for key in ("id", "parent", "dataset", "camera", "lr", "hr"):
            if not row.get(key):
                raise ValueError(f"Missing manifest field {key}")
        if row["id"] in ids:
            raise ValueError("Duplicate sample id")
        ids.add(row["id"])
        key = (row["dataset"], row["parent"])
        if key in parents and parents[key] != row["split"]:
            raise ValueError(f"Parent leakage across splits: {key}")
        parents[key] = row["split"]
        for field in ("lr", "hr", "mask"):
            if field in row:
                row[field] = str((path.parent / row[field]).resolve())
    return rows


def load_row(row, cfg):
    lr, hr = read_rgb(row["lr"]), read_rgb(row["hr"])
    if lr.shape != (3, 128, 128) or hr.shape != (3, 512, 512):
        raise ValueError("Manifest images must be aligned native 128/512 crops")
    targets = None
    if row.get("mask"):
        with Image.open(row["mask"]) as image:
            mask = np.asarray(image.convert("L"), dtype=np.float32) / 255
        if mask.shape != (512, 512):
            raise ValueError("Structure annotation must be a 512x512 foreground mask")
        targets = torch.from_numpy(soft_targets(mask, row.get("period", (0., 0.)), cfg.max_period))
    return lr, hr, targets


def synthetic(parent, cfg, size=128):
    """Analytic masks with real gaps, corners, loops and repeated strokes.

    A parent identifier fixes geometry, appearance and degradation together.
    Synthetic training is preparation; it does not qualify natural-image gains.
    """
    digest = hashlib.sha256(str(parent).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    h = size * cfg.scale
    mask = Image.new("L", (h, h)); draw = ImageDraw.Draw(mask)
    period = (0., 0.)
    kind = int(rng.integers(0, 4))
    width = int(rng.integers(2, 7))
    margin = max(10, h // 8)
    if kind == 0:
        pts = rng.integers(margin, h - margin, size=(5, 2))
        draw.line([tuple(p) for p in pts], fill=255, width=width)
        if rng.random() < .5:
            p = (pts[1] + pts[2]) // 2
            gap = int(rng.integers(2, 6))
            draw.ellipse((p[0]-gap, p[1]-gap, p[0]+gap, p[1]+gap), fill=0)
    elif kind == 1:
        draw.rectangle((margin, margin, h-margin, h-margin), outline=255, width=width)
        for u in (h // 3, 2 * h // 3):
            draw.line((u, margin, u, h-margin), fill=255, width=width)
            draw.line((margin, u, h-margin, u), fill=255, width=width)
    elif kind == 2:
        spacing = int(rng.integers(8, min(28, h // 3)))
        phase = int(rng.integers(0, spacing))
        period = (float(spacing), 0.)
        for u in range(margin + phase, h - margin, spacing):
            draw.line((u, margin, u, h-margin), fill=255, width=width)
    else:
        # Low-structure texture/flat examples teach abstention as well as strokes.
        if rng.random() < .5:
            draw.ellipse((margin, margin, h-margin, h-margin), outline=255, width=width)
    ink = np.asarray(mask, dtype=np.float32) / 255
    fg = rng.uniform(.05, .9, size=(3, 1, 1)); bg = rng.uniform(.1, .95, size=(3, 1, 1))
    yy, xx = np.mgrid[:h, :h]
    texture = .015 * np.sin(xx * rng.uniform(.4, 2.) + yy * rng.uniform(.4, 2.))
    rgb = np.clip(bg + (fg - bg) * ink + texture, 0, 1).astype(np.float32)
    hr = torch.from_numpy(rgb)
    cam = SharedCamera(scale=cfg.scale, sigma=float(rng.uniform(.4, 1.2)))
    lr = cam.nominal(hr[None])[0]
    noise = torch.from_numpy(rng.normal(0, rng.uniform(0., 2/255), size=tuple(lr.shape))).float()
    lr = ((lr + noise).clamp(0, 1) * 255).round() / 255
    return lr, hr, torch.from_numpy(soft_targets(ink, period, cfg.max_period))
