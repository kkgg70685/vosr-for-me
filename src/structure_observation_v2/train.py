"""Train structural proposals, then the frozen-backbone velocity residual."""
import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from .config import Config
from .data import load_row, read_manifest, synthetic
from .model import (SpatialVelocityAdapter, condition, decode_native, prepare,
                    shared_prefix, velocity)
from .runtime import freeze, load_checkpoint, load_runtime, save_checkpoint
from .structure import StructureHead


def structure_loss(pred, target):
    # Class-balanced foreground, endpoints, turns, boundary and affinity losses.
    truth = target[:, :7]
    positive = truth.mean((0, 2, 3), keepdim=True).clamp(.02, .98)
    weight = truth / positive + (1 - truth) / (1 - positive)
    semantic = (F.binary_cross_entropy(pred[:, :7], truth, reduction="none") * weight).mean()
    period_mask = target[:, 6:7]
    period = ((pred[:, 7:] - target[:, 7:]).abs() * period_mask).sum() / (2 * period_mask.sum()).clamp_min(1)
    # A link cannot be stronger than either endpoint's foreground support.
    ink = pred[:, :1]
    ax = F.pad(torch.minimum(ink[..., :-1], ink[..., 1:]), (0, 1, 0, 0))
    ay = F.pad(torch.minimum(ink[..., :-1, :], ink[..., 1:, :]), (0, 0, 0, 1))
    incidence = (pred[:, 4:6] - torch.cat((ax, ay), 1)).abs().mean()
    return semantic + period + .1 * incidence


def reconstruction_loss(output, hr, native, support):
    inside = support.expand_as(output)
    denom = inside.sum().clamp_min(1)
    image = ((output - hr).abs() * inside).sum() / denom
    dx = (output[..., 1:] - output[..., :-1]) - (hr[..., 1:] - hr[..., :-1])
    dy = (output[..., 1:, :] - output[..., :-1, :]) - (hr[..., 1:, :] - hr[..., :-1, :])
    gradient = (dx.abs() * inside[..., 1:]).sum() / denom + (dy.abs() * inside[..., 1:, :]).sum() / denom
    identity = ((output - native).abs() * (1 - inside)).mean()
    return image + .5 * gradient + 2 * identity


def velocity_loss(base, adapter, lr, hr, spatial, cfg, seed, index):
    """Single-state clean prediction; decoder gradients reach the velocity head.

    Sample native suffix states, with no backpropagation through frozen VOSR.
    This is a decoded clean-prediction surrogate, not flow-matching supervision.
    Full 9-step inference is assessed separately; no HR encoder API is assumed.
    """
    state = shared_prefix(base, lr, seed, cfg)
    cond, sem, z, times = state
    with torch.no_grad():
        for i in range(cfg.shared_steps, index):
            t, end = times[i:i+2]
            z = z - (t - end) * velocity(base, cond, sem, z, t, cfg.cfg)
        t = times[index]
        v = velocity(base, cond, sem, z, t, cfg.cfg)
        clean = z - t * v
        native = decode_native(base, clean)
    correction = adapter(z.detach(), cond.detach(), lr, spatial, t)
    output = decode_native(base, clean.detach() - t * correction)
    if not output.requires_grad:
        raise RuntimeError("Frozen decoder disables autograd; expose differentiable prefix/suffix for training")
    loss = reconstruction_loss(output, hr, native, spatial[:, 9:10])
    return loss + .01 * correction.square().mean()


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    cfg = Config()
    rows = read_manifest(args.manifest) if args.manifest else None
    train_rows = [r for r in rows if r["split"] == "train"] if rows else None
    if rows and not train_rows:
        raise ValueError("No train rows")
    if rows and any(not r.get("mask") for r in train_rows):
        raise ValueError("Training needs explicit HR stroke/structure masks; never invent topology labels")
    metadata = {"stage": args.stage, "seed": args.seed, "steps": args.steps,
                "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest() if rows else None,
                "data": "annotated_pairs" if rows else "synthetic_only"}
    if args.stage == "structure":
        if args.checkpoint:
            head, _, cfg, _ = load_checkpoint(args.checkpoint, str(device), require_adapter=False)
        else:
            head = StructureHead(cfg.width).to(device)
        head.train(); adapter = None
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        if not args.checkpoint or not args.runtime_root:
            raise ValueError("Velocity stage needs --checkpoint and --runtime-root")
        head, _, cfg, _ = load_checkpoint(args.checkpoint, str(device), require_adapter=False)
        freeze(head)
        base, camera = load_runtime(args.runtime_root, str(device), args.camera_mode)
        sample = torch.zeros(1, 3, 128, 128, device=device)
        with torch.no_grad():
            channels = base.conditions(sample)[0].shape[1]
        adapter = SpatialVelocityAdapter(channels, cfg.width, cfg.adapter_bound).to(device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
        metadata["structure_sha256"] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        metadata["camera_mode"] = args.camera_mode
    generator = torch.Generator().manual_seed(args.seed + 1)
    losses, updates, attempts = [], 0, 0
    # Empty/invalid object batches cannot silently produce a "trained" checkpoint.
    while updates < args.steps and attempts < args.steps * 10:
        attempts += 1
        batch = []
        for b in range(args.batch_size if args.stage == "structure" else 1):
            if train_rows:
                row = train_rows[int(torch.randint(len(train_rows), (), generator=generator))]
                batch.append(load_row(row, cfg))
            else:
                batch.append(synthetic(f"train/{args.seed}/{attempts}/{b}", cfg))
        lr, hr, target = [torch.stack([r[i] for r in batch]).to(device) for i in range(3)]
        if args.stage == "structure":
            loss = structure_loss(head(lr), target)
        else:
            with torch.no_grad():
                _, objects = prepare(lr, head, camera, cfg, args.seed + attempts)
                chosen, qualified = [], []
                for obj in objects:
                    # HR teacher is used ONLY in training; complete candidate choice.
                    scores = [float((2 * r[:, :1] * target[:, :1]).sum() /
                                    (r[:, :1].sum() + (target[:, :1] * obj.support).sum()).clamp_min(1))
                              for r in obj.rasters]
                    best = max(range(len(scores)), key=scores.__getitem__)
                    if scores[best] >= .5:
                        qualified.append(obj); chosen.append(best)
                if not qualified:
                    continue
                spatial = condition(qualified, hr.shape[-2:], lr, selected_override=chosen)
            index = int(torch.randint(cfg.shared_steps, cfg.steps, (), generator=generator))
            loss = velocity_loss(base, adapter, lr, hr, spatial, cfg, args.seed + attempts, index)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        parameters = list(head.parameters()) if adapter is None else list(adapter.parameters())
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if adapter is not None and float(norm) == 0:
            raise RuntimeError("No velocity gradient: check decoder contract and structure support")
        optimizer.step(); updates += 1; losses.append(float(loss.detach()))
        if updates == 1 or updates % 100 == 0:
            print(f"{args.stage} updates={updates} loss={losses[-1]:.6f}", flush=True)
    if updates != args.steps:
        raise RuntimeError(f"Only {updates}/{args.steps} usable updates; qualify proposal recall before training velocity")
    metadata.update(updates=updates, attempts=attempts, final_loss=losses[-1])
    save_checkpoint(args.output, head, cfg, adapter=adapter, metadata=metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("structure", "velocity"), required=True)
    p.add_argument("--manifest"); p.add_argument("--checkpoint"); p.add_argument("--runtime-root")
    p.add_argument("--camera-mode", choices=("deployed", "synthetic"), default="deployed")
    p.add_argument("--output", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=20000); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--seed", type=int, default=261111)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error("steps and batch-size must be positive")
    train(args)


if __name__ == "__main__":
    main()
