"""Observation- and direction-guided velocity sampling with frozen VOSR."""
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path("/home/zhangxu/hx/HX001")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
sys.path.insert(0, str(ROOT / "visual_region_condition_261108_v1/code"))
from probe import FrozenVOSR, measure, sha, write_json

def derivatives(gray):
    dx = F.pad((gray[..., 2:] - gray[..., :-2]) * .5, (1, 1, 0, 0), mode="replicate")
    dy = F.pad((gray[..., 2:, :] - gray[..., :-2, :]) * .5, (0, 0, 1, 1), mode="replicate")
    return dx, dy


def direction_field(lr, sem, method):
    dx, dy = derivatives(lr.mean(1, keepdim=True))
    moments = torch.cat((dx.square() - dy.square(), 2*dx*dy, dx.square()+dy.square()), 1)
    moments = F.avg_pool2d(F.pad(moments, (2,)*4, mode="replicate"), 5, stride=1)
    local = moments[:, :2] / (moments[:, 2:] + .0001)
    if method != "local":
        refs = F.avg_pool2d(local, 4).flatten(2).transpose(1, 2)
        desc = F.normalize(sem - sem.mean(1, keepdim=True), dim=-1)
        scores, indices = (desc @ desc.transpose(1, 2)).topk(32, dim=-1)
        votes = (refs[0][indices] * (scores*10).softmax(-1)[..., None]).sum(2)
        remote = votes.transpose(1, 2).reshape(1, 2, 32, 32)
        if method == "misplaced":
            remote = remote.roll((16,16), (-2,-1))
        remote = F.interpolate(remote, size=lr.shape[-2:], mode="bilinear", align_corners=False)
        # Strong local measurements retain priority. Remote votes fill weakly oriented regions.
        confidence = local.norm(dim=1, keepdim=True).clamp(max=1)
        local = confidence * local + (1-confidence) * remote
    return F.interpolate(local, scale_factor=4, mode="bilinear", align_corners=False)


def wrong_axis(rgb, field):
    dx, dy = derivatives(rgb.mean(1, keepdim=True))
    anisotropy = field.norm(dim=1, keepdim=True)
    return (.5 * (anisotropy*(dx.square()+dy.square()) - field[:, :1]*(dx.square()-dy.square())
                  - field[:, 1:]*(2*dx*dy))).mean()


def velocity(base, cond, sem, z, t, cfg):
    full = base.velocity(cond,z,t[None],sem)
    weak = base.velocity(.1*cond,z,t[None],torch.zeros_like(sem))
    return weak + cfg*(full-weak)


@torch.no_grad()
def start(base, lr, seed, cfg):
    cond, sem, _ = base.conditions(lr)
    gen = torch.Generator(device=lr.device).manual_seed(seed)
    z = torch.randn(cond.shape, device=lr.device, generator=gen)
    times = torch.linspace(1,0,26,device=lr.device)
    for t, end in zip(times[:16],times[1:17]):
        z = z - (t-end) * velocity(base,cond,sem,z,t,cfg)
    return cond,sem,z,times


@torch.no_grad()
def finish(base, state, lr, cfg, method, scale, weight, projector):
    cond, sem, z, times = state
    field = direction_field(lr, sem, method) if method in ("visual","local","misplaced") else None
    for index in range(16,25):
        t, end = times[index:index+2]
        v = velocity(base,cond,sem,z,t,cfg)
        next_z = z - (t-end)*v
        if method != "native" and scale > 0 and index % 2 == 0:
            with torch.enable_grad():
                clean = (z-t*v).detach().requires_grad_(True)
                predicted = base.decode(clean)
                loss = (projector(predicted)-lr).square().mean()
                if field is not None:
                    loss = loss + weight*wrong_axis(predicted, field)
                gradient, = torch.autograd.grad(loss, clean)
            # RMS normalization caps each correction in latent units.
            next_z = next_z - scale * gradient / gradient.square().mean().sqrt().clamp_min(1e-9)
        z = next_z
    return z
