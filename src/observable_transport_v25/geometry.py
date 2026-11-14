"""Continuous HR-pixel coordinates; positive velocity moves content right/down.

Stationary velocity is integrated as exp(-v), a backward sampling map. Discrete
integration and image interpolation do NOT prove topology or texture preservation.
"""
import math
import torch
from torch.nn import functional as F


def sample(image, displacement):
    h, w = image.shape[-2:]
    if displacement.shape != (image.shape[0], 2, h, w):
        raise ValueError("Expected Bx2xHxW backward displacement")
    yy, xx = torch.meshgrid(torch.arange(h, device=image.device, dtype=image.dtype),
                            torch.arange(w, device=image.device, dtype=image.dtype), indexing="ij")
    grid = torch.stack((2 * (xx + .5 + displacement[:, 0]) / w - 1,
                        2 * (yy + .5 + displacement[:, 1]) / h - 1), -1)
    return F.grid_sample(image, grid, align_corners=False, padding_mode="border")


def integrate(velocity, steps=6):
    d = -velocity / (2 ** steps)
    for _ in range(steps):
        d = d + sample(d, d)
    return d


def transport(image, velocity, steps=6):
    # Preserve exact identity without severing the differentiable training path.
    if not velocity.requires_grad and not bool(torch.count_nonzero(velocity)):
        return image.clone(), velocity.clone()
    d = integrate(velocity, steps)
    return sample(image, d), d


def cosine_basis(reference, modes=3):
    """K smooth whole-image fields, each scalar basis has unit spatial RMS."""
    h, w = reference.shape[-2:]
    y = (torch.arange(h, device=reference.device, dtype=reference.dtype) + .5) / h
    x = (torch.arange(w, device=reference.device, dtype=reference.dtype) + .5) / w
    scalar = []
    for iy in range(modes):
        for ix in range(modes):
            f = torch.cos(math.pi * iy * y)[:, None] * torch.cos(math.pi * ix * x)[None]
            scalar.append(f / f.square().mean().sqrt())
    scalar = torch.stack(scalar)
    z = torch.zeros_like(scalar)
    return torch.cat((torch.stack((scalar, z), 1), torch.stack((z, scalar), 1)))


def field(coefficients, basis):
    return torch.einsum("k,kchw->chw", coefficients, basis)[None]


def jacobian_determinant(d):
    """Forward cell differences of Id+d, not a continuous bijectivity proof."""
    dx = d[:, :, :-1, 1:] - d[:, :, :-1, :-1]
    dy = d[:, :, 1:, :-1] - d[:, :, :-1, :-1]
    return (1 + dx[:, 0]) * (1 + dy[:, 1]) - dy[:, 0] * dx[:, 1]


def inverse_error(v, steps=6, backward=None):
    d = integrate(v, steps) if backward is None else backward
    inv = integrate(-v, steps)
    return (d + sample(inv, d)).square().sum(1).mean().sqrt()
