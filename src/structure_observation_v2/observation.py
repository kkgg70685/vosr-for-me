"""Shared image formation and difference-specific, nuisance-projected evidence.

These are conservative finite-bank tests, NOT calibrated Bayesian probabilities.
Every candidate uses the same camera, appearance fit, noise and nuisance basis.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


def translate(x, dx, dy):
    h, w = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack((2 * (xx + 0.5 - dx) / w - 1,
                        2 * (yy + 0.5 - dy) / h - 1), -1)
    return F.grid_sample(x, grid[None].expand(x.shape[0], -1, -1, -1),
                         align_corners=False, padding_mode="border")


def blur(x, sigma):
    if sigma == 0:
        return x
    radius = max(1, int(3 * sigma + 0.5))
    q = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (q / sigma).square()); k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"),
                 k[None, None, None].expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"),
                    k[None, None, :, None].expand(c, 1, -1, 1), groups=c)


class SharedCamera:
    """One frozen sensor plus a small *common* blur/registration neighbourhood.

    The optional deployed sensor is called identically for all hypotheses. The
    default is a synthetic blur+area camera, not a fitted real-camera model.
    Spatially varying parameters can live inside the supplied frozen sensor.
    """
    # HR pixel shifts; 0.5 HR = 0.125 LR at x4.
    nuisance = ((0., 0., 0.), (0.5, 0., 0.), (-0.5, 0., 0.),
                (0., 0.5, 0.), (0., -0.5, 0.), (0., 0., 0.8))

    def __init__(self, sensor=None, scale=4, sigma=0.8):
        self.sensor = sensor
        self.scale = scale
        self.sigma = sigma

    def nominal(self, x):
        if self.sensor is not None:
            return self.sensor(x)
        return F.avg_pool2d(blur(x, self.sigma), self.scale)

    def project(self, x):
        return torch.stack([self.nominal(translate(blur(x, s), dx, dy))
                            for dx, dy, s in self.nuisance])


def observation_mask(mask, scale=4):
    # Include the PSF footprint: an outside-LR leak must not escape the test.
    halo = F.max_pool2d(mask, 9, stride=1, padding=4)
    return F.avg_pool2d(halo, scale).clamp(0, 1)


def shared_render(lr, rasters, camera, radius=2):
    """Fit ONE RGB foreground/background plane for the complete candidate set.

    The fit uses the mean occupancy, treating candidates symmetrically. It is
    never refit per candidate. A bad fit disables evidence downstream. This
    structural proxy is only for hypothesis testing, never the final SR image.
    """
    occupancy = torch.cat([r[:, :1] for r in rasters])
    k, _, h, w = occupancy.shape
    union = occupancy.amax(0, keepdim=True)
    support = F.max_pool2d(union, 2 * radius + 1, stride=1, padding=radius)
    lowmask = observation_mask(support, camera.scale)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=lr.device),
                            torch.linspace(-1, 1, w, device=lr.device), indexing="ij")
    basis = torch.stack((torch.ones_like(xx), xx, yy), 0)[None]
    # Each basis is passed as RGB: real sensors may not accept 1 or 4 channels.
    design_hr = torch.cat((basis, occupancy.mean(0, keepdim=True)), 1)
    up = F.interpolate(lr, (h, w), mode="bilinear", align_corners=False)
    active = lowmask[0, 0].flatten() > 0.01
    if active.sum() < 8:
        return up.expand(k, -1, -1, -1).clone(), support, lowmask
    # Fit on nominal sensor responses, including its color mixing / offsets.
    zero = camera.nominal(torch.zeros_like(up))
    columns = []
    for c in range(3):
        for j in range(4):
            b = torch.zeros_like(up); b[:, c:c+1] = design_hr[:, j:j+1]
            columns.append((camera.nominal(b) - zero)[0, :, active.reshape(lr.shape[-2:])].flatten())
    matrix = torch.stack(columns, 1)
    target = (lr - zero)[0, :, active.reshape(lr.shape[-2:])].flatten()
    weight = lowmask[0, 0].flatten()[active].sqrt().repeat(3)
    matrix = matrix * weight[:, None]; target = target * weight
    # Ridge stabilises tiny objects; no candidate-specific exposure or phase fitting.
    ridge = torch.eye(12, device=lr.device, dtype=lr.dtype) * 1e-3
    coef = torch.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ target).reshape(3, 4)
    bg = torch.einsum("cj,bjhw->bchw", coef[:, :3], basis)
    proxy = bg + coef[:, 3][None, :, None, None] * occupancy
    return (up + support * (proxy.clamp(0, 1) - up)), support, lowmask


@dataclass
class DifferenceEvidence:
    distinguish: float
    preference: float  # positive => b explains the LR better than a
    fit_rms: float
    reliable: bool


def _coordinates(a, b, lr, mask, cfg):
    """Build a pair's shared photometric quotient and finite covariance whitener."""
    if a.shape != b.shape or a.ndim != 5 or a.shape[1] != 1:
        raise ValueError("Expected matching [J,1,3,h,w] camera banks")
    active = mask[0, 0] > 0.01
    if int(active.sum()) < 4:
        return None
    weight = mask[0, 0][active].sqrt().repeat(3)
    def vectorize(x):
        return x[:, 0, :, active].flatten(1) / cfg.noise_floor * weight
    av, bv = vectorize(a), vectorize(b)
    yv = lr[0, :, active].flatten() / cfg.noise_floor * weight
    common = (av + bv) * 0.5
    # Linear colour fields absorb brightness/color/slow-shading ambiguity.
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, lr.shape[-2], device=lr.device),
                            torch.linspace(-1, 1, lr.shape[-1], device=lr.device), indexing="ij")
    spatial = torch.stack((torch.ones_like(xx), xx, yy))[:, active]
    color = torch.eye(3, device=lr.device)
    drift = torch.einsum("cd,jn->cjdn", color, spatial).reshape(9, -1).T
    u, s, _ = torch.linalg.svd(drift * weight[:, None], full_matrices=False)
    q = u[:, s > cfg.tangent_rtol * s.max().clamp_min(1e-6)]

    def remove_drift(v):
        return v - (v @ q) @ q.T

    # Finite nuisance variance, NOT an unbounded translation subspace. An
    # unbounded projector would erase arbitrarily large real displacements.
    tangent = remove_drift(common[1:] - common[:1]).T / max(1, common.shape[0] - 1) ** .5
    u, s, _ = torch.linalg.svd(tangent, full_matrices=False)
    keep = s > cfg.tangent_rtol * s.max().clamp_min(1e-6)
    u, s = u[:, keep], s[keep]

    def project(v):
        v = remove_drift(v)
        return v + ((v @ u) * ((1 + s.square()).rsqrt() - 1)) @ u.T

    delta = project(bv - av)
    residual = project(yv[None] - common)
    # Absolute fit before nuisance removal, conservatively over the common bank.
    rms = torch.minimum((yv - av).square().mean(1), (yv - bv).square().mean(1)).sqrt().min()
    return delta, residual, rms, project, vectorize


def compare(a, b, lr, mask, cfg):
    """a,b: [nuisance,1,3,h,w]. Test the PARTICULAR difference b-a.

    Remove shared photometric planes and whiten finite camera uncertainty. Take
    the worst paired preference and separation over the SAME nuisance settings.
    These engineering scores are not calibrated probabilities or p-values.
    """
    coordinates = _coordinates(a, b, lr, mask, cfg)
    if coordinates is None:
        return DifferenceEvidence(0., 0., 1e9, False)
    delta, residual, rms, _, _ = coordinates
    length = delta.norm(dim=1)
    preference = (residual * delta).sum(1) / length.clamp_min(1e-6)
    d, p, fit = float(length.min()), float(preference.min()), float(rms)
    reliable = d >= cfg.distinguish_z and fit <= cfg.max_fit_rms
    return DifferenceEvidence(d, p, fit, reliable)


def relation_movement(proxy_a, proxy_b, actual_a, actual_b, lr, mask, cfg):
    """Actual RGB movement along the SAME nuisance-whitened graph difference."""
    coordinates = _coordinates(proxy_a, proxy_b, lr, mask, cfg)
    if coordinates is None:
        return 0.
    delta, _, _, project, vectorize = coordinates
    movement = project(vectorize(actual_b - actual_a))
    score = (movement * delta).sum(1) / delta.norm(dim=1).clamp_min(1e-6)
    return float(score.min())


@torch.no_grad()
def choose_hypothesis(bank, projections, lr, mask, cfg, generator):
    """Retain indistinguishable alternatives; sample ONCE for the entire object.

    Each named family is compared to the same reference graph. A non-reference
    alternative is ruled out only by distinguishable paired evidence. Selection
    confidence does not become a universal texture confidence map.
    """
    evidence = [DifferenceEvidence(0., 0., 0., False)]
    plausible = [0]
    for k, hypothesis in enumerate(bank[1:], 1):
        ev = compare(projections[:, :1], projections[:, k:k+1], lr, mask, cfg)
        evidence.append(ev)
        # Reverse-direction evidence uses worst-case pairing too.
        reverse = compare(projections[:, k:k+1], projections[:, :1], lr, mask, cfg)
        if not (reverse.reliable and reverse.preference >= cfg.preference_z):
            plausible.append(k)
    # Pairwise pruning, not independent candidate-specific nuisance optimisation.
    winners = []
    for k in plausible:
        beaten = False
        for j in plausible:
            if j == k:
                continue
            ev = compare(projections[:, k:k+1], projections[:, j:j+1], lr, mask, cfg)
            if ev.reliable and ev.preference >= cfg.preference_z:
                beaten = True
                break
        if not beaten:
            winners.append(k)
    # Inconsistent finite-bank preferences cannot force an arbitrary winner.
    winners = winners or [0]
    draw = int(torch.randint(len(winners), (), generator=generator))
    selected = winners[draw]
    # Confidence must apply to the selected graph, not a different alternative.
    selected_strengths = {}
    for family in ("position", "topology", "phase"):
        # This bounded bank changes one factor relative to the LR graph. A
        # moved-vs-open comparison changes TWO factors and cannot certify either.
        if selected and bank[selected].family != family:
            selected_strengths[family] = 0.
            continue
        peers = [j for j, h in enumerate(bank) if h.family == family and j != selected]
        if selected and bank[selected].family == family:
            peers.append(0)
        values = []
        for j in peers:
            ev = compare(projections[:, j:j+1], projections[:, selected:selected+1], lr, mask, cfg)
            values.append(min(1., ev.distinguish / (2 * cfg.distinguish_z))
                          if ev.reliable and ev.preference >= cfg.preference_z else 0.)
        selected_strengths[family] = min(values, default=0.)
    return selected, winners, selected_strengths, evidence
