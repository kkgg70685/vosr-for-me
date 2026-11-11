"""Spatial velocity conditioning and a genuinely shared 16/9 trajectory."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .structure import extract_graphs, hypotheses, raster
from .observation import shared_render, choose_hypothesis, observation_mask, compare, relation_movement


CONDITION_CHANNELS = 12


class SpatialVelocityAdapter(nn.Module):
    """A zero-output residual beside the frozen VOSR velocity, not an RGB filter."""
    def __init__(self, latent_channels, width=48, bound=0.25):
        super().__init__()
        self.latent_channels = latent_channels
        self.bound = bound
        self.body = nn.Sequential(
            nn.Conv2d(2 * latent_channels + 3 + CONDITION_CHANNELS + 1, width, 3, padding=1),
            nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, latent_channels, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, z, cond, lr, spatial, t):
        size = z.shape[-2:]
        y = F.interpolate(lr, size, mode="bilinear", align_corners=False)
        g = F.interpolate(spatial, size, mode="area")
        tm = torch.ones_like(z[:, :1]) * t
        x = torch.cat((z, cond, y - 0.5, g, tm), 1)
        # Support multiplies the output AFTER convolutions: no off-object leakage.
        return self.bound * self.body(x).tanh() * g[:, 9:10]


@dataclass
class ObjectProposal:
    bank: list
    rasters: list
    support: torch.Tensor
    mask: torch.Tensor
    selected: int
    plausible: list
    strengths: dict
    evidence: list
    projections: torch.Tensor


@torch.no_grad()
def prepare(lr, head, camera, cfg, seed):
    maps = head(lr)
    graphs = extract_graphs(maps, cfg)
    generator = torch.Generator().manual_seed(seed ^ 0x5A17)
    objects = []
    for graph in graphs:
        bank = hypotheses(graph, cfg)
        rasters = [raster(h.graph, maps.shape[-2:], lr.device, cfg.radius) for h in bank]
        proxies, support, mask = shared_render(lr, rasters, camera, cfg.radius)
        projections = camera.project(proxies)
        selected, plausible, strengths, evidence = choose_hypothesis(
            bank, projections, lr, mask, cfg, generator)
        objects.append(ObjectProposal(bank, rasters, support, mask, selected,
                                      plausible, strengths, evidence, projections))
    # Phase displacements can exceed the extraction halo. Decline both colliding
    # objects rather than make order-dependent partial replacements.
    halos = [F.max_pool2d(o.support, 2 * cfg.feather + 1, stride=1, padding=cfg.feather)
             for o in objects]
    collisions = set()
    for i in range(len(halos)):
        for j in range(i):
            if bool((halos[i] * halos[j] > 0).any()):
                collisions.update((i, j))
    objects = [o for i, o in enumerate(objects) if i not in collisions]
    return condition(objects, maps.shape[-2:], lr), objects


def condition(objects, size, reference, *, uniform=False, selected_override=None):
    field = reference.new_zeros(1, CONDITION_CHANNELS, *size)
    for i, obj in enumerate(objects):
        selected = obj.selected if selected_override is None else selected_override[i]
        r = obj.rasters[selected]
        confidence = [obj.strengths[k] for k in ("position", "topology", "phase")]
        if uniform:
            confidence = [1., 1., 1.]
        p, t, f = confidence
        # Weak candidates remain coherent hypotheses; only observable differences
        # get strong conditioning. Final acceptance is an independent RGB test.
        weights = reference.new_tensor([p, t, t, p, t, f])[None, :, None, None]
        spatial = r * (0.15 + 0.85 * weights)
        scores = reference.new_tensor(confidence)[None, :, None, None] * obj.support
        period = reference.new_tensor(obj.bank[selected].graph.period)[None, :, None, None] / 32
        field += torch.cat((spatial, scores, obj.support, period * obj.support), 1)
    return field


def velocity(base, cond, sem, z, t, cfg):
    full = base.velocity(cond, z, t[None], sem)
    weak = base.velocity(0.1 * cond, z, t[None], torch.zeros_like(sem))
    return weak + cfg * (full - weak)


@torch.no_grad()
def shared_prefix(base, lr, seed, cfg):
    cond, sem, _ = base.conditions(lr)
    generator = torch.Generator(device=lr.device).manual_seed(seed)
    # Exact v1.5 random draw and Euler convention for the matched native branch.
    z = torch.randn(cond.shape, device=lr.device, generator=generator)
    times = torch.linspace(1, 0, cfg.steps + 1, device=lr.device)
    for i in range(cfg.shared_steps):
        t, end = times[i:i+2]
        z = z - (t - end) * velocity(base, cond, sem, z, t, cfg.cfg)
    return cond, sem, z, times


def suffix(base, state, lr, cfg, adapter=None, spatial=None):
    # Out-of-place updates ensure neither branch can mutate the shared state.
    cond, sem, z, times = state
    z = z.clone()
    for i in range(cfg.shared_steps, cfg.steps):
        t, end = times[i:i+2]
        v = velocity(base, cond, sem, z, t, cfg.cfg)
        if adapter is not None:
            v = v + adapter(z, cond, lr, spatial, t)
        z = z - (t - end) * v
    return z


def decode_native(base, z):
    # Matches bridge-before RGB in infer_system.py. Training needs gradients
    # through the frozen decoder, so do not call a no_grad convenience wrapper.
    return base.decoder_suffix(base.decoder_prefix(z))


def outer_alpha(support, feather):
    """Exactly one inside the full object; blending exists only outside it."""
    alpha = support.clone()
    for distance in range(1, feather + 1):
        ring = F.max_pool2d(support, 2 * distance + 1, stride=1, padding=distance)
        alpha = torch.maximum(alpha, ring * (1 - distance / (feather + 1)))
    return alpha


@torch.no_grad()
def compose(native, revised, objects, lr, camera, cfg, *, bypass_acceptance=False):
    """Atomic RGB+texture replacement after relation-specific LR validation.

    A real RGB residual improvement alone is insufficient: its projection on
    the selected structural counterfactual must also improve. An unchanged or
    ambiguous graph cannot earn a replacement through unrelated texture noise.
    """
    output = native.clone()
    decisions = []
    for obj in objects:
        alpha = outer_alpha(obj.support, cfg.feather)
        trial = torch.where(obj.support.bool(), revised, output + alpha * (revised - output))
        a = camera.project(output); b = camera.project(trial)
        mask = observation_mask((alpha > 0).to(alpha.dtype), cfg.scale)
        ev = compare(a, b, lr, mask, cfg)
        selected = obj.selected
        relation_ok = False
        relation_z = 0.
        # Test realised movement in the observable graph-difference direction.
        # Comparing native/revised against the selected proxy removes texture
        # improvements orthogonal to the intended geometric intervention.
        for j in range(len(obj.bank)):
            if j == selected:
                continue
            if selected and j and obj.bank[j].family != obj.bank[selected].family:
                continue
            proxy_ev = compare(obj.projections[:, j:j+1],
                               obj.projections[:, selected:selected+1], lr, obj.mask, cfg)
            if not proxy_ev.reliable or proxy_ev.preference < cfg.preference_z:
                continue
            movement = relation_movement(obj.projections[:, j:j+1],
                                         obj.projections[:, selected:selected+1], a, b, lr, obj.mask, cfg)
            relation_z = max(relation_z, movement)
            relation_ok |= movement >= cfg.preference_z
        # Validate the ACTUAL composite, including the outer seam, for every
        # common camera. A later accepted object is checked in the current image.
        error_a = ((a - lr).square() * mask).sum((1, 2, 3, 4))
        error_b = ((b - lr).square() * mask).sum((1, 2, 3, 4))
        global_a = (a - lr).square().sum((1, 2, 3, 4))
        global_b = (b - lr).square().sum((1, 2, 3, 4))
        supported = (ev.reliable and ev.preference >= cfg.preference_z and relation_ok
                     and bool((error_b < error_a).all()) and bool((global_b <= global_a).all()))
        accepted = bypass_acceptance or supported
        if accepted:
            output = trial
        decisions.append({"selected": obj.bank[selected].name,
                          "plausible": [obj.bank[i].name for i in obj.plausible],
                          "strengths": obj.strengths, "accepted": accepted,
                          "relation_z": relation_z, "rgb_evidence": asdict(ev)})
    return output, decisions


def coarse_feedback(rgb, lr, camera, cfg):
    """Optimise only a coarse RGB lattice; accept a bounded step on every camera.

    No frequency replacement, warping or high-frequency residual is permitted.
    Rejected objects can still receive this explicitly reported photometric step.
    """
    if cfg.feedback_bound <= 0:
        return rgb, 0.
    before = camera.project(rgb.detach())
    old = (before - lr).square().mean((1, 2, 3, 4))
    with torch.enable_grad():
        grid = rgb.new_zeros(1, 3, cfg.feedback_grid, cfg.feedback_grid, requires_grad=True)
        delta = F.interpolate(grid, rgb.shape[-2:], mode="bilinear", align_corners=False)
        loss = (camera.nominal(rgb.detach() + delta) - lr).square().mean()
        grad, = torch.autograd.grad(loss, grid)
    if not bool(torch.isfinite(grad).all()) or float(grad.abs().max()) == 0:
        return rgb, 0.
    coarse = -cfg.feedback_bound * grad / grad.abs().max()
    # Keep float output within [0,1] WITHOUT clipping individual high frequencies.
    delta = F.interpolate(coarse, rgb.shape[-2:], mode="bilinear", align_corners=False)
    with torch.no_grad():
        for step in (1., 0.5, 0.25, 0.125):
            trial = rgb + step * delta
            if bool((trial < 0).any()) or bool((trial > 1).any()):
                continue
            errors = (camera.project(trial) - lr).square().mean((1, 2, 3, 4))
            if bool((errors < old).all()):
                return trial, step
    return rgb, 0.


@torch.no_grad()
def restore(base, head, adapter, camera, lr, seed, cfg, *, control="full"):
    if lr.shape != (1, 3, 128, 128):
        raise ValueError("The deployed v2 contract is [1,3,128,128] RGB in [0,1]")
    if control not in ("full", "native", "uniform", "no_acceptance", "no_feedback", "wrong_graph"):
        raise ValueError(f"Unknown control: {control}")
    spatial, objects = prepare(lr, head, camera, cfg, seed) if control != "native" else (None, [])
    if control == "uniform":
        spatial = condition(objects, (512, 512), lr, uniform=True)
    elif control == "wrong_graph":
        spatial = spatial.roll((128, 128), (-2, -1))
    state = shared_prefix(base, lr, seed, cfg)
    native = decode_native(base, suffix(base, state, lr, cfg)).clamp(0, 1)
    if control == "native":
        return native, {"native": native, "revised": native, "composite": native,
                        "decisions": [], "feedback_step": 0.}
    revised = (decode_native(base, suffix(base, state, lr, cfg, adapter, spatial)).clamp(0, 1)
               if objects else native)
    composite, decisions = compose(native, revised, objects, lr, camera, cfg,
                                   bypass_acceptance=control == "no_acceptance")
    result, step = (composite, 0.) if control == "no_feedback" else coarse_feedback(composite, lr, camera, cfg)
    return result, {"native": native, "revised": revised, "composite": composite,
                    "decisions": decisions, "feedback_step": step}
