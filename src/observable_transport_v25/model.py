"""Training-free variable-projection Gauss-Newton geometry layer.

No HR, object annotations, semantic labels, or oracle commands enter restore().
The spectral cutoff expresses a deterministic local sensitivity scale, NOT a
calibrated posterior probability. Camera mismatch remains a failure mode.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch.nn import functional as F
from .geometry import cosine_basis, field, transport, jacobian_determinant, inverse_error


@dataclass(frozen=True)
class Config:
    scale: int = 4
    modes: int = 3
    iterations: int = 3
    integration_steps: int = 6
    difference_step: float = .125
    observation_floor: float = 2 / 255
    resolution: float = .25
    damping: float = 1e-4
    max_step: float = 1.
    max_motion: float = 2.
    min_jacobian: float = .2
    max_inverse_error: float = .1
    lr_margin: int = 3
    profile_color: bool = True
    spectral_filter: bool = True
    gain_min: float = .5
    gain_max: float = 2.

    def __post_init__(self):
        if any(not math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("Configuration must be finite")
        for name in ("scale", "modes", "iterations", "integration_steps", "lr_margin"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if min(self.difference_step, self.observation_floor, self.resolution,
               self.damping, self.max_step, self.max_motion, self.max_inverse_error) <= 0:
            raise ValueError("Positive numerical scales required")
        if not 0 < self.min_jacobian < 1:
            raise ValueError("Jacobian floor must lie in (0,1)")
        if not 0 < self.gain_min <= 1 <= self.gain_max:
            raise ValueError("Photometric gains must be positive and include identity")


class AreaCamera:
    """Explicit synthetic camera. Never silently substitute for a real sensor."""
    def __init__(self, scale=4):
        self.scale = scale

    def __call__(self, image):
        return F.avg_pool2d(image, self.scale)


def profiled_residual(prediction, target, enabled=True, gain_min=.5, gain_max=2.):
    """Eliminate bias and bounded positive RGB gain by centered regression.

    Photometric corrections affect the likelihood only, never the output image.
    Finite differences of this residual include the nuisance refit derivative.
    """
    if not enabled:
        return (prediction - target).flatten()
    p, y = prediction.reshape(3, -1), target.reshape(3, -1)
    pc, yc = p-p.mean(1, keepdim=True), y-y.mean(1, keepdim=True)
    variance = pc.square().mean(1, keepdim=True)
    gain = ((pc*yc).mean(1, keepdim=True)/variance.clamp_min(1e-12)).clamp(gain_min, gain_max)
    gain = torch.where(variance > 1e-12, gain, torch.ones_like(gain))
    return (gain*pc-yc).flatten()


class ObservableTransport:
    def __init__(self, camera, config=None):
        self.camera, self.config = camera, config or Config()

    @torch.no_grad()
    def restore(self, lr, native):
        c = self.config
        if lr.ndim != 4 or native.ndim != 4 or lr.shape[:2] != (1, 3) or native.shape[:2] != (1, 3):
            raise ValueError("Single-image RGB tensors required")
        if native.shape[-2:] != tuple(c.scale * d for d in lr.shape[-2:]):
            raise ValueError("LR/native scale mismatch")
        if lr.dtype not in (torch.float32, torch.float64) or native.dtype != lr.dtype or native.device != lr.device:
            raise ValueError("Use matching float32/float64 tensors on the same device")
        if not bool(torch.isfinite(lr).all() and torch.isfinite(native).all()):
            raise ValueError("Non-finite inputs")
        m = c.lr_margin
        if min(lr.shape[-2:]) <= 2*m + 2 or c.modes > min(native.shape[-2:]):
            raise ValueError("Image too small for basis / fixed valid observation crop")
        basis = cosine_basis(native, c.modes)
        a = native.new_zeros(len(basis))
        y = lr[..., m:-m, m:-m]

        def render(coeff):
            return transport(native, field(coeff, basis), c.integration_steps)

        def image_residual(image):
            prediction = self.camera(image)
            if prediction.shape != lr.shape or not bool(torch.isfinite(prediction).all()):
                raise ValueError("Camera must return finite RGB at the LR shape")
            return profiled_residual(prediction[..., m:-m, m:-m], y, c.profile_color, c.gain_min, c.gain_max)

        def residual(coeff):
            return image_residual(render(coeff)[0])

        r = residual(a)
        initial_loss = loss = float(r.square().mean())
        accepted = 0
        records = []
        revised = native.clone()
        for iteration in range(c.iterations):
            columns = []
            for k in range(len(a)):
                e = torch.zeros_like(a); e[k] = c.difference_step
                columns.append((residual(a + e) - residual(a - e)) / (2*c.difference_step))
            j = torch.stack(columns, 1) / math.sqrt(r.numel())
            u, s, vh = torch.linalg.svd(j, full_matrices=False)
            cutoff = c.observation_floor / c.resolution
            keep = s > (cutoff if c.spectral_filter else max(float(s.max()) * 1e-5, 1e-8))
            record = {"iteration": iteration, "rank": int(keep.sum()),
                      "singular_values": s.tolist(), "cutoff": cutoff, "accepted": False}
            records.append(record)
            if not bool(keep.any()):
                record["reason"] = "no_resolved_direction"; break
            weight = torch.where(keep, s / (s.square() + c.damping), torch.zeros_like(s))
            step = -vh.T @ (weight * (u.T @ (r / math.sqrt(r.numel()))))
            peak = field(step, basis).square().sum(1).sqrt().max()
            step *= min(1., c.max_step / max(float(peak), 1e-12))
            improved = False
            for rate in (1., .5, .25, .125):
                proposal = a + rate * step
                v = field(proposal, basis)
                revised, d = render(proposal)
                det = jacobian_determinant(d)
                motion = float(v.square().sum(1).sqrt().max())
                inv = float(inverse_error(v, c.integration_steps, backward=d))
                rr = image_residual(revised)
                trial = float(rr.square().mean())
                record.setdefault('trials', []).append(dict(rate=rate, loss=trial, min_jacobian=float(det.min()), inverse_error=inv, peak_motion=motion))
                record.update(trial_loss=trial, min_jacobian=float(det.min()),
                              inverse_error=inv, peak_motion=motion)
                safe = motion <= c.max_motion and float(det.min()) >= c.min_jacobian and inv <= c.max_inverse_error
                if safe and trial < loss - max(1e-12, loss * 1e-6):
                    a, r, loss = proposal, rr, trial
                    accepted += 1; improved = True
                    record.update(accepted=True, rate=rate, reason="profiled_residual_decreased")
                    break
            if not improved:
                record["reason"] = "no_safe_descent"; break
        final, displacement = render(a) if accepted else (native.clone(), torch.zeros_like(field(a, basis)))
        return {"native": native, "revised": revised, "final": final,
                "velocity": field(a, basis), "backward_displacement": displacement,
                "diagnostics": {"accepted_steps": accepted, "initial_loss": initial_loss,
                                "final_loss": loss, "valid_lr_fraction": y.numel()/lr.numel(),
                                "config": asdict(c), "iterations": records}}
