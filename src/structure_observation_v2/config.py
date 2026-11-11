"""One versioned architecture; experiment controls are explicit, not hidden recipes."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    scale: int = 4
    steps: int = 25
    shared_steps: int = 16
    cfg: float = -0.5  # Preserve the deployed FrozenVOSR velocity convention.
    width: int = 48
    proposal_threshold: float = 0.55
    min_pixels: int = 8
    max_pixels: int = 4096
    max_objects: int = 24
    radius: int = 2
    feather: int = 2
    shift: float = 1.0  # HR pixels, never a wrap-around roll.
    max_gap: float = 8.0
    max_period: float = 32.0
    noise_floor: float = 2.0 / 255
    distinguish_z: float = 3.0
    preference_z: float = 2.0
    max_fit_rms: float = 4.0
    tangent_rtol: float = 0.02
    adapter_bound: float = 0.25
    feedback_bound: float = 2.0 / 255
    feedback_grid: int = 4

    def __post_init__(self):
        if self.scale != 4 or not 0 <= self.shared_steps < self.steps:
            raise ValueError("v2 requires x4 and 0 <= shared_steps < steps")
        if min(self.noise_floor, self.distinguish_z, self.preference_z) <= 0:
            raise ValueError("Evidence thresholds must be positive")
        if self.radius < 1 or self.feather < 1 or self.max_objects < 1:
            raise ValueError("Invalid structure budget")

    def to_dict(self):
        return asdict(self)
