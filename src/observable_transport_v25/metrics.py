"""Explicit standard metric definitions and denominators, no learned judge."""
import math
import numpy as np
import torch


def rgb_psnr(prediction, target):
    if prediction.shape != target.shape or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("PSNR requires matching finite images")
    mse = float((prediction-target).square().mean())
    return float("inf") if mse == 0 else -10*math.log10(mse)


def error_auc(errors, thresholds=(3., 5., 10.)):
    """SuperGlue-style trapezoidal error CDF area; failures remain +inf.

    Thresholds have the error's unit: pixels for homographies, degrees for pose.
    Empty evaluation is an error, not a successful result.
    """
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or not len(values) or np.isnan(values).any() or (values < 0).any():
        raise ValueError("Nonempty nonnegative errors required; use inf for failures")
    values = np.r_[0., np.sort(values)]
    recall = np.arange(len(values), dtype=float)/(len(values)-1)
    out = {}
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Positive finite thresholds required")
        last = np.searchsorted(values, threshold)
        x = np.r_[values[:last], threshold]
        y = np.r_[recall[:last], recall[last-1]]
        # np.trapz remains compatible with the retained NumPy server environment.
        area = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
        out[str(threshold)] = float(area/threshold)
    return out


def homography_corner_error(estimated, reference, height, width):
    """Mean 4-corner reprojection distance, in HR pixels; failed fits are inf."""
    if estimated is None:
        return float("inf")
    pts = np.array([[0, 0, 1], [width-1, 0, 1], [width-1, height-1, 1], [0, height-1, 1]]).T
    warped = []
    for h in (estimated, reference):
        h = np.asarray(h, dtype=float)
        if h.shape != (3, 3) or not np.isfinite(h).all():
            raise ValueError("Expected finite 3x3 homography")
        q = h @ pts
        if (np.abs(q[2]) < 1e-10).any():
            return float("inf")
        warped.append((q[:2]/q[2]).T)
    return float(np.linalg.norm(warped[0]-warped[1], axis=1).mean())
