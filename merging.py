"""Vectorized burst warping and robust merge utilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from demosaicing import demosaic_bayer_bilinear


def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w]
    src_y = yy + flow[..., 1]
    src_x = xx + flow[..., 0]
    return map_coordinates(frame, [src_y, src_x], order=1, mode="reflect")


def robust_merge_bayer(
    imbs: np.ndarray,
    flows: np.ndarray,
    pattern: str = "RGGB",
    temporal_sigma: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp and merge binary burst into RGB accumulation + count map."""
    h, w, t = imbs.shape
    aligned = np.empty_like(imbs, dtype=np.float32)
    for i in range(t):
        aligned[..., i] = warp_frame(imbs[..., i].astype(np.float32), flows[..., :, i])

    med = np.median(aligned, axis=2, keepdims=True)
    mad = np.median(np.abs(aligned - med), axis=2, keepdims=True) + 1e-4
    wr = np.exp(-0.5 * ((aligned - med) / (temporal_sigma + mad)) ** 2)

    weighted = (aligned * wr).sum(axis=2) / (wr.sum(axis=2) + 1e-8)
    rgb = demosaic_bayer_bilinear(weighted, pattern=pattern)
    return rgb, weighted


def post_merge_rgb(
    rgb_prob: np.ndarray,
    total_frames: int,
    tau: float,
    eta: np.ndarray,
    dcr: float = 0.0,
) -> np.ndarray:
    """Invert SPAD response curve using MLE intensity mapping."""
    s = np.clip(rgb_prob * total_frames, 0, total_frames - 1e-3)
    p = np.clip(s / total_frames, 0.0, 1.0 - 1e-8)
    lam = -np.log(1.0 - p) / max(tau, 1e-12)
    lam = lam - dcr
    lam = np.maximum(lam, 0.0)
    eta = np.asarray(eta, dtype=np.float32).reshape(1, 1, 3)
    return lam / np.maximum(eta, 1e-8)


def post_merge_rgb_torch(rgb_prob, total_frames: int, tau: float, eta, dcr: float = 0.0):
    """Differentiable torch equivalent of post_merge_rgb."""
    import torch

    s = torch.clamp(rgb_prob * total_frames, 0.0, total_frames - 1e-3)
    p = torch.clamp(s / total_frames, 0.0, 1.0 - 1e-8)
    lam = -torch.log1p(-p) / max(tau, 1e-12)
    lam = torch.clamp(lam - dcr, min=0.0)
    eta_t = torch.as_tensor(eta, dtype=rgb_prob.dtype, device=rgb_prob.device).view(1, 1, 3)
    return lam / torch.clamp(eta_t, min=1e-8)
