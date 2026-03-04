"""Vectorized Bayer demosaicing utilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve


BAYER_KERNEL_GREEN = np.array(
    [[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]], dtype=np.float32
)
BAYER_KERNEL_RB = np.array(
    [[0.25, 0.0, 0.25], [0.0, 1.0, 0.0], [0.25, 0.0, 0.25]], dtype=np.float32
)


def bayer_masks(shape: tuple[int, int], pattern: str = "RGGB") -> np.ndarray:
    """Return boolean channel masks for Bayer images with shape [3, H, W]."""
    h, w = shape
    masks = np.zeros((3, h, w), dtype=bool)
    p = pattern.upper()
    if p == "RGGB":
        masks[0, 0::2, 0::2] = True
        masks[1, 0::2, 1::2] = True
        masks[1, 1::2, 0::2] = True
        masks[2, 1::2, 1::2] = True
    elif p == "BGGR":
        masks[2, 0::2, 0::2] = True
        masks[1, 0::2, 1::2] = True
        masks[1, 1::2, 0::2] = True
        masks[0, 1::2, 1::2] = True
    elif p == "GRBG":
        masks[1, 0::2, 0::2] = True
        masks[0, 0::2, 1::2] = True
        masks[2, 1::2, 0::2] = True
        masks[1, 1::2, 1::2] = True
    elif p == "GBRG":
        masks[1, 0::2, 0::2] = True
        masks[2, 0::2, 1::2] = True
        masks[0, 1::2, 0::2] = True
        masks[1, 1::2, 1::2] = True
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    return masks


def demosaic_bayer_bilinear(mosaicked: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Fast vectorized bilinear demosaicing for Bayer mosaics."""
    masks = bayer_masks(mosaicked.shape, pattern)
    out = np.empty((*mosaicked.shape, 3), dtype=np.float32)

    for c, kernel in enumerate((BAYER_KERNEL_RB, BAYER_KERNEL_GREEN, BAYER_KERNEL_RB)):
        mc = masks[c].astype(np.float32)
        raw = mosaicked * mc
        num = convolve(raw, kernel, mode="mirror")
        den = convolve(mc, kernel, mode="mirror") + 1e-8
        out[..., c] = num / den

    return out
