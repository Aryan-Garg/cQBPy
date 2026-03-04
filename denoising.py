"""Noise stabilization and lightweight chrominance denoising."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    m = np.array(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
        dtype=np.float32,
    )
    out = rgb @ m.T
    out[..., 1:] += 0.5
    return out


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    x = ycbcr.copy()
    x[..., 1:] -= 0.5
    m_inv = np.array(
        [[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]], dtype=np.float32
    )
    return x @ m_inv.T


def anscombe_binomial(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.clip(x, 0.0, None) + 3.0 / 8.0)


def anscombe_binomial_inv(x: np.ndarray) -> np.ndarray:
    return np.maximum((x * 0.5) ** 2 - 3.0 / 8.0, 0.0)


def chroma_denoise(rgb: np.ndarray, sigma_y: float = 0.6, sigma_c: float = 1.2) -> np.ndarray:
    """Denoise in YCbCr with stronger chroma smoothing."""
    ycc = rgb_to_ycbcr(np.clip(rgb, 0.0, 1.0).astype(np.float32))
    ycc[..., 0] = gaussian_filter(ycc[..., 0], sigma=sigma_y)
    ycc[..., 1] = gaussian_filter(ycc[..., 1], sigma=sigma_c)
    ycc[..., 2] = gaussian_filter(ycc[..., 2], sigma=sigma_c)
    return np.clip(ycbcr_to_rgb(ycc), 0.0, 1.0)
