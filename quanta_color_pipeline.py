"""Quanta Burst Color Photography Pipeline (NumPy-first reimplementation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import shift

from demosaicing import demosaic_bayer_bilinear
from denoising import chroma_denoise
from merging import post_merge_rgb, robust_merge_bayer


@dataclass
class QuantaParams:
    # Temporal window parameters
    alignTWSize: int = 7
    alignTWNum: int = 11
    mergeTWSize: int = 7
    mergeTWNum: int = 11
    srTWSize: int = 7
    srTWNum: int = 11
    refFrame: int = 35

    # Alignment / search parameters
    numLevels: int = 4
    searchRadii: Tuple[int, ...] = (1, 2, 3)

    # Quanta imaging parameters
    maxFlux: float = 0.7e5
    tau: float = 4.17e-5
    eta: np.ndarray = field(default_factory=lambda: np.array([0.125, 0.135, 0.09], dtype=np.float32))
    dcr: float = 2.0

    # Denoising
    bm3dSigma: float = 0.05

    # Processing flags
    doSR: bool = False
    computePSNR: bool = True
    correctDCR: bool = False
    cfa: str = "RGGB"


def load_quanta_data(mat_file: str):
    import scipy.io as sio

    data = sio.loadmat(mat_file)
    return data["imbs"], data.get("imgt", None), data.get("dcr", None)


class QuantaBurstPipeline:
    def __init__(self, params: QuantaParams):
        self.params = params
        self.flows = None

    def run(self, imbs: np.ndarray, imgt: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        results: Dict[str, np.ndarray] = {}

        results["naive_recons"] = self.naive_reconstruction(imbs)
        self.flows, _ = self.patch_align(imbs)
        results["flows"] = self.flows
        rgb_prob, merge_map = self.patch_merge(imbs, self.flows)
        results["merge_map"] = merge_map

        merged = self.post_merge(rgb_prob, apply_bm3d=False)
        results["merged"] = merged
        results["sr"] = merged
        results["sr_denoised"] = self.post_merge(rgb_prob, apply_bm3d=True)

        if self.params.computePSNR and imgt is not None:
            results["psnr"] = self.compute_psnr(results, imgt)
        return results

    def naive_reconstruction(self, imbs: np.ndarray) -> np.ndarray:
        avg = np.mean(imbs.astype(np.float32), axis=2)
        rgb_prob = demosaic_bayer_bilinear(avg, pattern=self.params.cfa)
        return self.post_merge(rgb_prob, apply_bm3d=False)

    def patch_align(self, imbs: np.ndarray):
        """Global translational alignment per frame (vectorized search on small window)."""
        h, w, t = imbs.shape
        ref_idx = min(max(self.params.refFrame, 0), t - 1)
        ref = imbs[..., ref_idx].astype(np.float32)

        flows = np.zeros((h, w, 2, t), dtype=np.float32)
        radii = list(self.params.searchRadii)

        for i in range(t):
            if i == ref_idx:
                continue
            src = imbs[..., i].astype(np.float32)
            best_cost = np.inf
            best_dx, best_dy = 0.0, 0.0
            for r in radii:
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        warped = shift(src, shift=(dy, dx), order=1, mode="nearest")
                        cost = np.mean(np.abs(warped - ref))
                        if cost < best_cost:
                            best_cost = cost
                            best_dx, best_dy = dx, dy
            flows[..., 0, i] = best_dx
            flows[..., 1, i] = best_dy

        return flows, flows.copy()

    def patch_merge(self, imbs: np.ndarray, flows: np.ndarray):
        return robust_merge_bayer(imbs, flows, pattern=self.params.cfa)

    def post_merge(self, rgb_prob: np.ndarray, apply_bm3d: bool = False) -> np.ndarray:
        total_frames = self.params.mergeTWNum * self.params.mergeTWSize
        total_frames = max(total_frames, 1)
        dcr = self.params.dcr if self.params.correctDCR else 0.0
        rgb = post_merge_rgb(rgb_prob, total_frames=total_frames, tau=self.params.tau, eta=self.params.eta, dcr=dcr)
        rgb = rgb / (np.max(rgb) + 1e-8)
        if apply_bm3d:
            rgb = chroma_denoise(rgb)
        return np.clip(rgb, 0.0, 1.0)

    def compute_psnr(self, results: Dict, imgt: np.ndarray) -> Dict[str, float]:
        def _psnr(a, b):
            mse = float(np.mean((a - b) ** 2))
            if mse <= 0:
                return float("inf")
            return 20 * np.log10(1.0 / np.sqrt(mse))

        out = {}
        for key in ("naive_recons", "merged", "sr_denoised"):
            if key in results:
                out[key] = _psnr(results[key], imgt)
        return out
