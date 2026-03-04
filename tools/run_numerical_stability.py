import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
import subprocess

import numpy as np
import scipy.io as sio

from quanta_color_pipeline import QuantaBurstPipeline, QuantaParams


def make_gt_video(h=96, w=96, t=40):
    y, x = np.mgrid[0:h, 0:w]
    video = np.zeros((h, w, 3, t), dtype=np.float64)
    for i in range(t):
        dx = 4 * np.sin(2 * np.pi * i / t)
        dy = 3 * np.cos(2 * np.pi * i / t)
        xs = np.clip(x + dx, 0, w - 1)
        ys = np.clip(y + dy, 0, h - 1)
        base = 0.5 + 0.5 * np.sin(0.07 * xs + 0.05 * ys + 0.3 * i)
        video[..., 0, i] = np.clip(base * (0.8 + 0.2 * np.sin(i / 7)), 0, 1)
        video[..., 1, i] = np.clip(0.6 * base + 0.3 * np.cos(0.03 * xs), 0, 1)
        video[..., 2, i] = np.clip(0.4 * base + 0.5 * np.sin(0.02 * ys + i / 11), 0, 1)
    return video


def mosaic_bayer(rgb):
    h, w, _ = rgb.shape
    m = np.zeros((h, w), dtype=np.float64)
    m[0::2, 0::2] = rgb[0::2, 0::2, 0]
    m[0::2, 1::2] = rgb[0::2, 1::2, 1]
    m[1::2, 0::2] = rgb[1::2, 0::2, 1]
    m[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return m


def simulate_photon_cube(gt_video, tau, max_flux, dcr, seed=0):
    rng = np.random.default_rng(seed)
    h, w, _, t = gt_video.shape
    imbs = np.zeros((h, w, t), dtype=np.uint8)
    for i in range(t):
        mosaicked = mosaic_bayer(gt_video[..., i])
        rate = mosaicked * max_flux + dcr
        p = 1.0 - np.exp(-rate * tau)
        imbs[..., i] = rng.binomial(1, np.clip(p, 0, 1)).astype(np.uint8)
    return imbs


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse <= 0 else 20 * np.log10(1.0 / np.sqrt(mse))


def compare_metrics(py, ref):
    d = py - ref
    return {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "max_abs": float(np.max(np.abs(d))),
        "ncc": float(np.sum(py * ref) / (np.linalg.norm(py) * np.linalg.norm(ref) + 1e-12)),
        "allclose_rtol1e-3_atol1e-5": bool(np.allclose(py, ref, rtol=1e-3, atol=1e-5)),
    }


def main():
    out_dir = Path("results/numerical_stability")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = QuantaParams(
        alignTWSize=7,
        alignTWNum=5,
        mergeTWSize=7,
        mergeTWNum=5,
        refFrame=20,
        searchRadii=(1, 2, 3),
        tau=4.17e-5,
        maxFlux=0.7e5,
        eta=np.array([0.125, 0.135, 0.09], dtype=np.float32),
        dcr=2.0,
        computePSNR=False,
        doSR=False,
    )

    gt_video = make_gt_video(t=40)
    gt_ref = gt_video[..., params.refFrame]
    imbs = simulate_photon_cube(gt_video, params.tau, params.maxFlux, params.dcr, seed=123)

    in_mat = out_dir / "parity_input.mat"
    sio.savemat(in_mat, {
        "imbs": imbs.astype(np.float64),
        "imgt": gt_ref.astype(np.float64),
        "searchRadii": np.array(params.searchRadii, dtype=np.float64),
        "refFrame": float(params.refFrame),
        "tau": float(params.tau),
        "eta": params.eta.astype(np.float64),
        "dcr": float(params.dcr),
    })

    py_start = time.perf_counter()
    py_results = QuantaBurstPipeline(params).run(imbs, gt_ref)
    py_runtime = time.perf_counter() - py_start

    out_mat = out_dir / "matlab_parity_output.mat"
    octave_cmd = [
        "octave",
        "--quiet",
        "--eval",
        f"addpath('matlab_src'); python_parity_pipeline('{in_mat.as_posix()}', '{out_mat.as_posix()}');",
    ]
    m_start = time.perf_counter()
    proc = subprocess.run(octave_cmd, capture_output=True, text=True)
    m_runtime = time.perf_counter() - m_start

    if proc.returncode != 0:
        raise RuntimeError(f"Octave parity run failed:\n{proc.stderr}\n{proc.stdout}")

    m = sio.loadmat(out_mat)
    m_naive = m["naive"]
    m_merged = m["merged"]
    m_flows = m["flows"]

    report = {
        "scene": {"h": int(imbs.shape[0]), "w": int(imbs.shape[1]), "t": int(imbs.shape[2]), "seed": 123},
        "runtime_seconds": {"python_pipeline": py_runtime, "matlab_parity": m_runtime},
        "psnr_to_gt_ref": {
            "python_naive": psnr(py_results["naive_recons"], gt_ref),
            "python_merged": psnr(py_results["merged"], gt_ref),
            "matlab_naive": psnr(m_naive, gt_ref),
            "matlab_merged": psnr(m_merged, gt_ref),
        },
        "numerical_closeness": {
            "naive": compare_metrics(py_results["naive_recons"], m_naive),
            "merged": compare_metrics(py_results["merged"], m_merged),
            "flows": compare_metrics(py_results["flows"], m_flows),
        },
        "octave_stdout": proc.stdout[-1000:],
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # Clean temporary MAT files to keep repo artifacts small
    if in_mat.exists():
        in_mat.unlink()
    if out_mat.exists():
        out_mat.unlink()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
