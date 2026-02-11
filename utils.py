import numpy as np
import hd5py
from typing import Tuple, Optional
from skimage.metrics import structural_similarity as ssim
import cv2


def load_mat_data(filepath: str) -> dict:
    """
        Load .mat file and extract quanta data using hd5py.
        NOTE: This function assumes the .mat file is in HDF5 format, which is common for MATLAB v7.3 and later.
    """
    with hd5py.File(filepath, 'r') as f:
        data = {}
        for key in f.keys():
            data[key] = f[key][()]
        return data


def load_npy_data(filepath: str) -> dict:
    """Load .npy file containing quanta data"""
    return np.load(filepath, allow_pickle=True).item()



def save_results(filepath: str, results: dict):
    """Save results to .mat format for comparison"""
    sio.savemat(filepath, results)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images"""
    return ssim(img1, img2, data_range=img1.max() - img1.min())


def visualize_flow(flow: np.ndarray) -> np.ndarray:
    """Visualize optical flow using HSV color space"""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    data = load_mat_data('dataBayer_0121_1.mat')
    print(data['imbs'].shape)
    assert data['imbs'].shape == (H, W, T)
    print(data['imbs'].dtype)
    assert data['imbs'].dtype == bool or data['imbs'].dtype == np.uint8
