import numpy as np
import scipy.io as sio
from typing import Tuple, Optional

def load_mat_data(filepath: str) -> dict:
    """Load .mat file and extract quanta data"""
    data = sio.loadmat(filepath)
    return {
        'imbs': data['imbs'],
        'imgt': data.get('imgt', None),
        'dcr': data.get('dcr', None),
        'dataParam': data.get('dataParam', None)
    }

def save_results(filepath: str, results: dict):
    """Save results to .mat format for comparison"""
    sio.savemat(filepath, results)

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))