"""
MATLAB vs Python Comparison Script

This script helps validate the Python reimplementation against
the original MATLAB codebase by comparing intermediate and final results.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json


class MATLABPythonComparator:
    """Compare MATLAB and Python pipeline outputs"""
    
    def __init__(self, matlab_dir: str, python_dir: str):
        """
        Args:
            matlab_dir: Directory with MATLAB results
            python_dir: Directory with Python results
        """
        self.matlab_dir = Path(matlab_dir)
        self.python_dir = Path(python_dir)
        
    def load_matlab_results(self) -> Dict:
        """Load MATLAB .mat result files"""
        results = {}
        
        # Load different stages
        mat_files = {
            'naive': 'naiveRecons.mat',
            'align': 'patchAlign.mat',
            'merge': 'patchMerge.mat',
            'sr': 'patchRgbSR.mat',
            'bm3d': 'bm3d.mat'
        }
        
        for key, filename in mat_files.items():
            filepath = self.matlab_dir / filename
            if filepath.exists():
                results[key] = sio.loadmat(str(filepath))
        
        return results
    
    def load_python_results(self) -> Dict:
        """Load Python result files"""
        # Assume Python saves as .npz
        results = {}
        
        npz_files = self.python_dir.glob('*.npz')
        for filepath in npz_files:
            key = filepath.stem
            results[key] = np.load(str(filepath))
        
        return results
    
    def compute_metrics(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Compute comparison metrics between two images"""
        # Ensure same shape
        if img1.shape != img2.shape:
            print(f"Warning: shape mismatch {img1.shape} vs {img2.shape}")
            # Resize to smaller
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(img1.shape, img2.shape))
            img1 = img1[:min_shape[0], :min_shape[1]]
            img2 = img2[:min_shape[0], :min_shape[1]]
        
        # Compute metrics
        mse = np.mean((img1 - img2) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # SSIM (simplified)
        ssim = self._compute_ssim_simple(img1, img2)
        
        # Max absolute error
        max_err = np.max(np.abs(img1 - img2))
        
        # Normalized cross-correlation
        ncc = np.sum(img1 * img2) / (np.linalg.norm(img1) * np.linalg.norm(img2))
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'max_error': float(max_err),
            'ncc': float(ncc)
        }
    
    def _compute_ssim_simple(self, img1: np.ndarray, img2: np.ndarray,
                            window_size: int = 11) -> float:
        """Simplified SSIM computation"""
        from scipy.ndimage import uniform_filter
        
        # Constants
        K1, K2 = 0.01, 0.03
        L = 1.0  # Dynamic range
        C1, C2 = (K1*L)**2, (K2*L)**2
        
        # Means
        mu1 = uniform_filter(img1, window_size)
        mu2 = uniform_filter(img2, window_size)
        
        # Variances and covariance
        sigma1_sq = uniform_filter(img1**2, window_size) - mu1**2
        sigma2_sq = uniform_filter(img2**2, window_size) - mu2**2
        sigma12 = uniform_filter(img1*img2, window_size) - mu1*mu2
        
        # SSIM
        ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    def compare_all_stages(self) -> Dict:
        """Compare all pipeline stages"""
        matlab_results = self.load_matlab_results()
        python_results = self.load_python_results()
        
        comparison = {}
        
        # Stage 1: Naive reconstruction
        if 'naive' in matlab_results and 'naive_recons' in python_results:
            matlab_img = matlab_results['naive']['ima']
            python_img = python_results['naive_recons']['arr_0']
            comparison['naive'] = self.compute_metrics(matlab_img, python_img)
        
        # Stage 2: Alignment (compare flows)
        if 'align' in matlab_results and 'flows' in python_results:
            matlab_flow = matlab_results['align']['flows']
            python_flow = python_results['flows']['arr_0']
            comparison['flows'] = self.compute_metrics(matlab_flow, python_flow)
        
        # Stage 3: Merged result
        if 'merge' in matlab_results and 'merged' in python_results:
            matlab_img = matlab_results['merge']['imr']
            python_img = python_results['merged']['arr_0']
            comparison['merged'] = self.compute_metrics(matlab_img, python_img)
        
        # Stage 4: SR result
        if 'sr' in matlab_results and 'sr_denoised' in python_results:
            matlab_img = matlab_results['sr']['imsr']
            python_img = python_results['sr_denoised']['arr_0']
            comparison['sr'] = self.compute_metrics(matlab_img, python_img)
        
        # Stage 5: BM3D denoised
        if 'bm3d' in matlab_results and 'sr_denoised' in python_results:
            matlab_img = matlab_results['bm3d']['imsrbm']
            python_img = python_results['sr_denoised']['arr_0']
            comparison['bm3d'] = self.compute_metrics(matlab_img, python_img)
        
        return comparison
    
    def generate_report(self, comparison: Dict, output_file: str = 'comparison_report.txt'):
        """Generate human-readable comparison report"""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MATLAB vs Python Comparison Report\n")
            f.write("=" * 80 + "\n\n")
            
            for stage, metrics in comparison.items():
                f.write(f"\n{stage.upper()}:\n")
                f.write("-" * 40 + "\n")
                for metric, value in metrics.items():
                    if metric == 'psnr':
                        status = "✓ EXCELLENT" if value > 40 else \
                                "✓ GOOD" if value > 30 else \
                                "⚠ NEEDS WORK" if value > 20 else \
                                "✗ POOR"
                        f.write(f"  {metric.upper()}: {value:.2f} dB {status}\n")
                    elif metric == 'ssim':
                        status = "✓ EXCELLENT" if value > 0.95 else \
                                "✓ GOOD" if value > 0.90 else \
                                "⚠ NEEDS WORK" if value > 0.80 else \
                                "✗ POOR"
                        f.write(f"  {metric.upper()}: {value:.4f} {status}\n")
                    else:
                        f.write(f"  {metric}: {value:.6f}\n")
        
        print(f"Report saved to {output_file}")
    
    def visualize_comparison(self, stage: str = 'merged'):
        """Create visual comparison plot"""
        matlab_results = self.load_matlab_results()
        python_results = self.load_python_results()
        
        # Get images
        if stage == 'naive':
            matlab_img = matlab_results['naive']['ima']
            python_img = python_results['naive_recons']['arr_0']
        elif stage == 'merged':
            matlab_img = matlab_results['merge']['imr']
            python_img = python_results['merged']['arr_0']
        elif stage == 'sr':
            matlab_img = matlab_results['sr']['imsr']
            python_img = python_results['sr_denoised']['arr_0']
        else:
            print(f"Unknown stage: {stage}")
            return
        
        # Compute difference
        diff = np.abs(matlab_img - python_img)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(np.clip(matlab_img, 0, 1))
        axes[0, 0].set_title('MATLAB Result')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(np.clip(python_img, 0, 1))
        axes[0, 1].set_title('Python Result')
        axes[0, 1].axis('off')
        
        im = axes[1, 0].imshow(diff, cmap='hot')
        axes[1, 0].set_title('Absolute Difference')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Histogram of differences
        axes[1, 1].hist(diff.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title('Difference Histogram')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'comparison_{stage}.png', dpi=150)
        print(f"Saved visualization to comparison_{stage}.png")
        plt.show()


def check_numerical_equivalence(matlab_val: np.ndarray,
                                python_val: np.ndarray,
                                rtol: float = 1e-3,
                                atol: float = 1e-5) -> bool:
    """
    Check if MATLAB and Python values are numerically equivalent
    
    Args:
        matlab_val: Value from MATLAB
        python_val: Value from Python
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if equivalent within tolerance
    """
    return np.allclose(matlab_val, python_val, rtol=rtol, atol=atol)


def debug_intermediate_values(matlab_dir: str, python_dir: str):
    """
    Debug by comparing intermediate computation values
    
    Useful for finding where Python diverges from MATLAB
    """
    print("Debugging intermediate values...")
    
    # Example: Compare block-sum images
    matlab_blocks = sio.loadmat(f"{matlab_dir}/debug_blocks.mat")['blocks']
    python_blocks = np.load(f"{python_dir}/debug_blocks.npz")['blocks']
    
    if check_numerical_equivalence(matlab_blocks, python_blocks):
        print("✓ Block-sum images match")
    else:
        diff = np.abs(matlab_blocks - python_blocks)
        print(f"✗ Block-sum mismatch: max diff = {np.max(diff)}")
        print(f"  Locations of large errors: {np.argwhere(diff > 0.01)[:10]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare MATLAB and Python results')
    parser.add_argument('--matlab_dir', type=str, required=True,
                       help='Directory with MATLAB results')
    parser.add_argument('--python_dir', type=str, required=True,
                       help='Directory with Python results')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['all', 'naive', 'align', 'merge', 'sr', 'bm3d'],
                       help='Which stage to compare')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization')
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = MATLABPythonComparator(args.matlab_dir, args.python_dir)
    
    # Compare
    if args.stage == 'all':
        comparison = comparator.compare_all_stages()
        comparator.generate_report(comparison)
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for stage, metrics in comparison.items():
            print(f"{stage:15s}: PSNR = {metrics['psnr']:6.2f} dB, " +
                  f"SSIM = {metrics['ssim']:.4f}")
    else:
        # Compare specific stage
        if args.visualize:
            comparator.visualize_comparison(args.stage)
    
    print("\nComparison complete!")
    print("\nInterpretation Guide:")
    print("  PSNR > 40 dB: Practically identical")
    print("  PSNR 30-40 dB: Very good agreement")
    print("  PSNR 20-30 dB: Acceptable for validation")
    print("  PSNR < 20 dB: Significant differences, needs debugging")
