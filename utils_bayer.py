
import numpy as np
from scipy import ndimage as ndi
from typing import Tuple


def generate_bayer_rggb(H: int, W: int) -> np.ndarray:
    """
    Generate Bayer RGGB pattern
    
    Pattern layout:
        R  G  R  G  ...
        G  B  G  B  ...
        R  G  R  G  ...
        G  B  G  B  ...
    
    Args:
        H: Height (should be even)
        W: Width (should be even)
        
    Returns:
        CFA pattern [H, W] where 0=R, 1=G, 2=B
    """
    if H % 2 != 0 or W % 2 != 0:
        print(f"Warning: Bayer pattern works best with even dimensions. Got {H}x{W}")
    
    cfa = np.zeros((H, W), dtype=np.uint8)
    cfa[0::2, 0::2] = 0  # R
    cfa[0::2, 1::2] = 1  # G
    cfa[1::2, 0::2] = 1  # G
    cfa[1::2, 1::2] = 2  # B
    
    return cfa


def get_color_name(color_idx: int) -> str:
    """Convert color index to name"""
    return ['R', 'G', 'B'][color_idx]


def bayer_to_grayscale_downsample(block_sum: np.ndarray, 
                                   cfa: np.ndarray = None) -> np.ndarray:
    """
    Convert Bayer block-sum to grayscale by downsampling
    
    This is the standard approach: average each 2x2 RGGB tile.
    Mentioned in Supplementary Section 1.1 of the paper.
    
    Args:
        block_sum: Mosaicked block-sum image [H, W]
        cfa: CFA pattern (not used, included for compatibility)
        
    Returns:
        Grayscale image [H//2, W//2]
    """
    H, W = block_sum.shape
    
    if H % 2 != 0 or W % 2 != 0:
        # Crop to even dimensions
        H = (H // 2) * 2
        W = (W // 2) * 2
        block_sum = block_sum[:H, :W]
    
    H_down = H // 2
    W_down = W // 2
    
    # Vectorized downsampling
    # Reshape to group 2x2 tiles
    reshaped = block_sum.reshape(H_down, 2, W_down, 2) 
    
    # Average over each tile (R + G + G + B) / 4
    gray = np.mean(reshaped, axis=(1, 3))
    
    return gray


def bayer_to_grayscale_demosaic(block_sum: np.ndarray,
                                 cfa: np.ndarray) -> np.ndarray:
    """
    Convert Bayer to grayscale by demosaicing first
    
    This maintains full resolution but is more computationally expensive.
    Use this if alignment precision is critical.
    
    Args:
        block_sum: Mosaicked block-sum image [H, W]
        cfa: CFA pattern [H, W]
        
    Returns:
        Grayscale image [H, W] (full resolution)
    """
    # Quick bilinear demosaic
    rgb = demosaic_bilinear(block_sum, cfa)
    
    # Convert to grayscale using standard weights
    gray = rgb_to_grayscale(rgb)
    
    return gray


def demosaic_bilinear(mosaicked: np.ndarray, 
                      cfa: np.ndarray) -> np.ndarray:
    """
    Fast bilinear demosaicing for Bayer pattern
    
    This is NOT for final image quality - just for alignment.
    Uses simple bilinear interpolation.
    
    Args:
        mosaicked: Mosaicked image [H, W]
        cfa: CFA pattern [H, W]
        
    Returns:
        RGB image [H, W, 3]
    """
    from scipy.ndimage import convolve
    
    H, W = mosaicked.shape
    rgb = np.zeros((H, W, 3))
    
    # Separate channels
    for c in range(3):
        # Mask for this color
        mask = (cfa == c).astype(float)
        
        # Bilinear interpolation kernel
        if c == 1:  # Green: 50% sampling, use 4-neighbor
            kernel = np.array([[0,    0.25, 0   ],
                              [0.25, 1,    0.25],
                              [0,    0.25, 0   ]], dtype=float)
        else:  # R or B: 25% sampling, use diagonal neighbors
            kernel = np.array([[0.25, 0,    0.25],
                              [0,    1,    0   ],
                              [0.25, 0,    0.25]], dtype=float)
        
        # Extract channel data
        channel_data = mosaicked * mask
        
        # Interpolate
        numerator = convolve(channel_data, kernel, mode='nearest')
        denominator = convolve(mask, kernel, mode='nearest') + 1e-10
        
        rgb[..., c] = numerator / denominator
    
    return rgb


def demosaic_gradient_based(mosaicked: np.ndarray,
                            cfa: np.ndarray) -> np.ndarray:
    """
    Gradient-based demosaicing (better quality than bilinear)
    
    Uses gradient information to reduce color artifacts.
    Based on Hamilton-Adams algorithm.
    
    Args:
        mosaicked: Mosaicked image [H, W]
        cfa: CFA pattern [H, W]
        
    Returns:
        RGB image [H, W, 3]
    """
    # For simplicity, fall back to bilinear
    # TODO: Implement proper gradient-based demosaicing
    # Reference: J. Hamilton and J. Adams, "Adaptive color plane interpolation"
    return demosaic_bilinear(mosaicked, cfa)


def rgb_to_grayscale(rgb: np.ndarray, 
                     weights: str = 'ITU-R') -> np.ndarray:
    """
    Convert RGB to grayscale
    
    Args:
        rgb: RGB image [H, W, 3]
        weights: 'ITU-R' for BT.709, 'uniform' for equal weights,
                'green' to emphasize green (exploit 50% sampling)
                
    Returns:
        Grayscale image [H, W]
    """
    if weights == 'ITU-R':
        # Standard ITU-R BT.709
        w_R, w_G, w_B = 0.2126, 0.7152, 0.0722
    elif weights == 'uniform':
        # Equal weights
        w_R, w_G, w_B = 1/3, 1/3, 1/3
    elif weights == 'green':
        # Emphasize green (it has 2x sampling in Bayer)
        w_R, w_G, w_B = 0.15, 0.70, 0.15
    else:
        raise ValueError(f"Unknown weights: {weights}")
    
    gray = w_R * rgb[..., 0] + w_G * rgb[..., 1] + w_B * rgb[..., 2]
    
    return gray


def apply_bayer_cfa(rgb: np.ndarray, cfa: np.ndarray) -> np.ndarray:
    """
    Apply Bayer CFA to RGB image (create mosaicked image)
    
    Useful for testing and simulation.
    
    Args:
        rgb: RGB image [H, W, 3]
        cfa: CFA pattern [H, W]
        
    Returns:
        Mosaicked image [H, W]
    """
    H, W = rgb.shape[:2]
    mosaicked = np.zeros((H, W))
    
    for c in range(3):
        mask = (cfa == c)
        mosaicked[mask] = rgb[mask, c] 
    
    return mosaicked


def get_bayer_statistics(cfa: np.ndarray) -> dict:
    """
    Get statistics about Bayer pattern
    
    Args:
        cfa: CFA pattern [H, W]
        
    Returns:
        Dictionary with channel statistics
    """
    total_pixels = cfa.size
    
    stats = {
        'R_count': np.sum(cfa == 0),
        'G_count': np.sum(cfa == 1),
        'B_count': np.sum(cfa == 2),
        'R_fraction': np.sum(cfa == 0) / total_pixels,
        'G_fraction': np.sum(cfa == 1) / total_pixels,
        'B_fraction': np.sum(cfa == 2) / total_pixels,
    }
    
    return stats


def visualize_bayer_pattern(cfa: np.ndarray, figsize: Tuple[int, int] = (8, 8)):
    """
    Visualize Bayer CFA pattern
    
    Args:
        cfa: CFA pattern [H, W]
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    # Create colored visualization
    H, W = cfa.shape
    viz = np.zeros((H, W, 3))
    
    viz[cfa == 0] = [1, 0, 0]  # Red
    viz[cfa == 1] = [0, 1, 0]  # Green
    viz[cfa == 2] = [0, 0, 1]  # Blue
    
    plt.figure(figsize=figsize)
    plt.imshow(viz, interpolation='nearest')
    plt.title('Bayer RGGB Pattern')
    plt.axis('off')
    
    # Add grid for small patterns
    if H <= 16 and W <= 16:
        for y in range(H+1):
            plt.axhline(y-0.5, color='gray', linewidth=0.5)
        for x in range(W+1):
            plt.axvline(x-0.5, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


def compare_grayscale_methods(block_sum: np.ndarray, 
                              cfa: np.ndarray) -> dict:
    """
    Compare different Bayer to grayscale conversion methods
    
    Args:
        block_sum: Mosaicked block-sum [H, W]
        cfa: CFA pattern [H, W]
        
    Returns:
        Dictionary with results from different methods
    """
    results = {}
    
    # Method 1: Downsample (standard)
    results['downsample'] = bayer_to_grayscale_downsample(block_sum, cfa)
    
    # Method 2: Demosaic first
    results['demosaic'] = bayer_to_grayscale_demosaic(block_sum, cfa)
    
    return results


# Bayer-specific denoising
def denoise_bayer_rgb(rgb: np.ndarray,
                      sigma_Y: float = 80,
                      sigma_C: float = 150,
                      use_bm3d: bool = True) -> np.ndarray:
    """
    Denoise Bayer-reconstructed RGB image
    
    Since there's no W channel, use standard YCbCr denoising.
    Can optionally emphasize Green channel's higher SNR.
    
    Args:
        rgb: RGB image [H, W, 3]
        sigma_Y: Luminance channel noise sigma
        sigma_C: Chrominance channel noise sigma
        use_bm3d: Use BM3D (if available) vs Gaussian fallback
        
    Returns:
        Denoised RGB image [H, W, 3]
    """
    # Convert to YCbCr
    ycbcr = rgb_to_ycbcr_standard(rgb)
    
    # Anscombe transform
    ycbcr_anscombe = anscombe_transform(ycbcr)
    
    # Denoise
    if use_bm3d:
        try:
            import bm3d
            Y_denoised = bm3d.bm3d(ycbcr_anscombe[..., 0], sigma_psd=sigma_Y/255)
            Cb_denoised = bm3d.bm3d(ycbcr_anscombe[..., 1], sigma_psd=sigma_C/255)
            Cr_denoised = bm3d.bm3d(ycbcr_anscombe[..., 2], sigma_psd=sigma_C/255)
        except ImportError:
            print("BM3D not available, using Gaussian filter")
            use_bm3d = False
    
    if not use_bm3d:
        # Fallback: Gaussian filter
        from scipy.ndimage import gaussian_filter
        Y_denoised = gaussian_filter(ycbcr_anscombe[..., 0], sigma_Y/100)
        Cb_denoised = gaussian_filter(ycbcr_anscombe[..., 1], sigma_C/100)
        Cr_denoised = gaussian_filter(ycbcr_anscombe[..., 2], sigma_C/100)
    
    # Inverse transforms
    ycbcr_denoised = np.stack([Y_denoised, Cb_denoised, Cr_denoised], axis=-1)
    ycbcr_denoised = inverse_anscombe_transform(ycbcr_denoised)
    rgb_denoised = ycbcr_to_rgb_standard(ycbcr_denoised)
    
    return rgb_denoised


def rgb_to_ycbcr_standard(rgb: np.ndarray) -> np.ndarray:
    """
    Standard RGB to YCbCr conversion (ITU-R BT.709)
    
    Args:
        rgb: RGB image [H, W, 3]
        
    Returns:
        YCbCr image [H, W, 3]
    """
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Cb = (B - Y) / 1.8556
    Cr = (R - Y) / 1.5748
    
    return np.stack([Y, Cb, Cr], axis=-1)


def ycbcr_to_rgb_standard(ycbcr: np.ndarray) -> np.ndarray:
    """
    Standard YCbCr to RGB conversion
    
    Args:
        ycbcr: YCbCr image [H, W, 3]
        
    Returns:
        RGB image [H, W, 3]
    """
    Y, Cb, Cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    
    R = Y + 1.5748 * Cr
    B = Y + 1.8556 * Cb
    G = (Y - 0.2126 * R - 0.0722 * B) / 0.7152
    
    return np.stack([R, G, B], axis=-1)


def anscombe_transform(img: np.ndarray) -> np.ndarray:
    """
    Anscombe transform for Poisson/binomial noise
    Converts to approximately Gaussian noise
    """
    return 2 * np.sqrt(np.maximum(0, img + 3/8))


def inverse_anscombe_transform(img: np.ndarray) -> np.ndarray:
    """Inverse Anscombe transform"""
    return np.maximum(0, (img / 2) ** 2 - 3/8)


if __name__ == "__main__":
    print("Testing Bayer utilities...")
    
    # Test 1: Generate pattern
    H, W = 256, 256
    cfa = generate_bayer_rggb(H, W)
    stats = get_bayer_statistics(cfa)
    
    print(f"\nBayer pattern statistics:")
    print(f"  R: {stats['R_fraction']*100:.1f}%")
    print(f"  G: {stats['G_fraction']*100:.1f}%")
    print(f"  B: {stats['B_fraction']*100:.1f}%")
    
    # Test 2: Grayscale conversion
    # Create synthetic Bayer image
    rgb_test = np.random.rand(H, W, 3)
    mosaicked = apply_bayer_cfa(rgb_test, cfa)
    
    # Test downsample method
    gray_down = bayer_to_grayscale_downsample(mosaicked, cfa)
    print(f"\nDownsampled shape: {gray_down.shape}")
    print(f"Expected: ({H//2}, {W//2})")
    
    # Test demosaic method
    gray_demosaic = bayer_to_grayscale_demosaic(mosaicked, cfa)
    print(f"Demosaic shape: {gray_demosaic.shape}")
    print(f"Expected: ({H}, {W})")
    
    print("\n✓ Bayer utilities test complete!")