import numpy as np

def identify_hot_pixels(dcr_map: np.ndarray, threshold: float = 30.0) -> np.ndarray:
    """
    Identify hot pixels from DCR calibration
    
    Args:
        dcr_map: Dark count rate map [H, W]
        threshold: DCR threshold (counts per second)
        
    Returns:
        Binary mask [H, W] where True = hot pixel
    """
    return dcr_map > threshold


def precompute_k_valid_neighbors(hot_pixel_mask: np.ndarray,
                              cfa_pattern: np.ndarray,
                              k: int = 3) -> dict:
    """
    For each hot pixel, find k-nearest non-hot pixels with same color.
    NOTE: This fn assumes CFA pattern to be bayer. 
    RGBW pattern from the paper is not supported as the color sensor employs a regular Bayer pattern.
    
    Args:
        hot_pixel_mask: Binary mask of hot pixels [H, W]
        cfa_pattern: CFA pattern [H, W] where 0=R, 1=G, 2=B
        k: Number of neighbors
        
    Returns:
        Dict mapping (y,x) -> list of (neighbor_y, neighbor_x) coordinates
    """
    from scipy.spatial import cKDTree
    
    neighbor_dict = {}
    H, W = hot_pixel_mask.shape
    
    # For each color channel
    for color in range(3):
        # Find hot pixels of this color
        hot_of_color = hot_pixel_mask & (cfa_pattern == color)

        # Find non-hot pixels of this color
        non_hot_of_color = (~hot_pixel_mask) & (cfa_pattern == color)
        
        if not hot_of_color.any() or not non_hot_of_color.any():
            continue
        
        # Build KDTree of non-hot pixels
        non_hot_coords = np.argwhere(non_hot_of_color)
        tree = cKDTree(non_hot_coords)
        
        # Query k-nearest for each hot pixel
        hot_coords = np.argwhere(hot_of_color)
        _, indices = tree.query(hot_coords, k=k)
        
        for i, (hy, hx) in enumerate(hot_coords):
            neighbor_coords = non_hot_coords[indices[i]]
            neighbor_dict[(hy, hx)] = [tuple(coord) for coord in neighbor_coords]
    
    return neighbor_dict


def correct_hot_pixels(imbs: np.ndarray,
                      neighbor_dict: dict,
                      seed: int = 0) -> np.ndarray:
    """
    Correct hot pixels by random replacement
    
    Args:
        imbs: Binary image sequence [H, W, T]
        neighbor_dict: Pre-computed neighbor lists
        seed: Random seed
        
    Returns:
        Corrected binary images [H, W, T]
    """
    rng = np.random.RandomState(seed)
    corrected = imbs.copy()
    
    for (y, x), neighbors in neighbor_dict.items():
        for t in range(imbs.shape[2]):
            # Randomly select one neighbor
            ny, nx = neighbors[rng.randint(len(neighbors))]
            # Replace with neighbor value
            corrected[y, x, t] = imbs[ny, nx, t]
    
    return corrected


if __name__ == "__main__":
    ### Test:
    H, W = 8, 8
    hot_mask = np.random.rand(H, W) > 0.97  # 3% hot pixels

    # Create dummy CFA
    cfa = np.mod(np.arange(H*W).reshape(H,W), 3)

    # Test neighbor finding
    neighbors = precompute_k_valid_neighbors(hot_mask, cfa, k=3)
    # assert all(len(v) == 3 for v in neighbors.values())

    # Test correction
    imbs = np.random.rand(H, W, 100) > 0.90
    corrected = correct_hot_pixels(imbs, neighbors)
    
    import matplotlib.pyplot as plt
    plt.subplot(1, 3, 1)
    plt.title("Og w HP")
    plt.imshow(imbs[:, :, 0], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("HP Mask")
    plt.imshow(hot_mask, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Corrected")
    plt.imshow(corrected[:, :, 0], cmap='gray')
    plt.savefig("hot_pixel_correction_test.png")