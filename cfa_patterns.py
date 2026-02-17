def generate_bayer_rgb(H: int, W: int) -> np.ndarray:
    """Generate standard Bayer RGGB pattern"""
    cfa = np.zeros((H, W), dtype=np.uint8)
    cfa[0::2, 0::2] = 0  # R
    cfa[0::2, 1::2] = 1  # G
    cfa[1::2, 0::2] = 1  # G
    cfa[1::2, 1::2] = 2  # B
    return cfa

# Purpose: testing, ML pipelines' input regularization and neighbor finding logic should be robust to different Bayer patterns
def get_bayer(H, W, pattern_type=None):
    """
        Generate a random mosaic pattern for testing. Randomly choose one of the 4 Bayer patterns:
        "RGGB", "GRBG", "BGGR", "GBRG"
    """
    bayer = np.zeros((H, W), dtype=np.uint8)
    if pattern_type == None:
        pattern_type = random.choice(["RGGB", "GRBG", "BGGR", "GBRG"])

    if pattern_type == "RGGB":
        # Red
        bayer[0::2, 0::2] = 0
        bayer[0::2, 1::2] = 1
        bayer[1::2, 0::2] = 1
        bayer[1::2, 1::2] = 2
    elif pattern_type == "GRBG":
        bayer[0::2, 1::2] = 0
        bayer[0::2, 0::2] = 1
        bayer[1::2, 1::2] = 1
        bayer[1::2, 0::2] = 2
        
    elif pattern_type == "BGGR":
        bayer[0::2, 0::2] = 2
        bayer[0::2, 1::2] = 1
        bayer[1::2, 0::2] = 1
        bayer[1::2, 1::2] = 0
    
    else: # GBRG
        bayer[0::2, 0::2] = 1
        bayer[1::2, 1::2] = 1
        bayer[0::2, 1::2] = 2
        bayer[1::2, 0::2] = 0

    return bayer


def get_neighbor_colors(cfa: np.ndarray, y: int, x: int, radius: int = 2) -> list:
    """Get colors of neighbors within radius"""
    H, W = cfa.shape
    neighbors = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                neighbors.append(cfa[ny, nx])
    return neighbors


# Not used in this re-implementation as the sensor employs a regular Bayer pattern, but provided for completeness
def generate_rgbw_75_bluenoise(H: int, W: int, seed: int = 0) -> np.ndarray:
    """
    Generate 75% W blue-noise RGBW pattern (proposed in paper)
    
    Algorithm:
    1. Place W pixels in 2x2 grid (75% coverage)
    2. Place RGB in remaining pixels using blue-noise constraint
    """
    rng = np.random.RandomState(seed)
    cfa = np.zeros((H, W), dtype=np.uint8)
    
    # Step 1: Place W pixels (3 out of 4 in each 2x2 tile)
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            # Place W at 3 positions in 2x2 tile
            cfa[y, x] = 3  # W
            cfa[y, x+1] = 3  # W
            cfa[y+1, x] = 3  # W
            # One position left for RGB
    
    # Step 2: Place RGB in remaining pixels with blue-noise
    # Use dart-throwing algorithm
    for y in range(1, H, 2):
        for x in range(1, W, 2):
            # Check neighbors to avoid same color adjacency
            neighbors = get_neighbor_colors(cfa, y, x, radius=2)
            available_colors = [0, 1, 2]  # R, G, B
            
            # Remove colors that are too close
            for color in neighbors:
                if color < 3 and color in available_colors:
                    available_colors.remove(color)
            
            if available_colors:
                cfa[y, x] = rng.choice(available_colors)
            else:
                cfa[y, x] = rng.choice([0, 1, 2])
    
    return cfa


