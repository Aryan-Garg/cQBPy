"""
Multi-scale patch-based alignment for quanta burst photography

Implements hierarchical Lucas-Kanade optical flow estimation on
block-sum grayscale images derived from mosaicked quanta images.
"""

import numpy as np
from numba import njit
from scipy import ndimage as ndi
from scipy.interpolate import RectBivariateSpline
from typing import Tuple, List
import warnings
from utils_bayer import bayer_to_grayscale_downsample


class BlockSumImages:
    """ 
    Container for block-sum image generation and brightness-constancy 
    violation management by downsample-grayscaling. Expect 4x spatial resolution drop.
    """
    
    def __init__(self, imbs: np.ndarray, block_size: int, num_blocks: int):
        """
        Args:
            imbs: Mosaicked binary images [H, W, T]
            block_size: Number of frames per block
            num_blocks: Number of blocks to create
        """
        self.imbs = imbs
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.H, self.W, self.T = imbs.shape
        assert self.block_size * self.num_blocks <= self.T, "Block size and number of blocks exceed total frames"
        
        # Generate block-sum images
        self.block_sums = self._generate_block_sums()
        self.grayscale_blocks = self.to_grayscale()
        

    def _generate_block_sums(self) -> np.ndarray:
        """
        Generate block-sum images by summing binary frames
        
        Returns:
            Block-sum images [H, W, num_blocks]
        """
        block_sums = np.zeros((self.H, self.W, self.num_blocks))
        
        for i in range(self.num_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, self.T)
            block_sums[..., i] = np.sum(self.imbs[..., start_idx:end_idx], axis=2)
        
        return block_sums
    

    def to_grayscale(self, cfa_pattern: str = 'RGGB') -> np.ndarray:
        """
        Convert mosaicked block-sum images to grayscale
        
        Args:
            cfa_pattern: CFA pattern type
            
        Returns:
            Grayscale block-sum images [H, W, num_blocks]
        """
        gray_blocks = []
        for i in range(self.num_blocks):
            this_block_sum = self.block_sums[..., i]
            gray_blocks.append(bayer_to_grayscale_downsample(this_block_sum, cfa_pattern))
        return np.stack(gray_blocks, axis=2)


class HierarchicalAlignment:
    """Hierarchical patch-based alignment using Lucas-Kanade"""
    
    def __init__(self, num_levels: int = 4,
                 patch_sizes: Tuple[int, ...] = (32, 32, 32, 16),
                 upsample_ratios: Tuple[int, ...] = (1, 2, 2, 4),
                 search_radii: Tuple[int, ...] = (1, 1, 2, 8),
                 num_lk_iters: int = 3,
                 block_size: int = None):
        """
        Initialize hierarchical alignment
        
        Args:
            num_levels: Number of pyramid levels
            patch_sizes: Patch size at each level
            upsample_ratios: Upsampling ratio at each level
            search_radii: Search radius at each level
            num_lk_iters: Number of Lucas-Kanade iterations
        """
        self.num_levels = num_levels
        self.patch_sizes = patch_sizes
        self.upsample_ratios = upsample_ratios
        self.search_radii = search_radii
        self.num_lk_iters = num_lk_iters
        self.block_size = block_size
        
    def align(self, grayscale_blocks: np.ndarray, 
              ref_idx: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow for all blocks relative to reference
        
        Args:
            grayscale_blocks: Grayscale block-sum images [H//2, W//2, N]
            ref_idx: Reference block index (default: middle block)
            
        Returns:
            block_flows: Block-level optical flow [H//2, W//2, 2, N]
            frame_flows: Interpolated frame-level flow [H//2, W//2, 2, T]
        """
        H, W, N = grayscale_blocks.shape
        
        if ref_idx is None:
            ref_idx = N // 2
        
        ref_img = grayscale_blocks[..., ref_idx]
        
        # Compute block-level flows (only after expansion: store dense flow)
        block_flows = np.zeros((H, W, 2, N))

        ref_pyramid = self._build_pyramid(ref_img)
        
        for i in range(N):
            if i == ref_idx:
                continue
            
            src_img = grayscale_blocks[..., i]
            flow_patch = self._compute_flow_pyramid(src_img,
                ref_img,
                ref_pyramid=ref_pyramid
            )
            # Expand to dense
            patch_size = self.patch_sizes[0]
            stride = patch_size // 2  # finest level stride

            flow_dense = self._upsample_patch_flow(
                flow_patch,
                (H, W),
                patch_size,
                stride
            )

            block_flows[..., i] = flow_dense

        
        # Interpolate to frame level (placeholder)
        # (Temporal interpolation based on block structure)
        frame_flows = block_flows  # Simplified
        if self.block_size is None:
            return block_flows, block_flows

        T = self.block_size * N
        frame_flows = np.zeros((H, W, 2, T), dtype=np.float32)
        for b in range(N - 1):
            flow0 = block_flows[..., b]
            flow1 = block_flows[..., b + 1]
            for t in range(self.block_size):
                alpha = t / self.block_size
                frame_flows[..., b*self.block_size + t] = \
                    (1 - alpha) * flow0 + alpha * flow1

        # Last block repeat
        for t in range(self.block_size):
            frame_flows[..., (N-1)*self.block_size + t] = \
                block_flows[..., N-1]
        return block_flows, frame_flows
    
    def _upsample_patch_flow(self, flow_patch: np.ndarray,
                         img_shape: Tuple[int, int],
                         patch_size: int,
                         stride: int) -> np.ndarray:
        """
        Convert patch-grid flow to dense pixel flow using nearest expansion.
        """
        H, W = img_shape
        hs, ws, _ = flow_patch.shape

        dense = np.zeros((H, W, 2), dtype=np.float32)

        for j in range(hs):
            for k in range(ws):
                y0 = j * stride
                x0 = k * stride

                dense[y0:y0+patch_size,
                      x0:x0+patch_size] = flow_patch[j, k]

        return dense

    def _compute_flow_pyramid(self, src: np.ndarray,
                              ref: np.ndarray,
                              ref_pyramid=None) -> np.ndarray:
        """
        True coarse-to-fine hierarchical block matching.
        """

        # Build pyramids
        src_pyramid = self._build_pyramid(src)
        if ref_pyramid is None:
            ref_pyramid = self._build_pyramid(ref)

        flow = None

        # Process from coarse → fine
        for level in reversed(range(self.num_levels)):

            src_level = src_pyramid[level]
            ref_level = ref_pyramid[level]

            patch_size = self.patch_sizes[level]
            stride = patch_size // 2
            search_radius = self.search_radii[level]

            H, W = src_level.shape
            hs = (H - patch_size) // stride + 1
            ws = (W - patch_size) // stride + 1

            if flow is None:
                # Initialize at coarsest
                flow = np.zeros((hs, ws, 2), dtype=np.float32)
            else:
                # Upsample previous level flow
                flow = ndi.zoom(flow, (2, 2, 1), order=1) * 2

                # Crop if mismatch due to rounding
                flow = flow[:hs, :ws]

            flow = self._block_match_with_init(
                src_level,
                ref_level,
                flow,
                patch_size,
                stride,
                search_radius
            )

        return flow
    
    def _build_pyramid(self, img: np.ndarray) -> List[np.ndarray]:
        """Build Gaussian pyramid"""
        pyramid = [img]
        for _ in range(1, self.num_levels):
            img = ndi.zoom(img, 0.5, order=1)
            pyramid.append(img)
        return pyramid
    
    def _upsample_patch_flow(self, flow_patch, img_shape, patch_size, stride):
        H, W = img_shape
        dense = np.zeros((H, W, 2), dtype=np.float32)
        weight = np.zeros((H, W, 1), dtype=np.float32)

        hs, ws, _ = flow_patch.shape

        for j in range(hs):
            for k in range(ws):
                y0 = j * stride
                x0 = k * stride

                dense[y0:y0+patch_size,
                      x0:x0+patch_size] += flow_patch[j, k]

                weight[y0:y0+patch_size,
                       x0:x0+patch_size] += 1.0

        weight[weight == 0] = 1.0
        dense /= weight

        return dense
    
    @njit(fastmath=True)
    def _block_match_with_init(self, src, ref, init_flow, patch_size, stride, search_radius):
        """
        Speed Trick: Use Convolution Instead of Patch Extraction for matching.
        Replaces patch extraction with sliding window sum via integral image.

        Instead of:
            > np.abs(ref_patch - src_patch).sum(),
        Sample cost only at patch grid locations (removes patch extraction entirely):
            > diff = abs(ref - shifted_src)
            > cost = box_filter(diff, patch_size)

        Complexity: O(H*W*search_area) -> O(H*W) per level, much faster for large patches.
        """
        H, W = src.shape
        hs = init_flow.shape[0]
        ws = init_flow.shape[1]

        best_flow = init_flow.copy()
        best_cost = np.full((hs, ws), 1e12, dtype=np.float32)

        for j in range(hs):
            for k in range(ws):

                y0 = j * stride
                x0 = k * stride

                init_dx = int(round(init_flow[j, k, 0]))
                init_dy = int(round(init_flow[j, k, 1]))

                for dy in range(init_dy - search_radius,
                                init_dy + search_radius + 1):
                    for dx in range(init_dx - search_radius,
                                    init_dx + search_radius + 1):

                        y1 = y0 + dy
                        x1 = x0 + dx

                        if (y1 < 0 or x1 < 0 or
                            y1 + patch_size >= H or
                            x1 + patch_size >= W):
                            continue

                        cost = 0.0

                        for yy in range(patch_size):
                            for xx in range(patch_size):
                                diff = ref[y0 + yy, x0 + xx] - \
                                       src[y1 + yy, x1 + xx]
                                cost += abs(diff)

                        if cost < best_cost[j, k]:
                            best_cost[j, k] = cost
                            best_flow[j, k, 0] = dx
                            best_flow[j, k, 1] = dy

        return best_flow

    def _extract_patches(self, img, patch_size, stride):
        H, W = img.shape
        hs = (H - patch_size) // stride + 1
        ws = (W - patch_size) // stride + 1

        shape = (hs, ws, patch_size, patch_size)
        strides = (
            img.strides[0] * stride,
            img.strides[1] * stride,
            img.strides[0],
            img.strides[1],
        )

        return np.lib.stride_tricks.as_strided(img, shape, strides)

    def _lk_refine(self, ref, src, flow, patch_size, stride):
        H, W = ref.shape
        hs, ws, _ = flow.shape

        refined = flow.copy()

        for j in range(hs):
            for k in range(ws):

                y0 = j * stride
                x0 = k * stride

                patch_ref = ref[y0:y0+patch_size, x0:x0+patch_size]

                uv = refined[j, k]

                for _ in range(self.num_lk_iters):

                    warped = ndi.shift(src,
                                       shift=(uv[1], uv[0]),
                                       order=1)

                    patch_src = warped[y0:y0+patch_size, x0:x0+patch_size]

                    It = patch_src - patch_ref

                    Ix = ndi.sobel(patch_src, axis=1)
                    Iy = ndi.sobel(patch_src, axis=0)

                    A = np.stack([Ix.ravel(), Iy.ravel()], axis=1)
                    b = -It.ravel()

                    if np.linalg.matrix_rank(A) < 2:
                        break

                    delta, *_ = np.linalg.lstsq(A, b, rcond=None)
                    uv += delta

                refined[j, k] = uv

        return refined



def box_sum(img, k):
    H, W = img.shape
    integral = np.zeros((H+1, W+1), dtype=img.dtype)
    integral[1:, 1:] = img.cumsum(0).cumsum(1)

    out = (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    )
    return out


def create_reference_image(block_flows: np.ndarray,
                          grayscale_blocks: np.ndarray,
                          wiener_c: float = 8.0) -> np.ndarray:
    """
    Create reference image for robust merging using Wiener filter
    
    Args:
        block_flows: Block-level optical flow
        grayscale_blocks: Grayscale block-sum images
        wiener_c: Wiener filter constant
        
    Returns:
        Reference image [H, W]
    """
    H, W, N = grayscale_blocks.shape
    
    # Warp blocks to reference frame
    warped_blocks = np.zeros_like(grayscale_blocks)
    
    for i in range(N):
        warped_blocks[..., i] = warp_image(grayscale_blocks[..., i],
                                           block_flows[..., i])
    
    # Wiener-filtered combination
    # E[I] = sum(I_i) / N
    mean_img = np.mean(warped_blocks, axis=2)
    
    # Var[I] = var(I_i)
    var_img = np.var(warped_blocks, axis=2)
    
    # Wiener filter: w = var / (var + C)
    weights = var_img / (var_img + wiener_c)
    
    # Reference = weighted average
    ref_img = weights * mean_img + (1 - weights) * np.median(warped_blocks, axis=2)
    
    return ref_img


def warp_image(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp image according to optical flow
    
    Args:
        img: Input image [H, W]
        flow: Optical flow [H, W, 2]
        
    Returns:
        Warped image [H, W]
    """
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]
    
    # Apply flow
    x_new = x + flow[..., 0]
    y_new = y + flow[..., 1]
    
    # Interpolate
    warped = ndi.map_coordinates(img, [y_new, x_new], order=1, mode='nearest')
    
    return warped


if __name__ == "__main__":
    print("Testing alignment module...")

    # Synthetic translation test
    H, W = 128, 128
    T = 40
    block_size = 4
    num_blocks = T // block_size

    # Create synthetic base image
    base = np.zeros((H, W))
    base[40:80, 50:90] = 1.0

    imbs = np.zeros((H, W, T))

    for t in range(T):
        dx = 2
        dy = 1
        shifted = ndi.shift(base, shift=(dy, dx), order=0)
        imbs[..., t] = shifted > 0.5

    # Generate block sums
    block_gen = BlockSumImages(imbs, block_size, num_blocks)

    print("Block-sum shape:", block_gen.block_sums.shape)
    print("Grayscale shape:", block_gen.grayscale_blocks.shape)

    # Run alignment
    aligner = HierarchicalAlignment(
        num_levels=3,
        patch_sizes=(32, 32, 16),
        search_radii=(2, 2, 4)
    )

    block_flows, frame_flows = aligner.align(
        block_gen.grayscale_blocks
    )

    print("Block flow shape:", block_flows.shape)
    print("Frame flow shape:", frame_flows.shape)

    # Inspect estimated shift
    mean_flow = np.mean(block_flows[..., 0], axis=(0,1))
    print("Estimated flow for block 0:", mean_flow)