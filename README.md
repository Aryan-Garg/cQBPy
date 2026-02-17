# Fast Python Implementation of Color Quanta Burst Photography (cQBP)
from *Seeing Photons in Color* (Ma et al., SIGGRAPH 2023)

> By Aryan Garg, Avery Gump

## Overview

This repository contains a Python reimplementation of the MATLAB codebase for color single-photon avalanche diode (SPAD) burst photography. The pipeline processes mosaicked binary quanta images to produce high-quality color reconstructions.

### Pipeline Stages

1. **Naive Reconstruction** - Simple temporal averaging + demosaicing
2. **Patch Alignment** - Multi-scale optical flow estimation  
3. **Joint Demosaicking & Merging** - Robust temporal fusion with color recovery
4. **Super-Resolution** (optional) - Higher resolution reconstruction
5. **Post-Processing** - Chrominance-focused BM3D denoising



---

## 🚀 Quick Start

```python
from quanta_color_pipeline import QuantaBurstPipeline, QuantaParams, load_quanta_data

# Load data
imbs, imgt, dcr = load_quanta_data('dataBayer_0121_1.mat')

# Configure parameters
params = QuantaParams(
    alignTWSize=7,
    mergeTWSize=7,
    tau=4.17e-5,
    doSR=True
)

# Run pipeline
pipeline = QuantaBurstPipeline(params)
results = pipeline.run(imbs, imgt)

# Access results
naive_recons = results['naive_recons']
merged = results['merged']
sr_denoised = results['sr_denoised']
```

---

## 📁 Project Structure

```
.
├── quanta_color_pipeline.py    # Main pipeline implementation
├── alignment.py                 # Optical flow alignment module
├── demosaicing.py              # Universal demosaicing algorithms
├── merging.py                  # Joint demosaic-merge with robust weighting
├── denoising.py                # Chrominance-focused BM3D
├── hot_pixel_correction.py     # Hot pixel preprocessing
├── utils.py                    # Utility functions
└── README.md                   # This file
```

---

## Key Components to Implement

### 1. **Multi-Scale Optical Flow** (`alignment.py`)

The alignment step is critical for motion compensation:

**Algorithm:**
1. Divide binary frames into temporal blocks
2. Sum frames in each block → multi-bit block-sum images
3. Convert mosaicked block-sums to grayscale (universal demosaicing)
4. Build Gaussian pyramid for coarse-to-fine matching
5. Lucas-Kanade optical flow at each pyramid level
6. Interpolate block-level flow to frame level

**Key Equations:**

Block-sum formation:
```python
S_block[x,y] = Σ(B_t[x,y])  # Sum over frames t in block
```

Optical flow estimation (Lucas-Kanade):
```python
∇I · Δp = -(I_src(x+u) - I_ref(x))
# Solve for Δp (flow update) iteratively
```

**Reference:** Section 4.1 of main paper, Supplementary Section 1.1

---

### 2. **Joint Demosaicking & Merging** (`merging.py`)

**Key Idea:** Treat each pixel in binary frames as a 1-bit color sample, warp to reference frame, and combine with adaptive weighting.

**Algorithm:**

For each pixel (x, y) in reference frame:
1. Collect warped color samples from neighborhood N
2. Compute weighted sum:

```python
S_c(x,y) = Σ(w_i · S_ci) / Σ(w_i)  # For channel c ∈ {R,G,B}
```

Where weight w_i = w_Gi · w_Ri consists of:

**a) Anisotropic Gaussian kernel** (based on local structure):
```python
w_Gi = exp(-0.5 · (x_i - x)^T · Ω^{-1} · (x_i - x))
```

Where Ω is the structure tensor from the reference image.

**b) Robustness term** (penalize misaligned patches):
```python
R = clamp(s · exp(-(x - μ_s)^2 / (s_c · (σ_s^2 + σ_b^2))), 0, 1)

# σ_b accounts for binomial noise:
σ_b = sqrt((S/T) · (1 - S/T) / T)
```

**Reference:** Section 4.2 of main paper, Supplementary Sections 1.2-1.3

---

### 3. **Universal Demosaicing** (`demosaicing.py`)

For arbitrary CFAs, use variational demosaicing (Condat 2009):

**Optimization problem:**
```
min_I  Σ ||∇I(x)|| + λ Σ (I_c(x) - S_c(x))^2
       x             x∈Ω_c
```

Where:
- ∇I: Image gradient (promotes smoothness)
- Ω_c: Set of pixels with color filter c
- λ: Data fidelity weight

**For RGBW with 75% W:** Can simplify by interpolating W, then using constraint:
```
w_R·R + w_G·G + w_B·B = W
```

**Reference:** Condat 2009 (cited in paper), Supplementary Section 1.3

---

### 4. **Chrominance-Focused BM3D** (`denoising.py`)

**Key Insight:** W channel has higher SNR than RGB. Denoise chrominance more aggressively.

**Algorithm:**

1. **Transform to YCbCr:**
```python
# Modified YCbCr where Y = weighted sum of W:
k_R, k_G, k_B = normalize(w_R, w_G, w_B)
Y = k_R·R + k_G·G + k_B·B  # ≈ W channel (high SNR)
Cb = (B - Y) / (2(1 - k_B))
Cr = (R - Y) / (2(1 - k_R))
```

2. **Apply Anscombe transform** (convert binomial → Gaussian noise):
```python
Z = 2 · sqrt(I + 3/8)
```

3. **BM3D with channel-specific σ:**
```python
σ_Y = 80    # Lower noise (from W pixels)
σ_Cb = 150  # Higher noise (from sparse RGB)
σ_Cr = 150
```

4. **Inverse Anscombe transform:**
```python
I = max(0, (Z/2)^2 - 3/8)
```

**Reference:** Supplementary Section 1.5

---

### 5. **Hot Pixel Correction** (`hot_pixel_correction.py`)

**Problem:** SPADs have hot pixels with exceptionally high dark count rate (DCR).

**Solution:** Random replacement *before* motion estimation

**Algorithm:**

For each hot pixel (x,y) with color filter c:
1. Pre-compute k-nearest non-hot pixels with same color c
2. For each binary frame:
   - Randomly select one neighbor from k-nearest
   - Replace B(x,y) with neighbor value

**Why before motion estimation?**  
Hot pixels bias alignment toward zero flow, causing blur.

**Reference:** Supplementary Section 1.4


---

## Parameter Guide

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 4.17e-5 | Exposure time per frame (seconds) |
| `eta` | [0.125, 0.135, 0.09] | PDE for R, G, B |
| `maxFlux` | 0.7e5 | Maximum photon flux |
| `dcr` | 2.0 | Average dark count rate |

### Alignment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alignTWSize` | 7 | Frames per alignment block |
| `alignTWNum` | 11 | Number of alignment blocks |
| `numLevels` | 4 | Pyramid levels |
| `patchSizes` | [32,32,32,16] | Patch size per level |
| `searchRadii` | [1,1,2,8] | Search radius per level |
| `flowLambda` | 0.01 | Flow smoothness weight |

### Merging Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mergeTWSize` | 7 | Frames per merge block |
| `wienerC` | 8.0 | Wiener filter constant |
| `combineRadius` | 1 | Neighborhood radius for merging |

### Denoising Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bm3dSigma` | 0.05 | Base BM3D sigma |
| `qbm3dSigmaW` | 80 | W channel sigma |
| `qbm3dSigmaC` | 150 | Chrominance sigma |

---

## Mathematical Details

### Imaging Model

Single-photon camera model:
```
B(x,y,t) ~ Bernoulli(1 - exp(-(ρ(x,y)·τ + r_d·τ)))
```

Where:
- ρ(x,y): Color intensity (photons/second)
- τ: Exposure time per frame
- r_d: Dark count rate

**Maximum Likelihood Estimator:**
```python
ρ_hat(x,y) = -ln(1 - S(x,y)/n) / τ - r_d
```

Where S(x,y) = Σ B_t(x,y) and n = number of frames.

### Color Intensity

For channel c ∈ {R, G, B, W}:
```
ρ_c = ∫ φ(λ) · η_c(λ) dλ
```

Where:
- φ(λ): Spectral photon flux
- η_c(λ): Photon detection efficiency (PDE) for color c

---

## Optimization Tips

### 1. Vectorization

**Bad:**
```python
for t in range(T):
    for y in range(H):
        for x in range(W):
            result[y,x,t] = process(img[y,x,t])
```

**Good:**
```python
result = process(img)  # Vectorized NumPy operation
```

### 2. Memory Management

Process in chunks for large sequences:
```python
chunk_size = 1000
for start in range(0, T, chunk_size):
    end = min(start + chunk_size, T)
    chunk = imbs[..., start:end]
    process_chunk(chunk)
```

### 3. Numba JIT Compilation

For critical loops:
```python
from numba import jit

@jit(nopython=True)
def lucas_kanade_iteration(img_grad, flow):
    # Inner loop with Numba acceleration
    ...
```

### 4. Parallel Processing

```python
from multiprocessing import Pool

with Pool(num_cores) as pool:
    results = pool.map(process_block, blocks)
```

---

## Setup & Dependencies

**Using the given env file:**

```bash
conda env create -f environment.yml
```

**Manual:**

```bash
conda create --name cqbpy python=3.12
```

```bash
pip install numpy scipy opencv-python scikit-image numba bm3d matplotlib h5py
```



---

## Testing & Validation

### Compare with MATLAB

```python
import scipy.io as sio

# Load MATLAB results
matlab_results = sio.loadmat('matlab_output.mat')
matlab_merged = matlab_results['imr']

# Run Python pipeline
python_merged = pipeline.run(imbs)['merged']

# Compute difference
mse = np.mean((matlab_merged - python_merged) ** 2)
psnr = 20 * np.log10(1.0 / np.sqrt(mse))

print(f"PSNR vs MATLAB: {psnr:.2f} dB")
# Target: >40 dB (practically identical)
```

### Reproduce Paper Results

Test on simulated data from paper:
- HDR scenes (Fig. 4 in paper)
- Fast motion (Fig. 18)
- Low light (Fig. 15)

---

## References

### Primary Paper
```bibtex
@article{ma2023seeing,
  title={Seeing Photons in Color},
  author={Ma, Sizhuo and Sundar, Varun and Mos, Paul and Bruschini, Claudio and Charbon, Edoardo and Gupta, Mohit},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  year={2023}
}
```

### Key Algorithms
1. **Optical Flow:** Lucas-Kanade pyramidal algorithm
2. **Demosaicing:** Condat (2009) - Universal variational demosaicing
3. **Denoising:** BM3D (Dabov et al. 2007)
4. **Wiener Merging:** Hasinoff et al. (2016) - Burst photography for HDR

---


## Known Issues & TODOs

### High Priority
- [ ] Implement full Lucas-Kanade optical flow
- [ ] Implement universal demosaicing (Condat's algorithm)
- [ ] Implement robust weighting for joint merge
- [ ] Add BM3D integration (use `bm3d` package)
- [ ] Hot pixel correction preprocessing

### Medium Priority
- [ ] Super-resolution implementation
- [ ] Flow refinement iteration
- [ ] Visualization tools (flow fields, error maps)
- [ ] Batch processing utilities

### Low Priority
- [ ] GPU acceleration (CuPy)
- [ ] Alternative demosaicing methods
- [ ] Interactive parameter tuning GUI

---

## Usage Examples

### Example 1: Process Real SPAD Data
```python
# Load .mat file from MATLAB codebase
imbs, imgt, dcr = load_quanta_data('real_spad_capture.mat')

# Run with default parameters
pipeline = QuantaBurstPipeline(QuantaParams())
results = pipeline.run(imbs)

# Save results
import imageio
imageio.imwrite('result_naive.png', (results['naive_recons'] * 255).astype(np.uint8))
imageio.imwrite('result_merged.png', (results['merged'] * 255).astype(np.uint8))
```

### Example 2: Optimize Parameters
```python
params = QuantaParams(
    alignTWSize=10,      # Larger blocks for smoother motion
    wienerC=4.0,         # Less aggressive Wiener filtering
    bm3dSigma=0.03,      # Less denoising
    doSR=True,
    srScale=2            # 2x super-resolution
)

pipeline = QuantaBurstPipeline(params)
results = pipeline.run(imbs)
```

---


## Contact

For questions about the algorithm, refer to the original paper.  
For implementation issues, please open a GitHub issue here.


---
