import json
import random
import warnings
from math import floor, ceil, log, exp
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import h5py
import numpy as np
import torch
import video_reader
from einops import rearrange, reduce
from jaxtyping import Float, Int
from loguru import logger
from natsort import natsorted
from scipy import ndimage, spatial
from torch import Tensor
from torch.utils.data import Dataset

from probabilistic_burst.filters.malvar_he_cutler import BayerPatternLiteral
from probabilistic_burst.ops.image import nearest_neighbor_indices

warnings.filterwarnings("ignore", category=UserWarning)


def bayer_nearest_neighbor_indices(
    bad_pixel_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes nearest valid neighbor indices for bad pixels, respecting the Bayer CFA pattern.
    """
    valid_mask = ~bad_pixel_mask.astype(bool)
    h, w = bad_pixel_mask.shape

    query_i, query_j = np.where(bad_pixel_mask.astype(bool))
    query_coords = np.stack([query_i, query_j], axis=-1)

    source_i, source_j = np.where(valid_mask)
    source_coords = np.stack([source_i, source_j], axis=-1)

    nearest_i = np.zeros_like(query_i)
    nearest_j = np.zeros_like(query_j)

    for i_offset in [0, 1]:
        for j_offset in [0, 1]:
            source_mask = (source_i % 2 == i_offset) & (source_j % 2 == j_offset)
            channel_source_coords = source_coords[source_mask]

            if channel_source_coords.shape[0] == 0:
                logger.warning("No valid source pixels found for Bayer channel.")
                continue

            tree = spatial.KDTree(channel_source_coords)

            query_mask = (query_i % 2 == i_offset) & (query_j % 2 == j_offset)
            channel_query_coords = query_coords[query_mask]

            if channel_query_coords.shape[0] == 0:
                continue

            dd, ii = tree.query(channel_query_coords, workers=-1)
            nearest_coords = tree.data[ii].astype(int)
            nearest_i[query_mask] = nearest_coords[:, 0]
            nearest_j[query_mask] = nearest_coords[:, 1]

    return nearest_i, nearest_j


def interpolate_white_pixels(photon_cube, sum_frames, cfa_mask, hot_pixel_mask):
    """
    Performs probabilistic inpainting of hot pixels in a photon cube.
    """
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    valid_pixel_mask = (~hot_pixel_mask.astype(bool)).astype(np.float32)
    neighbor_counts = ndimage.convolve(
        valid_pixel_mask, kernel, mode="constant", cval=0
    )
    masked_photon_cube = photon_cube * valid_pixel_mask[:, :, np.newaxis]
    neighbor_sums = ndimage.convolve(
        masked_photon_cube, kernel[..., np.newaxis], mode="constant", cval=0
    )
    neighbor_avg = np.divide(
        neighbor_sums,
        neighbor_counts[:, :, np.newaxis],
        out=np.zeros_like(neighbor_sums),
        where=neighbor_counts[:, :, np.newaxis] != 0,
    )
    prob_cube = np.clip(neighbor_avg / sum_frames, 0, 1)
    new_values = np.random.binomial(n=sum_frames, p=prob_cube)
    return np.where(cfa_mask[:, :, np.newaxis], new_values, photon_cube)


def colorSPAD_correct_func(photon_cube, sum_frames, colorSPAD_cfa_path, hot_pixel_mask):
    """
    Applies specific geometric corrections and inpainting for the 'Color SPAD' hardware prototype.
    """
    photon_cube_cropped = np.zeros(
        (254, 496, photon_cube.shape[-1]), dtype=photon_cube.dtype
    )
    photon_cube_cropped[:, :496] = photon_cube[2:, :496]
    photon_cube_cropped[:, 252:256] = photon_cube[2:, 260:264]
    photon_cube_cropped[:, 260:264] = photon_cube[2:, 252:256]
    if colorSPAD_cfa_path:
        cfa = cv2.imread(colorSPAD_cfa_path)[2:, :496, ::-1]
        cfa_mask = cfa.mean(axis=-1) < 255
        return interpolate_white_pixels(
            photon_cube_cropped.astype(int), sum_frames, cfa_mask, hot_pixel_mask
        )
    return photon_cube_cropped


def generate_photon_detection_probability_scale(
    photon_detection_prob_min: float,
    photon_detection_prob_max: float,
) -> float:
    return exp(
        random.uniform(log(photon_detection_prob_min), log(photon_detection_prob_max))
    )


def simulate_spad(
    video_cube: Float[Tensor, "... time"],
    spad_fps: int,
    video_fps: int,
    sum_frames: int,
    photon_detection_probability_scaling: float | tuple = 1.0,
) -> Tuple[
    Int[Tensor, "... summed_time"],
    Float[Tensor, "... summed_time"],
    float,
]:
    """
    Simulates SPAD sensor data from high-bitrate intensity video.
    """
    video_intensities = video_cube / 255.0
    oversampling_factor = spad_fps / video_fps
    frames_to_sum_over_video = sum_frames / oversampling_factor
    assert frames_to_sum_over_video.is_integer() and frames_to_sum_over_video >= 1
    frames_to_sum_over_video = int(frames_to_sum_over_video)

    if isinstance(photon_detection_probability_scaling, tuple):
        alpha = generate_photon_detection_probability_scale(
            *photon_detection_probability_scaling
        )
    else:
        alpha = photon_detection_probability_scaling

    prob = video_intensities * alpha
    simulated_frames = torch.binomial(
        count=torch.tensor(oversampling_factor, device=prob.device, dtype=prob.dtype),
        prob=prob,
    )
    summed_photon_cube = reduce(
        simulated_frames,
        "... (t m) -> ... t",
        "sum",
        m=frames_to_sum_over_video,
    )
    ground_truth = rearrange(prob, "... (t m) -> ... t m", m=frames_to_sum_over_video)[
        ..., -1
    ]
    return summed_photon_cube, ground_truth, alpha


class BaseCube(Dataset):
    """
    Base dataset class for managing 3D spatio-temporal data cubes.
    Supports rotations of 0, 90, 180, and 270 degrees.
    """

    def __init__(
        self,
        chunk_size: int,
        sum_frames: int,
        chunk_stride: int | None = None,
        initial_time_step: int = 0,
        temporal_stride: int = 1,
        rotation: int = 0,
        flip_lr=False,
        flip_ud=False,
        **kwargs,
    ):
        self.chunk_size = chunk_size
        self.sum_frames = sum_frames
        self.chunk_stride = chunk_stride or self.chunk_size
        self.initial_time_step = initial_time_step
        self.temporal_stride = temporal_stride

        if rotation not in [0, 90, 180, 270]:
            raise ValueError(f"Rotation must be 0, 90, 180, or 270. Got {rotation}")

        self.rotation = rotation
        self.k_rot = {0: 0, 90: 1, 180: 2, 270: 3}[rotation]
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self._h = self._w = self.t_out = self._num_chunks = self.num_time_step = 0

    def _initialize_timing(self, source_total_timesteps: int, num_time_step: int = -1):
        effective_source_t = source_total_timesteps - self.initial_time_step
        self.num_time_step = (
            effective_source_t
            if num_time_step == -1
            else min(num_time_step, effective_source_t)
        )
        chunk_source_span = (self.chunk_size - 1) * self.temporal_stride + 1
        if self.num_time_step >= chunk_source_span:
            self._num_chunks = (
                floor((self.num_time_step - chunk_source_span) / self.chunk_stride) + 1
            )
        else:
            self._num_chunks = 0

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._h, self._w, self.t_out

    @property
    def final_time_step(self) -> int:
        return self.initial_time_step + self.num_time_step

    @property
    def range_str(self) -> str:
        if self.temporal_stride > 1:
            return f"range({self.initial_time_step}, {self.final_time_step}, {self.temporal_stride})"
        return f"range({self.initial_time_step}, {self.final_time_step})"

    def _determine_dtype(self) -> torch.dtype:
        return torch.uint16 if self.sum_frames > 255 else torch.uint8

    def _apply_transforms(self, cube: torch.Tensor) -> torch.Tensor:
        if self.flip_lr:
            cube = torch.flip(cube, (1,))
        if self.flip_ud:
            cube = torch.flip(cube, (0,))
        if self.k_rot > 0:
            cube = torch.rot90(cube, k=self.k_rot, dims=(0, 1))
        return cube

    def __len__(self) -> int:
        return self._num_chunks


class RealCube(BaseCube):
    """
    Loads real SPAD data from NPY or H5 files.
    Handles Bayer pattern permutation and dimension swapping for rotations.
    """

    def __init__(
        self,
        file: str | Path,
        hot_pixel_mask_path: str | Path | None = None,
        colorSPAD_col_correct=False,
        colorSPAD_RGBW_CFA: str | Path | None = None,
        bayer_pattern: BayerPatternLiteral = "gray",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.file_path = Path(file)
        self.is_h5 = self.file_path.suffix.lower() == ".h5"
        self.hw_accumulation = 1

        if self.is_h5:
            self._h5_file = h5py.File(self.file_path, "r")
            self._group_path = self._find_data_group_path(self._h5_file)
            if not self._group_path:
                raise ValueError(f"No numeric dataset group found in {file}")

            meta = self.get_ubi_metadata(self.file_path)
            self.hw_accumulation = int(meta.get("sum_frames", 1))
            logger.info(f"H5 Hardware Accumulation detected: {self.hw_accumulation}")

            self._group = self._h5_file[self._group_path]

            all_keys = natsorted(self._group.keys())
            if len(all_keys) > 0:
                self._dataset_keys = all_keys[1:]
            else:
                self._dataset_keys = []
                logger.warning("H5 file contains no frames.")

            t = len(self._dataset_keys) * self.hw_accumulation

            if len(self._dataset_keys) > 0:
                first_frame = self._group[self._dataset_keys[0]][...].squeeze()
                self._source_h, self._source_w = first_frame.shape
            else:
                self._source_h, self._source_w = (0, 0)
        else:
            self._memmap_cube = np.load(file, mmap_mode="r")
            t, h, w_packed = self._memmap_cube.shape
            self._source_h, self._source_w = h, w_packed * 8

        self.colorSPAD_col_correct = colorSPAD_col_correct
        self.colorSPAD_RGBW_CFA = colorSPAD_RGBW_CFA

        self.bayer_pattern = bayer_pattern.lower()
        if self.bayer_pattern not in ["gray", "grey"]:
            self.bayer_pattern = self._transform_bayer_pattern(
                self.bayer_pattern, self.k_rot, self.flip_lr, self.flip_ud
            )
            logger.info(
                f"CFA Pattern adjusted (rot={self.rotation}): {self.bayer_pattern}"
            )

        self.is_bayer = self.bayer_pattern not in ["gray", "grey"]
        self.t_out = self.chunk_size // self.sum_frames
        self._initialize_timing(t, kwargs.get("num_time_step", -1))
        self.dtype = self._determine_dtype()

        if hot_pixel_mask_path:
            self.hot_pixel_mask = np.load(hot_pixel_mask_path).astype(np.uint8)
            query_i, query_j = np.where(self.hot_pixel_mask.astype(bool))
            if self.is_bayer:
                nearest_i, nearest_j = bayer_nearest_neighbor_indices(
                    self.hot_pixel_mask
                )
            else:
                nearest_i, nearest_j = nearest_neighbor_indices(self.hot_pixel_mask)
            self.hp_query = (
                torch.from_numpy(query_i).long(),
                torch.from_numpy(query_j).long(),
            )
            self.hp_nearest = (
                torch.from_numpy(nearest_i).long(),
                torch.from_numpy(nearest_j).long(),
            )
        else:
            self.hot_pixel_mask = None

        raw_h, raw_w = (
            (254, 496)
            if self.colorSPAD_col_correct
            else (self._source_h, self._source_w)
        )

        if self.k_rot % 2 != 0:
            self._h, self._w = raw_w, raw_h
        else:
            self._h, self._w = raw_h, raw_w

    def _transform_bayer_pattern(self, p, k, flr, fud):
        """
        Adjusts Bayer pattern string based on Flips and 90-degree CCW Rotations.
        Input Grid Indices: 0(TL) 1(TR) / 2(BL) 3(BR).
        """
        p = list(p.upper())

        if flr:
            p[0], p[1], p[2], p[3] = p[1], p[0], p[3], p[2]
        if fud:
            p[0], p[2], p[1], p[3] = p[2], p[0], p[3], p[1]

        old_p = p[:]
        if k == 1:
            p = [old_p[1], old_p[3], old_p[0], old_p[2]]
        elif k == 2:
            p = [old_p[3], old_p[2], old_p[1], old_p[0]]
        elif k == 3:
            p = [old_p[2], old_p[0], old_p[3], old_p[1]]

        return "".join(p).lower()

    def _find_data_group_path(self, f):
        found_path = None

        def visitor(name, obj):
            nonlocal found_path
            if found_path:
                return
            if isinstance(obj, h5py.Dataset) and Path(name).name.isdigit():
                found_path = str(Path(name).parent)

        f.visititems(visitor)
        return found_path

    @staticmethod
    def get_ubi_metadata(file_path: str | Path) -> Dict[str, Any]:
        overrides = {}
        with h5py.File(file_path, "r") as f:
            found_ds = None

            def visitor(name, obj):
                nonlocal found_ds
                if found_ds:
                    return
                if isinstance(obj, h5py.Dataset) and Path(name).name.isdigit():
                    found_ds = name

            f.visititems(visitor)
            if found_ds:
                raw_meta = f[found_ds].attrs.get("metadata")
                if raw_meta:
                    meta = json.loads(
                        raw_meta.decode("utf-8")
                        if isinstance(raw_meta, bytes)
                        else raw_meta
                    )
                    overrides["bayer_pattern"] = meta.get("cfa_pattern", "gray").lower()
                    poly_meta = meta.get("polycam_metadata", {})
                    overrides["sum_frames"] = poly_meta.get(
                        "num_integrated_binary_frames",
                        meta.get("num_integrated_binary_frames", None),
                    )
                    hist = meta.get("node_history", [])
                    if hist and isinstance(hist, list) and len(hist) > 0:
                        sched = hist[0].get("schedule", {}).get("base_schedule", {})
                        if "binary_frame_rate_fps" in sched:
                            overrides["binary_fps"] = sched["binary_frame_rate_fps"]
        return overrides

    def __getitem__(self, index: int) -> Int[Tensor, "height width time_out"]:
        start_idx = self.initial_time_step + index * self.chunk_stride
        end_idx = start_idx + self.chunk_size * self.temporal_stride

        if self.is_h5:
            h5_reduce = max(1, self.sum_frames // self.hw_accumulation)
            start_h5 = start_idx // self.hw_accumulation
            num_h5_frames_needed = self.t_out * h5_reduce

            frames = []
            for i in range(num_h5_frames_needed):
                idx = start_h5 + i
                if idx < len(self._dataset_keys):
                    data = self._group[self._dataset_keys[idx]][...].squeeze()
                else:
                    data = self._group[self._dataset_keys[-1]][...].squeeze()
                frames.append(data)

            photon_cube_raw = np.stack(frames, axis=-1)
            photon_cube = reduce(
                photon_cube_raw, "h w (t s) -> h w t", "sum", s=h5_reduce
            )
        else:
            raw_chunk = self._memmap_cube[start_idx : end_idx : self.temporal_stride]
            binary_cube = rearrange(np.unpackbits(raw_chunk, axis=-1), "t h w -> h w t")
            photon_cube = reduce(
                binary_cube, "h w (t s) -> h w t", "sum", s=self.sum_frames
            )

        if self.colorSPAD_col_correct:
            photon_cube = colorSPAD_correct_func(
                photon_cube,
                self.sum_frames,
                self.colorSPAD_RGBW_CFA,
                self.hot_pixel_mask,
            )

        photon_cube = torch.from_numpy(photon_cube.astype(np.int64))

        if self.hot_pixel_mask is not None:
            photon_cube[self.hp_query] = photon_cube[self.hp_nearest]

        return self._apply_transforms(photon_cube).to(self.dtype)


class SimulatedCube(BaseCube):
    """
    Generates synthetic SPAD data. Adjusts output dimensions for all rotations.
    """

    def __init__(
        self,
        file: str | Path,
        spad_fps: int,
        video_fps: int,
        bayer_pattern: BayerPatternLiteral = "gray",
        photon_detection_probability_scaling: float | tuple = 1.0,
        return_gt: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.file = Path(file)
        self.spad_fps = spad_fps
        self.video_fps = video_fps
        self.bayer_pattern = bayer_pattern.lower()
        self.photon_detection_probability_scaling = photon_detection_probability_scaling
        self.return_gt = return_gt

        self.oversampling_factor = spad_fps / video_fps
        frames_to_sum_over_video = self.sum_frames / self.oversampling_factor
        assert frames_to_sum_over_video.is_integer() and frames_to_sum_over_video >= 1

        if isinstance(self.photon_detection_probability_scaling, tuple):
            self.alpha = generate_photon_detection_probability_scale(
                *self.photon_detection_probability_scaling
            )
        else:
            self.alpha = self.photon_detection_probability_scaling

        self._data_source = None
        vr = video_reader.PyVideoReader(str(self.file))
        _, vr_h, vr_w = vr.get_shape()

        self._source_h, self._source_w = vr_h, vr_w

        self.raw_t_out = self.chunk_size // self.sum_frames
        self.t_out = ceil(self.raw_t_out / self.temporal_stride)
        self.video_chunk_len = int(self.raw_t_out * frames_to_sum_over_video)

        self._initialize_timing(
            int(len(vr) * self.oversampling_factor), kwargs.get("num_time_step", -1)
        )
        del vr

        self.dtype = self._determine_dtype()
        self.is_bayer = self.bayer_pattern not in ["gray", "grey"]

        if self.is_bayer:
            self.bayer_mask_r = torch.zeros(
                (self._source_h, self._source_w), dtype=torch.bool
            )
            self.bayer_mask_g = torch.zeros(
                (self._source_h, self._source_w), dtype=torch.bool
            )
            self.bayer_mask_b = torch.zeros(
                (self._source_h, self._source_w), dtype=torch.bool
            )

            if self.bayer_pattern == "rggb":
                self.bayer_mask_r[::2, ::2] = self.bayer_mask_g[::2, 1::2] = (
                    self.bayer_mask_g[1::2, ::2]
                ) = self.bayer_mask_b[1::2, 1::2] = True
            elif self.bayer_pattern == "bggr":
                self.bayer_mask_b[::2, ::2] = self.bayer_mask_g[::2, 1::2] = (
                    self.bayer_mask_g[1::2, ::2]
                ) = self.bayer_mask_r[1::2, 1::2] = True
            elif self.bayer_pattern == "grbg":
                self.bayer_mask_g[::2, ::2] = self.bayer_mask_r[::2, 1::2] = (
                    self.bayer_mask_b[1::2, ::2]
                ) = self.bayer_mask_g[1::2, 1::2] = True
            elif self.bayer_pattern == "gbrg":
                self.bayer_mask_g[::2, ::2] = self.bayer_mask_b[::2, 1::2] = (
                    self.bayer_mask_r[1::2, ::2]
                ) = self.bayer_mask_g[1::2, 1::2] = True
        else:
            self.grayscale_weights = torch.tensor([0.299, 0.587, 0.114])

        if self.k_rot % 2 != 0:
            self._h, self._w = self._source_w, self._source_h
        else:
            self._h, self._w = self._source_h, self._source_w

    def __getitem__(self, index: int):
        if self._data_source is None:
            self._data_source = video_reader.PyVideoReader(str(self.file))

        start_spad_frame = self.initial_time_step + index * self.chunk_stride
        start_video_frame = int(start_spad_frame // self.oversampling_factor)

        vr_indices = range(start_video_frame, start_video_frame + self.video_chunk_len)
        video_chunk_np = self._data_source[vr_indices.start : vr_indices.stop]

        if video_chunk_np.shape[0] < self.video_chunk_len:
            video_chunk_np = np.concatenate(
                [
                    video_chunk_np,
                    np.repeat(
                        video_chunk_np[-1:],
                        self.video_chunk_len - video_chunk_np.shape[0],
                        axis=0,
                    ),
                ],
                axis=0,
            )

        video_chunk = torch.from_numpy(video_chunk_np)

        if self.is_bayer:
            intensity_chunk = torch.empty(
                (self._source_h, self._source_w, video_chunk.shape[0]),
                device=video_chunk.device,
                dtype=torch.float,
            )
            # T H W C -> H W T C
            v_pt = rearrange(video_chunk.float(), "t h w c -> h w t c")
            intensity_chunk[self.bayer_mask_r] = v_pt[self.bayer_mask_r][..., 0]
            intensity_chunk[self.bayer_mask_g] = v_pt[self.bayer_mask_g][..., 1]
            intensity_chunk[self.bayer_mask_b] = v_pt[self.bayer_mask_b][..., 2]
        else:
            intensity_chunk = torch.einsum(
                "t h w c, c -> h w t",
                video_chunk.float(),
                self.grayscale_weights.to(video_chunk.device),
            )

        intensity_chunk = self._apply_transforms(intensity_chunk)

        summed_photon_cube, ground_truth, used_alpha = simulate_spad(
            intensity_chunk, self.spad_fps, self.video_fps, self.sum_frames, self.alpha
        )

        if self.return_gt and self.is_bayer:
            # T H W C -> H W C T for HWCT RGB GT
            rgb_chunk = rearrange(video_chunk.float(), "t h w c -> h w c t")
            rgb_chunk = self._apply_transforms(rgb_chunk)

            # Generate RGB GT using same alpha
            _, ground_truth_rgb, _ = simulate_spad(
                rgb_chunk,
                self.spad_fps,
                self.video_fps,
                self.sum_frames,
                photon_detection_probability_scaling=used_alpha,
            )
            ground_truth = ground_truth_rgb

        if self.temporal_stride > 1:
            summed_photon_cube = summed_photon_cube[..., :: self.temporal_stride]
            if ground_truth is not None:
                ground_truth = ground_truth[..., :: self.temporal_stride]

        return (
            (summed_photon_cube.to(self.dtype), ground_truth, used_alpha)
            if self.return_gt
            else summed_photon_cube.to(self.dtype)
        )
