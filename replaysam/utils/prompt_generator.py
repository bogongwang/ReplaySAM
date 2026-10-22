from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

import numpy as np
import dask.array as da
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import tqdm

from replaysam.utils.helpers import Crop3D


LOGGER = logging.getLogger(__name__)
PromptGenerationMethod = Literal["peak_local_max", "blockwise_argmax", "gradient_ascent"]
CandidateFinder = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class PromptGenerator:
    """Generate globally indexed particle prompts from a 3D tomogram.

    Input volumes may be NumPy arrays, Dask arrays, or PyTorch tensors.  All
    public coordinates are in ``(z, y, x)`` order.  Crops are evaluated one at
    a time and their candidate coordinates are translated back into the full
    input-volume coordinate system.
    """

    def __init__(self, tomo: np.ndarray | da.Array | torch.Tensor):
        supported = isinstance(tomo, (np.ndarray, da.Array, torch.Tensor))
        if not supported:
            raise TypeError(
                "PromptGenerator requires a numpy.ndarray, dask.array.Array, "
                f"or torch.Tensor, got {type(tomo)!r}."
            )
        if tomo.ndim != 3:
            raise ValueError(
                f"Expected a 3D tomogram in (Z, Y, X) order, got {tomo.shape}."
            )
        if any(int(size) <= 0 for size in tomo.shape):
            raise ValueError(f"Tomogram dimensions must be positive, got {tomo.shape}.")
        self.tomo = tomo

    def generate_prompts(
        self,
        *,
        method: PromptGenerationMethod = "peak_local_max",
        crop_size: tuple[int, int, int] | None = None,
        crop_overlap: tuple[int, int, int] | None = None,
        bin_thresh: int | float | None = None,
        min_peak_distance: float = 5.0,
        process_device: torch.device | str = torch.device("cpu"),
        show_progress: bool = False,
        return_peak_values: bool = False,
        downsample: bool = False,
        grid_shape: tuple[int, int, int] | None = None,
        dist_map_use_n_threads: int = 0,
        optimizer: str = "sgd",
        lr: float = 0.1,
        iterations: int = 2000,
        block_size: int = 12,
        triton_num_warps: int = 4,
        max_filter_size: int = 7,
    ):
        """Generate prompts with the selected candidate-finding algorithm.

        ``blockwise_argmax`` uses a Triton kernel on CUDA and a deterministic
        PyTorch block reduction on CPU.  The CPU fallback is intentional: it
        keeps the API usable for validation and small volumes while preserving
        the same blockwise selection semantics.
        """
        if method == "peak_local_max":
            return self.generate_peak_local_max_prompts(
                crop_size=crop_size,
                crop_overlap=crop_overlap,
                bin_thresh=bin_thresh,
                dist_val_thresh=min_peak_distance,
                max_filter_size=max_filter_size,
                show_progress=show_progress,
                return_max_values=return_peak_values,
            )
        if method == "blockwise_argmax":
            return self.generate_prompts_blockwise_argmax(
                crop_size=crop_size,
                crop_overlap=crop_overlap,
                bin_thresh=bin_thresh,
                min_peak_distance=min_peak_distance,
                block_size=block_size,
                process_device=process_device,
                triton_num_warps=triton_num_warps,
                show_progress=show_progress,
                return_peak_values=return_peak_values,
            )
        if method == "gradient_ascent":
            return self.generate_prompts_gradient_ascent(
                crop_size=crop_size,
                crop_overlap=crop_overlap,
                bin_thresh=bin_thresh,
                min_peak_distance=min_peak_distance,
                process_device=process_device,
                downsample=downsample,
                grid_shape=grid_shape,
                dist_map_use_n_threads=dist_map_use_n_threads,
                optimizer=optimizer,
                lr=lr,
                iterations=iterations,
                show_progress=show_progress,
                return_peak_values=return_peak_values,
            )
        raise ValueError(
            f"Unknown prompt generation method {method!r}; expected "
            "'peak_local_max', 'blockwise_argmax', or 'gradient_ascent'."
        )

    def generate_peak_local_max_prompts(
        self,
        crop_size: tuple[int, int, int] | None,
        crop_overlap: tuple[int, int, int] | None,
        bin_thresh: int | float | None = None,
        dist_val_thresh: float = 5.0,
        max_filter_size: int = 7,
        show_progress: bool = False,
        return_max_values: bool = False,
    ):
        """Generate prompts from local distance-transform maxima."""
        if max_filter_size <= 0:
            raise ValueError(f"max_filter_size must be positive, got {max_filter_size}.")

        def find_candidates(binary_mask: np.ndarray):
            return self._local_maximums_cupy(
                binary_mask,
                dist_threshold=dist_val_thresh,
                max_filter_size=max_filter_size,
            )

        return self._generate_prompts_over_crops(
            crop_size=crop_size,
            crop_overlap=crop_overlap,
            bin_thresh=bin_thresh,
            min_peak_distance=dist_val_thresh,
            show_progress=show_progress,
            return_peak_values=return_max_values,
            progress_description="Generating prompts from local maxima",
            candidate_finder=find_candidates,
        )

    def generate_prompts_blockwise_argmax(
        self,
        *,
        crop_size: tuple[int, int, int] | None = None,
        crop_overlap: tuple[int, int, int] | None = None,
        bin_thresh: int | float | None = None,
        min_peak_distance: float = 5.0,
        block_size: int = 12,
        process_device: torch.device | str = torch.device("cpu"),
        triton_num_warps: int = 4,
        show_progress: bool = False,
        return_peak_values: bool = False,
    ):
        """Generate one distance-map maximum per non-overlapping 3D block."""
        process_device = self._usable_device(process_device)

        def find_candidates(binary_mask: np.ndarray):
            return self._find_candidates_blockwise_argmax(
                binary_mask,
                min_peak_distance=min_peak_distance,
                block_size=block_size,
                process_device=process_device,
                triton_num_warps=triton_num_warps,
            )

        return self._generate_prompts_over_crops(
            crop_size=crop_size,
            crop_overlap=crop_overlap,
            bin_thresh=bin_thresh,
            min_peak_distance=min_peak_distance,
            show_progress=show_progress,
            return_peak_values=return_peak_values,
            progress_description="Generating prompts with Triton blockwise argmax",
            candidate_finder=find_candidates,
        )

    def generate_prompts_gradient_ascent(
        self,
        *,
        crop_size: tuple[int, int, int] | None = None,
        crop_overlap: tuple[int, int, int] | None = None,
        bin_thresh: int | float | None = None,
        min_peak_distance: float = 5.0,
        process_device: torch.device | str = torch.device("cpu"),
        downsample: bool = False,
        grid_shape: tuple[int, int, int] | None = None,
        dist_map_use_n_threads: int = 0,
        optimizer: str = "sgd",
        lr: float = 0.1,
        iterations: int = 2000,
        show_progress: bool = False,
        return_peak_values: bool = False,
    ):
        """Generate prompts by ascending a distance map from grid seeds."""
        process_device = self._usable_device(process_device)

        def find_candidates(binary_mask: np.ndarray):
            return self._find_candidates_gradient_ascent(
                binary_mask,
                downsample=downsample,
                grid_shape=grid_shape,
                process_device=process_device,
                dist_map_use_n_threads=dist_map_use_n_threads,
                optimizer=optimizer,
                lr=lr,
                iterations=iterations,
                min_peak_distance=min_peak_distance,
                show_progress=show_progress,
            )

        return self._generate_prompts_over_crops(
            crop_size=crop_size,
            crop_overlap=crop_overlap,
            bin_thresh=bin_thresh,
            min_peak_distance=min_peak_distance,
            show_progress=show_progress,
            return_peak_values=return_peak_values,
            progress_description="Generating prompts with gradient ascent",
            candidate_finder=find_candidates,
        )

    # ------------------------------------------------------------------
    # Shared crop orchestration
    # ------------------------------------------------------------------

    def _generate_prompts_over_crops(
        self,
        *,
        crop_size: tuple[int, int, int] | None,
        crop_overlap: tuple[int, int, int] | None,
        bin_thresh: int | float | None,
        min_peak_distance: float,
        show_progress: bool,
        return_peak_values: bool,
        progress_description: str,
        candidate_finder: CandidateFinder,
    ):
        if min_peak_distance < 0:
            raise ValueError("min_peak_distance must be non-negative.")

        crops = self.generate_crops(
            image_size=tuple(int(size) for size in self.tomo.shape),
            crop_size=crop_size,
            overlap=crop_overlap,
        )
        global_coords: list[np.ndarray] = []
        global_peak_values: list[np.ndarray] = []

        for crop in tqdm(crops, desc=progress_description, disable=not show_progress):
            crop_tomo = self._crop_to_numpy(crop)
            threshold = (
                float(threshold_otsu(crop_tomo))
                if bin_thresh is None and np.ptp(crop_tomo) > 0
                else float(np.min(crop_tomo) if bin_thresh is None else bin_thresh)
            )
            binary_mask = np.ascontiguousarray(crop_tomo > threshold)
            crop_coords, crop_peak_values = candidate_finder(binary_mask)

            crop_coords = np.asarray(crop_coords, dtype=np.int64).reshape(-1, 3)
            crop_peak_values = np.asarray(crop_peak_values, dtype=np.float32).reshape(-1)
            if len(crop_coords) != len(crop_peak_values):
                raise ValueError(
                    "Candidate finder returned different numbers of coordinates "
                    "and peak values."
                )
            if len(crop_coords) == 0:
                continue

            global_coords.append(crop_coords + np.asarray(crop.starts, dtype=np.int64))
            global_peak_values.append(crop_peak_values)

        return self._finalize_global_prompts(
            coord_batches=global_coords,
            value_batches=global_peak_values,
            min_peak_distance=min_peak_distance,
            return_peak_values=return_peak_values,
        )

    def _crop_to_numpy(self, crop: Crop3D) -> np.ndarray:
        crop_tomo = self.tomo[crop.slices]
        if isinstance(crop_tomo, da.Array):
            crop_tomo = crop_tomo.compute()
        elif isinstance(crop_tomo, torch.Tensor):
            crop_tomo = crop_tomo.detach().cpu().numpy()
        return np.asarray(crop_tomo)

    @staticmethod
    def _finalize_global_prompts(
        coord_batches: list[np.ndarray],
        value_batches: list[np.ndarray],
        min_peak_distance: float,
        return_peak_values: bool,
    ):
        if not coord_batches:
            empty_coords = np.empty((0, 3), dtype=np.int64)
            empty_values = np.empty((0,), dtype=np.float32)
            return (empty_coords, empty_values) if return_peak_values else empty_coords

        coords = np.concatenate(coord_batches, axis=0)
        values = np.concatenate(value_batches, axis=0)
        coords, values = PromptGenerator._suppress_close_points(
            coords,
            values,
            min_peak_distance=min_peak_distance,
        )
        if return_peak_values:
            return coords, values
        return coords

    # ------------------------------------------------------------------
    # Candidate detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _local_maximums_cupy(
        binary_image: np.ndarray,
        dist_threshold: float = 3.0,
        max_filter_size: int = 7,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return local distance-transform maxima, using CuPy when available."""
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import distance_transform_edt as cp_distance_transform_edt
            from cupyx.scipy.ndimage import maximum_filter as cp_maximum_filter

            bin_sl = cp.asarray(binary_image).astype(bool)
            dist_sl = cp_distance_transform_edt(bin_sl).astype(cp.float32)
            maximums = cp_maximum_filter(
                dist_sl,
                size=(max_filter_size,) * dist_sl.ndim,
                mode="nearest",
            )
            peaks = (maximums == dist_sl) & bin_sl & (dist_sl >= dist_threshold)
            peak_coords = cp.asnumpy(cp.argwhere(peaks))
            peak_values = cp.asnumpy(dist_sl[peaks])
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            return peak_coords, peak_values
        except Exception as exc:
            LOGGER.debug("Using CPU local maxima because CuPy failed: %s", exc)

        bin_sl = np.asarray(binary_image, dtype=bool)
        dist_sl = distance_transform_edt(bin_sl).astype(np.float32)
        maximums = maximum_filter(
            dist_sl,
            size=(max_filter_size,) * dist_sl.ndim,
            mode="nearest",
        )
        peaks = (maximums == dist_sl) & bin_sl & (dist_sl >= dist_threshold)
        return np.argwhere(peaks), dist_sl[peaks]

    @staticmethod
    def _compute_distance_map(
        binary_mask: np.ndarray,
        *,
        process_device: torch.device,
        dtype: torch.dtype = torch.float32,
        cpu_n_threads: int = 0,
    ) -> torch.Tensor:
        """Compute a distance map on CUDA when CuPy is available, else CPU."""
        binary_mask = np.ascontiguousarray(binary_mask, dtype=bool)
        if process_device.type == "cuda" and torch.cuda.is_available():
            try:
                import cupy as cp
                from cupyx.scipy.ndimage import distance_transform_edt as cp_distance_transform_edt

                device_index = process_device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                with cp.cuda.Device(device_index):
                    distance_map_cp = cp_distance_transform_edt(
                        cp.asarray(binary_mask, dtype=cp.uint8),
                        float64_distances=False,
                    )
                    distance_map = torch.as_tensor(
                        cp.asnumpy(distance_map_cp),
                        dtype=dtype,
                        device=process_device,
                    )
                    cp.cuda.Device(device_index).synchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    return distance_map
            except Exception as exc:
                LOGGER.warning("Falling back to CPU distance transform: %s", exc)

        # edt is faster than scipy for many CPU volumes, but scipy is already a
        # required dependency and is a reliable fallback for all supported dtypes.
        distance_map_np = distance_transform_edt(binary_mask).astype(np.float32)
        return torch.as_tensor(distance_map_np, dtype=dtype, device=process_device)

    @staticmethod
    def _compute_distance_map_torch(
        binary_mask: np.ndarray,
        process_device: torch.device | str,
        *,
        cpu_n_threads: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compatibility wrapper for the full implementation's helper name."""
        del cpu_n_threads  # scipy's EDT backend does not expose a thread knob.
        return PromptGenerator._compute_distance_map(
            binary_mask,
            process_device=PromptGenerator._usable_device(process_device),
            dtype=dtype,
        )

    @staticmethod
    def _find_candidates_blockwise_argmax(
        binary_mask: np.ndarray,
        *,
        min_peak_distance: float,
        block_size: int,
        process_device: torch.device,
        triton_num_warps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")
        if triton_num_warps <= 0:
            raise ValueError(f"triton_num_warps must be positive, got {triton_num_warps}.")

        distance_map = PromptGenerator._compute_distance_map(
            binary_mask,
            process_device=process_device,
            dtype=torch.float16 if process_device.type == "cuda" else torch.float32,
        ).contiguous()
        if distance_map.is_cuda:
            peak_coords, peak_values = PromptGenerator._argmax_3d_blocks_triton(
                distance_map,
                block_size=block_size,
                num_warps=triton_num_warps,
            )
        else:
            peak_coords, peak_values = PromptGenerator._argmax_3d_blocks_torch(
                distance_map,
                block_size=block_size,
            )

        # A zero-valued distance maximum is background, even when callers use
        # ``min_peak_distance=0`` to disable radius filtering.
        keep = peak_values > float(min_peak_distance)
        peak_coords_np = peak_coords[keep].detach().cpu().numpy().astype(np.int64)
        peak_values_np = peak_values[keep].detach().to(torch.float32).cpu().numpy()
        del distance_map, peak_coords, peak_values, keep
        if process_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return peak_coords_np, peak_values_np

    @staticmethod
    def _find_candidates_blockwise_argmax_triton(
        binary_mask: np.ndarray,
        *,
        min_peak_distance: float,
        block_size: int,
        process_device: torch.device,
        triton_num_warps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility wrapper retaining the full implementation's name."""
        return PromptGenerator._find_candidates_blockwise_argmax(
            binary_mask,
            min_peak_distance=min_peak_distance,
            block_size=block_size,
            process_device=PromptGenerator._usable_device(process_device),
            triton_num_warps=triton_num_warps,
        )

    @staticmethod
    def _argmax_3d_blocks_torch(
        distance_map: torch.Tensor,
        *,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference blockwise argmax used on CPU and as a Triton fallback."""
        if distance_map.ndim != 3:
            raise ValueError(f"distance_map must be 3D, got {tuple(distance_map.shape)}.")
        depth, height, width = (int(size) for size in distance_map.shape)
        coords: list[list[int]] = []
        values: list[torch.Tensor] = []
        for z0 in range(0, depth, block_size):
            for y0 in range(0, height, block_size):
                for x0 in range(0, width, block_size):
                    block = distance_map[
                        z0:min(z0 + block_size, depth),
                        y0:min(y0 + block_size, height),
                        x0:min(x0 + block_size, width),
                    ]
                    flat_idx = torch.argmax(block.reshape(-1))
                    local_coord = np.unravel_index(
                        int(flat_idx.detach().cpu()),
                        tuple(int(size) for size in block.shape),
                    )
                    coords.append([
                        z0 + int(local_coord[0]),
                        y0 + int(local_coord[1]),
                        x0 + int(local_coord[2]),
                    ])
                    values.append(block.reshape(-1)[flat_idx])
        return (
            torch.as_tensor(coords, dtype=torch.int64, device=distance_map.device),
            torch.stack(values).to(device=distance_map.device),
        )

    @staticmethod
    @triton.jit
    def _argmax_3d_blocks_triton_kernel(
        x_ptr,
        coords_ptr,
        values_ptr,
        depth: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        block_size: tl.constexpr,
        padded_block_size: tl.constexpr,
        block_elements: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        num_blocks_width = tl.cdiv(width, block_size)
        num_blocks_height = tl.cdiv(height, block_size)
        block_z = program_id // (num_blocks_height * num_blocks_width)
        remainder = program_id % (num_blocks_height * num_blocks_width)
        block_y = remainder // num_blocks_width
        block_x = remainder % num_blocks_width
        start_z = block_z * block_size
        start_y = block_y * block_size
        start_x = block_x * block_size

        offsets = tl.arange(0, block_elements)
        offset_z = offsets // (padded_block_size * padded_block_size)
        offset_y = (offsets // padded_block_size) % padded_block_size
        offset_x = offsets % padded_block_size
        z = start_z + offset_z
        y = start_y + offset_y
        x = start_x + offset_x
        valid = (
            (offset_z < block_size)
            & (offset_y < block_size)
            & (offset_x < block_size)
            & (z < depth)
            & (y < height)
            & (x < width)
        )
        linear_offsets = z * height * width + y * width + x
        values = tl.load(x_ptr + linear_offsets, mask=valid, other=-float("inf"))
        max_value = tl.max(values, axis=0)
        local_argmax = tl.min(
            tl.where(valid & (values == max_value), offsets, block_elements),
            axis=0,
        )
        max_offset_z = local_argmax // (padded_block_size * padded_block_size)
        max_offset_y = (local_argmax // padded_block_size) % padded_block_size
        max_offset_x = local_argmax % padded_block_size
        tl.store(coords_ptr + program_id * 3, start_z + max_offset_z)
        tl.store(coords_ptr + program_id * 3 + 1, start_y + max_offset_y)
        tl.store(coords_ptr + program_id * 3 + 2, start_x + max_offset_x)
        tl.store(values_ptr + program_id, max_value)

    @staticmethod
    def _argmax_3d_blocks_triton(
        distance_map: torch.Tensor,
        *,
        block_size: int = 12,
        num_warps: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one maximum coordinate and value per Triton block."""
        if not distance_map.is_cuda:
            raise ValueError("distance_map must be a CUDA tensor for Triton argmax.")
        if distance_map.ndim != 3 or not distance_map.is_contiguous():
            raise ValueError("distance_map must be a contiguous tensor with shape (D, H, W).")
        if not distance_map.is_floating_point():
            raise ValueError("distance_map must be floating point.")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")

        depth, height, width = (int(size) for size in distance_map.shape)
        num_blocks_width = triton.cdiv(width, block_size)
        num_blocks_height = triton.cdiv(height, block_size)
        num_blocks = triton.cdiv(depth, block_size) * num_blocks_height * num_blocks_width
        padded_block_size = triton.next_power_of_2(block_size)
        block_elements = padded_block_size**3
        if block_elements > 1_048_576:
            raise ValueError(f"block_size is too large for one Triton program: {block_size}.")

        coords = torch.empty((num_blocks, 3), device=distance_map.device, dtype=torch.int64)
        values = torch.empty((num_blocks,), device=distance_map.device, dtype=distance_map.dtype)
        PromptGenerator._argmax_3d_blocks_triton_kernel[(num_blocks,)](
            distance_map,
            coords,
            values,
            depth,
            height,
            width,
            block_size,
            padded_block_size,
            block_elements,
            num_warps=num_warps,
        )
        return coords, values

    @staticmethod
    def _find_candidates_gradient_ascent(
        binary_mask: np.ndarray,
        *,
        downsample: bool,
        grid_shape: tuple[int, int, int] | None,
        process_device: torch.device,
        dist_map_use_n_threads: int,
        optimizer: str,
        lr: float,
        iterations: int,
        min_peak_distance: float,
        show_progress: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find distance-map peaks using PyTorch gradient ascent."""
        if iterations < 0:
            raise ValueError(f"iterations must be non-negative, got {iterations}.")
        original_shape = tuple(int(size) for size in binary_mask.shape)
        processing_mask = binary_mask
        if downsample:
            scaled_shape = (1024, 1024, 1024)
            mask_tensor = torch.from_numpy(binary_mask)[None, None].float()
            processing_mask = (
                F.interpolate(mask_tensor, size=scaled_shape, mode="nearest")[0, 0]
                .to(torch.bool)
                .numpy()
            )

        distance_map = PromptGenerator._compute_distance_map(
            processing_mask,
            process_device=process_device,
            dtype=torch.float32,
            cpu_n_threads=dist_map_use_n_threads,
        )
        if grid_shape is None:
            grid_shape = tuple(max(1, size // 10) for size in distance_map.shape)
        start_coords = PromptGenerator._get_grid_coords(
            tuple(int(size) for size in distance_map.shape),
            grid_shape,
            process_device,
        )
        max_coords = PromptGenerator._gradient_ascend(
            distance_map,
            start_coords,
            optimizer=optimizer,
            lr=lr,
            iterations=iterations,
            show_progress=show_progress,
        )
        peak_values = PromptGenerator._sample_peak_distances(distance_map, max_coords)
        coords_np = max_coords.cpu().numpy()
        values_np = peak_values.cpu().numpy()
        coords_np, values_np = PromptGenerator._suppress_close_points(
            coords_np,
            values_np,
            min_peak_distance=min_peak_distance,
        )
        if downsample:
            coords_np = PromptGenerator._restore_coords(
                coords_np,
                original_shape=original_shape,
                scaled_shape=tuple(int(size) for size in processing_mask.shape),
            )
        return coords_np, values_np

    @staticmethod
    def _get_grid_coords(
        image_shape: tuple[int, int, int],
        grid_shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if len(grid_shape) != 3 or any(int(size) <= 0 for size in grid_shape):
            raise ValueError(f"grid_shape must contain three positive values, got {grid_shape}.")
        axes = [
            torch.linspace(0, size - 1, steps=count, device=device)
            for size, count in zip(image_shape, grid_shape, strict=True)
        ]
        grid = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(grid, dim=-1).reshape(-1, 3)

    @staticmethod
    def _gradient_ascend(
        distance_map: torch.Tensor,
        start_coords: torch.Tensor,
        *,
        optimizer: str,
        lr: float,
        iterations: int,
        show_progress: bool,
    ) -> torch.Tensor:
        depth, height, width = (int(size) for size in distance_map.shape)
        image = distance_map.float()[None, None]
        coords = start_coords.float().clone().detach().requires_grad_(True)
        if optimizer == "sgd":
            torch_optimizer = torch.optim.SGD([coords], lr=lr, momentum=0.9)
        elif optimizer == "adam":
            torch_optimizer = torch.optim.Adam([coords], lr=lr, weight_decay=1e-2)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer!r}.")

        for _ in tqdm(range(iterations), desc="Gradient ascending", disable=not show_progress):
            torch_optimizer.zero_grad()
            z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
            norm_z = 2.0 * z / max(depth - 1, 1) - 1.0
            norm_y = 2.0 * y / max(height - 1, 1) - 1.0
            norm_x = 2.0 * x / max(width - 1, 1) - 1.0
            grid = torch.stack([norm_x, norm_y, norm_z], dim=-1).view(1, -1, 1, 1, 3)
            sampled_values = F.grid_sample(
                image,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            loss = -sampled_values.sum()
            loss.backward()
            torch_optimizer.step()
            with torch.no_grad():
                coords[:, 0].clamp_(0, depth - 1)
                coords[:, 1].clamp_(0, height - 1)
                coords[:, 2].clamp_(0, width - 1)

        return torch.round(coords).to(torch.int64).detach()

    @staticmethod
    def _sample_peak_distances(distance_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        indices = coords.to(device=distance_map.device, dtype=torch.int64).clone()
        for axis, size in enumerate(distance_map.shape):
            indices[:, axis].clamp_(0, int(size) - 1)
        return distance_map[indices[:, 0], indices[:, 1], indices[:, 2]].float()

    @staticmethod
    def _suppress_close_points(
        coords: np.ndarray,
        values: np.ndarray,
        value_threshold: float | None = None,
        *,
        min_peak_distance: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Threshold candidates and greedily suppress nearby lower peaks."""
        threshold = min_peak_distance if min_peak_distance is not None else value_threshold
        threshold = 0.0 if threshold is None else float(threshold)
        coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(coords) != len(values):
            raise ValueError("coords and values must contain the same number of entries.")
        keep_threshold = values >= threshold
        if not np.any(keep_threshold):
            return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.float32)

        coords = coords[keep_threshold]
        values = values[keep_threshold]
        order = np.argsort(values)[::-1]
        tree = cKDTree(coords)
        suppressed = np.zeros(len(coords), dtype=bool)
        kept: list[int] = []
        for index in order:
            if suppressed[index]:
                continue
            kept.append(int(index))
            neighbours = tree.query_ball_point(coords[index], r=float(values[index]))
            suppressed[np.asarray(neighbours, dtype=np.int64)] = True
        kept_indices = np.asarray(kept, dtype=np.int64)
        return coords[kept_indices], values[kept_indices]

    @staticmethod
    def _restore_coords(
        coords: np.ndarray,
        original_shape: tuple[int, int, int],
        scaled_shape: tuple[int, int, int],
    ) -> np.ndarray:
        if len(coords) == 0:
            return np.empty((0, 3), dtype=np.int64)
        restored = np.round(
            np.asarray(coords, dtype=np.float64)
            * np.asarray(original_shape, dtype=np.float64)
            / np.asarray(scaled_shape, dtype=np.float64)
        ).astype(np.int64)
        for axis, size in enumerate(original_shape):
            restored[:, axis] = np.clip(restored[:, axis], 0, int(size) - 1)
        return restored

    @staticmethod
    def _usable_device(device: torch.device | str) -> torch.device:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA is unavailable; using CPU prompt generation instead.")
            return torch.device("cpu")
        return device

    # ------------------------------------------------------------------
    # Crop utilities
    # ------------------------------------------------------------------

    @staticmethod
    def generate_crops(
        image_size: tuple[int, int, int],
        crop_size: tuple[int, int, int] | None,
        overlap: tuple[int, int, int] | None,
    ) -> list[Crop3D]:
        image_z, image_y, image_x = image_size
        if crop_size is None or overlap is None:
            return [Crop3D(0, 0, 0, image_z, image_y, image_x)]

        crop_z, crop_y, crop_x = crop_size
        overlap_z, overlap_y, overlap_x = overlap
        z_starts = PromptGenerator._axis_starts(image_z, crop_z, overlap_z, "Z")
        y_starts = PromptGenerator._axis_starts(image_y, crop_y, overlap_y, "Y")
        x_starts = PromptGenerator._axis_starts(image_x, crop_x, overlap_x, "X")
        return [
            Crop3D(
                start_z=start_z,
                start_y=start_y,
                start_x=start_x,
                end_z=min(start_z + crop_z, image_z),
                end_y=min(start_y + crop_y, image_y),
                end_x=min(start_x + crop_x, image_x),
            )
            for start_z in z_starts
            for start_y in y_starts
            for start_x in x_starts
        ]

    @staticmethod
    def _axis_starts(
        image_len: int,
        crop_len: int,
        overlap: int,
        axis_name: str,
    ) -> list[int]:
        if image_len <= 0:
            raise ValueError(f"{axis_name}: image size must be positive.")
        if crop_len <= 0:
            raise ValueError(f"{axis_name}: crop size must be positive.")
        if overlap < 0:
            raise ValueError(f"{axis_name}: overlap must be non-negative.")
        if overlap >= crop_len:
            raise ValueError(
                f"{axis_name}: overlap must be smaller than crop size. "
                f"Got overlap={overlap}, crop_size={crop_len}."
            )
        if image_len <= crop_len:
            return [0]
        stride = crop_len - overlap
        starts = list(range(0, image_len - crop_len + 1, stride))
        final_start = image_len - crop_len
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts


__all__ = ["PromptGenerator", "PromptGenerationMethod"]
