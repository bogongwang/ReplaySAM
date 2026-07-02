from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from tqdm import tqdm

from replaysam.utils.helpers import Crop3D

LOGGER = logging.getLogger(__name__)


class PromptGenerator:
    """Generate particle prompt coordinates from a 3D tomogram."""

    def __init__(self, tomo: np.ndarray):
        self.tomo = np.asarray(tomo)
        if self.tomo.ndim != 3:
            raise ValueError(f"Expected a 3D tomogram in (Z, Y, X) order, got {self.tomo.shape}.")

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
        """Generate globally indexed prompts from local distance-transform maxima."""
        crops = self.generate_crops(
            image_size=tuple(self.tomo.shape),
            crop_size=crop_size,
            overlap=crop_overlap,
        )
        global_coords: list[np.ndarray] = []
        global_dist_vals: list[np.ndarray] = []

        for crop in tqdm(
            crops,
            desc="Generating prompts from local maxima",
            disable=not show_progress,
        ):
            crop_tomo = np.asarray(self.tomo[crop.slices])
            threshold = float(threshold_otsu(crop_tomo)) if bin_thresh is None else bin_thresh
            crop_mask = crop_tomo > threshold
            crop_coords, crop_dist_vals = self._local_maximums_cupy(
                crop_mask,
                dist_threshold=dist_val_thresh,
                max_filter_size=max_filter_size,
            )
            if len(crop_coords) == 0:
                continue
            global_coords.append(
                np.asarray(crop_coords, dtype=np.int64)
                + np.asarray(crop.starts, dtype=np.int64)
            )
            global_dist_vals.append(np.asarray(crop_dist_vals, dtype=np.float32))

        return self._finalize_global_prompts(
            coord_batches=global_coords,
            value_batches=global_dist_vals,
            value_threshold=dist_val_thresh,
            return_max_values=return_max_values,
        )

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
            dist_sl = cp_distance_transform_edt(bin_sl).astype(cp.float16)
            maximums = cp_maximum_filter(
                dist_sl,
                size=(max_filter_size,) * dist_sl.ndim,
                mode="nearest",
            )
            peaks = (maximums == dist_sl) & bin_sl & (dist_sl > dist_threshold)
            peak_coords = cp.asnumpy(cp.argwhere(peaks))
            peak_values = cp.asnumpy(dist_sl[peaks])
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            return peak_coords, peak_values
        except Exception as exc:
            LOGGER.warning("Falling back to CPU local maxima because CuPy failed: %s", exc)

        bin_sl = np.asarray(binary_image, dtype=bool)
        dist_sl = distance_transform_edt(bin_sl).astype(np.float32)
        maximums = maximum_filter(
            dist_sl,
            size=(max_filter_size,) * dist_sl.ndim,
            mode="nearest",
        )
        peaks = (maximums == dist_sl) & bin_sl & (dist_sl > dist_threshold)
        return np.argwhere(peaks), dist_sl[peaks]

    @staticmethod
    def _finalize_global_prompts(
        coord_batches: list[np.ndarray],
        value_batches: list[np.ndarray],
        value_threshold: float,
        return_max_values: bool,
    ):
        if not coord_batches:
            empty_coords = np.empty((0, 3), dtype=np.int64)
            empty_values = np.empty((0,), dtype=np.float32)
            if return_max_values:
                return empty_coords, empty_values
            return empty_coords

        coords = np.concatenate(coord_batches, axis=0)
        values = np.concatenate(value_batches, axis=0)
        coords, values = PromptGenerator._suppress_close_points(
            coords,
            values,
            value_threshold=value_threshold,
        )
        if return_max_values:
            return coords, values
        return coords

    @staticmethod
    def _suppress_close_points(
        coords: np.ndarray,
        values: np.ndarray,
        value_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(coords) == 0:
            return coords, values

        order = np.argsort(values)[::-1]
        coords = np.asarray(coords)[order]
        values = np.asarray(values)[order]
        tree = cKDTree(coords)
        keep = np.ones(len(coords), dtype=bool)
        for idx, coord in enumerate(coords):
            if not keep[idx]:
                continue
            close_indices = tree.query_ball_point(coord, r=float(values[idx]))
            for close_idx in close_indices:
                if close_idx != idx and values[close_idx] <= values[idx]:
                    keep[close_idx] = False

        value_keep = values > value_threshold
        keep &= value_keep
        return coords[keep], values[keep]

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

        crops = []
        for start_z in z_starts:
            for start_y in y_starts:
                for start_x in x_starts:
                    crops.append(
                        Crop3D(
                            start_z=start_z,
                            start_y=start_y,
                            start_x=start_x,
                            end_z=min(start_z + crop_z, image_z),
                            end_y=min(start_y + crop_y, image_y),
                            end_x=min(start_x + crop_x, image_x),
                        )
                    )
        return crops

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
