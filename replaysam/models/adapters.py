from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import dask.array as da
import torch

from replaysam.utils.configs import SAM2BackboneConfig
from replaysam.utils.dataloader import (
    SAM2DaskDataLoader,
    SAM2DataLoader,
    SAM2NumPyDataLoader,
    SAM2TorchTensorDataLoader,
)

class SAM2BackboneAdapter:
    """Run SAM2 inference across selected volume axes and fuse the masks."""

    def __init__(
        self,
        backbone_config: SAM2BackboneConfig,
        volume: np.ndarray | torch.Tensor | da.Array,
        *,
        dataloader_downsample: bool = False,
        dataloader_batch_size: int = 256,
        dataloader_persist: bool = False,
    ):
        self.dataloader = self._make_dataloader(
            volume=volume,
            backbone_config=backbone_config,
            dataloader_downsample=dataloader_downsample,
            dataloader_batch_size=dataloader_batch_size,
            dataloader_persist=dataloader_persist,
        )
        from replaysam.models.sam2_backbone import SAM2Backbone

        self.backbone = SAM2Backbone.from_config(config=backbone_config)
        self._last_merged_axis_masks: np.ndarray | None = None
        self._last_merged_axes: tuple[int, ...] = ()

    @staticmethod
    def _make_dataloader(
        *,
        volume: np.ndarray | torch.Tensor | da.Array,
        backbone_config: SAM2BackboneConfig,
        dataloader_downsample: bool,
        dataloader_batch_size: int,
        dataloader_persist: bool,
    ) -> SAM2DataLoader:
        if isinstance(volume, da.Array):
            return SAM2DaskDataLoader(
                data=volume,
                downsample=dataloader_downsample,
                batch_size=dataloader_batch_size,
                persist=dataloader_persist,
                process_device=backbone_config.compute_device,
                storage_device=backbone_config.storage_device,
            )
        if isinstance(volume, np.ndarray):
            return SAM2NumPyDataLoader(
                data=volume,
                process_device=backbone_config.compute_device,
                storage_device=backbone_config.storage_device,
            )
        if isinstance(volume, torch.Tensor):
            return SAM2TorchTensorDataLoader(
                data=volume,
                storage_device=backbone_config.storage_device,
            )
        expected = "dask.array.Array, numpy.ndarray, or torch.Tensor"
        raise TypeError(f"volume must be a {expected}, got {type(volume)!r}.")

    def set_crop(self, crop) -> None:
        """Set the active volume crop and invalidate crop-dependent features."""
        self.backbone.image_feats_cache.clear()
        self.backbone.clear_propagation_cache()
        self.dataloader.set_crop(crop)

    def run_multi_axes_inference(
        self,
        point_prompt: tuple[int, int, int],
        dist_map_val: float | None,
        axes: tuple[int, ...] = (0, 1, 2),
        majority_voting_threshold: int = 2,
        consistency_score: float | None = 0.7,
    ) -> "AdapterResult":
        axes = self._validate_axes(axes)
        self._validate_voting_threshold(majority_voting_threshold, len(axes))
        point_prompt = self._normalize_point(point_prompt, name="point_prompt")
        point_prompt = self.dataloader.to_active_point(point_prompt)
        self._validate_points_in_active_volume([point_prompt], name="point_prompt")

        axial_results = [
            self._run_single_axis_inference(
                point_prompt=point_prompt,
                axis=axis,
                rand_pts_radius=dist_map_val,
            )
            for axis in axes
        ]
        merged_mask, merged_mask_offset = self._merge_multi_axes_results(
            results=axial_results,
            axes=axes,
        )
        return self._postprocess_merged_results(
            mask=merged_mask,
            offset=merged_mask_offset,
            majority_voting_threshold=majority_voting_threshold,
            consistency_score=consistency_score,
        )

    def run_multi_axes_inference_with_prompts(
        self,
        positive_point_prompts: list[tuple[int, int, int]],
        negative_point_prompts: list[tuple[int, int, int]] | None = None,
        axes: tuple[int, ...] = (0, 1, 2),
        majority_voting_threshold: int = 2,
        consistency_score: float | None = 0.7,
    ) -> "AdapterResult":
        """Run multi-axis inference from multiple positive and negative 3D points."""
        axes = self._validate_axes(axes)
        self._validate_voting_threshold(majority_voting_threshold, len(axes))
        positive_point_prompts, negative_point_prompts = self._normalize_prompt_sets(
            positive_point_prompts=positive_point_prompts,
            negative_point_prompts=negative_point_prompts,
        )
        positive_point_prompts = [
            self.dataloader.to_active_point(point)
            for point in positive_point_prompts
        ]
        negative_point_prompts = [
            self.dataloader.to_active_point(point)
            for point in negative_point_prompts
        ]
        self._validate_points_in_active_volume(
            positive_point_prompts,
            name="positive_point_prompts",
        )
        self._validate_points_in_active_volume(
            negative_point_prompts,
            name="negative_point_prompts",
        )

        axial_results = [
            self._run_single_axis_inference_with_prompts(
                positive_point_prompts=positive_point_prompts,
                negative_point_prompts=negative_point_prompts,
                axis=axis,
            )
            for axis in axes
        ]
        merged_mask, merged_mask_offset = self._merge_multi_axes_results(
            results=axial_results,
            axes=axes,
        )
        return self._postprocess_merged_results(
            mask=merged_mask,
            offset=merged_mask_offset,
            majority_voting_threshold=majority_voting_threshold,
            consistency_score=consistency_score,
        )

    def _run_single_axis_inference(
        self,
        point_prompt: tuple[int, int, int],
        axis: int,
        rand_pts_radius: float | None = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]]:
        output_masks = {}
        scaled_radius = rand_pts_radius * 0.7 if rand_pts_radius is not None else None
        if self.backbone.config.zoom_enabled:
            segment_starting_slice = self.backbone.segment_starting_slice_zoomed
            propagate_all = self.backbone.propagate_all_zoomed
        else:
            segment_starting_slice = self.backbone.segment_starting_slice
            propagate_all = self.backbone.propagate_all

        first_slice_idx, output_mask = segment_starting_slice(
            dataloader=self.dataloader,
            point_prompt=point_prompt,
            axis=axis,
            rand_pts_radius=scaled_radius,
        )
        if output_mask is None:
            self.backbone.clear_propagation_cache()
            return output_masks

        output_masks[first_slice_idx] = self._rle_encode(output_mask > 0)
        for slice_idx, output_mask in propagate_all(
            dataloader=self.dataloader,
            start_slice_idx=first_slice_idx,
            axis=axis,
        ):
            output_masks[slice_idx] = self._rle_encode(output_mask > 0)
        return output_masks

    def _run_single_axis_inference_with_prompts(
        self,
        positive_point_prompts: list[tuple[int, int, int]],
        negative_point_prompts: list[tuple[int, int, int]],
        axis: int,
    ):
        output_masks = {}
        if self.backbone.config.zoom_enabled:
            segment_starting_slice = self.backbone.segment_starting_slice_zoomed_with_prompts
            propagate_all = self.backbone.propagate_all_zoomed
        else:
            segment_starting_slice = self.backbone.segment_starting_slice_with_prompts
            propagate_all = self.backbone.propagate_all

        first_slice_idx, output_mask = segment_starting_slice(
            dataloader=self.dataloader,
            positive_point_prompts=positive_point_prompts,
            negative_point_prompts=negative_point_prompts,
            axis=axis,
        )
        if output_mask is None:
            self.backbone.clear_propagation_cache()
            return output_masks

        output_masks[first_slice_idx] = self._rle_encode(output_mask > 0)
        for slice_idx, output_mask in propagate_all(
            dataloader=self.dataloader,
            start_slice_idx=first_slice_idx,
            axis=axis,
        ):
            output_masks[slice_idx] = self._rle_encode(output_mask > 0)
        return output_masks

    @staticmethod
    def _normalize_prompt_sets(
        positive_point_prompts: list[tuple[int, int, int]],
        negative_point_prompts: list[tuple[int, int, int]] | None,
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        positive_points = SAM2BackboneAdapter._normalize_points(
            positive_point_prompts,
            name="positive_point_prompts",
            allow_empty=False,
        )
        negative_points = SAM2BackboneAdapter._normalize_points(
            negative_point_prompts or [],
            name="negative_point_prompts",
            allow_empty=True,
        )
        return positive_points, negative_points

    @staticmethod
    def _normalize_points(
        points,
        *,
        name: str,
        allow_empty: bool,
    ) -> list[tuple[int, int, int]]:
        if points is None:
            points = []
        normalized = []
        for point in points:
            try:
                point = tuple(int(coord) for coord in point)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain integer 3D points, got {point!r}.") from exc
            if len(point) != 3:
                raise ValueError(f"{name} must contain 3D points, got {point!r}.")
            normalized.append(point)
        if not normalized and not allow_empty:
            raise ValueError(f"{name} must contain at least one 3D point.")
        return normalized

    @staticmethod
    def _normalize_point(point, *, name: str) -> tuple[int, int, int]:
        normalized = SAM2BackboneAdapter._normalize_points(
            [point], name=name, allow_empty=False
        )
        return normalized[0]

    def _validate_points_in_active_volume(
        self,
        points: list[tuple[int, int, int]],
        *,
        name: str,
    ) -> None:
        shape = self.dataloader.get_crop_shape()
        for point in points:
            if any(coord < 0 or coord >= int(shape[axis]) for axis, coord in enumerate(point)):
                raise ValueError(
                    f"{name} contains point {point} outside active volume shape {shape}."
                )

    @staticmethod
    def _validate_axes(axes: tuple[int, ...]) -> tuple[int, ...]:
        axes = tuple(int(axis) for axis in axes)
        if not axes:
            raise ValueError("axes must contain at least one of 0, 1, or 2.")
        if any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError(f"axes must contain only 0, 1, or 2, got {axes}.")
        if len(set(axes)) != len(axes):
            raise ValueError(f"axes must not contain duplicates, got {axes}.")
        return axes

    @staticmethod
    def _validate_voting_threshold(threshold: int, axis_count: int) -> None:
        if not isinstance(threshold, (int, np.integer)):
            raise TypeError(f"majority_voting_threshold must be an integer, got {type(threshold)!r}.")
        if threshold < 1 or threshold > axis_count:
            raise ValueError(
                "majority_voting_threshold must be between 1 and the number of axes "
                f"({axis_count}), got {threshold}."
            )

    def _merge_multi_axes_results(
        self,
        results: list[dict],
        axes: tuple[int, ...],
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        by_axis = {axis: result for axis, result in zip(axes, results, strict=True)}
        offset, shape = self.find_min_bbox(by_axis)
        output_arr = np.zeros(tuple(int(value) for value in shape), dtype=np.uint8)
        axis_masks = np.zeros((len(axes), *tuple(int(value) for value in shape)), dtype=bool)
        axis_positions = {axis: position for position, axis in enumerate(axes)}
        z_off, y_off, x_off = (int(value) for value in offset)

        for slice_idx, encoded in by_axis.get(0, {}).items():
            starts, lengths, crop_origin, cropped_shape = encoded
            mask = self._rle_decode(starts, lengths, cropped_shape)
            y0, x0 = crop_origin
            dy, dx = cropped_shape
            zz = int(slice_idx) - z_off
            yy0, xx0 = int(y0) - y_off, int(x0) - x_off
            output_arr[zz, yy0:yy0 + dy, xx0:xx0 + dx] += mask
            axis_masks[axis_positions[0], zz, yy0:yy0 + dy, xx0:xx0 + dx] |= mask.astype(bool)

        for slice_idx, encoded in by_axis.get(1, {}).items():
            starts, lengths, crop_origin, cropped_shape = encoded
            mask = self._rle_decode(starts, lengths, cropped_shape)
            z0, x0 = crop_origin
            dz, dx = cropped_shape
            yy = int(slice_idx) - y_off
            zz0, xx0 = int(z0) - z_off, int(x0) - x_off
            output_arr[zz0:zz0 + dz, yy, xx0:xx0 + dx] += mask
            axis_masks[axis_positions[1], zz0:zz0 + dz, yy, xx0:xx0 + dx] |= mask.astype(bool)

        for slice_idx, encoded in by_axis.get(2, {}).items():
            starts, lengths, crop_origin, cropped_shape = encoded
            mask = self._rle_decode(starts, lengths, cropped_shape)
            z0, y0 = crop_origin
            dz, dy = cropped_shape
            xx = int(slice_idx) - x_off
            zz0, yy0 = int(z0) - z_off, int(y0) - y_off
            output_arr[zz0:zz0 + dz, yy0:yy0 + dy, xx] += mask
            axis_masks[axis_positions[2], zz0:zz0 + dz, yy0:yy0 + dy, xx] |= mask.astype(bool)

        self._last_merged_axis_masks = axis_masks
        self._last_merged_axes = axes
        return output_arr, tuple(int(value) for value in offset)

    def _postprocess_merged_results(
        self,
        mask: np.ndarray,
        offset: tuple[int, int, int],
        majority_voting_threshold: int = 2,
        consistency_score: float | None = 0.7,
    ) -> "AdapterResult":
        if consistency_score is not None and not 0 <= consistency_score <= 1:
            raise ValueError(f"consistency_score must be in [0, 1], got {consistency_score}.")
        if (
            consistency_score is not None
            and self._last_merged_axis_masks is not None
            and not self._passes_pair_consistency(
                mask=mask,
                consistency_score=consistency_score,
            )
        ):
            return AdapterResult(volume=np.zeros_like(mask, dtype=bool), offset=offset)
        return AdapterResult(
            volume=np.asarray(mask >= majority_voting_threshold, dtype=bool),
            offset=offset,
        )

    def _passes_pair_consistency(
        self,
        mask: np.ndarray | None = None,
        consistency_score: float = 0.7,
    ) -> bool:
        """Require at least one pair of selected axes to agree sufficiently."""
        axis_masks = self._last_merged_axis_masks
        if axis_masks is not None and axis_masks.shape[0] < 2:
            # A single selected axis has no pair to compare; leave the
            # ordinary majority-voting result intact.
            return True
        if axis_masks is None:
            if mask is not None:
                occupied = np.count_nonzero(mask >= 1)
                if occupied == 0:
                    return True
                return np.count_nonzero(mask >= 2) / occupied >= consistency_score
            return True
        for first_axis, second_axis in combinations(range(axis_masks.shape[0]), 2):
            first = axis_masks[first_axis]
            second = axis_masks[second_axis]
            union = first | second
            union_size = int(np.count_nonzero(union))
            if union_size == 0:
                continue
            pair_iou = float(np.count_nonzero(first & second) / union_size)
            if pair_iou >= consistency_score:
                return True
        return False

    @staticmethod
    def _rle_encode(mask: torch.Tensor):
        ndim = mask.ndim
        device = mask.device
        empty = (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            tuple(0 for _ in range(ndim)),
            tuple(0 for _ in range(ndim)),
        )
        if mask.numel() == 0:
            return empty

        binary = mask != 0
        nz = torch.nonzero(binary, as_tuple=False)
        if nz.numel() == 0:
            return empty

        mins = nz.min(dim=0).values
        maxs = nz.max(dim=0).values + 1
        crop_origin = tuple(int(value) for value in mins.tolist())
        crop_shape = tuple(int(value) for value in (maxs - mins).tolist())
        slices = tuple(
            slice(int(lo), int(hi))
            for lo, hi in zip(mins.tolist(), maxs.tolist(), strict=True)
        )
        flat = binary[slices].reshape(-1).to(torch.uint8)
        padded = torch.empty(flat.numel() + 2, dtype=torch.uint8, device=device)
        padded[0] = 0
        padded[-1] = 0
        padded[1:-1] = flat
        transitions = torch.nonzero(padded[1:] != padded[:-1], as_tuple=False).flatten()
        starts = transitions[0::2]
        ends = transitions[1::2]
        return starts.cpu(), (ends - starts).cpu(), crop_origin, crop_shape

    @staticmethod
    def _rle_decode(starts, lengths, crop_shape):
        starts = np.asarray(starts)
        lengths = np.asarray(lengths)
        flat = np.zeros(int(np.prod(crop_shape)), dtype=np.uint8)
        if starts.size > 0:
            for start, end in zip(starts, starts + lengths, strict=True):
                flat[int(start):int(end)] = 1
        return flat.reshape(crop_shape)

    @staticmethod
    def find_min_bbox(by_axis: dict[int, dict]) -> tuple[np.ndarray, np.ndarray]:
        """Return the tight 3D bounding box covering all per-axis masks."""
        z_mins, z_maxs = [], []
        y_mins, y_maxs = [], []
        x_mins, x_maxs = [], []

        def update(mins: list, maxs: list, start_vals, end_vals) -> None:
            mins.append(int(np.min(start_vals)))
            maxs.append(int(np.max(end_vals)))

        axis0 = SAM2BackboneAdapter._extract_axis_data(by_axis.get(0, {}))
        if axis0 is not None:
            indices, offsets, shapes = axis0
            update(z_mins, z_maxs, indices, indices + 1)
            update(y_mins, y_maxs, offsets[:, 0], offsets[:, 0] + shapes[:, 0])
            update(x_mins, x_maxs, offsets[:, 1], offsets[:, 1] + shapes[:, 1])

        axis1 = SAM2BackboneAdapter._extract_axis_data(by_axis.get(1, {}))
        if axis1 is not None:
            indices, offsets, shapes = axis1
            update(z_mins, z_maxs, offsets[:, 0], offsets[:, 0] + shapes[:, 0])
            update(y_mins, y_maxs, indices, indices + 1)
            update(x_mins, x_maxs, offsets[:, 1], offsets[:, 1] + shapes[:, 1])

        axis2 = SAM2BackboneAdapter._extract_axis_data(by_axis.get(2, {}))
        if axis2 is not None:
            indices, offsets, shapes = axis2
            update(z_mins, z_maxs, offsets[:, 0], offsets[:, 0] + shapes[:, 0])
            update(y_mins, y_maxs, offsets[:, 1], offsets[:, 1] + shapes[:, 1])
            update(x_mins, x_maxs, indices, indices + 1)

        if not z_mins:
            return np.zeros(3, dtype=np.int32), np.zeros(3, dtype=np.int32)
        bbox_min = np.array([min(z_mins), min(y_mins), min(x_mins)], dtype=np.int32)
        bbox_max = np.array([max(z_maxs), max(y_maxs), max(x_maxs)], dtype=np.int32)
        return bbox_min, bbox_max - bbox_min

    @staticmethod
    def _extract_axis_data(axis_result: dict):
        if not axis_result:
            return None
        indices = np.asarray(list(axis_result.keys()), dtype=np.int32)
        offsets = np.asarray([result[2] for result in axis_result.values()], dtype=np.int32)
        shapes = np.asarray([result[3] for result in axis_result.values()], dtype=np.int32)
        return indices, offsets, shapes


@dataclass
class AdapterResult:
    volume: np.ndarray
    offset: tuple[int, int, int]


__all__ = ["AdapterResult", "SAM2BackboneAdapter"]
