from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from replaysam.utils.configs import SAM2BackboneConfig
from replaysam.utils.dataloader import SAM2NumPyDataLoader


class SAM2BackboneAdapter:
    """Run vanilla SAM2 inference across selected volume axes."""

    def __init__(
        self,
        backbone_config: SAM2BackboneConfig,
        volume: np.ndarray,
    ):
        if not isinstance(volume, np.ndarray):
            raise TypeError(f"volume must be a numpy.ndarray, got {type(volume)!r}.")
        from replaysam.models.sam2_backbone import SAM2Backbone

        self.backbone = SAM2Backbone.from_config(config=backbone_config)
        self.dataloader = SAM2NumPyDataLoader(
            data=volume,
            process_device=backbone_config.compute_device,
        )

    def run_multi_axes_inference(
        self,
        point_prompt: tuple[int, int, int],
        dist_map_val: float | None,
        axes: tuple[int, ...] = (0, 1, 2),
        majority_voting_threshold: int = 2,
    ) -> AdapterResult:
        axial_results = []
        for axis in axes:
            axial_results.append(
                self._run_single_axis_inference(
                    point_prompt=point_prompt,
                    axis=axis,
                    rand_pts_radius=dist_map_val,
                )
            )
        merged_mask, merged_mask_offset = self._merge_multi_axes_results(
            results=axial_results,
            axes=axes,
        )
        return self._postprocess_merged_results(
            mask=merged_mask,
            offset=merged_mask_offset,
            majority_voting_threshold=majority_voting_threshold,
        )

    def _run_single_axis_inference(
        self,
        point_prompt: tuple[int, int, int],
        axis: int,
        rand_pts_radius: float | None = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]]:
        output_masks = {}
        scaled_radius = rand_pts_radius * 0.7 if rand_pts_radius is not None else None
        first_slice_idx, output_mask = self.backbone.segment_starting_slice(
            dataloader=self.dataloader,
            point_prompt=point_prompt,
            axis=axis,
            rand_pts_radius=scaled_radius,
        )
        if output_mask is None:
            self.backbone.clear_propagation_cache()
            return output_masks

        output_masks[first_slice_idx] = self._rle_encode(output_mask > 0)
        for slice_idx, output_mask in self.backbone.propagate_all(
            dataloader=self.dataloader,
            start_slice_idx=first_slice_idx,
            axis=axis,
        ):
            output_masks[slice_idx] = self._rle_encode(output_mask > 0)
        return output_masks

    def _merge_multi_axes_results(
        self,
        results: list[dict],
        axes: tuple[int, ...],
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        by_axis = {axis: result for axis, result in zip(axes, results, strict=True)}
        offset, shape = self.find_min_bbox(by_axis)
        output_arr = np.zeros(shape, dtype=np.uint8)
        z_off, y_off, x_off = offset

        for slice_idx, (starts, lengths, crop_origin, cropped_shape) in by_axis.get(0, {}).items():
            mask = self._rle_decode(starts, lengths, cropped_shape)
            y0, x0 = crop_origin
            dy, dx = cropped_shape
            output_arr[slice_idx - z_off, y0 - y_off:y0 - y_off + dy, x0 - x_off:x0 - x_off + dx] += mask

        for slice_idx, (starts, lengths, crop_origin, cropped_shape) in by_axis.get(1, {}).items():
            mask = self._rle_decode(starts, lengths, cropped_shape)
            z0, x0 = crop_origin
            dz, dx = cropped_shape
            output_arr[z0 - z_off:z0 - z_off + dz, slice_idx - y_off, x0 - x_off:x0 - x_off + dx] += mask

        for slice_idx, (starts, lengths, crop_origin, cropped_shape) in by_axis.get(2, {}).items():
            mask = self._rle_decode(starts, lengths, cropped_shape)
            z0, y0 = crop_origin
            dz, dy = cropped_shape
            output_arr[z0 - z_off:z0 - z_off + dz, y0 - y_off:y0 - y_off + dy, slice_idx - x_off] += mask

        return output_arr, tuple(int(x) for x in offset)

    @staticmethod
    def _postprocess_merged_results(
        mask: np.ndarray,
        offset: tuple[int, int, int],
        majority_voting_threshold: int = 2,
    ) -> "AdapterResult":
        return AdapterResult(volume=mask >= majority_voting_threshold, offset=offset)

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
        crop_origin = tuple(int(x) for x in mins.tolist())
        crop_shape = tuple(int(x) for x in (maxs - mins).tolist())
        slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins.tolist(), maxs.tolist(), strict=True))
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
            for start, end in zip(starts, starts + lengths):
                flat[start:end] = 1
        return flat.reshape(crop_shape)

    @staticmethod
    def find_min_bbox(by_axis: dict[int, dict]) -> tuple[np.ndarray, np.ndarray]:
        z_mins, z_maxs = [], []
        y_mins, y_maxs = [], []
        x_mins, x_maxs = [], []

        def update(mins: list, maxs: list, start_vals, end_vals) -> None:
            mins.append(np.min(start_vals))
            maxs.append(np.max(end_vals))

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
            return np.array([0, 0, 0], dtype=np.int32), np.array([0, 0, 0], dtype=np.int32)

        bbox_min = np.array([min(z_mins), min(y_mins), min(x_mins)], dtype=np.int32)
        bbox_max = np.array([max(z_maxs), max(y_maxs), max(x_maxs)], dtype=np.int32)
        return bbox_min, bbox_max - bbox_min

    @staticmethod
    def _extract_axis_data(axis_result: dict):
        if not axis_result:
            return None
        indices = np.asarray(list(axis_result.keys()), dtype=np.int32)
        offsets = np.asarray([res[2] for res in axis_result.values()], dtype=np.int32)
        shapes = np.asarray([res[3] for res in axis_result.values()], dtype=np.int32)
        return indices, offsets, shapes


@dataclass
class AdapterResult:
    volume: np.ndarray
    offset: tuple[int, int, int]
