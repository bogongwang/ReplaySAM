from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Iterator
import logging

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    import sam2
except ModuleNotFoundError as exc:
    if exc.name != "sam2":
        raise
    vendored_models_dir = Path(__file__).resolve().parent
    if not (vendored_models_dir / "sam2" / "__init__.py").exists():
        raise
    sys.path.insert(0, str(vendored_models_dir))
    import sam2

from sam2.build_sam import _load_checkpoint
from sam2.modeling.sam2_base import NO_OBJ_SCORE
from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.amg import batch_iterator
from sam2.utils.misc import fill_holes_in_mask_scores
from replaysam.utils.configs import SAM2BackboneConfig
from replaysam.utils.dataloader import SAM2DataLoader
from replaysam.utils.helpers import (
    LRUDict,
    LimitedOrderedDict,
    resize_2d,
    pad_to_orig_size,
    create_random_points
)


LOGGER = logging.getLogger(__name__)


# MARK: SAM2 Backbone
class SAM2Backbone(SAM2VideoPredictor):
    """SAM2 video backbone adapted for volumetric slice-by-slice segmentation."""

    def __init__(
            self,
            backbone_config: SAM2BackboneConfig,
            **kwargs,
    ):
        """
        Initialise the backbone with inference config, cache, and data access.

        To initialise externally, use `SAM2Backbone.from_config`.
        Args:
            backbone_config: Runtime configuration for SAM2 inference.
            compile: Whether to compile model components during initialisation.
            **kwargs: Additional keyword arguments passed to `SAM2VideoPredictor`.
        """
        self.config = backbone_config
        super().__init__(**kwargs)
        if self.config.compile:
            self._compile_all_components()
        self.image_feats_cache = SAM2ImageFeatureCache()
        self.propagation_cache = SAM2PropagationCache()

    @torch.inference_mode()
    def segment_starting_slice(
            self,
            dataloader: SAM2DataLoader,
            point_prompt: tuple[int, int, int],  # requires in original range
            axis: int = 0,
            rand_pts_radius: float | None = None,
    ) -> tuple[int, torch.Tensor] | tuple[None, None]:
        """Run prompt-based segmentation on the slice containing the seed point.

        Args:
            point_prompt: Seed point in original tomogram coordinates.
            axis: Axis used to extract 2D slices from the volume.
            rand_pts_radius: Radius used to sample extra prompt points (usually use distance value).

        Returns:
            A tuple of the selected slice index and the resized high-resolution mask.
            Returns `(None, None)` when no valid object is found.
        """
        slice_idx = point_prompt[axis]
        crop_boxes = [(0, 0, self.image_size, self.image_size)]
        input_points = self._generate_input_points(
            input_point=point_prompt,
            tomo_size=self._dataloader_shape(dataloader),
            crop_boxes=crop_boxes,
            axis=axis,
            n_sample_points=self.config.n_sample_points,
            sample_radius=rand_pts_radius
        )
        sam_results = self._predict_with_points(
            dataloader=dataloader,
            slice_idx=slice_idx,
            axis=axis,
            point_prompts=input_points,
            crop_boxes=crop_boxes,
        )
        postprocessed_sam_result = self._postprocess_sam_results(
            sam_results=sam_results,
            pred_iou_thresh=self.config.pred_iou_thresh
        )
        if postprocessed_sam_result is None:
            # Unable to find starting object
            return None, None
        self._save_seg_res_to_cache(
            dataloader=dataloader,
            slice_idx=slice_idx,
            axis=axis,
            sam_result=postprocessed_sam_result,
        )
        output_high_res_mask = resize_2d(
            image=postprocessed_sam_result.high_res_masks,
            size=dataloader.get_curr_hw(axis=axis),
        ).squeeze()
        del postprocessed_sam_result
        torch.cuda.empty_cache()
        return slice_idx, output_high_res_mask

    @staticmethod
    def _dataloader_shape(dataloader: SAM2DataLoader) -> tuple[int, int, int]:
        return (
            dataloader.get_num_slices(0),
            dataloader.get_num_slices(1),
            dataloader.get_num_slices(2),
        )

    @torch.inference_mode()
    def propagate_all(
            self,
            dataloader: SAM2DataLoader,
            start_slice_idx: int,
            axis: int,
    ) -> Iterator[tuple[int, torch.Tensor]]:
        for tracking_order in [False, True]:
            for (slice_idx, output_mask) in self.propagate(
                dataloader=dataloader,
                start_slice_idx=start_slice_idx,
                axis=axis,
                reverse=tracking_order
            ):
                yield slice_idx, output_mask
        self.clear_propagation_cache()

    @torch.inference_mode()
    def propagate(
            self,
            dataloader: SAM2DataLoader,
            start_slice_idx: int,
            axis: int,
            reverse: bool = False,
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Propagate a conditioned mask forward or backward through the volume.

        Args:
            start_slice_idx: Slice index to start propagation from.
            reverse: Whether to propagate toward lower slice indices.

        Yields:
            Tuples of slice index and resized propagated mask.
        """
        processing_range = tqdm(
            self._prepare_processing_range(
                num_slices=dataloader.get_num_slices(axis=axis),
                start_frame_idx=start_slice_idx,
                reverse=reverse,
            ),
            desc=f'{"Reverse" if reverse else "Forward"} Propagation at axis {axis}',
            leave=False
        )

        for slice_idx in processing_range:
            tracking_result = self._propagate(
                dataloader=dataloader,
                axis=axis,
                slice_idx=slice_idx,
                reverse=reverse,
            )

            if tracking_result is None:
                continue
            is_tracking_terminated = self._check_tracking_finished(
                tracking_result=tracking_result,
                slice_idx=slice_idx,
                reverse=reverse,
                termination_iou=self.config.termination_iou,
                termination_mask_size=self.config.termination_mask_size,
            )

            if is_tracking_terminated:
                return
            else:
                self._save_track_res_to_cache(
                    slice_idx=slice_idx,
                    tracking_result=tracking_result,
                )
            # restore to original resolution
            output_high_res_mask = resize_2d(
                image=tracking_result.high_res_masks,
                size=dataloader.get_curr_hw(axis=axis),
            ).squeeze()
            yield slice_idx, output_high_res_mask

    def clear_propagation_cache(self):
        self.propagation_cache.clear_tracking_records()

    def _propagate(
            self,
            dataloader: SAM2DataLoader,
            slice_idx: int,
            axis: int,
            reverse: bool = False,
    ):
        """Predict the object mask for one slice using the memory bank.

        Args:
            slice_idx: Slice index to process.
            reverse: Whether propagation is running toward lower slice indices.

        Returns:
            A tracking result for the slice, or `None` for conditioning slices.
        """
        if slice_idx in self.propagation_cache.cond_frame_outputs:
            # We have SAM results for this frame, so we can directly use them without propagation
            return None
        else:
            # We don't have SAM results for this frame, so we need to propagate from the nearest frame with SAM results
            (
                _, _, current_vision_feats, current_vision_pos_embeds, feat_sizes
            ) = self._get_image_feature(
                dataloader=dataloader,
                slice_idx=slice_idx,
                axis=axis,
                crop_box=None
            )
            # Fuse visual feature with previous memory features in memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                slice_idx=slice_idx,
                num_slices=dataloader.get_num_slices(axis=axis),
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                is_init_cond_frame=False,
                track_in_reverse=reverse
            )
            # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=False,
            )
            (
                _, _,
                ious, low_res_masks, high_res_masks,
                obj_ptr, object_score_logits,
            ) = sam_outputs

            maskmem_features, maskmem_pos_enc = self._encode_memory_in_output(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                point_inputs=None,
                run_mem_encoder=True,
                high_res_masks=high_res_masks,
                object_score_logits=object_score_logits,
            )
            return TrackingResult(
                ious=ious,
                low_res_masks=low_res_masks,
                high_res_masks=high_res_masks,
                obj_ptrs=obj_ptr,
                obj_score_logits=object_score_logits,
                maskmem_features=maskmem_features,
                maskmem_pos_enc=maskmem_pos_enc,
            )

    def _check_tracking_finished(
            self,
            tracking_result: TrackingResult,
            slice_idx: int,
            reverse: bool,
            termination_mask_size: int,
            termination_iou: float,
    ) -> bool:
        """Check whether single-slice propagation should terminate.

        Args:
            tracking_result: Tracking result predicted for the current slice.
            slice_idx: Current slice index.
            reverse: Whether propagation is running toward lower slice indices.
            termination_mask_size: Minimum foreground-pixel count required to continue.
            termination_iou: Minimum IoU with the previous mask required to continue.

        Returns:
            Whether tracking should stop at the current slice.

        Raises:
            ValueError: If the previous slice result is missing from the cache.
        """
        low_res_masks = tracking_result.low_res_masks
        prev_slice_idx = slice_idx - 1 if not reverse else slice_idx + 1

        if prev_slice_idx in self.propagation_cache.cond_frame_outputs:
            prev_out_mask = self.propagation_cache.cond_frame_outputs[prev_slice_idx]["pred_masks"]
        elif prev_slice_idx in self.propagation_cache.non_cond_frame_outputs:
            prev_out_mask = self.propagation_cache.non_cond_frame_outputs[prev_slice_idx]["pred_masks"]
        else:
            raise ValueError("Previous slice not found in outputs")

        pred_mask = low_res_masks > 0
        prev_mask = prev_out_mask > 0

        pred_size = pred_mask.sum()
        prev_size = prev_mask.sum()
        inter = (pred_mask & prev_mask).sum()
        union = pred_size + prev_size - inter
        iou_val = inter.float() / union.clamp_min(1).float()
        should_stop = (pred_size < termination_mask_size) | (iou_val < termination_iou)
        should_stop = bool(should_stop.item())
        if should_stop:
            LOGGER.info(
                f'{"Reverse" if reverse else "Forward"} propagation terminated at {slice_idx}. Size: {pred_size} (thresh: {termination_mask_size}), IoU: {iou_val:.2f} (thresh: {termination_iou}).')
        return should_stop

    def _save_track_res_to_cache(
            self,
            slice_idx: int,
            tracking_result: TrackingResult,
    ):
        """Store a propagated tracking result in the non-conditioning cache.

        Args:
            slice_idx: Slice index associated with the tracking result.
            tracking_result: Tracking result to cache for later propagation.
        """
        out = {
            "maskmem_features": tracking_result.maskmem_features,
            "maskmem_pos_enc": tracking_result.maskmem_pos_enc,
            "pred_masks": tracking_result.low_res_masks,
            "obj_ptr": tracking_result.obj_ptrs,
            "object_score_logits": tracking_result.obj_score_logits,
        }
        self.propagation_cache.non_cond_frame_outputs.add(key=slice_idx, value=out)

    def _encode_memory_in_output(
            self,
            current_vision_feats: list[torch.Tensor],
            feat_sizes: list[tuple[int, int]],
            point_inputs: torch.Tensor | None,
            run_mem_encoder: bool,
            high_res_masks: torch.Tensor,
            object_score_logits: torch.Tensor,
            **kwargs
    ):
        """Encode the predicted mask into memory features for later propagation.

        Args:
            current_vision_feats: Current slice feature pyramid from the image encoder.
            feat_sizes: Spatial feature sizes corresponding to `current_vision_feats`.
            point_inputs: Optional point prompts used to produce the mask.
            run_mem_encoder: Whether to run the memory encoder.
            high_res_masks: High-resolution mask logits to encode.
            object_score_logits: Object-presence score logits from the mask decoder.

        Returns:
            A tuple of memory features and positional encodings, or `(None, None)`.
        """
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            return (maskmem_features, maskmem_pos_enc)
        else:
            return (None, None)

    def _prepare_memory_conditioned_features(
            self,
            slice_idx: int,
            num_slices: int,
            current_vision_feats: list[torch.Tensor],
            current_vision_pos_embeds: list[torch.Tensor],
            feat_sizes: list[tuple[int, int]],
            is_init_cond_frame: bool = False,
            track_in_reverse: bool = False,
            **kwargs
    ):
        """Fuse the current frame features with cached temporal memory.

        Args:
            slice_idx: Current slice index being processed.
            current_vision_feats: Backbone feature maps for the current slice.
            current_vision_pos_embeds: Positional encodings for the current slice.
            feat_sizes: Spatial sizes corresponding to the feature pyramid.
            is_init_cond_frame: Whether this is the initial conditioned slice.
            track_in_reverse: Whether propagation is running in reverse order.

        Returns:
            Memory-conditioned pixel features for the SAM mask heads.
        """
        num_frames = num_slices
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(self.propagation_cache.cond_frame_outputs) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = self.propagation_cache.cond_frame_outputs
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                slice_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = slice_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = slice_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((slice_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(slice_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = self.propagation_cache.non_cond_frame_outputs.get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                feats = prev["maskmem_features"]
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev["maskmem_pos_enc"][-1]
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                        maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= slice_idx if track_in_reverse else t <= slice_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (slice_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(slice_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = slice_idx + t_diff if track_in_reverse else slice_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = self.propagation_cache.non_cond_frame_outputs.get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.clone()
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _prepare_processing_range(
            self,
            num_slices: int,
            start_frame_idx: int,
            reverse: bool,
    ) -> range:
        """Build the slice iteration order for forward or reverse propagation.

        Args:
            start_frame_idx: Slice index to start from.
            reverse: Whether to iterate toward lower indices.

        Returns:
            A range describing the processing order.

        Raises:
            AssertionError: If the start index is out of bounds.
        """
        if not reverse:
            # forward propagate
            assert start_frame_idx < num_slices, (
                "start_frame_idx must be less than num_frames for forward tracking"
            )
            processing_order = range(start_frame_idx, num_slices)
        else:
            # reverse propagate
            assert start_frame_idx >= 0, (
                "start_frame_idx must be greater than 0 for reverse tracking"
            )
            processing_order = range(start_frame_idx, -1, -1)
        return processing_order

    @torch.inference_mode()
    def _predict_with_points(
            self,
            dataloader: SAM2DataLoader,
            slice_idx: int,
            axis: int,
            point_prompts: list[np.ndarray],
            crop_boxes: list,
            point_labels: list[np.ndarray] | None = None,
            points_per_batch: int = 16,
            original_size: tuple[int, int] | None = None,
    ) -> SegmentationResult:
        """Run SAM mask prediction for prompt points across all crop boxes.

        Args:
            slice_idx: Slice index to segment.
            point_prompts: Prompt points per crop, normalized to each crop.
            crop_boxes: Crop boxes generated for SAM automatic mask proposals.
            points_per_batch: Number of prompt points evaluated per decoder batch.

        Returns:
            Aggregated SAM predictions across all prompt batches and crops.
        """
        # ious, low_res_masks, high_res_masks, obj_ptrs, object_score_logits
        all_sam_results = SegmentationResult()
        if point_labels is None:
            point_labels = [None] * len(point_prompts)
        for points, labels, crop_box in zip(point_prompts, point_labels, crop_boxes):
            # Get the image features for the selected slice
            _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
                dataloader=dataloader,
                slice_idx=slice_idx,
                axis=axis,
                crop_box=crop_box
            )
            features = [
                feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
                for feat, feat_size in zip(current_vision_feats[::-1], feat_sizes[::-1])
            ][::-1]
            image_embeddings = features[-1]
            high_res_features = features[:-1]

            if len(points) == 0:
                continue
            if labels is None:
                for (points_batch,) in batch_iterator(points_per_batch, points):
                    sam_results = self._forward_sam_model(
                        points_batch=points_batch,
                        crop_box=crop_box,
                        image_embeddings=image_embeddings,
                        high_res_features=high_res_features,
                        multimask_output=False,
                        original_size=original_size,
                    )
                    all_sam_results.cat(sam_results)
            else:
                sam_results = self._forward_sam_model(
                    points_batch=np.expand_dims(points, axis=0),
                    crop_box=crop_box,
                    image_embeddings=image_embeddings,
                    high_res_features=high_res_features,
                    multimask_output=False,
                    point_labels_batch=np.expand_dims(labels, axis=0),
                    original_size=original_size,
                )
                all_sam_results.cat(sam_results)
        return all_sam_results

    def _postprocess_sam_results(
            self,
            sam_results: SegmentationResult,
            pred_iou_thresh: float,
            softmax_temp: float = 1.0
    ) -> SegmentationResult | None:
        """Merge multiple SAM proposals into a single weighted segmentation result.

        Args:
            sam_results: Raw SAM proposals collected from prompt sampling.
            pred_iou_thresh: Minimum predicted IoU required to keep a proposal.
            softmax_temp: Temperature applied before proposal weighting.

        Returns:
            A weighted segmentation result, or `None` when no proposal survives filtering.
        """
        # Remove low IoU preds
        if pred_iou_thresh > 0:
            keep_mask = sam_results.ious > pred_iou_thresh
            if keep_mask.sum() == 0:
                LOGGER.warning(f"Ignoring segmentation result: No proposal survives IoU threshold {pred_iou_thresh:.2f}.")
                return None
            sam_results.filter(keep_mask)
        iou_softmax = F.softmax(sam_results.ious / softmax_temp, dim=0)
        weighted_iou = torch.sum(iou_softmax * sam_results.ious, dim=0)
        weighted_high_res_mask = torch.sum(iou_softmax[:, None, None] * sam_results.high_res_masks, dim=0)
        weighted_low_res_mask = resize_2d(weighted_high_res_mask, size=sam_results.low_res_masks.shape[-2:]).squeeze()
        weighted_obj_ptr = torch.sum(iou_softmax[:, None] * sam_results.obj_ptrs, dim=0)
        weighted_obj_score_logit = torch.sum(iou_softmax[:, None] * sam_results.obj_score_logits, dim=0)
        # Remove extremely large predictions
        mask_area_ratio = (weighted_low_res_mask > 0).sum() / 65336  # normalize by the total number of pixels in low-res mask (256x256)
        if mask_area_ratio > 0.5:
            LOGGER.warning(f"Ignoring segmentation result: Predicted mask area ratio {mask_area_ratio:.2f} is larger than 0.5, likely due to failed prompt sampling.")
            return None
        return SegmentationResult(
            ious=weighted_iou,
            low_res_masks=weighted_low_res_mask,
            high_res_masks=weighted_high_res_mask,
            obj_ptrs=weighted_obj_ptr,
            obj_score_logits=weighted_obj_score_logit,
        )

    @torch.inference_mode()
    def _save_seg_res_to_cache(
            self,
            dataloader: SAM2DataLoader,
            slice_idx: int,
            axis: int,
            sam_result: SegmentationResult,
    ):
        """Encode and cache the initial conditioned segmentation result.

        Args:
            slice_idx: Conditioned slice index.
            sam_result: Prompt-based segmentation result to encode and cache.
        """
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            dataloader=dataloader,
            slice_idx=slice_idx,
            axis=axis,
            crop_box=None
        )
        # Move to compute device for encoding
        high_res_mask = sam_result.high_res_masks[None, None].to(self.config.compute_device)
        obj_score_logit = sam_result.obj_score_logits.to(self.config.compute_device)
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_mask,
            object_score_logits=obj_score_logit,
            is_mask_from_pts=True,
        )
        maskmem_features = maskmem_features.to(torch.bfloat16)

        if maskmem_pos_enc is not None:
            batch_size = maskmem_pos_enc[0].size(0)
            if self.propagation_cache.maskmem_pos_enc is None:
                assert isinstance(maskmem_pos_enc, list)
                maskmem_pos_enc = [x[0:1].clone() for x in maskmem_pos_enc]
                self.propagation_cache.maskmem_pos_enc = maskmem_pos_enc
            else:
                maskmem_pos_enc = self.propagation_cache.maskmem_pos_enc
            expended_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expended_maskmem_pos_enc = None
        self.propagation_cache.cond_frame_outputs[slice_idx] = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": expended_maskmem_pos_enc,
            "pred_masks": sam_result.low_res_masks.unsqueeze(0),
            "obj_ptr": sam_result.obj_ptrs.unsqueeze(0),
            "object_score_logits": sam_result.obj_score_logits.unsqueeze(0),
        }

    def _forward_sam_model(
            self,
            points_batch: list[np.ndarray],
            crop_box: tuple[int, int, int, int],
            image_embeddings: torch.Tensor,
            high_res_features: list[torch.Tensor],
            multimask_output: bool,
            point_labels_batch: np.ndarray | None = None,
            original_size: tuple[int, int] | None = None,
    ) -> SegmentationResult:
        """Run SAM prompt encoding and mask decoding for one batch of points.

        Args:
            points_batch: Normalized prompt points for one decoder batch.
            crop_box: Crop box in `(x0, y0, x1, y1)` format.
            image_embeddings: Image embedding tensor for the current crop.
            high_res_features: High-resolution feature maps for the SAM decoder.
            multimask_output: Whether the decoder should emit multiple masks per prompt.

        Returns:
            SAM segmentation outputs for the prompt batch.
        """
        point_coords_normed = torch.as_tensor(
            points_batch,
            dtype=torch.float32,
            device=self.config.compute_device,
        )
        point_coords_normed = point_coords_normed * self.image_size
        if point_coords_normed.ndim == 2:
            point_coords_normed = point_coords_normed.unsqueeze(1)
        if point_labels_batch is None:
            point_labels = torch.ones(
                point_coords_normed.shape[:2],
                dtype=torch.int,
                device=self.config.compute_device,
            )
        else:
            point_labels = torch.as_tensor(
                point_labels_batch,
                dtype=torch.int,
                device=self.config.compute_device,
            )
        # Encode the points using the SAM prompt encoder
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(point_coords_normed, point_labels),
            boxes=None,
            masks=None,
        )
        # Clone image_pe and the outputs of sam_prompt_encoder
        # to enable compilation
        sparse_embeddings = sparse_embeddings.clone()
        dense_embeddings = dense_embeddings.clone()
        image_pe = self.sam_prompt_encoder.get_dense_pe().clone()
        # Decode the masks using the SAM mask decoder
        (
            low_res_masks,
            ious,
            sam_output_tokens,
            object_score_logits
        ) = self.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        # Clone the output of sam_mask_decoder
        # to enable compilation
        low_res_masks = low_res_masks.clone()
        ious = ious.clone()
        sam_output_tokens = sam_output_tokens.clone()
        object_score_logits = object_score_logits.clone()
        # Fill small holes in the low-res masks
        # This function uses CUDA kernels. Build before use.
        low_res_masks = fill_holes_in_mask_scores(
            low_res_masks, self.fill_hole_area
        )
        # Set NO_OBJ_SCORE for low-res masks where the object is not appearing
        is_obj_appearing = object_score_logits > 0
        low_res_masks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_masks,
            NO_OBJ_SCORE,
        ).float()
        # Resize the low-res masks to get the high-res masks
        cropped_im_hw = (crop_box[3] - crop_box[1], crop_box[2] - crop_box[0])
        high_res_masks = resize_2d(
            image=low_res_masks,
            size=cropped_im_hw
        )
        if original_size is None:
            original_size = (self.image_size, self.image_size)
        high_res_masks = pad_to_orig_size(
            cropped_im=high_res_masks,
            original_size=original_size,
            crop_box=crop_box,
        )
        # Compute object pointers
        sam_output_token = sam_output_tokens[:, 0]
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.soft_no_obj_ptr:
            lambda_is_obj_appearing = object_score_logits.sigmoid()
        else:
            lambda_is_obj_appearing = is_obj_appearing.float()
        if self.fixed_no_obj_ptr:
            obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        # Move to storage device to avoid GPU memory overflow
        return SegmentationResult(
            ious.squeeze(1),
            low_res_masks.squeeze(1),
            high_res_masks.squeeze(1),
            obj_ptr,
            object_score_logits
        )

    def _get_image_feature(
            self,
            dataloader: SAM2DataLoader,
            slice_idx: int,
            axis: int,
            crop_box: tuple[int] | None = None,
            **kwargs
    ):
        """Load or compute backbone features for a slice and optional crop box.

        Args:
            slice_idx: Slice index to read from the volume.
            crop_box: Optional crop box in `(x0, y0, x1, y1)` format.

        Returns:
            The expanded image tensor and prepared backbone features expected by SAM2.
        """
        if crop_box is None:
            crop_box = (0, 0, self.image_size, self.image_size)
        else:
            crop_box = tuple(crop_box)
        feature_map = self.image_feats_cache.get_cache(axis).get(slice_idx, None)
        image, backbone_out = None, None
        if feature_map is not None:
            image, backbone_out = feature_map.get(crop_box, (None, None))
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = self.config.compute_device
            x0, y0, x1, y1 = crop_box
            image = dataloader.get(axis, slice_idx)
            if crop_box != (0, 0, self.image_size, self.image_size):
                image = image[:, y0:y1, x0:x1]
            image = image.to(device)
            image = resize_2d(
                image=image,
                size=(self.image_size, self.image_size),
                mode='bilinear'
            )
            backbone_out = self._forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            if slice_idx not in self.image_feats_cache.get_cache(axis):
                self.image_feats_cache.get_cache(axis)[slice_idx] = {}
            # Only save the first 8 crops to avoid memory leaking
            if len(self.image_feats_cache.get_cache(axis)[slice_idx]) < 6:
                self.image_feats_cache.get_cache(axis)[slice_idx][crop_box] = (image, backbone_out)
        # expand the features to have the same dimension as the number of objects
        batch_size = 1
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _generate_full_image_points(
            self,
            input_point: tuple[int, int, int],
            tomo_size: tuple[int, int, int],
            axis: int = 0,
            n_sample_points: int = 3,
            sample_radius: float | None = 5,
            output_hw: tuple[int, int] | None = None,
    ) -> np.ndarray:
        assert axis in [0, 1, 2], "Axis must be one of 0, 1, or 2"
        if output_hw is None:
            output_hw = (self.image_size, self.image_size)

        remaining_axes = [i for i in range(3) if i != axis]
        input_point_2d = (
            input_point[remaining_axes[1]],
            input_point[remaining_axes[0]],
        )
        tomo_size_2d = (
            tomo_size[remaining_axes[1]],
            tomo_size[remaining_axes[0]],
        )

        if sample_radius is None:
            input_points = np.asarray([input_point_2d], dtype=float)
        else:
            sampled_points = create_random_points(
                x=input_point_2d[0],
                y=input_point_2d[1],
                r=sample_radius,
                n=n_sample_points,
            )
            input_points = np.vstack(([input_point_2d], sampled_points))

        output_h, output_w = output_hw
        return input_points / tomo_size_2d * (output_w, output_h)

    def _generate_input_points(
            self,
            input_point: tuple[int, int, int],  # requires original range
            tomo_size: tuple[int, int, int],
            crop_boxes: list,
            axis: int = 0,
            n_sample_points: int = 3,
            sample_radius: float | None = 5,
    ) -> list:
        """Create prompt points in full-image space and map them into each crop.

        If sample_radius is None, only the original input point is translated,
        and n_sample_points is ignored.

        Args:
            input_point: Seed point in original tomogram coordinates.
            tomo_size: Full tomogram shape.
            crop_boxes: Crop boxes that will receive translated prompt points.
            axis: Slice axis for the current inference view.
            n_sample_points: Number of extra random prompt points to generate.
                Ignored when sample_radius is None.
            sample_radius: Sampling radius around the seed point. If None, no
                extra random prompt points are generated.

        Returns:
            A list of prompt arrays, one per crop box, in relative scale.
        """
        input_points = self._generate_full_image_points(
            input_point=input_point,
            tomo_size=tomo_size,
            axis=axis,
            n_sample_points=n_sample_points,
            sample_radius=sample_radius,
        )

        translated_points = self._translate_points(
            points=input_points,
            crop_boxes=crop_boxes,
        )

        return translated_points

    def _translate_points(
            self,
            points: np.ndarray,
            crop_boxes: list,
            clamp_to_crop: bool = False,
            image_hw: tuple[int, int] | None = None,
    ) -> list:
        """Translate full-image prompt points into crop-relative normalized coordinates.

        Args:
            points: Prompt points in full-image pixel coordinates.
            crop_boxes: Crop boxes in `(x0, y0, x1, y1)` format.

        Returns:
            A list of arrays containing crop-relative normalized points.
        """
        if image_hw is None:
            image_hw = (self.image_size, self.image_size)
        image_h, image_w = image_hw
        all_normalized_points = []
        for crop_box in crop_boxes:
            normalized_points = []
            x0, y0, x1, y1 = crop_box
            for point in points:
                x, y = point
                clipped_x = np.clip(x, 0, image_w)
                clipped_y = np.clip(y, 0, image_h)
                norm_x = (clipped_x - x0) / (x1 - x0)
                norm_y = (clipped_y - y0) / (y1 - y0)
                if clamp_to_crop:
                    norm_x = np.clip(norm_x, 0, 1)
                    norm_y = np.clip(norm_y, 0, 1)
                normalized_points.append([
                    norm_x,
                    norm_y,
                ])
            all_normalized_points.append(np.asarray(normalized_points))
        return all_normalized_points


    # --- Initialisation method --- #
    @staticmethod
    def from_config(
            config: SAM2BackboneConfig,
            hydra_overrides_extra: list[str] | None = None,
    ) -> SAM2Backbone:
        """Construct, configure, and load a SAM2 backbone from inference config.

        Args:
            config: Runtime configuration for model loading and inference.
            compile: Whether to compile the model for faster inference.
            hydra_overrides_extra: Extra Hydra overrides applied during instantiation.

        Returns:
            A loaded `SAM2Backbone` instance on the configured compute device.
        """
        if hydra_overrides_extra is None:
            hydra_overrides_extra = []

        model_sizes = {
            "t": "tiny",
            "s": "small",
            "b+": "base_plus",
            "l": "large",
        }
        config_paths = {
            "tiny": (
                "configs/sam2.1/sam2.1_hiera_t.yaml",
                "sam2.1_hiera_tiny.pt",
            ),
            "small": (
                "configs/sam2.1/sam2.1_hiera_s.yaml",
                "sam2.1_hiera_small.pt",
            ),
            "base_plus": (
                "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "sam2.1_hiera_base_plus.pt",
            ),
            "large": (
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "sam2.1_hiera_large.pt",
            ),
        }
        if config.model_size in model_sizes:
            config.model_size = model_sizes[config.model_size]

        module_dir = Path(sam2.__file__).resolve().parent
        config_dir = "/" + str(module_dir / config_paths[config.model_size][0])
        checkpt_dir = str(
            module_dir / "checkpoints" / config_paths[config.model_size][1]
        )

        hydra_overrides = [f"++model._target_={__name__}.SAM2Backbone"]

        if config.compile:
            hydra_overrides += [
                "++model.compile_image_encoder=True",
            ]

        if config.apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                "++model.binarize_mask_from_pts_for_mem_enc=true",
                "++model.fill_hole_area=8",
            ]
        hydra_overrides.extend(hydra_overrides_extra)

        cfg = compose(config_name=config_dir, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(
            cfg.model,
            _recursive_=True,
            backbone_config=config,
        )
        _load_checkpoint(model, checkpt_dir)
        model = model.to(config.compute_device)
        model.eval()
        return model

    def _compile_all_components(self):
        """Compile SAM2 model components used during inference."""
        LOGGER.info("Compiling all components. First time may be very slow.")
        self.memory_encoder.forward = torch.compile(
            self.memory_encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        self.memory_attention.forward = torch.compile(
            self.memory_attention.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=True,  # Num. of memories varies
        )

        self.sam_prompt_encoder.forward = torch.compile(
            self.sam_prompt_encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

        self.sam_mask_decoder.forward = torch.compile(
            self.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

    def _forward_image(self, img_batch: torch.Tensor):
        """Run the image encoder and clone outputs for compilation compatibility.

        Args:
            img_batch: Batched input images in SAM2 tensor format.

        Returns:
            Backbone feature and positional encoding dictionary.
        """
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        # Clone to help torch.compile
        for i in range(len(backbone_out["backbone_fpn"])):
            backbone_out["backbone_fpn"][i] = backbone_out["backbone_fpn"][i].clone()
            backbone_out["vision_pos_enc"][i] = backbone_out["vision_pos_enc"][
                i
            ].clone()
        return backbone_out

    def _forward_sam_heads(
            self,
            backbone_features,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=None,
            multimask_output=False,
    ):
        """Run SAM heads and clone outputs for compilation compatibility.

        Args:
            backbone_features: Memory-conditioned image features for the SAM decoder.
            point_inputs: Optional point prompt dictionary with coordinates and labels.
            mask_inputs: Optional mask prompt tensor.
            high_res_features: Optional high-resolution feature maps for decoding.
            multimask_output: Whether to return multiple candidate masks.

        Returns:
            A tuple containing low-resolution multimasks, high-resolution multimasks,
            IoUs, selected low-resolution masks, selected high-resolution masks,
            object pointers, and object-score logits.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        # Clone image_pe and the outputs of sam_prompt_encoder
        # to enable compilation
        sparse_embeddings = sparse_embeddings.clone()
        dense_embeddings = dense_embeddings.clone()
        image_pe = self.sam_prompt_encoder.get_dense_pe().clone()
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        # Clone the output of sam_mask_decoder
        # to enable compilation
        low_res_multimasks = low_res_multimasks.clone()
        ious = ious.clone()
        sam_output_tokens = sam_output_tokens.clone()
        object_score_logits = object_score_logits.clone()

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _encode_new_memory(
            self,
            current_vision_feats: list[torch.Tensor],
            feat_sizes: list[tuple[int, int]],
            pred_masks_high_res: torch.Tensor,
            object_score_logits: torch.Tensor,
            is_mask_from_pts: bool,
    ):
        """Encode a predicted mask into temporal memory features.

        Args:
            current_vision_feats: Current slice feature pyramid from the image encoder.
            feat_sizes: Spatial feature sizes corresponding to `current_vision_feats`.
            pred_masks_high_res: High-resolution predicted mask logits.
            object_score_logits: Object-presence score logits from the mask decoder.
            is_mask_from_pts: Whether the mask came from point prompts.

        Returns:
            A tuple of memory features and memory positional encodings.
        """
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        # Clone the feats and pos_enc to enable compilation
        maskmem_features = maskmem_out["vision_features"].clone()
        maskmem_pos_enc = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                                        1 - is_obj_appearing[..., None, None]
                                ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

# MARK: Inference Cache
class SAM2PropagationCache:
    """Cache for image features and temporal memory used during inference."""

    def __init__(
            self,
            num_cond_frames: int = 7,
            num_non_cond_frames: int = 7,
    ):
        """Initialise feature and tracking caches with bounded storage.

        Args:
            num_non_cond_frames: Maximum number of non-conditioning frames to keep.
        """
        # `maskmem_pos_enc` neither depend on the frame nor the object currently tracking
        self.maskmem_pos_enc = None
        # cond_frames and non_cond_frames for tracking
        self.cond_frame_outputs = LimitedOrderedDict(max_size=num_cond_frames)
        self.non_cond_frame_outputs = LimitedOrderedDict(max_size=num_non_cond_frames)
        self.temp_cond_frame_outputs = LimitedOrderedDict(max_size=num_non_cond_frames)

    def clear_tracking_records(self):
        """Clear cached tracking state while keeping reusable image features."""
        self.maskmem_pos_enc = None
        self.cond_frame_outputs.clear()
        self.non_cond_frame_outputs.clear()
        torch.cuda.empty_cache()

class SAM2ImageFeatureCache:
    def __init__(
        self,
        cache_size: tuple[int, int, int] = (16, 16, 16)
    ):
        self.cache_size = cache_size
        self.caches = [
            LRUDict(max_size=self.cache_size[i])
            for i in range(3)
        ]

    def get_cache(self, axis: int):
        return self.caches[axis]

    def clear(self):
        for cache in self.caches:
            cache.clear()

# MARK: Segmentation Result
@dataclass
class SegmentationResult:
    """Container for SAM segmentation outputs at multiple resolutions."""

    ious: torch.Tensor = None
    low_res_masks: torch.Tensor = None
    high_res_masks: torch.Tensor = None
    obj_ptrs: torch.Tensor = None
    obj_score_logits: torch.Tensor = None

    @staticmethod
    def _cat_optional(a: torch.Tensor | None, b: torch.Tensor | None):
        """Concatenate two optional tensors, preserving `None` values.

        Args:
            a: First optional tensor.
            b: Second optional tensor.

        Returns:
            The non-`None` tensor or a concatenation of both tensors along dim 0.
        """
        assert a is None or isinstance(a, torch.Tensor), \
            f"Expected a to be None or torch.Tensor, got {type(a)}"
        assert b is None or isinstance(b, torch.Tensor), \
            f"Expected b to be None or torch.Tensor, got {type(b)}"

        if a is None:
            return b
        if b is None:
            return a
        return torch.cat([a, b], dim=0)

    def cat(self, new_results: SegmentationResult) -> None:
        """Append another result object onto each populated tensor field.

        Args:
            new_results: Segmentation results to append to this instance.
        """
        for k, v in self.__dict__.items():
            self.__dict__[k] = self._cat_optional(v, getattr(new_results, k))

    def filter(self, keep_mask: torch.Tensor):
        """Filter every populated tensor field using the same boolean mask.

        Args:
            keep_mask: Boolean mask selecting entries to keep in each tensor field.
        """
        for k, v in self.__dict__.items():
            if v is None:
                continue
            self.__dict__[k] = v[torch.as_tensor(keep_mask, device=v.device)]


# MARK: Tracking Result
@dataclass
class TrackingResult:
    """Container for propagated tracking outputs and encoded memory tensors."""
    ious: torch.Tensor = None
    low_res_masks: torch.Tensor = None
    high_res_masks: torch.Tensor = None
    obj_ptrs: torch.Tensor = None
    obj_score_logits: torch.Tensor = None
    maskmem_features: torch.Tensor = None
    maskmem_pos_enc: list[torch.Tensor] = None
    crop_box: tuple[int, int, int, int] | None = None
