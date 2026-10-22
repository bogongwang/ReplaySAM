from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import dask.array as da

from replaysam.models.adapters import AdapterResult, SAM2BackboneAdapter
from replaysam.utils.configs import SAM2PipelineConfig
from replaysam.utils.helpers import ObjectIDDispatcher, overlap_ratio
from replaysam.utils.io import VolumeReader, ZarrVolumeWriter
from replaysam.utils.prompt_generator import PromptGenerator

LOGGER = logging.getLogger(__name__)


class Pipeline:
    """SAM2-only particle segmentation pipeline."""

    def __init__(self, pipeline_config: SAM2PipelineConfig):
        if not isinstance(pipeline_config, SAM2PipelineConfig):
            raise TypeError(f"Pipeline only supports SAM2PipelineConfig, got {type(pipeline_config).__name__}.")

        self.pipeline_config = pipeline_config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.input_volume = self._load_input_volume(
            self.pipeline_config.volume_path,
            loader=self.pipeline_config.volume_loader,
        )
        self.output_dir = Path(self.pipeline_config.output_parent_dir) / f"result_sam2_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "segmentation.zarr"
        adapter_kwargs = {}
        if (
            self.pipeline_config.dataloader_downsample
            or self.pipeline_config.dataloader_batch_size != 256
            or self.pipeline_config.dataloader_persist
        ):
            adapter_kwargs = {
                "dataloader_downsample": self.pipeline_config.dataloader_downsample,
                "dataloader_batch_size": self.pipeline_config.dataloader_batch_size,
                "dataloader_persist": self.pipeline_config.dataloader_persist,
            }
        self.backbone_adapter = SAM2BackboneAdapter(
            backbone_config=self.pipeline_config.backbone_config,
            volume=self.input_volume,
            **adapter_kwargs,
        )
        self._init_logger()
        self._setup_output_writer()

    @staticmethod
    def _load_input_volume(volume_path, *, loader: str = "numpy"):
        """Load an array-backed volume while preserving the requested backend."""
        if loader not in {"numpy", "dask", "torch"}:
            raise ValueError(f"volume_loader must be 'numpy', 'dask', or 'torch', got {loader!r}.")
        if isinstance(volume_path, np.ndarray):
            return volume_path
        if isinstance(volume_path, torch.Tensor):
            return volume_path
        if isinstance(volume_path, da.Array):
            return volume_path
        if not isinstance(volume_path, (str, Path)):
            raise TypeError(
                "volume_path must be a path, numpy.ndarray, dask.array.Array, "
                f"or torch.Tensor, got {type(volume_path)!r}."
            )
        path = Path(volume_path)
        if path.suffix.lower() in {".pt", ".pth"}:
            try:
                volume = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # older torch versions
                volume = torch.load(path, map_location="cpu")
            if not isinstance(volume, torch.Tensor):
                raise TypeError(f"Expected {path} to contain a torch.Tensor, got {type(volume)!r}.")
            if volume.ndim != 3:
                raise ValueError(f"Expected a 3D tensor in (Z, Y, X) order, got {tuple(volume.shape)}.")
            return volume
        reader = VolumeReader(path)
        if loader == "dask":
            return reader.get_dask_array()
        if loader == "torch":
            return reader.get_torch_tensor()
        return reader.get_numpy_array()

    def run(self) -> None:
        # Prompt generation
        prompt_config = self.pipeline_config.prompt_generator_config
        prompt_generator = PromptGenerator(tomo=self.input_volume)
        min_peak_distance = (
            prompt_config.dist_val_thresh
            if prompt_config.min_peak_distance is None
            else prompt_config.min_peak_distance
        )
        if hasattr(prompt_generator, "generate_prompts"):
            prompt_coords, prompt_vals = prompt_generator.generate_prompts(
                method=prompt_config.method,
                crop_size=prompt_config.crop_size,
                crop_overlap=prompt_config.crop_overlap,
                bin_thresh=prompt_config.binarisation_threshold,
                min_peak_distance=min_peak_distance,
                process_device=self.pipeline_config.backbone_config.compute_device,
                show_progress=prompt_config.show_progress,
                return_peak_values=True,
                downsample=prompt_config.downsample,
                grid_shape=prompt_config.grid_shape,
                dist_map_use_n_threads=prompt_config.dist_map_use_n_threads,
                optimizer=prompt_config.optimizer,
                lr=prompt_config.lr,
                iterations=prompt_config.iterations,
                block_size=prompt_config.block_size,
                triton_num_warps=prompt_config.triton_num_warps,
                max_filter_size=prompt_config.max_filter_size,
            )
        else:  # Keep older injected prompt generators usable.
            prompt_coords, prompt_vals = prompt_generator.generate_peak_local_max_prompts(
                crop_size=prompt_config.crop_size,
                crop_overlap=prompt_config.crop_overlap,
                bin_thresh=prompt_config.binarisation_threshold,
                dist_val_thresh=min_peak_distance,
                max_filter_size=prompt_config.max_filter_size,
                show_progress=prompt_config.show_progress,
                return_max_values=True,
            )

        max_prompts = self.pipeline_config.max_prompts or len(prompt_coords)
        max_prompts = min(max_prompts, len(prompt_coords))
        id_dispatcher = ObjectIDDispatcher(start_id=1)
        prompt_loop_start = time.perf_counter()

        # Main `REPLAY` loop: replay SAM2 inference for each prompt
        for prompt_idx in range(max_prompts):
            self._log_prompt_progress(prompt_idx, max_prompts, prompt_loop_start)
            prompt_coord = tuple(int(coord) for coord in prompt_coords[prompt_idx])
            prompt_val = float(prompt_vals[prompt_idx])
            LOGGER.info(
                "Prompt %s: prompted location: %s with distance %s",
                prompt_idx + 1,
                prompt_coord,
                prompt_val,
            )

            if self._is_prompt_in_segmented_region(prompt_coord):
                LOGGER.warning("Prompt %s: %s already in segmented region, skipping.", prompt_idx + 1, prompt_coord)
                continue

            # Secondary `REPLAY` loop: run SAM2 inference 3 times for the current prompt
            adapter_result = self._run_multi_axes_inference(
                point_prompt=prompt_coord,
                dist_map_val=prompt_val,
            )
            adapter_result = self.postprocess_adapter_result(
                adapter_result=adapter_result,
                mask_size_threshold=self.pipeline_config.postprocess_mask_size_threshold,
                overlap_ratio_threshold=self.pipeline_config.postprocess_overlap_ratio_threshold,
            )
            if adapter_result is None:
                LOGGER.warning("Prompt %s: %s failed to generate a valid result, skipping.", prompt_idx + 1, prompt_coord)
                continue

            curr_obj_id = id_dispatcher.get()
            output_mask = adapter_result.volume.astype(np.int32) * curr_obj_id
            self.writer.write(
                data=output_mask,
                offset=adapter_result.offset,
                write_positive_regions_only=True,
            )
            id_dispatcher.increment()
            LOGGER.info("Prompt %s: Object %s written.", prompt_idx + 1, curr_obj_id)

        LOGGER.info(
            "Inference completed!\nNO. objects segmented: %s\nResults saved to: %s",
            id_dispatcher.get() - 1,
            self.output_path,
        )

    def run_with_prompt_points(
        self,
        positive_point_prompts: list[tuple[int, int, int]],
        negative_point_prompts: list[tuple[int, int, int]] | None = None,
    ) -> list[AdapterResult]:
        """Run one object inference from explicit positive and negative 3D points."""
        positive_point_prompts, negative_point_prompts = (
            SAM2BackboneAdapter._normalize_prompt_sets(
                positive_point_prompts=positive_point_prompts,
                negative_point_prompts=negative_point_prompts,
            )
        )
        volume_shape = tuple(int(size) for size in self.input_volume.shape)
        for name, points in (
            ("positive_point_prompts", positive_point_prompts),
            ("negative_point_prompts", negative_point_prompts),
        ):
            for point in points:
                if any(coord < 0 or coord >= volume_shape[axis] for axis, coord in enumerate(point)):
                    raise ValueError(
                        f"{name} contains point {point} outside volume shape {volume_shape}."
                    )

        adapter_result = self._run_multi_axes_inference_with_prompts(
            positive_point_prompts=positive_point_prompts,
            negative_point_prompts=negative_point_prompts,
        )
        adapter_result = self.postprocess_adapter_result(
            adapter_result=adapter_result,
            mask_size_threshold=self.pipeline_config.postprocess_mask_size_threshold,
            overlap_ratio_threshold=self.pipeline_config.postprocess_overlap_ratio_threshold,
        )
        if adapter_result is None:
            return []

        object_id = 1
        self.writer.write(
            data=adapter_result.volume.astype(np.int32) * object_id,
            offset=adapter_result.offset,
            write_positive_regions_only=True,
        )
        LOGGER.info("Explicit prompt object %s written.", object_id)
        return [adapter_result]

    def _init_logger(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(self.output_dir / "inference_log.log"),
                logging.StreamHandler(),
            ],
            force=True,
        )

    def _setup_output_writer(self) -> None:
        output_shape = (
            self.backbone_adapter.dataloader.get_crop_shape()
            if self.pipeline_config.dataloader_downsample
            else tuple(self.input_volume.shape)
        )
        self.writer = ZarrVolumeWriter(
            path=self.output_path,
            shape=output_shape,
            dtype=np.int32,
            chunks=tuple(min(128, int(size)) for size in output_shape),
            shards=tuple(min(512, int(size)) for size in output_shape),
            compression_level=1,
            fill_value=0,
            overwrite=True,
            attributes={
                "description": "Generated ReplaySAM segmentation masks.",
                "timestamp": datetime.now().isoformat(),
                "output_path": str(self.output_path),
                "pipeline_config": self.pipeline_config.as_dict(),
            },
        )

    def _is_prompt_in_segmented_region(self, prompt_coord: tuple[int, int, int]) -> bool:
        dataloader = getattr(self.backbone_adapter, "dataloader", None)
        active_coord = (
            dataloader.to_active_point(prompt_coord)
            if dataloader is not None
            else prompt_coord
        )
        active_coord = tuple(
            min(max(int(coord), 0), int(size) - 1)
            for coord, size in zip(active_coord, self.writer.shape, strict=True)
        )
        return bool(self.writer.zarr_array[active_coord] > 0)

    def _run_multi_axes_inference(
        self,
        *,
        point_prompt: tuple[int, int, int],
        dist_map_val: float | None,
    ) -> AdapterResult:
        """Call the adapter while tolerating older injected adapter shims."""
        kwargs = {
            "point_prompt": point_prompt,
            "dist_map_val": dist_map_val,
            "axes": self.pipeline_config.inference_axes,
            "majority_voting_threshold": self.pipeline_config.majority_voting_threshold,
        }
        try:
            return self.backbone_adapter.run_multi_axes_inference(
                **kwargs,
                consistency_score=self.pipeline_config.consistency_score,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'consistency_score'" not in str(exc):
                raise
            return self.backbone_adapter.run_multi_axes_inference(**kwargs)

    def _run_multi_axes_inference_with_prompts(
        self,
        *,
        positive_point_prompts: list[tuple[int, int, int]],
        negative_point_prompts: list[tuple[int, int, int]],
    ) -> AdapterResult:
        """Call explicit-prompt inference with the configured consistency gate."""
        kwargs = {
            "positive_point_prompts": positive_point_prompts,
            "negative_point_prompts": negative_point_prompts,
            "axes": self.pipeline_config.inference_axes,
            "majority_voting_threshold": self.pipeline_config.majority_voting_threshold,
        }
        try:
            return self.backbone_adapter.run_multi_axes_inference_with_prompts(
                **kwargs,
                consistency_score=self.pipeline_config.consistency_score,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'consistency_score'" not in str(exc):
                raise
            return self.backbone_adapter.run_multi_axes_inference_with_prompts(**kwargs)

    def _log_prompt_progress(
        self,
        prompt_idx: int,
        max_prompts: int,
        prompt_loop_start: float,
    ) -> None:
        elapsed_seconds = time.perf_counter() - prompt_loop_start
        seconds_per_prompt = elapsed_seconds / prompt_idx if prompt_idx > 0 else 0.0
        remaining_prompts = max_prompts - prompt_idx
        eta_seconds = remaining_prompts * seconds_per_prompt if seconds_per_prompt > 0 else None
        LOGGER.info(
            "%s Prompt %s / %s | %s<%s | %.2f s/prompt %s",
            "*" * 30,
            prompt_idx + 1,
            max_prompts,
            self._format_duration(elapsed_seconds),
            self._format_duration(eta_seconds),
            seconds_per_prompt,
            "*" * 30,
        )

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "--:--:--"
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def postprocess_adapter_result(
        self,
        adapter_result: AdapterResult | None,
        mask_size_threshold: int = 100,
        overlap_ratio_threshold: float = 0.5,
    ) -> AdapterResult | None:
        if adapter_result is None or adapter_result.volume.size == 0:
            return None
        offset = adapter_result.offset
        volume = adapter_result.volume
        if np.sum(volume) < mask_size_threshold:
            return None
        existing_region = self.writer.zarr_array[
            offset[0]:offset[0] + volume.shape[0],
            offset[1]:offset[1] + volume.shape[1],
            offset[2]:offset[2] + volume.shape[2],
        ] > 0
        if overlap_ratio(mask1=volume, mask2=existing_region) >= overlap_ratio_threshold:
            return None
        return adapter_result
