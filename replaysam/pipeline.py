from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from replaysam.models.adapters import AdapterResult, SAM2BackboneAdapter
from replaysam.utils.configs import SAM2PipelineConfig
from replaysam.utils.helpers import ObjectIDDispatcher, overlap_ratio
from replaysam.utils.io import VolumeReader, ZarrVolumeWriter
from replaysam.utils.prompt_generator import PromptGenerator

LOGGER = logging.getLogger(__name__)


class Pipeline:
    """SAM2-only particle segmentation pipeline."""

    def __init__(self, config: SAM2PipelineConfig):
        if not isinstance(config, SAM2PipelineConfig):
            raise TypeError(f"Pipeline only supports SAM2PipelineConfig, got {type(config).__name__}.")

        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.input_volume = VolumeReader(config.volume_path).get_numpy_array()
        self.output_dir = Path(config.output_parent_dir) / f"result_sam2_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "segmentation.zarr"
        self.backbone_adapter = SAM2BackboneAdapter(
            backbone_config=self.config.backbone_config,
            volume=self.input_volume,
        )
        self._init_logger()
        self._setup_output_writer()

    def run(self) -> None:
        prompt_config = self.config.prompt_generator_config
        prompt_generator = PromptGenerator(tomo=self.input_volume)
        prompt_coords, prompt_vals = prompt_generator.generate_peak_local_max_prompts(
            crop_size=prompt_config.crop_size,
            crop_overlap=prompt_config.crop_overlap,
            bin_thresh=prompt_config.binarisation_threshold,
            dist_val_thresh=prompt_config.dist_val_thresh,
            max_filter_size=prompt_config.max_filter_size,
            show_progress=True,
            return_max_values=True,
        )

        max_prompts = self.config.max_prompts or len(prompt_coords)
        max_prompts = min(max_prompts, len(prompt_coords))
        id_dispatcher = ObjectIDDispatcher(start_id=1)
        prompt_loop_start = time.perf_counter()

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

            adapter_result = self.backbone_adapter.run_multi_axes_inference(
                point_prompt=prompt_coord,
                dist_map_val=prompt_val,
                axes=self.config.inference_axes,
                majority_voting_threshold=self.config.majority_voting_threshold,
            )
            adapter_result = self.postprocess_adapter_result(
                adapter_result=adapter_result,
                mask_size_threshold=self.config.postprocess_mask_size_threshold,
                overlap_ratio_threshold=self.config.postprocess_overlap_ratio_threshold,
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
        self.writer = ZarrVolumeWriter(
            path=self.output_path,
            shape=tuple(self.input_volume.shape),
            dtype=np.int32,
            chunks=tuple(min(128, int(size)) for size in self.input_volume.shape),
            shards=tuple(min(512, int(size)) for size in self.input_volume.shape),
            compression_level=1,
            fill_value=0,
            overwrite=True,
            attributes={
                "description": "Generated ReplaySAM segmentation masks.",
                "timestamp": datetime.now().isoformat(),
                "output_path": str(self.output_path),
                "pipeline_config": self.config.as_dict(),
            },
        )

    def _is_prompt_in_segmented_region(self, prompt_coord: tuple[int, int, int]) -> bool:
        return bool(self.writer.zarr_array[prompt_coord] > 0)

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
