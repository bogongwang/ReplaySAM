from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import torch


@dataclass
class Config:
    def as_dict(self) -> dict[str, Any]:
        return self._json_safe(self)

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, Config):
            return {
                config_field.name: self._json_safe(getattr(value, config_field.name))
                for config_field in fields(value)
            }
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        return value


@dataclass
class PromptGeneratorConfig(Config):
    """Configuration for automatic 3D prompt generation.

    ``peak_local_max`` is the original CPU/CuPy implementation.  The
    ``blockwise_argmax`` method selects one distance-map maximum per 3D block
    and uses Triton when the processing device is CUDA.  A small PyTorch
    fallback is used on CPU so the method remains testable and usable without
    a GPU.
    """

    method: Literal["peak_local_max", "blockwise_argmax", "gradient_ascent"] = "blockwise_argmax"
    binarisation_threshold: int | float | None = None
    dist_val_thresh: float = 5.0
    min_peak_distance: float | None = None
    max_filter_size: int = 7
    crop_size: tuple[int, int, int] | None = (1024, 1024, 1024)
    crop_overlap: tuple[int, int, int] | None = (128, 128, 128)
    show_progress: bool = True

    # Optional gradient-ascent prompt generation settings.  They are kept
    # here for API compatibility with the full ReplaySAM implementation.
    downsample: bool = False
    grid_shape: tuple[int, int, int] | None = None
    dist_map_use_n_threads: int = 0
    optimizer: Literal["sgd", "adam"] = "sgd"
    lr: float = 0.1
    iterations: int = 2000

    # Triton blockwise-argmax settings.
    block_size: int = 12
    triton_num_warps: int = 4


@dataclass
class SAM2BackboneConfig(Config):
    model_size: str = "large"
    apply_postprocessing: bool = True
    compile: bool = False
    compute_device: torch.device = torch.device("cuda:0")
    storage_device: torch.device = torch.device("cpu")
    n_sample_points: int = 3
    pred_iou_thresh: float = 0.8
    termination_iou: float = 0.7
    termination_mask_size: int = 20

    # SAM crop-layer and zoomed/multi-scale inference settings.
    n_crop_layers: int = 0
    crop_layers_overlap_ratio: float = 0.5
    zoom_enabled: bool = False
    zoom_padding: int = 32
    zoom_min_crop_size: int = 128
    zoom_fallback_to_coarse: bool = True
    zoom_sizes: tuple[int, ...] = (256, 512, 1024, 1536, 2048)
    zoom_termination_mask_sizes: dict[int, int] = field(
        default_factory=lambda: {
            256: 4,
            512: 10,
            1024: 20,
            1536: 20,
            2048: 20,
        }
    )


@dataclass
class SAM2PipelineConfig(Config):
    volume_path: str | Path
    output_parent_dir: str | Path
    backbone_config: SAM2BackboneConfig = field(default_factory=SAM2BackboneConfig)
    prompt_generator_config: PromptGeneratorConfig = field(default_factory=PromptGeneratorConfig)
    max_prompts: int | None = None
    postprocess_mask_size_threshold: int = 100
    postprocess_overlap_ratio_threshold: float = 0.6
    inference_axes: tuple[int, ...] = (0, 1, 2)
    majority_voting_threshold: int = 2
    consistency_score: float | None = 0.7
    dataloader_downsample: bool = False
    dataloader_batch_size: int = 256
    dataloader_persist: bool = False
    volume_loader: Literal["numpy", "dask", "torch"] = "numpy"
    note: str | None = None
