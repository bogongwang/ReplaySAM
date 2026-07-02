from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

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
    binarisation_threshold: int | float | None = None
    dist_val_thresh: float = 5.0
    max_filter_size: int = 7
    crop_size: tuple[int, int, int] | None = (1024, 1024, 1024)
    crop_overlap: tuple[int, int, int] | None = (128, 128, 128)


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


@dataclass
class SAM2PipelineConfig(Config):
    volume_path: str | Path
    output_parent_dir: str | Path
    backbone_config: SAM2BackboneConfig = field(default_factory=SAM2BackboneConfig)
    prompt_generator_config: PromptGeneratorConfig = field(default_factory=PromptGeneratorConfig)
    max_prompts: int | None = None
    postprocess_mask_size_threshold: int = 100
    postprocess_overlap_ratio_threshold: float = 0.6
    inference_axes: tuple[int, int, int] = (0, 1, 2)
    majority_voting_threshold: int = 2
    note: str | None = None
