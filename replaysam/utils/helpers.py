from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Crop3D:
    start_z: int
    start_y: int
    start_x: int
    end_z: int
    end_y: int
    end_x: int

    @property
    def starts(self) -> tuple[int, int, int]:
        return self.start_z, self.start_y, self.start_x

    @property
    def ends(self) -> tuple[int, int, int]:
        return self.end_z, self.end_y, self.end_x

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.start_z, self.end_z),
            slice(self.start_y, self.end_y),
            slice(self.start_x, self.end_x),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            self.end_z - self.start_z,
            self.end_y - self.start_y,
            self.end_x - self.start_x,
        )


class LRUDict(OrderedDict):
    def __init__(self, max_size: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


class LimitedOrderedDict(OrderedDict):
    def __init__(self, max_size: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def add(self, key, value):
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)


class ObjectIDDispatcher:
    def __init__(self, start_id: int = 0):
        self.id = np.int32(start_id)

    def increment(self) -> None:
        self.id += 1

    def get(self) -> np.int32:
        return self.id


def resize_2d(
    image: torch.Tensor | np.ndarray,
    size: Tuple[int, int] | torch.Size,
    mode: str = "bilinear",
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor or numpy.ndarray.")
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)
    if image.shape[-2:] == tuple(size):
        return image
    return F.interpolate(
        image.float(),
        size=tuple(size),
        mode=mode,
        align_corners=align_corners,
    )


def create_random_points(x: float, y: float, r: float, n: int) -> np.ndarray:
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = r * np.sqrt(np.random.uniform(0, 1, n))
    xs = x + radii * np.cos(angles)
    ys = y + radii * np.sin(angles)
    return np.column_stack((xs, ys))


def pad_to_orig_size(
    cropped_im: torch.Tensor,
    original_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> torch.Tensor:
    h, w = original_size
    x0, y0, x1, y1 = crop_box
    if cropped_im.shape[-2:] != (y1 - y0, x1 - x0):
        raise ValueError("Cropped image size does not match crop box dimensions.")
    if (y1 - y0 == h) and (x1 - x0 == w):
        return cropped_im
    padded = torch.zeros(
        (*cropped_im.shape[:-2], h, w),
        dtype=cropped_im.dtype,
        device=cropped_im.device,
    )
    padded[..., y0:y1, x0:x1] = cropped_im
    return padded


def overlap_ratio(mask1: np.ndarray, mask2: np.ndarray, eps: float = 1e-6) -> float:
    if mask1.shape != mask2.shape:
        raise ValueError(f"Shape mismatch: {mask1.shape} != {mask2.shape}")
    mask1 = np.ma.filled(mask1, fill_value=0).astype(bool)
    mask2 = np.ma.filled(mask2, fill_value=0).astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    size = mask1.sum()
    return float((intersection + eps) / (size + eps))
