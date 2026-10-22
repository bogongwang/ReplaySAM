from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import dask.array as da
from dask_image.ndinterp import affine_transform

from replaysam.utils.helpers import Crop3D


_IMAGE_MEAN = 0.485
_IMAGE_STD = 0.229


def _normalise_slice(
    image: Any,
    *,
    process_device: torch.device,
    storage_device: torch.device,
) -> torch.Tensor:
    """Convert one scalar slice to the normalized 3-channel SAM format."""
    tensor = torch.as_tensor(image, dtype=torch.float32, device=process_device)
    tensor = (tensor / 255.0 - _IMAGE_MEAN) / _IMAGE_STD
    if tensor.device != storage_device:
        tensor = tensor.to(storage_device)
    return tensor


def _resize_dask_block(block: da.Array, output_shape: tuple[int, int, int]) -> da.Array:
    """Resize a Dask volume lazily with endpoint-aligned linear interpolation."""
    if tuple(block.shape) == tuple(output_shape):
        return block

    scales = [
        0.0 if output_size == 1 else (input_size - 1) / (output_size - 1)
        for input_size, output_size in zip(block.shape, output_shape, strict=True)
    ]
    matrix = np.diag(scales)
    output_chunks = tuple(min(128, int(size)) for size in output_shape)
    return affine_transform(
        block,
        matrix=matrix,
        output_shape=output_shape,
        output_chunks=output_chunks,
        order=1,
    )


class SAM2DataLoader(ABC):
    """Common interface for SAM2 volumetric slice data loaders.

    Coordinates and shapes use ``(Z, Y, X)`` order. ``set_crop`` changes the
    active coordinate frame so adapters can run on a crop without materializing
    the full input volume.
    """

    @abstractmethod
    def get(self, axis: int, idx: int) -> torch.Tensor:
        """Return a normalized three-channel image for one volume slice."""

    @abstractmethod
    def get_curr_hw(self, axis: int) -> tuple[int, int]:
        """Return the native image height and width for an axis."""

    @abstractmethod
    def get_num_slices(self, axis: int) -> int:
        """Return the number of slices along an axis."""

    @abstractmethod
    def get_crop_shape(self) -> tuple[int, int, int]:
        """Return the shape of the active crop or full volume."""

    @abstractmethod
    def set_crop(self, crop: Crop3D | None = None) -> None:
        """Set or clear the active volume crop."""

    def to_active_point(self, point: tuple[int, int, int]) -> tuple[int, int, int]:
        """Map a point in the active input frame to the model frame.

        Eager loaders preserve resolution and therefore use the identity map.
        The Dask loader overrides this when optional 1024-cubed downsampling is
        enabled.
        """
        return tuple(int(coord) for coord in point)


class _ArrayDataLoader(SAM2DataLoader):
    """Shared crop metadata and validation for array-backed loaders."""

    def _set_crop_metadata(self, shape: tuple[int, int, int]) -> None:
        self.num_slices = [int(size) for size in shape]
        self.orig_hws = [
            (int(shape[1]), int(shape[2])),
            (int(shape[0]), int(shape[2])),
            (int(shape[0]), int(shape[1])),
        ]

    def get_curr_hw(self, axis: int) -> tuple[int, int]:
        self._validate_axis(axis)
        return self.orig_hws[axis]

    def get_num_slices(self, axis: int) -> int:
        self._validate_axis(axis)
        return self.num_slices[axis]

    def get_crop_shape(self) -> tuple[int, int, int]:
        return tuple(int(size) for size in self._block.shape)

    def _validate_axis_idx(self, axis: int, idx: int) -> None:
        self._validate_axis(axis)
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"Slice index must be an integer, got {type(idx)!r}.")
        if idx < 0 or idx >= self.num_slices[axis]:
            raise IndexError(
                f"Slice index {idx} out of range for axis {axis} "
                f"with {self.num_slices[axis]} slices."
            )

    @staticmethod
    def _validate_axis(axis: int) -> None:
        if axis not in (0, 1, 2):
            raise ValueError(f"Expected axis to be 0, 1, or 2, got {axis}.")


class SAM2NumPyDataLoader(_ArrayDataLoader):
    """Serve normalized SAM2 slices from a 3D NumPy volume."""

    downsample = False

    def __init__(
        self,
        data: np.ndarray,
        process_device: torch.device | str = torch.device("cpu"),
        storage_device: torch.device | str | None = None,
    ):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a numpy.ndarray, got {type(data)!r}.")
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D volume in (Z, Y, X) order, got {data.shape}.")
        if any(int(size) <= 0 for size in data.shape):
            raise ValueError(f"Volume dimensions must be positive, got {data.shape}.")

        self.process_device = torch.device(process_device)
        self.storage_device = (
            self.process_device
            if storage_device is None
            else torch.device(storage_device)
        )
        self.data_torch = torch.from_numpy(np.ascontiguousarray(data))
        self._block = self.data_torch
        self.crop: Crop3D | None = None
        self._set_crop_metadata(tuple(self._block.shape))

    def get(self, axis: int, idx: int) -> torch.Tensor:
        self._validate_axis_idx(axis, idx)
        if axis == 0:
            image = self._block[idx, :, :]
        elif axis == 1:
            image = self._block[:, idx, :]
        else:
            image = self._block[:, :, idx]
        image = _normalise_slice(
            image,
            process_device=self.process_device,
            storage_device=self.storage_device,
        )
        return image.repeat(3, 1, 1)

    def set_crop(self, crop: Crop3D | None = None) -> None:
        self.crop = crop
        self._block = self.data_torch if crop is None else self.data_torch[crop.slices]
        if any(int(size) <= 0 for size in self._block.shape):
            raise ValueError(f"Crop must contain at least one voxel, got {self._block.shape}.")
        self._set_crop_metadata(tuple(self._block.shape))


class SAM2TorchTensorDataLoader(_ArrayDataLoader):
    """Serve normalized SAM2 slices from a 3D PyTorch tensor."""

    downsample = False

    def __init__(
        self,
        data: torch.Tensor,
        storage_device: torch.device | str = torch.device("cpu"),
    ):
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be a torch.Tensor, got {type(data)!r}.")
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D tensor in (Z, Y, X) order, got {tuple(data.shape)}.")
        if any(int(size) <= 0 for size in data.shape):
            raise ValueError(f"Tensor dimensions must be positive, got {tuple(data.shape)}.")

        self.storage_device = torch.device(storage_device)
        self.data = data.detach().to(self.storage_device)
        self._block = self.data
        self.crop: Crop3D | None = None
        self._set_crop_metadata(tuple(self._block.shape))

    def get(self, axis: int, idx: int) -> torch.Tensor:
        self._validate_axis_idx(axis, idx)
        if axis == 0:
            image = self._block[idx, :, :]
        elif axis == 1:
            image = self._block[:, idx, :]
        else:
            image = self._block[:, :, idx]
        image = _normalise_slice(
            image,
            process_device=self.storage_device,
            storage_device=self.storage_device,
        )
        return image.repeat(3, 1, 1)

    def set_crop(self, crop: Crop3D | None = None) -> None:
        self.crop = crop
        self._block = self.data if crop is None else self.data[crop.slices]
        if any(int(size) <= 0 for size in self._block.shape):
            raise ValueError(f"Crop must contain at least one voxel, got {self._block.shape}.")
        self._set_crop_metadata(tuple(self._block.shape))


class SAM2DaskDataLoader(_ArrayDataLoader):
    """Serve normalized SAM2 slices lazily from a 3D Dask array.

    Only the requested slice is computed. A small per-axis cache avoids
    recomputing a slice when SAM2 asks for it again during propagation.
    """

    def __init__(
        self,
        data: da.Array,
        *,
        downsample: bool = False,
        batch_size: int = 256,
        persist: bool = False,
        process_device: torch.device | str = torch.device("cpu"),
        storage_device: torch.device | str = torch.device("cpu"),
        cache_size: int = 32,
    ):
        if not isinstance(data, da.Array):
            raise TypeError(f"data must be a dask.array.Array, got {type(data)!r}.")
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D Dask array in (Z, Y, X) order, got {data.shape}.")
        if any(int(size) <= 0 for size in data.shape):
            raise ValueError(f"Dask volume dimensions must be positive, got {data.shape}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}.")

        self.data = data
        self.downsample = bool(downsample)
        self.batch_size = int(batch_size)
        self.persist = bool(persist)
        self.process_device = torch.device(process_device)
        self.storage_device = torch.device(storage_device)
        self.cache_size = int(cache_size)
        self.crop: Crop3D | None = None
        self._cache: dict[tuple[int, int], torch.Tensor] = {}
        self._block = self._prepare_block(self.data)
        self._set_crop_metadata(tuple(self._block.shape))

        if self.persist:
            self._block = self._block.persist()

    def _prepare_block(self, block: da.Array) -> da.Array:
        if self.downsample:
            return _resize_dask_block(block, (1024, 1024, 1024))
        return block

    def get(self, axis: int, idx: int) -> torch.Tensor:
        self._validate_axis_idx(axis, idx)
        key = (axis, int(idx))
        image = self._cache.get(key)
        if image is None:
            axis_block = da.moveaxis(self._block, axis, 0)
            computed = axis_block[idx].compute()
            image = _normalise_slice(
                computed,
                process_device=self.process_device,
                storage_device=self.storage_device,
            ).detach()
            self._cache[key] = image
            while len(self._cache) > self.cache_size:
                self._cache.pop(next(iter(self._cache)))
        return image.repeat(3, 1, 1)

    def set_crop(self, crop: Crop3D | None = None) -> None:
        self._cache.clear()
        self.crop = crop
        raw_block = self.data if crop is None else self.data[crop.slices]
        if any(int(size) <= 0 for size in raw_block.shape):
            raise ValueError(f"Crop must contain at least one voxel, got {raw_block.shape}.")
        self._block = self._prepare_block(raw_block)
        if any(int(size) <= 0 for size in self._block.shape):
            raise ValueError(f"Crop must contain at least one voxel, got {self._block.shape}.")
        self._set_crop_metadata(tuple(self._block.shape))
        if self.persist:
            self._block = self._block.persist()

    def to_active_point(self, point: tuple[int, int, int]) -> tuple[int, int, int]:
        if not self.downsample:
            return super().to_active_point(point)
        source_shape = tuple(
            int(size) for size in (self.crop.shape if self.crop is not None else self.data.shape)
        )
        active_shape = self.get_crop_shape()
        mapped = []
        for coord, source_size, active_size in zip(point, source_shape, active_shape, strict=True):
            if source_size <= 1 or active_size <= 1:
                mapped.append(0)
            else:
                mapped.append(
                    int(round(int(coord) * (active_size - 1) / (source_size - 1)))
                )
        return tuple(mapped)


__all__ = [
    "SAM2DataLoader",
    "SAM2DaskDataLoader",
    "SAM2NumPyDataLoader",
    "SAM2TorchTensorDataLoader",
]
