from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class SAM2DataLoader(ABC):
    """Common interface for SAM2 volumetric slice data loaders."""

    @abstractmethod
    def get(self, axis: int, idx: int) -> torch.Tensor:
        """Return a normalized three-channel image for one volume slice."""

    @abstractmethod
    def get_curr_hw(self, axis: int) -> tuple[int, int]:
        """Return the native image height and width for an axis."""

    @abstractmethod
    def get_num_slices(self, axis: int) -> int:
        """Return the number of slices along an axis."""


class SAM2NumPyDataLoader(SAM2DataLoader):
    """Serve normalized SAM2 slices from a 3D NumPy volume."""

    def __init__(
        self,
        data: np.ndarray,
        process_device: torch.device = torch.device("cpu"),
    ):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a numpy.ndarray, got {type(data)!r}.")
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D volume in (Z, Y, X) order, got {data.shape}.")

        self.process_device = torch.device(process_device)
        self.data_torch = torch.from_numpy(np.asarray(data))
        self._set_metadata(tuple(self.data_torch.shape))

    def get(self, axis: int, idx: int) -> torch.Tensor:
        self._validate_axis_idx(axis, idx)
        if axis == 0:
            tensor = self.data_torch[idx, :, :]
        elif axis == 1:
            tensor = self.data_torch[:, idx, :]
        else:
            tensor = self.data_torch[:, :, idx]

        tensor = tensor.to(device=self.process_device, dtype=torch.bfloat16)
        tensor = tensor / 255.0
        tensor = (tensor - 0.485) / 0.229
        return tensor.repeat(3, 1, 1)

    def get_curr_hw(self, axis: int) -> tuple[int, int]:
        self._validate_axis(axis)
        return self.orig_hws[axis]

    def get_num_slices(self, axis: int) -> int:
        self._validate_axis(axis)
        return self.num_slices[axis]

    def _set_metadata(self, shape: tuple[int, int, int]) -> None:
        self.num_slices = [shape[0], shape[1], shape[2]]
        self.orig_hws = [
            (shape[1], shape[2]),
            (shape[0], shape[2]),
            (shape[0], shape[1]),
        ]

    def _validate_axis_idx(self, axis: int, idx: int) -> None:
        self._validate_axis(axis)
        if idx < 0 or idx >= self.num_slices[axis]:
            raise IndexError(
                f"Slice index {idx} out of range for axis {axis} "
                f"with {self.num_slices[axis]} slices."
            )

    @staticmethod
    def _validate_axis(axis: int) -> None:
        if axis not in (0, 1, 2):
            raise ValueError(f"Expected axis to be 0, 1, or 2, got {axis}.")
