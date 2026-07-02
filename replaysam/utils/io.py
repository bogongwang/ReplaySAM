from __future__ import annotations

from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import torch
import zarr
from zarr.codecs import BloscCodec

PathLike = str | Path
SUPPORTED_EXTENSIONS = (".zarr", ".nc", ".npy", ".tif", ".tiff", ".nii", ".nii.gz")

class VolumeReader:
    """Read supported volume-like array files into Dask, NumPy, or torch."""

    def __init__(
        self,
        path: PathLike,
        *,
        chunks: Any = "auto",
        variable: str | None = None,
        dtype: Any | None = None,
    ):
        self.path = Path(path)
        self.chunks = chunks
        self.variable = variable
        self.dtype = dtype

    def get_dask_array(self) -> da.Array:
        extension = _normalised_extension(self.path)
        if extension == ".zarr":
            array = load_zarr_as_dask(self.path, chunks=self.chunks)
        elif extension == ".nc":
            array = load_netcdf_as_dask(
                self.path,
                chunks=self.chunks,
                variable=self.variable,
            )
        elif extension == ".npy":
            array = load_npy_as_dask(self.path, chunks=self.chunks)
        elif extension in {".tif", ".tiff"}:
            array = load_tiff_as_dask(self.path, chunks=self.chunks)
        elif extension in {".nii", ".nii.gz"}:
            array = load_nifti_as_dask(self.path, chunks=self.chunks)
        else:
            supported = ", ".join(SUPPORTED_EXTENSIONS)
            raise ValueError(
                f"Unsupported array file extension {extension!r} for {self.path}. "
                f"Supported extensions: {supported}."
            )

        if self.dtype is not None:
            array = array.astype(self.dtype)
        return array

    def get_numpy_array(self) -> np.ndarray:
        return np.asarray(self.get_dask_array().compute())

    def get_torch_tensor(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(self.get_numpy_array())
        if dtype is not None or device is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor

class ZarrVolumeWriter:
    """Create and write volume data to a Zarr array."""

    def __init__(
        self,
        path: PathLike,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
        *,
        chunks: tuple[int, ...] = (128, 128, 128),
        shards: tuple[int, ...] = (512, 512, 512),
        compressor_name: str | None = "zstd",
        compression_level: int = 0,
        fill_value: Any = 0,
        overwrite: bool = False,
        attributes: dict[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if shape is None and dtype is None and self.path.exists() and not overwrite:
            self.zarr_array = zarr.open_array(str(self.path), mode="r+")
        else:
            if shape is None or dtype is None:
                raise ValueError("shape and dtype are required when creating a Zarr array.")
            create_kwargs: dict[str, Any] = {
                "store": str(self.path),
                "shape": tuple(shape),
                "dtype": np.dtype(dtype),
                "chunks": chunks,
                "fill_value": fill_value,
                "overwrite": overwrite,
                "attributes": attributes,
            }
            if shards is not None:
                create_kwargs["shards"] = shards
            if compressor_name is not None:
                create_kwargs["compressors"] = [
                    BloscCodec(cname=compressor_name, clevel=compression_level)
                ]
            self.zarr_array = zarr.create_array(**create_kwargs)

        self.chunks = tuple(self.zarr_array.chunks)
        self.shards = tuple(self.zarr_array.shards) if self.zarr_array.shards else None
        self.write_chunks = self.shards or self.chunks

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.zarr_array.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.zarr_array.dtype)

    def write(
        self,
        data: np.ndarray | da.Array | torch.Tensor,
        *,
        offset: tuple[int, ...] | None = None,
        write_positive_regions_only: bool = False,
    ) -> None:
        """Write a full array or an offset region into the Zarr volume."""
        data = _normalise_write_data(data)
        region = self._region_for(data.shape, offset)

        if isinstance(data, da.Array):
            self._write_dask(data, region, write_positive_regions_only)
        else:
            self._write_numpy(data, region, write_positive_regions_only)

    def write_region(
        self,
        data: np.ndarray | da.Array | torch.Tensor,
        *,
        offset: tuple[int, ...],
        write_positive_regions_only: bool = False,
    ) -> None:
        """Write an offset region into the Zarr volume."""
        self.write(
            data,
            offset=offset,
            write_positive_regions_only=write_positive_regions_only,
        )

    def update_attributes(self, attributes: dict[str, Any]) -> None:
        self.zarr_array.attrs.update(attributes)

    def _region_for(
        self,
        data_shape: tuple[int, ...],
        offset: tuple[int, ...] | None,
    ) -> tuple[slice, ...]:
        if offset is None:
            offset = (0,) * len(data_shape)
        if len(data_shape) != len(self.shape):
            raise ValueError(
                f"Data rank {len(data_shape)} does not match Zarr rank {len(self.shape)}."
            )
        if len(offset) != len(self.shape):
            raise ValueError(f"Offset must have {len(self.shape)} values, got {offset!r}.")
        if any(start < 0 for start in offset):
            raise ValueError(f"Offset values must be non-negative, got {offset!r}.")

        end = tuple(start + size for start, size in zip(offset, data_shape, strict=True))
        if any(stop > limit for stop, limit in zip(end, self.shape, strict=True)):
            raise ValueError(
                f"Write region {tuple(zip(offset, end, strict=True))} exceeds "
                f"Zarr shape {self.shape}."
            )
        return tuple(slice(start, stop) for start, stop in zip(offset, end, strict=True))

    def _write_numpy(
        self,
        data: np.ndarray,
        region: tuple[slice, ...],
        write_positive_regions_only: bool,
    ) -> None:
        data = data.astype(self.dtype, copy=False)
        if write_positive_regions_only:
            mask = data > 0
            if not np.any(mask):
                return
            current = self.zarr_array[region]
            data = np.where(mask, data, current)
        self.zarr_array[region] = data

    def _write_dask(
        self,
        data: da.Array,
        region: tuple[slice, ...],
        write_positive_regions_only: bool,
    ) -> None:
        data = data.astype(self.dtype).rechunk(self.write_chunks)
        if write_positive_regions_only:
            current = da.from_array(self.zarr_array[region], chunks=self.write_chunks)
            data = da.where(data > 0, data, current)

        with dask.config.set({"array.chunk-size": self._write_chunk_nbytes()}):
            data.to_zarr(self.zarr_array, region=region, compute=True)

    def _write_chunk_nbytes(self) -> int:
        return int(np.prod(self.write_chunks) * self.dtype.itemsize)

def load_zarr_as_dask(path: PathLike, *, chunks: Any = "auto") -> da.Array:
    """Load a Zarr array as a Dask array."""
    return da.from_zarr(str(path), chunks=chunks)

def load_netcdf_as_dask(
    path: PathLike,
    *,
    chunks: Any = "auto",
    variable: str | None = None,
) -> da.Array:
    """Load a NetCDF variable as a Dask array."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Reading NetCDF files requires xarray and h5netcdf. "
            "Install the project dependencies before loading .nc files."
        ) from exc

    open_chunks = None if _is_sequence_chunks(chunks) else chunks
    dataset = xr.open_dataset(path, chunks=open_chunks, engine="h5netcdf")
    data_array = _select_netcdf_variable(dataset, variable)
    data_array = _chunk_netcdf_variable(data_array, chunks)
    data = data_array.data
    if not isinstance(data, da.Array):
        data = da.from_array(data, chunks=chunks)
    return data

def load_npy_as_dask(path: PathLike, *, chunks: Any = "auto") -> da.Array:
    """Load a .npy file as a Dask array using NumPy memory mapping."""
    array = np.load(path, mmap_mode="r")
    return da.from_array(array, chunks=chunks)


def load_tiff_as_dask(path: PathLike, *, chunks: Any = "auto") -> da.Array:
    """Load a TIFF stack as a Dask array."""
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "Reading TIFF files requires tifffile. "
            "Install the project dependencies before loading .tif or .tiff files."
        ) from exc

    try:
        array = tifffile.memmap(path)
    except Exception:
        array = tifffile.imread(path)
    return da.from_array(array, chunks=chunks)


def load_nifti_as_dask(path: PathLike, *, chunks: Any = "auto") -> da.Array:
    """Load a NIfTI image as a Dask array."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Reading NIfTI files requires nibabel. "
            "Install the project dependencies before loading .nii or .nii.gz files."
        ) from exc

    image = nib.load(str(path))
    return da.from_array(image.dataobj, chunks=chunks)

def read_dask_array(path: PathLike, **kwargs: Any) -> da.Array:
    return VolumeReader(path, **kwargs).get_dask_array()


def read_numpy_array(path: PathLike, **kwargs: Any) -> np.ndarray:
    return VolumeReader(path, **kwargs).get_numpy_array()


def read_torch_tensor(
    path: PathLike,
    *,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    return VolumeReader(path, **kwargs).get_torch_tensor(
        device=device,
        dtype=torch_dtype,
    )

def write_zarr(
    path: PathLike,
    data: np.ndarray | da.Array | torch.Tensor,
    *,
    chunks: tuple[int, ...] = (128, 128, 128),
    shards: tuple[int, ...] = (512, 512, 512),
    compressor_name: str | None = "zstd",
    compression_level: int = 0,
    fill_value: Any = 0,
    overwrite: bool = False,
    attributes: dict[str, Any] | None = None,
) -> ZarrVolumeWriter:
    data = _normalise_write_data(data)
    writer = ZarrVolumeWriter(
        path,
        shape=tuple(data.shape),
        dtype=data.dtype,
        chunks=chunks,
        shards=shards,
        compressor_name=compressor_name,
        compression_level=compression_level,
        fill_value=fill_value,
        overwrite=overwrite,
        attributes=attributes,
    )
    writer.write(data)
    return writer

def _normalise_write_data(data: np.ndarray | da.Array | torch.Tensor) -> np.ndarray | da.Array:
    if isinstance(data, da.Array):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    raise TypeError(
        "ZarrVolumeWriter supports numpy.ndarray, dask.array.Array, or torch.Tensor data."
    )

def _normalised_extension(path: PathLike) -> str:
    path = Path(path)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if len(suffixes) >= 2 and suffixes[-2:] == [".nii", ".gz"]:
        return ".nii.gz"
    return path.suffix.lower()

def _format_array_candidates(dataset: Any) -> str:
    candidates = []
    for name, data_array in dataset.data_vars.items():
        candidates.append(f"{name}: shape={tuple(data_array.shape)}")
    return ", ".join(candidates) if candidates else "no data variables"

def _select_netcdf_variable(dataset: Any, variable: str | None) -> Any:
    if variable is not None:
        if variable not in dataset.data_vars:
            candidates = _format_array_candidates(dataset)
            raise ValueError(
                f"NetCDF variable {variable!r} was not found. Available variables: {candidates}."
            )
        return dataset[variable]

    for data_array in dataset.data_vars.values():
        if data_array.ndim == 3:
            return data_array

    candidates = _format_array_candidates(dataset)
    raise ValueError(
        "NetCDF file does not contain a 3D data variable. "
        f"Available variables: {candidates}."
    )


def _is_sequence_chunks(chunks: Any) -> bool:
    return isinstance(chunks, (tuple, list))


def _chunk_netcdf_variable(data_array: Any, chunks: Any) -> Any:
    if not _is_sequence_chunks(chunks):
        return data_array

    if len(chunks) != data_array.ndim:
        raise ValueError(
            "Tuple/list chunks for NetCDF must match the selected variable rank. "
            f"Got chunks={chunks!r} for shape={tuple(data_array.shape)}."
        )

    return data_array.chunk(dict(zip(data_array.dims, chunks, strict=True)))
