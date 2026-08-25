import logging
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from utils import asinh_normalize

mp.set_sharing_strategy("file_system")


H5_IMAGES_NAME = "images"
H5_PRELOAD_BLOCK_BYTES = 512 * 1024**2


def _metadata_path(data_dir, slug, split):
    if split is None:
        return data_dir / "info.csv"
    if not slug:
        raise ValueError("A split slug is required when loading a dataset split.")
    return data_dir / "splits" / f"{slug}-{split}.csv"


def _load_hdf5_metadata(data_dir, slug, split, label_col, load_labels):
    """Load only metadata used to address and label HDF5 image rows."""
    catalog = _metadata_path(data_dir, slug, split)
    required_columns = ["object_id", "h5_index"]
    if load_labels:
        required_columns.append(label_col)
    required_columns = list(dict.fromkeys(required_columns))
    required_set = set(required_columns)

    metadata = pd.read_csv(
        catalog,
        usecols=lambda column: column in required_set,
        dtype={column: "string" for column in required_columns},
    )
    missing_columns = required_set.difference(metadata.columns)
    if missing_columns:
        raise KeyError(
            f"Metadata CSV {catalog} is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )
    return metadata


def _validate_h5_images(images, h5_path):
    if getattr(images, "ndim", None) != 4:
        raise ValueError(
            "HDF5 dataset 'images' must have shape (N, C, H, W); "
            f"received {getattr(images, 'shape', None)} in {h5_path}"
        )
    if any(dimension <= 0 for dimension in images.shape):
        raise ValueError(
            "HDF5 dataset 'images' must have non-empty N, C, H, and W "
            f"dimensions; received {images.shape} in {h5_path}"
        )
    if np.dtype(images.dtype) != np.dtype(np.float32):
        raise ValueError(
            "HDF5 dataset 'images' must use float32 values; "
            f"received {images.dtype} in {h5_path}"
        )
    return tuple(int(dimension) for dimension in images.shape)


def preload_h5_images(
    data_dir,
    *,
    normalization_kwargs,
    device,
    block_bytes=H5_PRELOAD_BLOCK_BYTES,
):
    """Read and normalize all HDF5 images into one C-contiguous array."""
    if block_bytes <= 0:
        raise ValueError("block_bytes must be greater than zero.")
    if not normalization_kwargs or not {
        "vmin",
        "vmax",
        "softening",
    }.issubset(normalization_kwargs):
        raise ValueError(
            "HDF5 preload requires fixed vmin/vmax/softening normalization."
        )
    device = torch.device(device)

    h5_path = Path(data_dir) / "tensors" / "tensors.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"HDF5 dataset not found at {h5_path}. "
            "Please run create_cutouts.py first."
        )

    started = time.monotonic()
    with h5py.File(
        h5_path,
        "r",
        swmr=True,
        rdcc_nbytes=16 * 1024**2,
        rdcc_w0=1.0,
    ) as h5_file:
        if H5_IMAGES_NAME not in h5_file:
            raise KeyError(f"HDF5 dataset 'images' not found in {h5_path}")
        images = h5_file[H5_IMAGES_NAME]
        if not isinstance(images, h5py.Dataset):
            raise TypeError(
                f"HDF5 object 'images' is not a dataset in {h5_path}"
            )

        image_shape = _validate_h5_images(images, h5_path)
        image_bytes = (
            int(np.prod(image_shape, dtype=np.int64)) * images.dtype.itemsize
        )
        logging.info(
            "Preloading and normalizing complete HDF5 image tensor: "
            "shape=%s size=%.2f GiB device=%s path=%s",
            image_shape,
            image_bytes / 2**30,
            device,
            h5_path,
        )

        preloaded = np.empty(image_shape, dtype=images.dtype, order="C")
        row_bytes = (
            int(np.prod(image_shape[1:], dtype=np.int64))
            * images.dtype.itemsize
        )
        storage_chunk_rows = int(images.chunks[0]) if images.chunks else 1
        target_rows = max(1, block_bytes // row_bytes)
        rows_per_block = max(
            storage_chunk_rows,
            (target_rows // storage_chunk_rows) * storage_chunk_rows,
        )

        last_reported_decile = 0
        with torch.inference_mode():
            for start in range(0, image_shape[0], rows_per_block):
                stop = min(start + rows_per_block, image_shape[0])
                selection = (
                    slice(start, stop),
                    slice(None),
                    slice(None),
                    slice(None),
                )
                images.read_direct(
                    preloaded,
                    source_sel=selection,
                    dest_sel=selection,
                )

                host_block = torch.from_numpy(preloaded[start:stop])
                device_block = host_block.to(device)
                normalized_block = asinh_normalize(
                    device_block,
                    **normalization_kwargs,
                )
                host_block.copy_(normalized_block)
                del normalized_block, device_block, host_block

                completed_decile = min(10, (stop * 10) // image_shape[0])
                if completed_decile > last_reported_decile:
                    logging.info(
                        "HDF5 preload and normalization progress: %d%%",
                        completed_decile * 10,
                    )
                    last_reported_decile = completed_decile

    if device.type == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.monotonic() - started
    logging.info(
        "Completed HDF5 preload and normalization in %.1f seconds "
        "(%.2f GiB/s).",
        elapsed,
        image_bytes / 2**30 / max(elapsed, 1e-9),
    )
    return preloaded


class HDF5Dataset(Dataset):
    """Read catalog-selected tensors from HDF5 or a shared memory preload."""

    def __init__(
            self,
            data_dir,
            label_col="class",
            slug=None,
            split=None,
            load_labels=True,
    ):
        self.data_dir = Path(data_dir)

        self.data_info = _load_hdf5_metadata(
            self.data_dir,
            slug,
            split,
            label_col,
            load_labels,
        )

        if load_labels:
            label_info_path = self.data_dir / "labels.csv"
            if label_info_path.is_file():
                label_df = pd.read_csv(
                    label_info_path,
                    dtype={"key": "string", "value": "Int64"},
                )
                label_mapping = dict(zip(label_df["key"], label_df["value"]))
                self.labels = np.asarray(
                    [label_mapping[value] for value in self.data_info[label_col]],
                    dtype=np.int64,
                )
            else:
                self.labels = np.asarray(self.data_info[label_col])
        else:
            self.labels = np.ones(len(self.data_info), dtype=int)

        self.h5_path = self.data_dir / "tensors" / "tensors.h5"
        self.h5_file = None
        self.h5_images = None

        if not self.h5_path.is_file():
            raise FileNotFoundError(
                f"HDF5 dataset not found at {self.h5_path}. "
                "Please run create_cutouts.py first."
            )

        h5_index = self.data_info["h5_index"]
        numeric_h5_index = pd.to_numeric(h5_index, errors="coerce")
        invalid_mask = numeric_h5_index.isna() | ~np.isfinite(numeric_h5_index)
        invalid_mask |= numeric_h5_index % 1 != 0
        invalid_mask |= numeric_h5_index < np.iinfo(np.int64).min
        invalid_mask |= numeric_h5_index > np.iinfo(np.int64).max
        if invalid_mask.any():
            invalid_rows = self.data_info.index[invalid_mask].tolist()[:5]
            raise ValueError(
                "Metadata column 'h5_index' must contain finite integers; "
                f"invalid CSV row index(es): {invalid_rows}"
            )

        self.h5_indices = numeric_h5_index.to_numpy(dtype=np.int64)
        self.data_info["h5_index"] = self.h5_indices
        negative_mask = self.h5_indices < 0
        if negative_mask.any():
            invalid_rows = self.data_info.index[negative_mask].tolist()[:5]
            raise ValueError(
                "Metadata column 'h5_index' cannot contain negative values; "
                f"invalid CSV row index(es): {invalid_rows}"
            )

        duplicated_mask = pd.Series(self.h5_indices).duplicated(keep=False)
        if duplicated_mask.any():
            duplicate_values = np.unique(self.h5_indices[duplicated_mask])[:5].tolist()
            raise ValueError(
                "Metadata column 'h5_index' must map each row to a unique image; "
                f"duplicate value(s): {duplicate_values}"
            )

        with h5py.File(self.h5_path, "r") as h5_file:
            if H5_IMAGES_NAME not in h5_file:
                raise KeyError(f"HDF5 dataset 'images' not found in {self.h5_path}")
            images = h5_file[H5_IMAGES_NAME]
            if not isinstance(images, h5py.Dataset):
                raise TypeError(
                    f"HDF5 object 'images' is not a dataset in {self.h5_path}"
                )
            self.h5_image_shape = _validate_h5_images(images, self.h5_path)
            self.h5_image_dtype = np.dtype(images.dtype)
            image_count = self.h5_image_shape[0]

        out_of_bounds_mask = self.h5_indices >= image_count
        if out_of_bounds_mask.any():
            invalid_rows = self.data_info.index[out_of_bounds_mask].tolist()[:5]
            invalid_values = self.h5_indices[out_of_bounds_mask][:5].tolist()
            raise IndexError(
                f"Metadata 'h5_index' exceeds HDF5 image count ({image_count}); "
                f"CSV row index(es) {invalid_rows} contain value(s) {invalid_values}"
            )

        logging.info(
            "Initialized HDF5 dataset: rows=%d path=%s",
            len(self.data_info),
            self.h5_path,
        )

    @property
    def h5_preloaded(self):
        return isinstance(self.h5_images, np.ndarray)

    def attach_preloaded_h5_images(self, images):
        """Attach one shared, complete HDF5 image array before iteration."""
        if self.h5_file is not None:
            raise RuntimeError(
                "Cannot attach preloaded images after streaming HDF5 access started."
            )
        if not isinstance(images, np.ndarray):
            raise TypeError("Preloaded HDF5 images must be a NumPy array.")

        image_shape = _validate_h5_images(images, self.h5_path)
        if image_shape != self.h5_image_shape:
            raise ValueError(
                "Preloaded HDF5 shape does not match the validated file: "
                f"{image_shape} != {self.h5_image_shape}"
            )
        if np.dtype(images.dtype) != self.h5_image_dtype:
            raise ValueError(
                "Preloaded HDF5 dtype does not match the validated file: "
                f"{images.dtype} != {self.h5_image_dtype}"
            )
        if not images.flags.c_contiguous:
            raise ValueError("Preloaded HDF5 images must be C-contiguous.")

        self.h5_images = images

    def _lazy_init_h5(self):
        """Initializes the HDF5 file lazily when a PyTorch background worker asks for it."""
        if self.h5_images is None:
            # swmr=True enables Single-Writer Multiple-Reader, which is ideal for PyTorch DataLoaders
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, rdcc_nbytes=1024 ** 2 * 512)
            self.h5_images = self.h5_file[H5_IMAGES_NAME]

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            idx = index
            if idx < 0:
                idx += len(self.labels)
            if idx < 0 or idx >= len(self.labels):
                raise IndexError(f"Dataset index out of range: {index}")

            self._lazy_init_h5()
            true_h5_idx = int(self.h5_indices[idx])
            pt_np = self.h5_images[true_h5_idx]
            pt = torch.from_numpy(pt_np)

            label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return pt, label
        else:
            raise TypeError(f"Invalid argument type: {type(index)}")

    def __len__(self):
        return len(self.labels)

    def __del__(self):
        """Gracefully close the HDF5 file handle upon garbage collection."""
        if getattr(self, "h5_file", None) is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
