import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import NamedTuple

import click
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from astropy.io import fits
from tqdm import tqdm

MAX_BATCH_ROWS = 64
TARGET_BATCH_BYTES = 8 * 1024 * 1024
PREFETCH_BATCHES_PER_WORKER = 4
TENSOR_STORE_COLUMN = "tensor_store"
TENSOR_INDEX_COLUMN = "tensor_index"
TENSOR_OBJECT_ID_COLUMN = "tensor_object_id"
PAIR_IMAGES_DATASET = "images"
PAIR_OBJECT_IDS_DATASET = "object_ids"


class ObjectTask(NamedTuple):
    """One FITS-backed or HDF5-backed input row."""

    df_index: int
    fits_paths: tuple[str, ...] | None
    tensor_store: str | None
    tensor_index: int | None
    tensor_object_id: str | None
    cutout_size: int
    bands: tuple[str, ...]


_PAIR_H5_CACHE = {}


def _load_fits_tensor(filename):
    """Load the first usable 2D FITS image as a float32 tensor."""
    image = None
    try:
        image = fits.getdata(filename, memmap=False)
    except (IndexError, KeyError, TypeError, ValueError):
        pass

    if image is None or not hasattr(image, "shape") or image.ndim != 2:
        with fits.open(filename, memmap=False) as hdus:
            image = next(
                (
                    hdu.data
                    for hdu in hdus
                    if hdu.data is not None
                    and hasattr(hdu.data, "shape")
                    and hdu.data.ndim == 2
                ),
                None,
            )
        if image is None:
            raise ValueError(f"No valid 2D image array in {filename}")

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(image.astype(np.float32))


def _center_crop_or_pad_tensor(tensor, size):
    """Center-crop or zero-pad a 2D tensor to a square of ``size`` pixels."""
    height, width = tensor.shape
    pad_height = max(0, size - height)
    pad_width = max(0, size - width)
    if pad_height or pad_width:
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        tensor = F.pad(
            tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
        height, width = tensor.shape

    start_y = height // 2 - size // 2
    start_x = width // 2 - size // 2
    return tensor[start_y:start_y + size, start_x:start_x + size]


def _build_runtime_metadata(df, successful_indices, bands):
    """Drop tensor-construction references from aligned runtime metadata."""
    source_columns = {
        *bands,
        TENSOR_STORE_COLUMN,
        TENSOR_INDEX_COLUMN,
        TENSOR_OBJECT_ID_COLUMN,
    }
    runtime_columns = [
        column for column in df.columns if column not in source_columns
    ]
    aligned_info = df.iloc[successful_indices].loc[:, runtime_columns].copy()
    aligned_info["h5_index"] = np.arange(
        len(aligned_info),
        dtype=np.int64,
    )
    return aligned_info


def _load_pair_h5_tensor(task):
    """Load and validate one generated-pair tensor in a worker process."""
    cached = _PAIR_H5_CACHE.get(task.tensor_store)
    if cached is None:
        handle = h5py.File(task.tensor_store, "r")
        try:
            if PAIR_IMAGES_DATASET not in handle:
                raise KeyError(
                    f"Dataset {PAIR_IMAGES_DATASET!r} is missing from "
                    f"{task.tensor_store}"
                )
            if PAIR_OBJECT_IDS_DATASET not in handle:
                raise KeyError(
                    f"Dataset {PAIR_OBJECT_IDS_DATASET!r} is missing from "
                    f"{task.tensor_store}"
                )
            stored_bands = handle.attrs.get("bands", "")
            if isinstance(stored_bands, bytes):
                stored_bands = stored_bands.decode("utf-8")
            if tuple(str(stored_bands).split(",")) != task.bands:
                raise ValueError(
                    f"Pair HDF5 bands {stored_bands!r} do not match "
                    f"requested {task.bands}: {task.tensor_store}"
                )
            images = handle[PAIR_IMAGES_DATASET]
            object_ids = handle[PAIR_OBJECT_IDS_DATASET]
            if not isinstance(images, h5py.Dataset) or images.ndim != 4:
                raise ValueError(
                    "Pair HDF5 images must have shape (N, C, H, W): "
                    f"{task.tensor_store}"
                )
            if images.dtype != np.dtype("float32"):
                raise ValueError(
                    f"Pair HDF5 images must be float32: {task.tensor_store}"
                )
            if tuple(object_ids.shape) != (images.shape[0],):
                raise ValueError(
                    "Pair HDF5 object_ids must align with images: "
                    f"{task.tensor_store}"
                )
            expected_shape = (
                len(task.bands),
                task.cutout_size,
                task.cutout_size,
            )
            if tuple(images.shape[1:]) != expected_shape:
                raise ValueError(
                    f"Pair tensor shape {tuple(images.shape[1:])} does not "
                    f"match expected {expected_shape}: {task.tensor_store}"
                )
            cached = (handle, images, object_ids)
            _PAIR_H5_CACHE[task.tensor_store] = cached
        except BaseException:
            handle.close()
            raise

    _, images, object_ids = cached
    if task.tensor_index < 0 or task.tensor_index >= images.shape[0]:
        raise IndexError(
            f"Pair tensor index {task.tensor_index} is outside [0, "
            f"{images.shape[0]}): {task.tensor_store}"
        )
    stored_object_id = object_ids[task.tensor_index]
    if isinstance(stored_object_id, bytes):
        stored_object_id = stored_object_id.decode("utf-8")
    else:
        stored_object_id = str(stored_object_id)
    if stored_object_id != task.tensor_object_id:
        raise ValueError(
            f"Pair tensor object-ID mismatch at index {task.tensor_index}: "
            f"catalog={task.tensor_object_id!r}, "
            f"HDF5={stored_object_id!r}, store={task.tensor_store}"
        )
    array = np.asarray(images[task.tensor_index])
    if not np.isfinite(array).all():
        raise ValueError(
            f"Pair tensor contains NaN or Inf at index {task.tensor_index}: "
            f"{task.tensor_store}"
        )
    return array


def process_single_object(task):
    """Load one FITS-backed or pair-HDF5-backed tensor row."""
    if task.tensor_store is not None:
        return task.df_index, _load_pair_h5_tensor(task)

    if task.fits_paths is None:
        raise ValueError(f"Input row {task.df_index} has no tensor source")
    channels = []

    for fits_path in task.fits_paths:
        try:
            tensor_2d = _load_fits_tensor(fits_path)
            tensor_2d = _center_crop_or_pad_tensor(
                tensor_2d,
                task.cutout_size,
            )
            channels.append(tensor_2d)
        except Exception:
            # Signal failure by returning None instead of zero-padding
            return task.df_index, None

    stacked_tensor = torch.stack(channels, dim=0)

    return task.df_index, stacked_tensor.numpy()


def process_object_batch(task_batch):
    """Process one ordered task batch into a contiguous HDF5 write block."""
    successful_indices = []
    arrays = []
    for task_args in task_batch:
        df_index, array = process_single_object(task_args)
        if array is not None:
            successful_indices.append(df_index)
            arrays.append(array)

    stacked = np.stack(arrays, axis=0) if arrays else None
    return len(task_batch), successful_indices, stacked


def _rows_per_batch(channel_count, cutout_size):
    bytes_per_row = (
        channel_count
        * cutout_size
        * cutout_size
        * np.dtype('float32').itemsize
    )
    return max(1, min(MAX_BATCH_ROWS, TARGET_BATCH_BYTES // bytes_per_row))


def _iter_task_batches(
    df,
    data_dir,
    bands,
    cutout_size,
    batch_rows,
    tensor_rows,
    tensor_indices,
):
    """Yield lightweight ordered task batches without materializing all rows."""
    batch = []
    if set(bands).issubset(df.columns):
        band_values = df.loc[:, list(bands)].to_numpy(dtype=object)
    else:
        band_values = None
    if TENSOR_STORE_COLUMN in df:
        tensor_stores = df[TENSOR_STORE_COLUMN].to_numpy(dtype=object)
        tensor_object_ids = df[TENSOR_OBJECT_ID_COLUMN].to_numpy(dtype=object)
    else:
        tensor_stores = tensor_object_ids = None

    for df_index in range(len(df)):
        if tensor_rows[df_index]:
            tensor_store = Path(str(tensor_stores[df_index]))
            if not tensor_store.is_absolute():
                tensor_store = data_dir / tensor_store
            task = ObjectTask(
                df_index=df_index,
                fits_paths=None,
                tensor_store=str(tensor_store),
                tensor_index=int(tensor_indices[df_index]),
                tensor_object_id=str(tensor_object_ids[df_index]),
                cutout_size=cutout_size,
                bands=tuple(bands),
            )
        else:
            row_paths = band_values[df_index]
            task = ObjectTask(
                df_index=df_index,
                fits_paths=tuple(
                    str(data_dir / str(path)) for path in row_paths
                ),
                tensor_store=None,
                tensor_index=None,
                tensor_object_id=None,
                cutout_size=cutout_size,
                bands=tuple(bands),
            )
        batch.append(task)
        if len(batch) == batch_rows:
            yield batch
            batch = []
    if batch:
        yield batch


def _validate_input_rows(df, bands):
    """Classify and validate FITS-backed versus tensor-backed metadata rows."""
    tensor_columns = {
        TENSOR_STORE_COLUMN,
        TENSOR_INDEX_COLUMN,
        TENSOR_OBJECT_ID_COLUMN,
    }
    present_tensor_columns = tensor_columns.intersection(df.columns)
    if present_tensor_columns and present_tensor_columns != tensor_columns:
        missing = tensor_columns.difference(df.columns)
        raise ValueError(
            "Metadata has an incomplete tensor reference; missing column(s): "
            + ", ".join(sorted(missing))
        )

    if present_tensor_columns:
        store_text = df[TENSOR_STORE_COLUMN].astype("string")
        store_present = store_text.notna() & store_text.str.strip().ne("")
        index_present = df[TENSOR_INDEX_COLUMN].notna()
        object_id_text = df[TENSOR_OBJECT_ID_COLUMN].astype("string")
        object_id_present = (
            object_id_text.notna() & object_id_text.str.strip().ne("")
        )
        partially_tensor_backed = (
            store_present.astype(np.int8)
            + index_present.astype(np.int8)
            + object_id_present.astype(np.int8)
        ).isin((1, 2))
        if partially_tensor_backed.any():
            examples = df.index[partially_tensor_backed].tolist()[:5]
            raise ValueError(
                "Each tensor-backed row must provide tensor_store, "
                "tensor_index, and tensor_object_id; invalid row index(es): "
                f"{examples}"
            )
        tensor_rows = (
            store_present & index_present & object_id_present
        ).to_numpy(dtype=bool)
    else:
        tensor_rows = np.zeros(len(df), dtype=bool)

    fits_rows = ~tensor_rows
    missing_band_columns = [band for band in bands if band not in df.columns]
    if fits_rows.any() and missing_band_columns:
        raise ValueError(
            "FITS-backed metadata rows require band column(s): "
            + ", ".join(missing_band_columns)
        )
    if fits_rows.any():
        for band in bands:
            values = df[band].astype("string")
            missing = values.isna() | values.str.strip().eq("")
            invalid = fits_rows & missing.to_numpy(dtype=bool)
            if invalid.any():
                examples = df.index[invalid].tolist()[:5]
                raise ValueError(
                    f"FITS-backed rows require a non-empty {band!r} path; "
                    f"invalid row index(es): {examples}"
                )

    tensor_indices = np.full(len(df), -1, dtype=np.int64)
    if tensor_rows.any():
        numeric = pd.to_numeric(
            df.loc[tensor_rows, TENSOR_INDEX_COLUMN],
            errors="coerce",
        )
        invalid = (
            numeric.isna()
            | ~np.isfinite(numeric)
            | (numeric % 1 != 0)
            | (numeric < 0)
            | (numeric > np.iinfo(np.int64).max)
        )
        if invalid.any():
            examples = numeric.index[invalid].tolist()[:5]
            raise ValueError(
                "tensor_index must contain non-negative integers; invalid "
                f"row index(es): {examples}"
            )
        tensor_indices[tensor_rows] = numeric.to_numpy(dtype=np.int64)
    return tensor_rows, tensor_indices


def _bounded_ordered_map(executor, function, items, max_pending):
    """Map in submission order while bounding queued tasks and results."""
    iterator = iter(items)
    pending = deque(
        executor.submit(function, item)
        for item in islice(iterator, max_pending)
    )
    while pending:
        future = pending.popleft()
        result = future.result()
        try:
            item = next(iterator)
        except StopIteration:
            pass
        else:
            pending.append(executor.submit(function, item))
        yield result


def create_cutout_tensors(
    data_dir,
    csv_path,
    out_dir,
    bands,
    info_path=None,
    cutout_size=96,
    workers=4,
):
    """Publish final HDF5 tensors and their row-aligned ``info.csv``."""
    data_dir = Path(data_dir)
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    info_path = (
        Path(info_path)
        if info_path is not None
        else out_dir.parent / "info.csv"
    )
    bands = tuple(bands)

    if not data_dir.is_dir():
        raise ValueError(f"Data directory not found: {data_dir}")
    if not csv_path.is_file():
        raise ValueError(f"Metadata CSV not found: {csv_path}")
    if not bands:
        raise ValueError("At least one band column is required")
    if cutout_size <= 0:
        raise ValueError("cutout_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        csv_path,
        dtype={
            "object_id": "string",
            **{band: "string" for band in bands},
            TENSOR_STORE_COLUMN: "string",
            TENSOR_INDEX_COLUMN: "Int64",
            TENSOR_OBJECT_ID_COLUMN: "string",
        },
        low_memory=False,
    )
    if df.empty:
        raise ValueError(f"Metadata CSV contains no rows: {csv_path}")
    if "object_id" not in df.columns:
        raise ValueError(f"Metadata CSV is missing object_id: {csv_path}")
    tensor_rows, tensor_indices = _validate_input_rows(df, bands)
    fits_count = int((~tensor_rows).sum())
    tensor_count = int(tensor_rows.sum())

    click.echo(
        f"Processing up to {len(df)} objects into "
        f"{len(bands)}-channel tensors..."
    )
    click.echo(
        f"Inputs: FITS={fits_count}, pair_HDF5={tensor_count}."
    )
    click.echo(f"Using {workers} CPU workers.")

    h5_path = out_dir / "tensors.h5"
    temporary_h5_path = out_dir / "tensors.h5.incomplete"
    temporary_info_path = info_path.with_name(info_path.name + ".incomplete")

    try:
        # Only this parent process writes the final HDF5 file. Workers return
        # bounded ordered batches loaded from either FITS or pair HDF5 stores.
        with h5py.File(temporary_h5_path, 'w') as h5f:
            max_len = len(df)

            chunk_rows = min(MAX_BATCH_ROWS, max_len)
            batch_rows = _rows_per_batch(len(bands), cutout_size)
            dset = h5f.create_dataset(
                "images",
                shape=(max_len, len(bands), cutout_size, cutout_size),
                maxshape=(max_len, len(bands), cutout_size, cutout_size),
                dtype='float32',
                chunks=(chunk_rows, len(bands), cutout_size, cutout_size)
            )

            current_h5_idx = 0
            successful_indices = []

            task_batches = _iter_task_batches(
                df,
                data_dir,
                bands,
                cutout_size,
                batch_rows,
                tensor_rows,
                tensor_indices,
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                results = _bounded_ordered_map(
                    executor,
                    process_object_batch,
                    task_batches,
                    max_pending=workers * PREFETCH_BATCHES_PER_WORKER,
                )
                with tqdm(total=max_len, unit='object') as progress:
                    for processed_count, batch_indices, numpy_batch in results:
                        progress.update(processed_count)
                        if numpy_batch is None:
                            continue
                        next_h5_idx = current_h5_idx + len(batch_indices)
                        dset[current_h5_idx:next_h5_idx] = numpy_batch
                        successful_indices.extend(batch_indices)
                        current_h5_idx = next_h5_idx

            if current_h5_idx < max_len:
                dset.resize(current_h5_idx, axis=0)

        # Publish only runtime metadata plus the exact row-to-HDF5 map. Source
        # FITS and pair-HDF5 references remain in the input raw_info.csv.
        aligned_info = _build_runtime_metadata(
            df,
            successful_indices,
            bands,
        )
        info_path.parent.mkdir(parents=True, exist_ok=True)
        aligned_info.to_csv(temporary_info_path, index=False)
        temporary_h5_path.replace(h5_path)
        temporary_info_path.replace(info_path)
    finally:
        temporary_h5_path.unlink(missing_ok=True)
        temporary_info_path.unlink(missing_ok=True)

    click.echo(
        f"Finished! Packed {len(successful_indices)} tensors "
        f"sequentially into {h5_path}"
    )
    click.echo(
        f"Saved aligned metadata to {info_path}"
    )
    return h5_path, info_path


@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True),
    required=True,
    help='Base directory for relative FITS and pair-HDF5 paths.',
)
@click.option(
    '--csv-path',
    type=click.Path(exists=True),
    required=True,
    help='Path to the metadata CSV.',
)
@click.option(
    '--out-dir',
    type=click.Path(),
    required=True,
    help='Output directory for the HDF5 tensor file.',
)
@click.option(
    '--info-path',
    type=click.Path(dir_okay=False),
    default=None,
    help='Aligned metadata output (default: OUT_DIR/../info.csv).',
)
@click.option(
    '--bands',
    multiple=True,
    default=['i_band', 'r_band', 'g_band'],
    help='List of columns for channels.',
)
@click.option(
    '--cutout-size',
    type=int,
    default=96,
    show_default=True,
    help='Final square size of the cutouts.',
)
@click.option(
    '--workers',
    type=int,
    default=4,
    help='Number of parallel CPU workers for disk I/O.',
)
def generate_tensors(
    data_dir,
    csv_path,
    out_dir,
    info_path,
    bands,
    cutout_size,
    workers,
):
    """Assemble FITS and pair tensors into HDF5 plus aligned metadata."""
    try:
        create_cutout_tensors(
            data_dir=data_dir,
            csv_path=csv_path,
            out_dir=out_dir,
            bands=bands,
            info_path=info_path,
            cutout_size=cutout_size,
            workers=workers,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == '__main__':
    generate_tensors()
