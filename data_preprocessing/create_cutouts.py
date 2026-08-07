import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import center_crop_or_pad_torch

try:
    from .dataset import FITSDataset
except ImportError:  # Support direct execution as a script.
    from dataset import FITSDataset


MAX_BATCH_ROWS = 64
TARGET_BATCH_BYTES = 8 * 1024 * 1024
PREFETCH_BATCHES_PER_WORKER = 4


def process_single_object(task_args):
    """Process one object and return its HDF5 array, or ``None`` on failure."""
    df_index, fits_paths, cutout_size = task_args
    channels = []

    for fits_path in fits_paths:
        try:
            tensor_2d = FITSDataset.load_fits_as_tensor(fits_path)
            tensor_2d = center_crop_or_pad_torch(tensor_2d, cutout_size)
            channels.append(tensor_2d)
        except Exception:
            # Signal failure by returning None instead of zero-padding
            return df_index, None

    stacked_tensor = torch.stack(channels, dim=0)

    return df_index, stacked_tensor.numpy()


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
):
    """Yield lightweight ordered task batches without materializing all rows."""
    batch = []
    band_rows = df.loc[:, list(bands)].itertuples(index=False, name=None)
    for df_index, row_paths in enumerate(band_rows):
        fits_paths = tuple(str(data_dir / str(path)) for path in row_paths)
        batch.append((df_index, fits_paths, cutout_size))
        if len(batch) == batch_rows:
            yield batch
            batch = []
    if batch:
        yield batch


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
    cutout_size=96,
    workers=4,
):
    """Pack FITS files into HDF5 and write row-aligned clean metadata."""
    data_dir = Path(data_dir)
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
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

    df = pd.read_csv(csv_path, dtype={"object_id": str})
    missing_columns = [band for band in bands if band not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Metadata CSV is missing band column(s): {', '.join(missing_columns)}"
        )
    if df.empty:
        raise ValueError(f"Metadata CSV contains no rows: {csv_path}")

    click.echo(
        f"Processing up to {len(df)} objects into "
        f"{len(bands)}-channel tensors..."
    )
    click.echo(f"Using {workers} CPU workers.")

    h5_path = out_dir / "tensors.h5"

    # Open HDF5 file in write mode
    with h5py.File(h5_path, 'w') as h5f:
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

    # Filter out the corrupted rows to ensure perfect alignment
    clean_df = df.iloc[successful_indices].copy()
    clean_df["h5_index"] = np.arange(len(clean_df), dtype=np.int64)
    clean_csv_path = out_dir / "clean_info.csv"
    clean_df.to_csv(clean_csv_path, index=False)

    click.echo(
        f"Finished! Packed {len(successful_indices)} tensors "
        f"sequentially into {h5_path}"
    )
    click.echo(
        f"Saved aligned metadata to {clean_csv_path} (use this for training!)"
    )
    return h5_path, clean_csv_path


@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True),
    required=True,
    help='Base directory containing FITS files.',
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
def generate_tensors(data_dir, csv_path, out_dir, bands, cutout_size, workers):
    """Preprocess FITS files into HDF5 and aligned clean metadata."""
    try:
        create_cutout_tensors(
            data_dir=data_dir,
            csv_path=csv_path,
            out_dir=out_dir,
            bands=bands,
            cutout_size=cutout_size,
            workers=workers,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == '__main__':
    generate_tensors()
