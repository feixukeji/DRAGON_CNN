import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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


def process_single_object(task_args):
    """Process one object and return its HDF5 array, or ``None`` on failure."""
    df_index, row, data_dir, bands, cutout_size, device_str = task_args
    channels = []

    for band in bands:
        fits_path = data_dir / str(row[band])
        try:
            tensor_2d = FITSDataset.load_fits_as_tensor(fits_path, device=device_str)
            tensor_2d = center_crop_or_pad_torch(tensor_2d, cutout_size)
            channels.append(tensor_2d)
        except Exception:
            # Signal failure by returning None instead of zero-padding
            return df_index, None

    stacked_tensor = torch.stack(channels, dim=0)

    # Return numpy array for saving to HDF5. cpu() ensures it's off the GPU.
    return df_index, stacked_tensor.cpu().numpy()


def create_cutout_tensors(
    data_dir,
    csv_path,
    out_dir,
    bands,
    cutout_size=94,
    workers=4,
    use_gpu=False,
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

    device_str = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    click.echo(
        f"Processing up to {len(df)} objects into "
        f"{len(bands)}-channel tensors..."
    )
    click.echo(f"Using {workers} workers. Compute Device: {device_str.upper()}")

    # Package tasks with their original dataframe index
    tasks = [
        (i, row, data_dir, bands, cutout_size, device_str)
        for i, (_, row) in enumerate(df.iterrows())
    ]

    h5_path = out_dir / "tensors.h5"

    # Open HDF5 file in write mode
    with h5py.File(h5_path, 'w') as h5f:
        max_len = len(df)

        # 1. Pre-allocate the FULL shape immediately (no shape=(0,...))
        # 2. Chunk it by a larger number, like 64 or 128, to optimize batch reading
        chunk_rows = min(64, max_len)
        dset = h5f.create_dataset(
            "images",
            shape=(max_len, len(bands), cutout_size, cutout_size),
            maxshape=(max_len, len(bands), cutout_size, cutout_size),
            dtype='float32',
            chunks=(chunk_rows, len(bands), cutout_size, cutout_size)
        )

        current_h5_idx = 0
        successful_indices = []

        # executor.map ensures results are returned in order
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            results = executor.map(process_single_object, tasks)

            for df_index, numpy_array in tqdm(results, total=len(tasks)):
                if numpy_array is not None:
                    # 3. Assign directly to the pre-allocated space (NO resizing in loop!)
                    dset[current_h5_idx] = numpy_array
                    successful_indices.append(df_index)
                    current_h5_idx += 1

        # 4. Do one final resize at the very end to trim any dropped/corrupted files
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
    default=['g_band', 'i_band', 'r_band'],
    help='List of columns for channels.',
)
@click.option(
    '--cutout-size',
    type=int,
    default=94,
    help='Final square size of the cutouts.',
)
@click.option(
    '--workers',
    type=int,
    default=4,
    help='Number of parallel CPU workers for disk I/O.',
)
@click.option(
    '--use-gpu/--no-gpu',
    default=False,
    help='Flag to push tensor operations to GPU.',
)
def generate_tensors(data_dir, csv_path, out_dir, bands, cutout_size, workers, use_gpu):
    """Preprocess FITS files into HDF5 and aligned clean metadata."""
    try:
        create_cutout_tensors(
            data_dir=data_dir,
            csv_path=csv_path,
            out_dir=out_dir,
            bands=bands,
            cutout_size=cutout_size,
            workers=workers,
            use_gpu=use_gpu,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == '__main__':
    generate_tensors()
