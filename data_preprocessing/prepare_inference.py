"""Prepare a compact catalog and HDF5 store for DRAGON inference."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click
import pandas as pd

from data_preprocessing.create_cutouts import create_cutout_tensors


@dataclass(frozen=True)
class PreparedInferenceData:
    output_dir: Path
    info_path: Path
    h5_path: Path
    rows: int


def prepare_inference_data(
    *,
    catalog_path: Path | str,
    cutout_dir: Path | str,
    output_dir: Path | str,
    bands: tuple[str, ...] | list[str],
    cutout_size: int,
    workers: int,
) -> PreparedInferenceData:
    """Build ``info.csv`` and recreate tensors for one inference catalog."""
    catalog = Path(catalog_path).expanduser().resolve()
    cutouts = Path(cutout_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    normalized_bands = tuple(dict.fromkeys(str(band).strip() for band in bands))

    if not catalog.is_file():
        raise FileNotFoundError(f"Catalog not found: {catalog}")
    if not cutouts.is_dir():
        raise FileNotFoundError(f"Cutout directory not found: {cutouts}")
    if not normalized_bands or any(not band for band in normalized_bands):
        raise ValueError("At least one non-empty band is required")
    if cutout_size <= 0:
        raise ValueError("cutout_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    info = pd.read_csv(
        catalog,
        dtype={"object_id": "string"},
        low_memory=False,
    )
    if "object_id" not in info.columns:
        raise ValueError(f"Catalog is missing object_id column: {catalog}")
    if info["object_id"].isna().any():
        raise ValueError(f"Catalog contains missing object_id values: {catalog}")
    if info.empty:
        raise ValueError(f"Catalog contains no rows: {catalog}")

    for band in normalized_bands:
        info[band] = [
            str(cutouts / f"{object_id}_{band}.fits")
            for object_id in info["object_id"].astype(str)
        ]

    output.mkdir(parents=True, exist_ok=True)
    for stale_file in (
        "raw_info.csv",
        "predictions.csv",
        "summary_counts.csv",
        "labels.csv",
        "normalization_stats.json",
    ):
        (output / stale_file).unlink(missing_ok=True)
    for stale_dir in ("predictions", "heatmaps"):
        stale_path = output / stale_dir
        if stale_path.is_symlink():
            stale_path.unlink()
        elif stale_path.is_dir():
            shutil.rmtree(stale_path)

    tensors_dir = output / "tensors"
    h5_path = tensors_dir / "tensors.h5"
    info_path = output / "info.csv"
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as temp_dir:
        raw_info_path = Path(temp_dir) / "raw_info.csv"
        info.to_csv(raw_info_path, index=False)
        create_cutout_tensors(
            data_dir=output,
            csv_path=raw_info_path,
            out_dir=tensors_dir,
            bands=normalized_bands,
            info_path=info_path,
            cutout_size=cutout_size,
            workers=workers,
        )

    aligned_info = pd.read_csv(
        info_path,
        dtype={"object_id": "string"},
        low_memory=False,
    )
    if aligned_info.empty:
        raise ValueError("No readable FITS cutouts remain after preprocessing")

    return PreparedInferenceData(
        output_dir=output,
        info_path=info_path,
        h5_path=h5_path,
        rows=len(aligned_info),
    )


@click.command()
@click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option("--cutout-dir", type=click.Path(file_okay=False), required=True)
@click.option("--output-dir", type=click.Path(file_okay=False), required=True)
@click.option("--band", "bands", multiple=True, required=True)
@click.option("--cutout-size", type=int, default=96, show_default=True)
@click.option("--workers", type=int, default=4, show_default=True)
def main(**kwargs) -> None:
    """Prepare metadata and tensors for DRAGON inference."""
    try:
        prepared = prepare_inference_data(**kwargs)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Prepared {prepared.rows} inference rows in {prepared.output_dir}")
    click.echo(f"Rebuilt inference tensors: {prepared.h5_path}")


if __name__ == "__main__":
    main()
