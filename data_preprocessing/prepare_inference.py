"""Prepare a catalog and HDF5 tensors for label-free DRAGON inference."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import click
import pandas as pd

from data_preprocessing import create_cutout_tensors
from utils import label_mapping_frame, load_asinh_stats, load_label_mapping


SIGNATURE_NAME = "preprocessing.sha256"


@dataclass(frozen=True)
class PreparedInferenceData:
    run_dir: Path
    raw_info_path: Path
    info_path: Path
    labels_path: Path
    normalization_stats_path: Path
    h5_path: Path
    rows: int
    rebuilt_tensors: bool


def _resolve_file(path: Path | str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def resolve_model_artifact(
    model_path: Path | str,
    filename: str,
    explicit_path: Path | str | None = None,
) -> Path:
    """Resolve a run artifact beside a model or in its parent run directory."""
    if explicit_path is not None:
        return _resolve_file(explicit_path, filename)

    model = _resolve_file(model_path, "Model")
    candidates = [model.parent / filename, model.parent.parent / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Unable to find {filename}; searched: {searched}")


def _copy_file(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _preprocessing_signature(
    raw_info_path: Path,
    bands: tuple[str, ...],
    cutout_size: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(raw_info_path.read_bytes())
    config = json.dumps(
        {"bands": bands, "cutout_size": cutout_size},
        sort_keys=True,
    ).encode("utf-8")
    digest.update(config)
    return digest.hexdigest()


def prepare_inference_data(
    *,
    catalog_path: Path | str,
    cutout_dir: Path | str,
    run_dir: Path | str,
    model_path: Path | str,
    bands: tuple[str, ...] | list[str],
    cutout_size: int,
    channels: int,
    workers: int,
    labels_path: Path | str | None = None,
    normalization_stats_path: Path | str | None = None,
    force_preprocess: bool = False,
) -> PreparedInferenceData:
    """Build inference metadata, reuse valid tensors, and copy model artifacts."""
    catalog = _resolve_file(catalog_path, "Catalog")
    cutouts = Path(cutout_dir).expanduser().resolve()
    output = Path(run_dir).expanduser().resolve()
    model = _resolve_file(model_path, "Model")
    normalized_bands = tuple(dict.fromkeys(str(band).strip() for band in bands))

    if not cutouts.is_dir():
        raise FileNotFoundError(f"Cutout directory not found: {cutouts}")
    if not normalized_bands or any(not band for band in normalized_bands):
        raise ValueError("At least one non-empty band is required")
    if channels != len(normalized_bands):
        raise ValueError(
            f"channels={channels} does not match {len(normalized_bands)} bands"
        )
    if cutout_size <= 0:
        raise ValueError("cutout_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    source_labels = resolve_model_artifact(model, "labels.csv", labels_path)
    source_stats = resolve_model_artifact(
        model,
        "normalization_stats.json",
        normalization_stats_path,
    )
    label_mapping = load_label_mapping(source_labels)
    load_asinh_stats(source_stats, channels=channels)

    info = pd.read_csv(catalog, dtype={"object_id": str})
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
    raw_info_path = output / "raw_info.csv"
    info.to_csv(raw_info_path, index=False)

    destination_labels = output / "labels.csv"
    label_mapping_frame(label_mapping).to_csv(destination_labels, index=False)
    destination_stats = output / "normalization_stats.json"
    _copy_file(source_stats, destination_stats)

    signature = _preprocessing_signature(
        raw_info_path,
        normalized_bands,
        cutout_size,
    )
    current_signature_path = output / SIGNATURE_NAME
    current_signature_path.write_text(signature + "\n", encoding="ascii")

    tensors_dir = output / "tensors"
    h5_path = tensors_dir / "tensors.h5"
    clean_info_path = tensors_dir / "clean_info.csv"
    cached_signature_path = tensors_dir / SIGNATURE_NAME
    cached_signature = (
        cached_signature_path.read_text(encoding="ascii").strip()
        if cached_signature_path.is_file()
        else None
    )
    rebuild = (
        force_preprocess
        or not h5_path.is_file()
        or not clean_info_path.is_file()
        or cached_signature != signature
    )

    if rebuild:
        create_cutout_tensors(
            data_dir=output,
            csv_path=raw_info_path,
            out_dir=tensors_dir,
            bands=normalized_bands,
            cutout_size=cutout_size,
            workers=workers,
            use_gpu=False,
        )

    clean_info = pd.read_csv(clean_info_path, dtype={"object_id": str})
    if clean_info.empty:
        raise ValueError("No readable FITS cutouts remain after preprocessing")
    if rebuild:
        cached_signature_path.write_text(signature + "\n", encoding="ascii")
    info_path = output / "info.csv"
    clean_info.to_csv(info_path, index=False)

    return PreparedInferenceData(
        run_dir=output,
        raw_info_path=raw_info_path,
        info_path=info_path,
        labels_path=destination_labels,
        normalization_stats_path=destination_stats,
        h5_path=h5_path,
        rows=len(clean_info),
        rebuilt_tensors=rebuild,
    )


@click.command()
@click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option("--cutout-dir", type=click.Path(file_okay=False), required=True)
@click.option("--run-dir", type=click.Path(file_okay=False), required=True)
@click.option("--model-path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--band", "bands", multiple=True, required=True)
@click.option("--cutout-size", type=int, default=94, show_default=True)
@click.option("--channels", type=int, required=True)
@click.option("--workers", type=int, default=4, show_default=True)
@click.option("--labels", "labels_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--normalization-stats",
    "normalization_stats_path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--force-preprocess", is_flag=True)
def main(**kwargs) -> None:
    """Prepare metadata and tensors for DRAGON inference."""
    try:
        prepared = prepare_inference_data(**kwargs)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    action = "rebuilt" if prepared.rebuilt_tensors else "reused"
    click.echo(f"Prepared {prepared.rows} inference rows in {prepared.run_dir}")
    click.echo(f"Tensor cache: {action} ({prepared.h5_path})")
    click.echo(f"Normalization stats: {prepared.normalization_stats_path}")


if __name__ == "__main__":
    main()
