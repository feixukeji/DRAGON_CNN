"""Build DRAGON training metadata from survey-specific catalog definitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .catalog_validation import (
    DEFAULT_COORDINATE_TOLERANCE_ARCSEC,
    add_class_prefix_to_object_ids,
    discard_coordinate_equivalent_duplicates,
    require_unique_object_ids,
)


DRAGON_CLASS_ORDER = (
    "rubbish",
    "empty",
    "single_galaxy",
    "single_star",
    "single_agn",
    "agn_galaxy",
    "merger",
    "agn_star",
    "dual_agn",
)


@dataclass(frozen=True)
class ClassSpec:
    """Survey-local catalog and cutout paths for one DRAGON class."""

    name: str
    csv_path: Path
    cutout_dir: Path


@dataclass(frozen=True)
class PreparedTrainingCatalog:
    """Paths and per-class counts produced by metadata preparation."""

    raw_info_path: Path
    labels_path: Path
    rows: int
    row_counts: dict[str, int]
    discarded_counts: dict[str, int]


class EmptyTrainingCatalogError(ValueError):
    """Raised when all configured class catalogs are empty after filtering."""


def _load_objects(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, dtype={"object_id": str})
    if "object_id" not in frame:
        raise ValueError(
            f"Missing required column(s) in {csv_path}: object_id"
        )

    coordinate_columns = {"ra", "dec"}.intersection(frame.columns)
    if len(coordinate_columns) == 1:
        missing_coordinate = ({"ra", "dec"} - coordinate_columns).pop()
        raise ValueError(
            f"Coordinate columns must be provided together in {csv_path}; "
            f"missing {missing_coordinate}"
        )
    return frame


def _build_rows(
    objects: pd.DataFrame,
    class_name: str,
    cutout_dir: Path,
    bands: Sequence[str],
) -> pd.DataFrame:
    original_object_ids = objects["object_id"].astype(str)
    metadata_columns = ["object_id"]
    if {"ra", "dec"}.issubset(objects.columns):
        metadata_columns.extend(["ra", "dec"])
    rows = add_class_prefix_to_object_ids(
        objects.loc[:, metadata_columns],
        class_name,
    )
    rows["class"] = class_name
    for band in bands:
        rows[band] = [
            str(cutout_dir / f"{object_id}_{band}.fits")
            for object_id in original_object_ids
        ]
    return rows


def _validate_class_configuration(
    class_specs: Sequence[ClassSpec],
    class_order: Sequence[str],
) -> None:
    spec_names = [spec.name for spec in class_specs]
    if not spec_names:
        raise ValueError("At least one class specification is required")
    if len(spec_names) != len(set(spec_names)):
        raise ValueError("Class specifications contain duplicate names")

    label_names = list(class_order)
    if len(label_names) != len(set(label_names)):
        raise ValueError("Class order contains duplicate names")
    if set(spec_names) != set(label_names):
        missing_specs = sorted(set(label_names) - set(spec_names))
        unexpected_specs = sorted(set(spec_names) - set(label_names))
        details = []
        if missing_specs:
            details.append(f"missing specifications: {', '.join(missing_specs)}")
        if unexpected_specs:
            details.append(
                f"unexpected specifications: {', '.join(unexpected_specs)}"
            )
        raise ValueError(
            "Class specifications do not match class order; " + "; ".join(details)
        )


def prepare_training_catalog(
    *,
    class_specs: Sequence[ClassSpec],
    bands: Sequence[str],
    output_dir: Path | str,
    class_order: Sequence[str] = DRAGON_CLASS_ORDER,
    coordinate_tolerance_arcsec: float = DEFAULT_COORDINATE_TOLERANCE_ARCSEC,
    context: str = "DRAGON training",
) -> PreparedTrainingCatalog:
    """Write ``raw_info.csv`` and ``labels.csv`` for DRAGON training.

    Band names are used exactly as supplied. Catalog and cutout paths remain
    survey-owned configuration supplied through ``class_specs``. Every catalog
    must contain ``object_id``. Catalogs that also contain both ``ra`` and
    ``dec`` participate in coordinate-based duplicate filtering; catalogs with
    neither coordinate column skip that filtering.
    """
    specs = list(class_specs)
    ordered_classes = list(class_order)
    band_names = list(bands)
    _validate_class_configuration(specs, ordered_classes)

    objects_by_class = {
        spec.name: _load_objects(spec.csv_path)
        for spec in specs
    }
    objects_with_coordinates = {
        class_name: objects
        for class_name, objects in objects_by_class.items()
        if {"ra", "dec"}.issubset(objects.columns)
    }
    if objects_with_coordinates:
        filtered_objects, coordinate_discarded_counts = (
            discard_coordinate_equivalent_duplicates(
                objects_with_coordinates,
                tolerance_arcsec=coordinate_tolerance_arcsec,
                context=f"{context} catalogs",
            )
        )
        objects_by_class.update(filtered_objects)
    else:
        coordinate_discarded_counts = {}
    discarded_counts = {
        spec.name: coordinate_discarded_counts.get(spec.name, 0)
        for spec in specs
    }

    rows_by_class = {
        spec.name: _build_rows(
            objects_by_class[spec.name],
            spec.name,
            spec.cutout_dir,
            band_names,
        )
        for spec in specs
    }
    row_counts = {
        class_name: len(rows)
        for class_name, rows in rows_by_class.items()
    }
    if not any(row_counts.values()):
        raise EmptyTrainingCatalogError("No rows found. Check the input catalogs.")

    prepared_info = pd.concat(
        [rows_by_class[spec.name] for spec in specs],
        ignore_index=True,
        sort=False,
    )
    require_unique_object_ids(
        prepared_info,
        context=f"{context} dataset",
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    raw_info_path = destination / "raw_info.csv"
    prepared_info.to_csv(raw_info_path, index=False)

    labels_path = destination / "labels.csv"
    pd.DataFrame(
        {
            "key": ordered_classes,
            "value": list(range(len(ordered_classes))),
        }
    ).to_csv(labels_path, index=False)

    return PreparedTrainingCatalog(
        raw_info_path=raw_info_path,
        labels_path=labels_path,
        rows=len(prepared_info),
        row_counts=row_counts,
        discarded_counts=discarded_counts,
    )
