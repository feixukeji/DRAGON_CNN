"""Shared catalog filtering for DRAGON training datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


# One millionth of a degree. This covers ordinary CSV rounding noise without
# treating genuinely distinct sky positions as the same source.
DEFAULT_COORDINATE_TOLERANCE_ARCSEC = 0.0036


def _pairwise_separations_arcsec(
    ra_degrees: np.ndarray,
    dec_degrees: np.ndarray,
) -> np.ndarray:
    """Return the great-circle separation matrix for one object-ID group."""
    ra = np.deg2rad(np.mod(ra_degrees, 360.0))
    dec = np.deg2rad(dec_degrees)
    delta_ra = ra[:, None] - ra[None, :]
    delta_dec = dec[:, None] - dec[None, :]
    haversine = (
        np.sin(delta_dec / 2.0) ** 2
        + np.cos(dec[:, None])
        * np.cos(dec[None, :])
        * np.sin(delta_ra / 2.0) ** 2
    )
    haversine = np.clip(haversine, 0.0, 1.0)
    separation = 2.0 * np.arctan2(
        np.sqrt(haversine),
        np.sqrt(1.0 - haversine),
    )
    return separation * (180.0 / math.pi) * 3600.0


def _duplicate_coordinate_positions(
    separations_arcsec: np.ndarray,
    tolerance_arcsec: float,
) -> set[int]:
    """Find rows belonging to a coordinate-equivalent component of size > 1."""
    adjacent = separations_arcsec <= tolerance_arcsec
    np.fill_diagonal(adjacent, False)
    discarded: set[int] = set()
    visited: set[int] = set()

    for start in range(len(adjacent)):
        if start in visited:
            continue
        component: set[int] = set()
        pending = [start]
        while pending:
            position = pending.pop()
            if position in component:
                continue
            component.add(position)
            pending.extend(np.flatnonzero(adjacent[position]).tolist())
        visited.update(component)
        if len(component) > 1:
            discarded.update(component)

    return discarded


def discard_coordinate_equivalent_duplicates(
    catalogs: Mapping[str, pd.DataFrame],
    *,
    tolerance_arcsec: float = DEFAULT_COORDINATE_TOLERANCE_ARCSEC,
    context: str = "DRAGON training catalogs",
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Discard all repeated-ID rows that describe the same sky position.

    Catalog names are used only to split the result back into classes. For a
    repeated original ``object_id``, every coordinate-connected component with
    at least two rows is discarded in full. Rows with that ID at positions
    farther apart than ``tolerance_arcsec`` remain available for later class
    prefixing.
    """
    if not math.isfinite(tolerance_arcsec) or tolerance_arcsec < 0:
        raise ValueError("Coordinate tolerance must be a finite non-negative value")
    if not catalogs:
        raise ValueError("At least one class catalog is required")

    marker = "__dragon_class__"
    required_columns = ("object_id", "ra", "dec")
    combined: list[pd.DataFrame] = []
    original_columns: dict[str, list[str]] = {}
    input_counts: dict[str, int] = {}

    for class_name, catalog in catalogs.items():
        missing = [column for column in required_columns if column not in catalog]
        if missing:
            raise ValueError(
                f"Missing required column(s) in {context} ({class_name}): "
                f"{', '.join(missing)}"
            )
        if marker in catalog:
            raise ValueError(f"Reserved catalog column is already present: {marker}")

        prepared = catalog.copy()
        if prepared["object_id"].isna().any():
            raise ValueError(f"Missing object_id value in {context} ({class_name})")
        prepared["object_id"] = prepared["object_id"].astype(str)
        for coordinate in ("ra", "dec"):
            prepared[coordinate] = pd.to_numeric(
                prepared[coordinate],
                errors="coerce",
            )
        invalid_coordinates = (
            ~np.isfinite(prepared["ra"])
            | ~np.isfinite(prepared["dec"])
            | (prepared["dec"] < -90.0)
            | (prepared["dec"] > 90.0)
        )
        if invalid_coordinates.any():
            bad_ids = prepared.loc[invalid_coordinates, "object_id"].head(5).tolist()
            raise ValueError(
                f"Invalid ra/dec values in {context} ({class_name}); "
                f"example object_id values: {', '.join(bad_ids)}"
            )

        original_columns[class_name] = list(prepared.columns)
        input_counts[class_name] = len(prepared)
        combined.append(prepared.assign(**{marker: class_name}))

    merged = pd.concat(combined, ignore_index=True, sort=False)
    discarded_indices: set[int] = set()
    duplicate_rows = merged.loc[merged["object_id"].duplicated(keep=False)]
    for _object_id, group in duplicate_rows.groupby("object_id", sort=False):
        separations = _pairwise_separations_arcsec(
            group["ra"].to_numpy(dtype=float),
            group["dec"].to_numpy(dtype=float),
        )
        discarded_positions = _duplicate_coordinate_positions(
            separations,
            tolerance_arcsec,
        )
        group_indices = group.index.to_numpy()
        discarded_indices.update(
            int(group_indices[position]) for position in discarded_positions
        )

    if discarded_indices:
        merged = merged.drop(index=sorted(discarded_indices))

    filtered: dict[str, pd.DataFrame] = {}
    discarded_counts: dict[str, int] = {}
    for class_name in catalogs:
        selected = merged.loc[
            merged[marker] == class_name,
            original_columns[class_name],
        ].reset_index(drop=True)
        filtered[class_name] = selected
        discarded_counts[class_name] = input_counts[class_name] - len(selected)
    return filtered, discarded_counts


def add_class_prefix_to_object_ids(
    frame: pd.DataFrame,
    class_name: str,
) -> pd.DataFrame:
    """Return a copy whose object IDs are prefixed with ``<class_name>_``."""
    if not class_name:
        raise ValueError("Class name cannot be empty")
    if "object_id" not in frame:
        raise ValueError("Missing object_id column")
    if frame["object_id"].isna().any():
        raise ValueError(f"Missing object_id value in class {class_name}")

    prefixed = frame.copy()
    prefixed["object_id"] = class_name + "_" + prefixed["object_id"].astype(str)
    return prefixed


def require_unique_object_ids(
    frame: pd.DataFrame,
    *,
    context: str = "DRAGON dataset",
) -> None:
    """Raise when class prefixing did not produce unique object IDs."""
    if "object_id" not in frame:
        raise ValueError(f"Missing object_id column in {context}")
    duplicate_ids = (
        frame.loc[
            frame["object_id"].duplicated(keep=False),
            "object_id",
        ]
        .astype(str)
        .unique()
    )
    if len(duplicate_ids):
        examples = ", ".join(duplicate_ids[:5])
        raise ValueError(
            f"Class-prefixed object_id values are not unique in {context}; "
            f"duplicate IDs remain within a class: {examples}"
        )
