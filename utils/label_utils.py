"""Class-label mapping helpers shared by preparation and inference."""

from pathlib import Path

import pandas as pd


def load_label_mapping(path, expected_classes=None):
    """Load and validate a contiguous ``labels.csv`` key/value mapping."""
    labels_path = Path(path)
    labels = pd.read_csv(labels_path)
    required = {"key", "value"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            f"Labels CSV is missing column(s) {sorted(missing)}: {labels_path}"
        )
    if labels.empty:
        raise ValueError(f"Labels CSV contains no rows: {labels_path}")
    values = pd.to_numeric(labels["value"], errors="coerce")
    if values.isna().any() or (values % 1 != 0).any():
        raise ValueError(f"Labels CSV contains a non-integer value: {labels_path}")
    values = values.astype(int)
    if sorted(values.tolist()) != list(range(len(labels))):
        raise ValueError(
            f"Labels CSV values must be contiguous from zero: {labels_path}"
        )
    keys = labels["key"].astype(str)
    if keys.duplicated().any():
        raise ValueError(f"Labels CSV contains duplicate keys: {labels_path}")
    if expected_classes is not None and len(labels) != expected_classes:
        raise ValueError(
            f"Labels CSV contains {len(labels)} classes, expected {expected_classes}: "
            f"{labels_path}"
        )
    return dict(sorted(zip(values.tolist(), keys.tolist())))


def label_mapping_frame(mapping):
    """Convert an index-to-name mapping to canonical labels.csv columns."""
    return pd.DataFrame(
        {
            "key": [mapping[index] for index in sorted(mapping)],
            "value": sorted(mapping),
        }
    )
