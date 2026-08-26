#!/usr/bin/env python3
"""Calibrate class confidence thresholds from a labelled dataset split.

The command performs one model pass, retains every softmax column, and then
computes one-vs-rest confidence curves for every class declared by
``labels.csv``.  Thresholds are calibrated against the worst individual
negative class rather than the pooled negative population.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

# Batch/HPC nodes frequently have neither a display nor a writable home
# directory.  Configure matplotlib before importing pyplot and give its cache a
# private, automatically cleaned temporary directory unless the caller already
# selected one.
os.environ.setdefault("MPLBACKEND", "Agg")
_MPL_CONFIG_TMPDIR = None
if "MPLCONFIGDIR" not in os.environ:
    _MPL_CONFIG_TMPDIR = tempfile.TemporaryDirectory(
        prefix="dragon-cnn-matplotlib-"
    )
    os.environ["MPLCONFIGDIR"] = _MPL_CONFIG_TMPDIR.name

import click
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from cnn import DRAGON, DRAGON_CUTOUT_SIZE
from data_preprocessing import HDF5Dataset, get_data_loader
from utils import (
    asinh_normalize,
    discover_devices,
    load_label_mapping,
    load_model_state,
    normalization_kwargs_from_stats,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "dragon_confidence_thresholds"
PROBABILITY_COLUMN_PREFIX = "probability__"


@dataclass(frozen=True)
class NegativeClassThreshold:
    """The FPR constraint contributed by one negative class."""

    class_index: int
    n_negative: int
    allowed_false_positives: int
    required_threshold: float
    false_positives: int
    empirical_fpr: float


@dataclass(frozen=True)
class ClassThreshold:
    """Worst-negative-class calibration result for one target class."""

    class_index: int
    threshold: float
    n_positive: int
    true_positives: int
    empirical_tpr: float
    pooled_fpr: float
    macro_fpr: float
    empirical_worst_class_fpr: float
    worst_negative_class_index: int
    per_negative_class: dict[int, NegativeClassThreshold]


def _validate_target_fpr(target_fpr: float) -> float:
    if isinstance(target_fpr, bool):
        raise ValueError("target_fpr must be a finite number between zero and one.")
    target_fpr = float(target_fpr)
    if not math.isfinite(target_fpr) or not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must be a finite number between zero and one.")
    return target_fpr


def _allowed_false_positives(target_fpr: float, sample_count: int) -> int:
    """Return floor(target_fpr * sample_count) without binary-float drift."""
    product = Decimal(str(target_fpr)) * Decimal(sample_count)
    return int(product.to_integral_value(rounding=ROUND_FLOOR))


def _next_score(score: np.generic | float) -> float:
    """Return the next finite float64 value above ``score``."""
    result = math.nextafter(float(score), math.inf)
    if not math.isfinite(result):
        raise ValueError("Could not construct a finite confidence threshold.")
    return result


def threshold_for_group_fpr(
    negative_scores: np.ndarray,
    target_fpr: float,
) -> tuple[float, int]:
    """Find a tie-safe ``>=`` threshold for one negative population.

    The returned threshold is the float64 value immediately above the first
    disallowed score.  Consequently tied scores are always accepted or
    rejected together.
    """
    target_fpr = _validate_target_fpr(target_fpr)
    scores = np.asarray(negative_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Each negative class must contain at least one score.")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Softmax scores must be finite and between zero and one.")

    allowed = _allowed_false_positives(target_fpr, scores.size)
    if allowed >= scores.size:
        return 0.0, allowed

    descending = np.sort(scores)[::-1]
    first_disallowed = descending[allowed]
    threshold = _next_score(first_disallowed)
    return threshold, allowed


def _count_at_or_above(scores: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(scores >= threshold))


def calibrate_worst_class_threshold(
    labels: np.ndarray,
    target_scores: np.ndarray,
    *,
    target_class_index: int,
    class_indices: list[int] | tuple[int, ...] | range,
    target_fpr: float,
) -> ClassThreshold:
    """Calibrate one target class by constraining every negative class."""
    target_fpr = _validate_target_fpr(target_fpr)
    labels = np.asarray(labels)
    scores = np.asarray(target_scores, dtype=np.float64)
    class_indices = tuple(int(index) for index in class_indices)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and target_scores must be aligned 1-D arrays.")
    if target_class_index not in class_indices:
        raise ValueError("target_class_index is absent from class_indices.")
    if len(class_indices) < 2 or len(set(class_indices)) != len(class_indices):
        raise ValueError("At least two unique class indices are required.")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Softmax scores must be finite and between zero and one.")

    n_positive = int(np.count_nonzero(labels == target_class_index))
    if n_positive == 0:
        raise ValueError(
            f"Calibration split contains no rows for class {target_class_index}."
        )

    provisional: dict[int, tuple[np.ndarray, float, int]] = {}
    for negative_index in class_indices:
        if negative_index == target_class_index:
            continue
        negative_scores = scores[labels == negative_index]
        if negative_scores.size == 0:
            raise ValueError(
                f"Calibration split contains no rows for negative class "
                f"{negative_index} when calibrating class {target_class_index}."
            )
        required_threshold, allowed = threshold_for_group_fpr(
            negative_scores,
            target_fpr,
        )
        provisional[negative_index] = (
            negative_scores,
            required_threshold,
            allowed,
        )

    # Enforcing every per-negative-class constraint is equivalent to taking
    # the maximum of their independently calibrated thresholds.
    threshold = max(item[1] for item in provisional.values())
    per_negative: dict[int, NegativeClassThreshold] = {}
    total_false_positives = 0
    total_negatives = 0
    for negative_index, (
        negative_scores,
        required_threshold,
        allowed,
    ) in provisional.items():
        false_positives = _count_at_or_above(negative_scores, threshold)
        n_negative = int(negative_scores.size)
        empirical_fpr = false_positives / n_negative
        # Compare integer counts here: ``allowed`` is the exact finite-sample
        # constraint, whereas comparing two rounded binary floats can report a
        # spurious violation when the empirical FPR lies exactly on the limit.
        if false_positives > allowed:
            raise AssertionError(
                "Internal error: calibrated threshold violates a negative-class "
                "FPR constraint."
            )
        per_negative[negative_index] = NegativeClassThreshold(
            class_index=negative_index,
            n_negative=n_negative,
            allowed_false_positives=allowed,
            required_threshold=required_threshold,
            false_positives=false_positives,
            empirical_fpr=empirical_fpr,
        )
        total_false_positives += false_positives
        total_negatives += n_negative

    positive_scores = scores[labels == target_class_index]
    true_positives = _count_at_or_above(positive_scores, threshold)
    empirical_tpr = true_positives / n_positive
    pooled_fpr = total_false_positives / total_negatives
    per_negative_fprs = [item.empirical_fpr for item in per_negative.values()]
    macro_fpr = float(np.mean(per_negative_fprs))
    worst_negative_index = max(
        per_negative,
        key=lambda index: per_negative[index].empirical_fpr,
    )
    worst_fpr = per_negative[worst_negative_index].empirical_fpr

    return ClassThreshold(
        class_index=target_class_index,
        threshold=threshold,
        n_positive=n_positive,
        true_positives=true_positives,
        empirical_tpr=empirical_tpr,
        pooled_fpr=pooled_fpr,
        macro_fpr=macro_fpr,
        empirical_worst_class_fpr=worst_fpr,
        worst_negative_class_index=worst_negative_index,
        per_negative_class=per_negative,
    )


def calibrate_all_classes(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_fpr: float,
) -> dict[int, ClassThreshold]:
    """Calibrate all probability columns using one shared labelled split."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or labels.shape != (probabilities.shape[0],):
        raise ValueError("probabilities must be N x C and aligned with labels.")
    if probabilities.shape[1] < 2:
        raise ValueError("Confidence calibration requires at least two classes.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Softmax probabilities contain non-finite values.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Softmax probabilities must be between zero and one.")
    expected = set(range(probabilities.shape[1]))
    observed = set(np.unique(labels).astype(int).tolist())
    unexpected = observed - expected
    if unexpected:
        raise ValueError(f"Labels contain out-of-range class indices: {unexpected}")
    missing = expected - observed
    if missing:
        raise ValueError(
            "Worst-class FPR calibration requires every labels.csv class in the "
            f"selected split; missing indices: {sorted(missing)}"
        )

    class_indices = range(probabilities.shape[1])
    return {
        index: calibrate_worst_class_threshold(
            labels,
            probabilities[:, index],
            target_class_index=index,
            class_indices=class_indices,
            target_fpr=target_fpr,
        )
        for index in class_indices
    }


def _counts_at_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    sorted_scores = np.sort(np.asarray(scores))
    return sorted_scores.size - np.searchsorted(
        sorted_scores,
        thresholds,
        side="left",
    )


def compute_curve_tables(
    labels: np.ndarray,
    probabilities: np.ndarray,
    label_names: dict[int, str],
    calibrations: dict[int, ClassThreshold],
    *,
    curve_points: int = 1001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sampled aggregate and per-negative-class confidence curves."""
    if curve_points < 2:
        raise ValueError("curve_points must be at least two.")
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    aggregate_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []

    base_thresholds = np.linspace(0.0, 1.0, curve_points, dtype=np.float64)
    for target_index in range(probabilities.shape[1]):
        target_name = label_names[target_index]
        calibration = calibrations[target_index]
        thresholds = np.unique(
            np.append(base_thresholds, calibration.threshold)
        )
        target_scores = probabilities[:, target_index]
        positive_scores = target_scores[labels == target_index]
        true_positives = _counts_at_thresholds(positive_scores, thresholds)
        false_negatives = positive_scores.size - true_positives
        recall = true_positives / positive_scores.size

        negative_indices = [
            index for index in range(probabilities.shape[1])
            if index != target_index
        ]
        false_positives_by_class = []
        negative_sizes = []
        for negative_index in negative_indices:
            negative_scores = target_scores[labels == negative_index]
            false_positives = _counts_at_thresholds(
                negative_scores,
                thresholds,
            )
            false_positives_by_class.append(false_positives)
            negative_sizes.append(negative_scores.size)
            negative_fpr = false_positives / negative_scores.size
            for row_index, threshold in enumerate(thresholds):
                negative_rows.append(
                    {
                        "target_class_index": target_index,
                        "target_class": target_name,
                        "negative_class_index": negative_index,
                        "negative_class": label_names[negative_index],
                        "threshold": float(threshold),
                        "false_positives": int(false_positives[row_index]),
                        "n_negative": int(negative_scores.size),
                        "fpr": float(negative_fpr[row_index]),
                        "is_calibrated_threshold": bool(
                            threshold == calibration.threshold
                        ),
                    }
                )

        fp_matrix = np.asarray(false_positives_by_class, dtype=np.int64)
        size_array = np.asarray(negative_sizes, dtype=np.int64)[:, None]
        fpr_matrix = fp_matrix / size_array
        false_positives = fp_matrix.sum(axis=0)
        total_negatives = int(size_array.sum())
        true_negatives = total_negatives - false_positives
        pooled_fpr = false_positives / total_negatives
        macro_fpr = fpr_matrix.mean(axis=0)
        worst_positions = np.argmax(fpr_matrix, axis=0)
        worst_fpr = fpr_matrix[
            worst_positions,
            np.arange(thresholds.size),
        ]
        predicted_positive = true_positives + false_positives
        precision = np.divide(
            true_positives,
            predicted_positive,
            out=np.ones_like(true_positives, dtype=np.float64),
            where=predicted_positive != 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )

        for row_index, threshold in enumerate(thresholds):
            worst_negative_index = negative_indices[worst_positions[row_index]]
            aggregate_rows.append(
                {
                    "target_class_index": target_index,
                    "target_class": target_name,
                    "threshold": float(threshold),
                    "tp": int(true_positives[row_index]),
                    "fp": int(false_positives[row_index]),
                    "tn": int(true_negatives[row_index]),
                    "fn": int(false_negatives[row_index]),
                    "precision": float(precision[row_index]),
                    "recall": float(recall[row_index]),
                    "tpr": float(recall[row_index]),
                    "f1": float(f1[row_index]),
                    "pooled_fpr": float(pooled_fpr[row_index]),
                    "macro_fpr": float(macro_fpr[row_index]),
                    "worst_class_fpr": float(worst_fpr[row_index]),
                    "worst_negative_class_index": worst_negative_index,
                    "worst_negative_class": label_names[worst_negative_index],
                    "is_calibrated_threshold": bool(
                        threshold == calibration.threshold
                    ),
                }
            )

    return pd.DataFrame(aggregate_rows), pd.DataFrame(negative_rows)


def predict_all_probabilities(
    model_path: Path,
    dataset: HDF5Dataset,
    *,
    channels: int,
    num_classes: int,
    normalization_kwargs: dict[str, object],
    batch_size: int,
    n_workers: int,
    parallel: bool,
) -> np.ndarray:
    """Run one inference pass and return the complete N x C softmax array."""
    device = discover_devices()
    model = DRAGON(channels=channels, num_classes=num_classes)
    logging.info("Loading model from %s", model_path)
    load_model_state(model, model_path, device=device)
    if parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
    )
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _labels in tqdm(loader, desc="Calibration inference"):
            images = images.to(device)
            images = asinh_normalize(images, **normalization_kwargs)
            logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise ValueError(
                    "Model output shape does not match labels.csv: "
                    f"{tuple(logits.shape)}"
                )
            batch_probabilities = nn.functional.softmax(logits, dim=1)
            batches.append(batch_probabilities.float().cpu().numpy())

    if not batches:
        raise ValueError("Calibration split contains no rows.")
    probabilities = np.concatenate(batches, axis=0)
    if probabilities.shape != (len(dataset), num_classes):
        raise AssertionError("Inference output is not aligned with the dataset split.")
    return probabilities


def _write_scores(
    path: Path,
    dataset: HDF5Dataset,
    probabilities: np.ndarray,
    label_names: dict[int, str],
) -> None:
    labels = np.asarray(dataset.labels, dtype=np.int64)
    result = pd.DataFrame(
        {
            "object_id": dataset.data_info["object_id"].astype("string"),
            "true_label": labels,
            "true_class": [label_names[int(index)] for index in labels],
        }
    )
    for class_index in range(probabilities.shape[1]):
        result[f"{PROBABILITY_COLUMN_PREFIX}{class_index}"] = probabilities[
            :, class_index
        ]
    # Preserve the exact float32 score when pandas later reads it as float64.
    # A short decimal representation can round to the other side of a
    # float64-nextafter threshold and invalidate the documented >= rule.
    result.to_csv(path, index=False, float_format="%.17g")


def build_manifest(
    *,
    model_path: Path,
    data_dir: Path,
    labels_path: Path,
    normalization_stats_path: Path,
    split_path: Path,
    split_slug: str,
    split: str,
    sample_count: int,
    label_names: dict[int, str],
    target_fpr: float,
    calibrations: dict[int, ClassThreshold],
) -> dict[str, object]:
    """Create the stable threshold-manifest payload consumed downstream."""
    classes: dict[str, object] = {}
    for class_index, class_name in label_names.items():
        calibration = calibrations[class_index]
        false_positives = sum(
            item.false_positives
            for item in calibration.per_negative_class.values()
        )
        precision_denominator = calibration.true_positives + false_positives
        precision = (
            calibration.true_positives / precision_denominator
            if precision_denominator
            else 1.0
        )
        f1_denominator = precision + calibration.empirical_tpr
        f1 = (
            2.0 * precision * calibration.empirical_tpr / f1_denominator
            if f1_denominator
            else 0.0
        )
        per_negative = {}
        for negative_index, result in calibration.per_negative_class.items():
            per_negative[label_names[negative_index]] = {
                "class_index": negative_index,
                "n_negative": result.n_negative,
                "allowed_false_positives": result.allowed_false_positives,
                "threshold_for_target_fpr": result.required_threshold,
                "false_positives": result.false_positives,
                "empirical_fpr": result.empirical_fpr,
            }
        classes[class_name] = {
            "class_index": class_index,
            "threshold": calibration.threshold,
            "empirical_fpr": calibration.empirical_worst_class_fpr,
            "empirical_worst_class_fpr": (
                calibration.empirical_worst_class_fpr
            ),
            "worst_negative_class": label_names[
                calibration.worst_negative_class_index
            ],
            "pooled_fpr": calibration.pooled_fpr,
            "macro_fpr": calibration.macro_fpr,
            "n_positive": calibration.n_positive,
            "true_positives": calibration.true_positives,
            "empirical_tpr": calibration.empirical_tpr,
            "precision": precision,
            "f1": f1,
            "per_negative_class": per_negative,
        }

    resolved_data_dir = data_dir.resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_fpr": target_fpr,
        "fpr_aggregation": "worst_class",
        "comparison": ">=",
        "model": {
            "path": str(model_path.resolve()),
        },
        "labels": {
            "path": str(labels_path.resolve()),
            "mapping": {
                class_name: class_index
                for class_index, class_name in label_names.items()
            },
        },
        "data": {
            "path": str(resolved_data_dir),
            "split_slug": split_slug,
            "split": split,
            "n_samples": sample_count,
            "split_path": str(split_path.resolve()),
            "normalization_stats_path": str(
                normalization_stats_path.resolve()
            ),
        },
        "calibration": {
            "split_slug": split_slug,
            "split": split,
            "n_samples": sample_count,
        },
        "classes": classes,
    }


def _finish_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_curves(
    curves: pd.DataFrame,
    output_dir: Path,
    label_names: dict[int, str],
    target_fpr: float,
) -> None:
    specifications = (
        ("FPR_curve.png", "threshold", "worst_class_fpr", "Confidence", "Worst-class FPR"),
        ("Precision_curve.png", "threshold", "precision", "Confidence", "Precision"),
        ("Recall_curve.png", "threshold", "recall", "Confidence", "Recall / TPR"),
        ("F1_curve.png", "threshold", "f1", "Confidence", "F1"),
        ("ROC_curve.png", "pooled_fpr", "tpr", "Pooled FPR", "TPR"),
        ("PR_curve.png", "recall", "precision", "Recall", "Precision"),
    )
    colors = plt.get_cmap("tab20")
    for filename, x_column, y_column, x_label, y_label in specifications:
        fig, axis = plt.subplots(figsize=(8.0, 6.0))
        for class_index, class_name in label_names.items():
            class_rows = curves[
                curves["target_class_index"] == class_index
            ]
            axis.plot(
                class_rows[x_column],
                class_rows[y_column],
                label=class_name,
                color=colors(class_index % 20),
                linewidth=1.6,
            )
        if filename == "FPR_curve.png":
            axis.axhline(
                target_fpr,
                color="black",
                linestyle=":",
                linewidth=1.2,
                label=f"target FPR = {target_fpr:g}",
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        _finish_figure(fig, output_dir / filename)


def _plot_negative_class_fpr(
    aggregate_curves: pd.DataFrame,
    negative_curves: pd.DataFrame,
    output_dir: Path,
    label_names: dict[int, str],
    calibrations: dict[int, ClassThreshold],
    target_fpr: float,
) -> None:
    classes_dir = output_dir / "classes"
    classes_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab20")
    for target_index, target_name in label_names.items():
        fig, axis = plt.subplots(figsize=(8.5, 6.0))
        calibration = calibrations[target_index]
        worst_fpr = calibration.empirical_worst_class_fpr
        worst_negative_indices = {
            negative_index
            for negative_index, result in calibration.per_negative_class.items()
            if math.isclose(
                result.empirical_fpr,
                worst_fpr,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        }
        worst_negative_names = ", ".join(
            label_names[index] for index in sorted(worst_negative_indices)
        )
        target_negative_rows = negative_curves[
            negative_curves["target_class_index"] == target_index
        ]
        for negative_index, negative_name in label_names.items():
            if negative_index == target_index:
                continue
            rows = target_negative_rows[
                target_negative_rows["negative_class_index"] == negative_index
            ]
            axis.plot(
                rows["threshold"],
                rows["fpr"],
                color=colors(negative_index % 20),
                linewidth=(
                    1.8 if negative_index in worst_negative_indices else 1.25
                ),
                alpha=1.0 if negative_index in worst_negative_indices else 0.9,
                zorder=3,
                label=(
                    f"negative: {negative_name}"
                    + (
                        " (worst @ calibrated threshold)"
                        if negative_index in worst_negative_indices
                        else ""
                    )
                ),
            )
        aggregate_rows = aggregate_curves[
            aggregate_curves["target_class_index"] == target_index
        ]
        for column, label, linestyle in (
            ("pooled_fpr", "pooled", "-"),
            ("macro_fpr", "macro over negative classes", "-."),
            ("worst_class_fpr", "worst negative class", "--"),
        ):
            axis.plot(
                aggregate_rows["threshold"],
                aggregate_rows[column],
                color="black",
                linestyle=linestyle,
                linewidth=0.7,
                alpha=0.35,
                zorder=4,
                label=label,
            )
        axis.axhline(
            target_fpr,
            color="grey",
            linestyle=":",
            linewidth=0.9,
            alpha=0.7,
            zorder=1.5,
            label=f"target FPR = {target_fpr:g}",
        )
        axis.axvline(
            calibrations[target_index].threshold,
            color="red",
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
            zorder=1.5,
            label=(
                "calibrated threshold = "
                f"{calibration.threshold:.6g}"
            ),
        )
        axis.scatter(
            [calibration.threshold],
            [worst_fpr],
            marker="*",
            s=110,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            label=f"worst at calibrated threshold: {worst_negative_names}",
        )
        axis.set_title(
            f"{target_name}: FPR by negative class\n"
            f"Worst at calibrated threshold: {worst_negative_names}"
        )
        axis.set_xlabel("Confidence")
        axis.set_ylabel("FPR")
        axis.set_xlim(
            0.0,
            max(1.0, calibration.threshold),
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_axisbelow(True)
        axis.grid(alpha=0.25)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        class_dir = classes_dir / f"class_{target_index:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        _finish_figure(fig, class_dir / "FPR_by_negative_class.png")


@click.command()
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Best model.pt checkpoint selected during training.",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Dataset directory containing labels.csv, splits/, and tensors/.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory (default: MODEL_PATH parent/confidence_curves).",
)
@click.option("--split-slug", type=str, required=True)
@click.option(
    "--split",
    type=click.Choice(["train", "devel", "test"]),
    default="devel",
    show_default=True,
)
@click.option(
    "--normalization-stats",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Normalization JSON (default: DATA_DIR/normalization_stats.json).",
)
@click.option(
    "--target-fpr",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.001,
    show_default=True,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--n-workers", type=click.IntRange(min=0), default=4, show_default=True)
@click.option("--parallel/--no-parallel", default=True, show_default=True)
@click.option(
    "--curve-points",
    type=click.IntRange(min=2),
    default=1001,
    show_default=True,
    help="Number of evenly spaced display thresholds from zero to one.",
)
def main(
    model_path: Path,
    data_dir: Path,
    output_dir: Path | None,
    split_slug: str,
    split: str,
    normalization_stats: Path | None,
    target_fpr: float,
    batch_size: int,
    n_workers: int,
    parallel: bool,
    curve_points: int,
) -> None:
    """Create all-class confidence curves and worst-class FPR thresholds."""
    split_slug = split_slug.strip()
    if not split_slug:
        raise click.BadParameter("must not be empty", param_hint="--split-slug")

    labels_path = data_dir / "labels.csv"
    if not labels_path.is_file():
        raise click.ClickException(f"Dataset labels.csv not found: {labels_path}")
    try:
        label_names = load_label_mapping(labels_path)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if len(label_names) < 2:
        raise click.ClickException("labels.csv must declare at least two classes.")

    split_path = data_dir / "splits" / f"{split_slug}-{split}.csv"
    if not split_path.is_file():
        raise click.ClickException(f"Dataset split not found: {split_path}")
    normalization_stats_path = (
        normalization_stats
        if normalization_stats is not None
        else data_dir / "normalization_stats.json"
    )
    if not normalization_stats_path.is_file():
        raise click.ClickException(
            "Normalization statistics not found: "
            f"{normalization_stats_path}"
        )

    try:
        dataset = HDF5Dataset(
            data_dir=data_dir,
            slug=split_slug,
            split=split,
            load_labels=True,
        )
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise click.ClickException(f"Cannot load calibration split: {exc}") from exc

    stored_shape = dataset.h5_image_shape[1:]
    channels, height, width = stored_shape
    if (height, width) != (DRAGON_CUTOUT_SIZE, DRAGON_CUTOUT_SIZE):
        raise click.ClickException(
            "Stored HDF5 cutout shape is incompatible with DRAGON: "
            f"{stored_shape}"
        )
    try:
        normalization_kwargs = normalization_kwargs_from_stats(
            normalization_stats_path,
            channels=channels,
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"Invalid normalization statistics: {exc}"
        ) from exc

    logging.info(
        "Running one inference pass for %d rows and %d classes on split %s/%s.",
        len(dataset),
        len(label_names),
        split_slug,
        split,
    )
    try:
        probabilities = predict_all_probabilities(
            model_path,
            dataset,
            channels=channels,
            num_classes=len(label_names),
            normalization_kwargs=normalization_kwargs,
            batch_size=batch_size,
            n_workers=n_workers,
            parallel=parallel,
        )
        calibrations = calibrate_all_classes(
            np.asarray(dataset.labels, dtype=np.int64),
            probabilities,
            target_fpr=target_fpr,
        )
        curves, negative_curves = compute_curve_tables(
            np.asarray(dataset.labels, dtype=np.int64),
            probabilities,
            label_names,
            calibrations,
            curve_points=curve_points,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    output_dir = output_dir or model_path.parent / "confidence_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.csv"
    curves_path = output_dir / "curves.csv"
    negative_curves_path = output_dir / "negative_class_fpr.csv"
    manifest_path = output_dir / "thresholds.json"

    _write_scores(scores_path, dataset, probabilities, label_names)
    curves.to_csv(curves_path, index=False, float_format="%.17g")
    negative_curves.to_csv(
        negative_curves_path,
        index=False,
        float_format="%.17g",
    )
    manifest = build_manifest(
        model_path=model_path,
        data_dir=data_dir,
        labels_path=labels_path,
        normalization_stats_path=normalization_stats_path,
        split_path=split_path,
        split_slug=split_slug,
        split=split,
        sample_count=len(dataset),
        label_names=label_names,
        target_fpr=target_fpr,
        calibrations=calibrations,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot_combined_curves(curves, output_dir, label_names, target_fpr)
    _plot_negative_class_fpr(
        curves,
        negative_curves,
        output_dir,
        label_names,
        calibrations,
        target_fpr,
    )

    logging.info("Saved calibration scores to %s", scores_path)
    logging.info("Saved confidence curves to %s", curves_path)
    logging.info("Saved worst-class threshold manifest to %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
