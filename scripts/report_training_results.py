"""Report metrics from experiment-level ``best_metrics.json`` files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


SPLITS = ("devel", "test")
BEST_METRICS_NAME = "best_metrics.json"
REQUIRED_SPLIT_METRICS = ("accuracy", "precision", "recall", "loss", "f1")


class ReportDataError(ValueError):
    """Raised when an experiment report cannot be interpreted reliably."""


@dataclass(frozen=True)
class ExperimentMetrics:
    directory: Path
    metrics_path: Path
    best_epoch: int
    best_devel_macro_f1: float
    stored_metrics: dict[str, float]
    confusion_matrices: dict[str, list[list[float]]]


@dataclass(frozen=True)
class SplitMetrics:
    split: str
    confusion_matrix: list[list[float]]
    total: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class: list[dict[str, float | str]]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: object, *, field: str, source: Path) -> float:
    if not _is_number(value):
        raise ReportDataError(f"{field} must be numeric in {source}")
    result = float(value)
    if not math.isfinite(result):
        raise ReportDataError(f"{field} must be finite in {source}")
    return result


def _load_labels_csv(path: Path) -> dict[int, str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = {"key", "value"} - fieldnames
            if missing:
                raise ReportDataError(
                    f"Labels CSV is missing column(s) {sorted(missing)}: {path}"
                )
            rows = list(reader)
    except OSError as exc:
        raise ReportDataError(f"Cannot read labels CSV {path}: {exc}") from exc

    if not rows:
        raise ReportDataError(f"Labels CSV contains no rows: {path}")

    mapping: dict[int, str] = {}
    seen_names: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        name = (row.get("key") or "").strip()
        raw_value = (row.get("value") or "").strip()
        if not name:
            raise ReportDataError(
                f"Labels CSV contains an empty key at line {line_number}: {path}"
            )
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ReportDataError(
                f"Labels CSV contains a non-integer value at line "
                f"{line_number}: {path}"
            ) from exc
        if value in mapping:
            raise ReportDataError(
                f"Labels CSV contains duplicate value {value}: {path}"
            )
        if name in seen_names:
            raise ReportDataError(
                f"Labels CSV contains duplicate key {name!r}: {path}"
            )
        mapping[value] = name
        seen_names.add(name)

    expected_values = list(range(len(mapping)))
    if sorted(mapping) != expected_values:
        raise ReportDataError(
            f"Labels CSV values must be contiguous from zero: {path}"
        )
    return {value: mapping[value] for value in expected_values}


def _discover_experiments(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ReportDataError(f"Experiment path is not a directory: {root}")

    direct_metrics = root / BEST_METRICS_NAME
    if direct_metrics.is_file():
        return [root]

    experiments = sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and (child / BEST_METRICS_NAME).is_file()
        ),
        key=lambda path: path.name,
    )
    if not experiments:
        raise ReportDataError(
            f"No {BEST_METRICS_NAME} found in {root} or its direct child directories"
        )
    return experiments


def _validate_confusion_matrix(
    raw_matrix: object,
    *,
    split: str,
    source: Path,
) -> list[list[float]]:
    if not isinstance(raw_matrix, list) or not raw_matrix:
        raise ReportDataError(
            f"confusion_matrices.{split} must be a non-empty matrix in {source}"
        )

    size = len(raw_matrix)
    matrix: list[list[float]] = []
    for row_index, raw_row in enumerate(raw_matrix):
        if not isinstance(raw_row, list) or len(raw_row) != size:
            raise ReportDataError(
                f"confusion_matrices.{split} must be square; row {row_index} "
                f"has length {len(raw_row) if isinstance(raw_row, list) else 'invalid'} "
                f"instead of {size} in {source}"
            )
        row: list[float] = []
        for column_index, raw_count in enumerate(raw_row):
            count = _finite_number(
                raw_count,
                field=(
                    f"confusion_matrices.{split}[{row_index}]"
                    f"[{column_index}]"
                ),
                source=source,
            )
            if count < 0:
                raise ReportDataError(
                    f"confusion_matrices.{split} contains a negative count in {source}"
                )
            row.append(count)
        matrix.append(row)
    return matrix


def _load_experiment(directory: Path) -> ExperimentMetrics:
    metrics_path = directory / BEST_METRICS_NAME
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReportDataError(f"Cannot read {metrics_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReportDataError(f"Invalid JSON in {metrics_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ReportDataError(f"{metrics_path} must contain a JSON object")

    required = {
        "best_epoch",
        "best_devel_macro_f1",
        "metrics",
        "confusion_matrices",
    }
    missing = required - payload.keys()
    if missing:
        raise ReportDataError(
            f"{metrics_path} is missing required field(s): {sorted(missing)}"
        )

    raw_epoch = payload["best_epoch"]
    if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int) or raw_epoch < 1:
        raise ReportDataError(
            f"best_epoch must be a positive integer in {metrics_path}"
        )
    best_score = _finite_number(
        payload["best_devel_macro_f1"],
        field="best_devel_macro_f1",
        source=metrics_path,
    )
    if not 0.0 <= best_score <= 1.0:
        raise ReportDataError(
            f"best_devel_macro_f1 must be between zero and one in {metrics_path}"
        )

    raw_metrics = payload["metrics"]
    if not isinstance(raw_metrics, dict):
        raise ReportDataError(f"metrics must be an object in {metrics_path}")
    stored_metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if not isinstance(key, str):
            raise ReportDataError(f"metrics keys must be strings in {metrics_path}")
        stored_metrics[key] = _finite_number(
            value,
            field=f"metrics.{key}",
            source=metrics_path,
        )

    required_metric_keys = {
        f"{split}_{metric}"
        for split in SPLITS
        for metric in REQUIRED_SPLIT_METRICS
    }
    required_metric_keys.add("train_loss")
    missing_metric_keys = required_metric_keys - stored_metrics.keys()
    if missing_metric_keys:
        raise ReportDataError(
            f"metrics is missing required field(s) "
            f"{sorted(missing_metric_keys)} in {metrics_path}"
        )
    if not math.isclose(
        stored_metrics["devel_f1"],
        best_score,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ReportDataError(
            f"metrics.devel_f1 does not match best_devel_macro_f1 in "
            f"{metrics_path}"
        )

    raw_matrices = payload["confusion_matrices"]
    if not isinstance(raw_matrices, dict):
        raise ReportDataError(
            f"confusion_matrices must be an object in {metrics_path}"
        )
    missing_splits = set(SPLITS) - raw_matrices.keys()
    if missing_splits:
        raise ReportDataError(
            f"confusion_matrices is missing split(s) {sorted(missing_splits)} "
            f"in {metrics_path}"
        )

    matrices = {
        split: _validate_confusion_matrix(
            raw_matrices[split], split=split, source=metrics_path
        )
        for split in SPLITS
    }
    dimensions = {len(matrix) for matrix in matrices.values()}
    if len(dimensions) != 1:
        raise ReportDataError(
            f"All confusion matrices must have the same dimensions in {metrics_path}"
        )

    return ExperimentMetrics(
        directory=directory,
        metrics_path=metrics_path,
        best_epoch=raw_epoch,
        best_devel_macro_f1=best_score,
        stored_metrics=stored_metrics,
        confusion_matrices=matrices,
    )


def _compute_metrics(
    split: str,
    matrix: list[list[float]],
    labels: list[str],
) -> SplitMetrics:
    size = len(matrix)
    if len(labels) != size:
        raise ReportDataError(
            f"Labels contain {len(labels)} classes but confusion matrices contain "
            f"{size} classes"
        )

    supports = [sum(row) for row in matrix]
    predicted = [
        sum(matrix[row_index][column_index] for row_index in range(size))
        for column_index in range(size)
    ]
    per_class: list[dict[str, float | str]] = []
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        precision = true_positive / predicted[index] if predicted[index] else 0.0
        recall = true_positive / supports[index] if supports[index] else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append(
            {
                "class": label,
                "support": supports[index],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    total = sum(supports)
    correct = sum(matrix[index][index] for index in range(size))
    accuracy = correct / total if total else 0.0
    macro_precision = sum(float(row["precision"]) for row in per_class) / size
    macro_recall = sum(float(row["recall"]) for row in per_class) / size
    macro_f1 = sum(float(row["f1"]) for row in per_class) / size
    weighted_precision = (
        sum(
            float(row["precision"]) * float(row["support"])
            for row in per_class
        )
        / total
        if total
        else 0.0
    )
    weighted_recall = (
        sum(float(row["recall"]) * float(row["support"]) for row in per_class)
        / total
        if total
        else 0.0
    )
    weighted_f1 = (
        sum(float(row["f1"]) * float(row["support"]) for row in per_class)
        / total
        if total
        else 0.0
    )
    return SplitMetrics(
        split=split,
        confusion_matrix=matrix,
        total=total,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        per_class=per_class,
    )


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


def _format_count(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _format_confusion_cell(count: float, actual_class_total: float) -> str:
    percentage = (
        100.0 * count / actual_class_total if actual_class_total else 0.0
    )
    return f"{_format_count(count)} ({percentage:.1f}%)"


def _print_confusion_matrix(matrix: list[list[float]], labels: list[str]) -> None:
    values = []
    for row in matrix:
        actual_class_total = sum(row)
        values.append(
            [
                _format_confusion_cell(count, actual_class_total)
                for count in row
            ]
        )
    corner_label = "actual\\predicted"
    label_width = max(len(corner_label), *(len(label) for label in labels))
    column_widths = [
        max(len(labels[index]), *(len(row[index]) for row in values))
        for index in range(len(labels))
    ]
    header = f"      {corner_label:<{label_width}}  " + "  ".join(
        f"{label:>{column_widths[index]}}" for index, label in enumerate(labels)
    )
    print("    confusion matrix (absolute count (percentage of actual class)):")
    print(header)
    for row_index, label in enumerate(labels):
        print(
            f"      {label:<{label_width}}  "
            + "  ".join(
                f"{values[row_index][column_index]:>{column_widths[column_index]}}"
                for column_index in range(len(labels))
            )
        )


def _print_split(metrics: SplitMetrics, stored_metrics: dict[str, float]) -> None:
    print(f"  Split: {metrics.split}")
    print(
        f"    samples={_format_count(metrics.total)} | "
        f"accuracy={_format_metric(metrics.accuracy)} | "
        f"macro P/R/F1={_format_metric(metrics.macro_precision)}/"
        f"{_format_metric(metrics.macro_recall)}/"
        f"{_format_metric(metrics.macro_f1)} | "
        f"weighted P/R/F1={_format_metric(metrics.weighted_precision)}/"
        f"{_format_metric(metrics.weighted_recall)}/"
        f"{_format_metric(metrics.weighted_f1)}"
    )
    loss = stored_metrics.get(f"{metrics.split}_loss")
    if loss is not None:
        print(f"    loss={loss:.6f}")

    class_width = max(
        len("class"),
        *(len(str(row["class"])) for row in metrics.per_class),
    )
    print("    per-class:")
    print(f"      {'class':<{class_width}}  support  precision  recall      f1")
    for row in metrics.per_class:
        support = _format_count(float(row["support"]))
        print(
            f"      {str(row['class']):<{class_width}}  "
            f"{support:>7}  "
            f"{_format_metric(float(row['precision'])):>9}  "
            f"{_format_metric(float(row['recall'])):>6}  "
            f"{_format_metric(float(row['f1'])):>6}"
        )
    _print_confusion_matrix(
        metrics.confusion_matrix,
        [str(row["class"]) for row in metrics.per_class],
    )


def _resolve_labels(args: argparse.Namespace, class_count: int) -> list[str]:
    labels_path: Path | None = args.labels
    if labels_path is None and args.data_dir is not None:
        labels_path = args.data_dir / "labels.csv"

    if labels_path is None:
        raise ReportDataError(
            "Class names are required; provide --labels PATH or --data-dir DIR"
        )
    if not labels_path.is_file():
        raise ReportDataError(f"Labels CSV not found: {labels_path}")

    mapping = _load_labels_csv(labels_path)
    if len(mapping) != class_count:
        raise ReportDataError(
            f"Labels CSV contains {len(mapping)} classes but confusion matrices "
            f"contain {class_count}: {labels_path}"
        )
    return [mapping[index] for index in range(class_count)]


def _requested_splits(selected: str) -> tuple[str, ...]:
    return SPLITS if selected == "all" else (selected,)


def _report_experiment(
    experiment: ExperimentMetrics,
    *,
    labels: list[str],
    selected_split: str,
) -> None:
    print(f"Experiment: {experiment.directory.name}")
    print(f"  directory: {experiment.directory}")
    print(f"  metrics: {experiment.metrics_path}")
    print(f"  best epoch: {experiment.best_epoch}")
    train_loss = experiment.stored_metrics.get("train_loss")
    if train_loss is not None:
        print(f"  recorded train loss at best epoch: {train_loss:.6f}")
    print(
        "  recorded best devel macro-F1: "
        f"{experiment.best_devel_macro_f1:.6f}"
    )
    for split in _requested_splits(selected_split):
        split_metrics = _compute_metrics(
            split,
            experiment.confusion_matrices[split],
            labels,
        )
        _print_split(split_metrics, experiment.stored_metrics)
    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report experiment-level best metrics. The path may be one experiment "
            "directory containing best_metrics.json or a parent whose direct child "
            "directories contain best_metrics.json."
        )
    )
    parser.add_argument("experiment_path", type=Path)
    parser.add_argument(
        "--labels",
        type=Path,
        help="Explicit labels.csv path; overrides --data-dir/labels.csv.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Dataset directory containing the labels.csv used for class names when "
            "--labels is omitted."
        ),
    )
    parser.add_argument(
        "--split",
        choices=[*SPLITS, "all"],
        default="all",
        help="Which split(s) to report (default: all).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        experiment_dirs = _discover_experiments(args.experiment_path)
    except ReportDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    reported = 0
    failures: list[tuple[Path, str]] = []
    for directory in experiment_dirs:
        try:
            experiment = _load_experiment(directory)
            class_count = len(experiment.confusion_matrices["devel"])
            labels = _resolve_labels(args, class_count)
            _report_experiment(
                experiment,
                labels=labels,
                selected_split=args.split,
            )
            reported += 1
        except ReportDataError as exc:
            failures.append((directory, str(exc)))
            print(f"Experiment failed: {directory}: {exc}", file=sys.stderr)

    print(f"Summary: reported={reported}, failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
