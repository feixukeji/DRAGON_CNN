#!/usr/bin/env python3
"""Summarize locally stored W&B training results.

Example:
    python -m scripts.report_training_results /path/to/dragon_runs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TABLE_NAME_RE = re.compile(
    r"^(?P<split>.+)_confusion_matrix_table_(?P<step>\d+)_.+\.table\.json$"
)
TABLE_HISTORY_KEY_RE = re.compile(r"^(?P<split>.+)_confusion_matrix_table$")


class ReportDataError(ValueError):
    """Raised when a stored run cannot be reported reliably."""


@dataclass
class ConfusionTable:
    labels: list[str]
    matrix: list[list[float]]


@dataclass
class RunTransactionData:
    path: Path
    tables: dict[str, dict[int, Path]]
    summary: dict[str, float]


@dataclass
class SplitMetrics:
    split: str
    step: int
    confusion_matrix: list[list[float]]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class: list[dict[str, float | str]]


def _load_labels_csv(path: Path) -> dict[int, str]:
    """Load labels without importing the pandas-dependent training utilities."""

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
                f"Labels CSV contains a non-integer value at line {line_number}: {path}"
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

    ordered_values = sorted(mapping)
    if ordered_values != list(range(len(mapping))):
        raise ReportDataError(
            f"Labels CSV values must be contiguous from zero: {path}"
        )
    return {value: mapping[value] for value in ordered_values}


def _find_labels_csv(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        candidate = parent / "labels.csv"
        if candidate.is_file():
            return candidate
    return None


def _find_best_metrics_json(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        candidate = parent / "best_metrics.json"
        if candidate.is_file():
            return candidate
    return None


def _find_run_dirs(root: Path) -> list[Path]:
    if (root / "files").is_dir():
        return [root]
    if not root.is_dir():
        return []

    run_dirs = {
        files_dir.parent
        for files_dir in root.glob("**/files")
        if files_dir.is_dir()
        and files_dir.parent.name.startswith(("run-", "offline-run-"))
    }
    return sorted(run_dirs)


def _parse_table_name(name: str) -> tuple[str, int] | None:
    match = TABLE_NAME_RE.fullmatch(name)
    if match is None:
        return None
    return match.group("split"), int(match.group("step"))


def _collect_tables(table_dir: Path) -> dict[str, dict[int, Path]]:
    tables: dict[str, dict[int, Path]] = {}
    for item in sorted(table_dir.glob("*_confusion_matrix_table_*.table.json")):
        parsed = _parse_table_name(item.name)
        if parsed is None:
            continue
        split, step = parsed
        split_tables = tables.setdefault(split, {})
        previous = split_tables.get(step)
        if previous is not None:
            raise ReportDataError(
                f"Multiple {split!r} confusion tables exist for step {step}: "
                f"{previous.name}, {item.name}"
            )
        split_tables[step] = item
    return tables


def _history_item_key(item: object) -> tuple[str, ...]:
    key = getattr(item, "key", "")
    if key:
        return (str(key),)
    return tuple(str(part) for part in getattr(item, "nested_key", ()))


def _history_item_value(item: object, transaction_path: Path) -> object:
    try:
        return json.loads(getattr(item, "value_json"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReportDataError(
            f"Invalid JSON value in W&B transaction log {transaction_path}"
        ) from exc


def _transaction_table_path(
    files_dir: Path, relative_path: object, transaction_path: Path
) -> Path:
    if not isinstance(relative_path, str):
        raise ReportDataError(
            f"Invalid table path in W&B transaction log {transaction_path}"
        )
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReportDataError(
            f"Unsafe table path {relative_path!r} in {transaction_path}"
        )
    return files_dir / relative


def _apply_summary_record(
    summary_values: dict[str, object], summary_record: object, transaction_path: Path
) -> None:
    for item in getattr(summary_record, "update", ()):
        key_parts = _history_item_key(item)
        if len(key_parts) == 1:
            summary_values[key_parts[0]] = _history_item_value(
                item, transaction_path
            )
    for item in getattr(summary_record, "remove", ()):
        key_parts = _history_item_key(item)
        if len(key_parts) == 1:
            summary_values.pop(key_parts[0], None)


def _load_run_transaction(run_dir: Path, files_dir: Path) -> RunTransactionData | None:
    transaction_paths = sorted(run_dir.glob("run-*.wandb"))
    if not transaction_paths:
        return None
    if len(transaction_paths) != 1:
        raise ReportDataError(
            f"Expected one W&B transaction log under {run_dir}, found "
            f"{len(transaction_paths)}"
        )
    transaction_path = transaction_paths[0]

    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal import datastore
    except ImportError as exc:
        raise ReportDataError(
            "A W&B transaction log is present, but the wandb package is unavailable. "
            "Run this command with the project virtual environment."
        ) from exc

    tables: dict[str, dict[int, Path]] = {}
    summary_values: dict[str, object] = {}
    store = datastore.DataStore()
    try:
        store.open_for_scan(str(transaction_path))
        while True:
            raw_record = store.scan_data()
            if raw_record is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw_record)
            record_type = record.WhichOneof("record_type")

            if record_type == "run":
                _apply_summary_record(
                    summary_values, record.run.summary, transaction_path
                )
                continue
            if record_type == "summary":
                _apply_summary_record(
                    summary_values, record.summary, transaction_path
                )
                continue
            if record_type != "history":
                continue

            history = record.history
            step = int(history.step.num) if history.HasField("step") else None
            for item in history.item:
                key_parts = _history_item_key(item)
                if key_parts == ("_step",):
                    value = _history_item_value(item, transaction_path)
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ReportDataError(
                            f"Invalid _step value in W&B transaction log "
                            f"{transaction_path}"
                        )
                    if step is not None and step != value:
                        raise ReportDataError(
                            f"Conflicting history step values in {transaction_path}: "
                            f"{step} != {value}"
                        )
                    step = value

            for item in history.item:
                key_parts = _history_item_key(item)
                table_key: str | None = None
                table_path_value: object | None = None
                if len(key_parts) == 2 and key_parts[1] == "path":
                    table_key = key_parts[0]
                    table_path_value = _history_item_value(item, transaction_path)
                elif len(key_parts) == 1:
                    value = _history_item_value(item, transaction_path)
                    if isinstance(value, dict) and "path" in value:
                        table_key = key_parts[0]
                        table_path_value = value["path"]

                match = (
                    TABLE_HISTORY_KEY_RE.fullmatch(table_key)
                    if table_key is not None
                    else None
                )
                if match is None:
                    continue
                if step is None or step < 0:
                    raise ReportDataError(
                        f"Confusion table history has no valid step in "
                        f"{transaction_path}"
                    )
                split = match.group("split")
                tables.setdefault(split, {})[step] = _transaction_table_path(
                    files_dir, table_path_value, transaction_path
                )
    except ReportDataError:
        raise
    except Exception as exc:
        raise ReportDataError(
            f"Cannot parse W&B transaction log {transaction_path}: {exc}"
        ) from exc
    finally:
        store.close()

    numeric_summary: dict[str, float] = {}
    for key, value in summary_values.items():
        if (
            isinstance(key, str)
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ):
            numeric_summary[key] = float(value)
    return RunTransactionData(
        path=transaction_path,
        tables=tables,
        summary=numeric_summary,
    )


def _select_latest_table_for_split(
    tables: dict[str, dict[int, Path]], split: str
) -> tuple[int, Path] | None:
    split_tables = tables.get(split)
    if not split_tables:
        return None
    step = max(split_tables)
    return step, split_tables[step]


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportDataError(f"Cannot read JSON file {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReportDataError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReportDataError(f"Expected a JSON object in {path}")
    return data


def _find_column(columns: list[object], names: set[str], path: Path) -> int:
    matches = [
        index
        for index, column in enumerate(columns)
        if isinstance(column, str) and column.strip().casefold() in names
    ]
    if len(matches) != 1:
        expected = "/".join(sorted(names))
        raise ReportDataError(
            f"Expected exactly one {expected!r} column in table {path}"
        )
    return matches[0]


def _class_value(value: object, path: Path) -> tuple[str, int | None]:
    if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
        raise ReportDataError(f"Invalid class value {value!r} in table {path}")

    numeric_index: int | None = None
    if isinstance(value, int):
        numeric_index = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ReportDataError(f"Non-finite class value {value!r} in table {path}")
        if value.is_integer():
            numeric_index = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ReportDataError(f"Empty class value in table {path}")
        try:
            numeric_index = int(stripped)
        except ValueError:
            pass

    if numeric_index is not None and numeric_index < 0:
        raise ReportDataError(
            f"Class indices must be non-negative, found {numeric_index} in {path}"
        )
    return str(value), numeric_index


def _count_value(value: object, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportDataError(f"Invalid confusion-matrix count {value!r} in {path}")
    count = float(value)
    if not math.isfinite(count) or count < 0:
        raise ReportDataError(f"Invalid confusion-matrix count {value!r} in {path}")
    return count


def _load_confusion_table(
    path: Path, label_mapping: dict[int, str] | None
) -> ConfusionTable:
    data = _load_json_object(path)
    columns = data.get("columns")
    rows = data.get("data")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ReportDataError(
            f"Table must contain list-valued 'columns' and 'data' fields: {path}"
        )

    actual_index = _find_column(columns, {"actual"}, path)
    predicted_index = _find_column(columns, {"predicted"}, path)
    count_index = _find_column(columns, {"npredictions", "count"}, path)
    required_width = max(actual_index, predicted_index, count_index) + 1

    parsed_rows: list[tuple[str, int | None, str, int | None, float]] = []
    ordered_names: list[str] = []
    seen_names: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, list) or len(row) < required_width:
            raise ReportDataError(f"Malformed table row {row_number} in {path}")
        actual_name, actual_numeric = _class_value(row[actual_index], path)
        predicted_name, predicted_numeric = _class_value(row[predicted_index], path)
        count = _count_value(row[count_index], path)
        parsed_rows.append(
            (actual_name, actual_numeric, predicted_name, predicted_numeric, count)
        )
        for name in (actual_name, predicted_name):
            if name not in seen_names:
                seen_names.add(name)
                ordered_names.append(name)

    if not parsed_rows and not label_mapping:
        raise ReportDataError(f"Confusion table contains no class rows: {path}")

    expected_label_names = (
        {label_mapping[index] for index in range(len(label_mapping))}
        if label_mapping
        else set()
    )
    table_values_match_label_names = bool(label_mapping) and seen_names.issubset(
        expected_label_names
    )
    numeric_classes = not table_values_match_label_names and all(
        actual_numeric is not None and predicted_numeric is not None
        for _, actual_numeric, _, predicted_numeric, _ in parsed_rows
    )

    if numeric_classes:
        observed_indices = {
            class_index
            for _, actual_numeric, _, predicted_numeric, _ in parsed_rows
            for class_index in (actual_numeric, predicted_numeric)
            if class_index is not None
        }
        if label_mapping:
            unknown = observed_indices - set(label_mapping)
            if unknown:
                raise ReportDataError(
                    f"Table contains class indices absent from labels.csv "
                    f"({sorted(unknown)}): {path}"
                )
            labels = [label_mapping[index] for index in range(len(label_mapping))]
        else:
            if not observed_indices:
                raise ReportDataError(f"Confusion table contains no classes: {path}")
            largest = max(observed_indices)
            if min(observed_indices) != 0 or len(observed_indices) != largest + 1:
                raise ReportDataError(
                    f"Numeric table classes must be contiguous from zero: {path}"
                )
            labels = [str(index) for index in range(largest + 1)]
        class_lookup: dict[object, int] = {
            index: index for index in range(len(labels))
        }
        row_keys = [
            (actual_numeric, predicted_numeric, count)
            for _, actual_numeric, _, predicted_numeric, count in parsed_rows
        ]
    else:
        if label_mapping:
            expected_names = [
                label_mapping[index] for index in range(len(label_mapping))
            ]
            unknown = seen_names - set(expected_names)
            if unknown:
                raise ReportDataError(
                    f"Table class names do not match labels.csv ({sorted(unknown)}): {path}"
                )
            labels = expected_names
        else:
            labels = ordered_names
        class_lookup = {name: index for index, name in enumerate(labels)}
        row_keys = [
            (actual_name, predicted_name, count)
            for actual_name, _, predicted_name, _, count in parsed_rows
        ]

    matrix = [[0.0 for _ in labels] for _ in labels]
    for actual, predicted, count in row_keys:
        try:
            row_index = class_lookup[actual]
            column_index = class_lookup[predicted]
        except KeyError as exc:
            raise ReportDataError(
                f"Table contains an unmapped class value {exc.args[0]!r}: {path}"
            ) from exc
        matrix[row_index][column_index] += count
    return ConfusionTable(labels=labels, matrix=matrix)


def _compute_metrics(
    split: str,
    step: int,
    matrix: list[list[float]],
    labels: list[str],
) -> SplitMetrics:
    n = len(matrix)
    if len(labels) != n or any(len(row) != n for row in matrix):
        raise ReportDataError("Confusion matrix and labels must have matching dimensions")

    totals = [sum(row) for row in matrix]
    col_sums = [sum(matrix[row][column] for row in range(n)) for column in range(n)]

    per_class: list[dict[str, float | str]] = []
    for index in range(n):
        tp = matrix[index][index]
        fn = totals[index] - tp
        fp = col_sums[index] - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class.append(
            {
                "class": labels[index],
                "support": totals[index],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    total = sum(totals)
    accuracy = sum(matrix[index][index] for index in range(n)) / total if total else 0.0

    macro_precision = sum(float(row["precision"]) for row in per_class) / n if n else 0.0
    macro_recall = sum(float(row["recall"]) for row in per_class) / n if n else 0.0
    macro_f1 = sum(float(row["f1"]) for row in per_class) / n if n else 0.0

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
        step=step,
        confusion_matrix=matrix,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        per_class=per_class,
    )


def _metric_score(metric: str, metrics: SplitMetrics) -> float:
    if metric == "accuracy":
        return metrics.accuracy
    if metric == "macro_f1":
        return metrics.macro_f1
    if metric == "weighted_f1":
        return metrics.weighted_f1
    raise ValueError(f"Unsupported metric: {metric}")


def _load_cached_table(
    path: Path,
    label_mapping: dict[int, str] | None,
    cache: dict[Path, ConfusionTable],
) -> ConfusionTable:
    if path not in cache:
        cache[path] = _load_confusion_table(path, label_mapping)
    return cache[path]


def _select_best_step(
    tables: dict[str, dict[int, Path]],
    label_mapping: dict[int, str] | None,
    metric: str,
    cache: dict[Path, ConfusionTable] | None = None,
) -> tuple[int, float] | None:
    devel_tables = tables.get("devel")
    if not devel_tables:
        return None

    table_cache = cache if cache is not None else {}
    best_step = -1
    best_score: float | None = None
    for step in sorted(devel_tables):
        path = devel_tables[step]
        table = _load_cached_table(path, label_mapping, table_cache)
        metrics = _compute_metrics("devel", step, table.matrix, table.labels)
        score = _metric_score(metric, metrics)
        # The trainer keeps the first epoch on a tie; reporting must do the same.
        if best_score is None or score > best_score:
            best_score = score
            best_step = step
    if best_score is None:
        return None
    return best_step, best_score


def _load_summary(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    data = _load_json_object(path)
    metrics: dict[str, float] = {}
    for key, value in data.items():
        if (
            isinstance(key, str)
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ):
            metrics[key] = float(value)
    return metrics


def _summary_step(summary: dict[str, float], key: str) -> int | None:
    value = summary.get(key)
    if value is None or value < 0 or not value.is_integer():
        return None
    return int(value)


def _load_best_metrics(path: Path) -> dict[str, float]:
    data = _load_json_object(path)
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        raise ReportDataError(
            f"best_metrics.json must contain an object-valued 'metrics' field: {path}"
        )

    result: dict[str, float] = {}
    for key in ("best_epoch", "best_devel_accuracy"):
        value = data.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ReportDataError(
                f"best_metrics.json contains an invalid {key!r} value: {path}"
            )
        result[key] = float(value)

    if _summary_step(result, "best_epoch") is None:
        raise ReportDataError(
            f"best_metrics.json contains a non-integer best_epoch: {path}"
        )

    for key, value in metrics.items():
        if not isinstance(key, str):
            raise ReportDataError(
                f"best_metrics.json contains a non-string metric key: {path}"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ReportDataError(
                f"best_metrics.json contains an invalid metric {key!r}: {path}"
            )
        result[f"best_{key}"] = float(value)
    return result


def _merge_summary_metadata(
    wandb_summary: dict[str, float] | None,
    best_metrics: dict[str, float] | None,
) -> dict[str, float] | None:
    if wandb_summary is None and best_metrics is None:
        return None

    merged = dict(wandb_summary or {})
    for key, value in (best_metrics or {}).items():
        previous = merged.get(key)
        if previous is not None and not math.isclose(
            previous, value, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ReportDataError(
                f"Conflicting metadata values for {key!r}: "
                f"{previous:.12g} != {value:.12g}"
            )
        merged[key] = value
    return merged


def _loss_for_step(
    summary: dict[str, float] | None, split: str, step: int
) -> tuple[float | None, bool]:
    if not summary:
        return None, False

    best_loss_key = f"best_{split}_loss"
    latest_loss_key = f"{split}_loss"
    has_loss = best_loss_key in summary or latest_loss_key in summary

    if (
        _summary_step(summary, "best_epoch") == step
        and best_loss_key in summary
    ):
        return summary[best_loss_key], True
    if _summary_step(summary, "_step") == step and latest_loss_key in summary:
        return summary[latest_loss_key], True
    return None, has_loss


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _format_matrix_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}"


def _format_matrix_cell(value: float, row_total: float) -> str:
    pct = (value / row_total * 100.0) if row_total else 0.0
    return f"{_format_matrix_value(value)} ({pct:.1f}%)"


def _print_confusion_matrix(matrix: list[list[float]], labels: list[str]) -> None:
    print("    Confusion Matrix (rows=actual, columns=predicted; cell=count (row %)):")
    if not matrix:
        print("      <empty>")
        return

    n = len(matrix)
    row_totals = [sum(row) for row in matrix]
    values = [
        [
            _format_matrix_cell(matrix[row_index][column_index], row_totals[row_index])
            for column_index in range(n)
        ]
        for row_index in range(n)
    ]

    row_header = "actual\\pred"
    label_width = max(len(row_header), *(len(label) for label in labels))
    col_widths = [
        max(
            len(labels[column_index]),
            *(len(values[row_index][column_index]) for row_index in range(n)),
        )
        for column_index in range(n)
    ]

    header = (
        f"      {row_header:<{label_width}}  "
        + "  ".join(
            f"{labels[column_index]:>{col_widths[column_index]}}"
            for column_index in range(n)
        )
    )
    print(header)
    for row_index, label in enumerate(labels):
        print(
            f"      {label:<{label_width}}  "
            + "  ".join(
                f"{values[row_index][column_index]:>{col_widths[column_index]}}"
                for column_index in range(n)
            )
        )


def _print_split_metrics(
    metrics: SplitMetrics, summary: dict[str, float] | None
) -> None:
    print(f"  Split: {metrics.split} (step {metrics.step})")
    print(
        f"    accuracy={_format_metric(metrics.accuracy)} | "
        f"macro P/R/F1={_format_metric(metrics.macro_precision)}/"
        f"{_format_metric(metrics.macro_recall)}/"
        f"{_format_metric(metrics.macro_f1)} | "
        f"weighted P/R/F1={_format_metric(metrics.weighted_precision)}/"
        f"{_format_metric(metrics.weighted_recall)}/"
        f"{_format_metric(metrics.weighted_f1)}"
    )
    loss, has_unaligned_loss = _loss_for_step(
        summary, metrics.split, metrics.step
    )
    if loss is not None:
        print(f"    loss={loss:.6f}")
    elif has_unaligned_loss:
        print(f"    loss=<unavailable for step {metrics.step}>")
    print("    per-class:")
    class_width = max(
        len("class"),
        max((len(str(row["class"])) for row in metrics.per_class), default=5),
    )
    print(f"      {'class':<{class_width}}  support  precision  recall  f1")
    for row in metrics.per_class:
        print(
            "      "
            f"{str(row['class']):<{class_width}}  "
            f"{_format_matrix_value(float(row['support'])):>7}  "
            f"{_format_metric(float(row['precision'])):>9}  "
            f"{_format_metric(float(row['recall'])):>6}  "
            f"{_format_metric(float(row['f1'])):>6}"
        )
    _print_confusion_matrix(
        metrics.confusion_matrix,
        [str(row["class"]) for row in metrics.per_class],
    )


def _report_run(
    run_dir: Path,
    args: argparse.Namespace,
    explicit_labels: dict[int, str] | None,
    best_metrics_path: Path | None = None,
) -> None:
    files_dir = run_dir / "files"
    table_dir = files_dir / "media" / "table"
    if not table_dir.is_dir():
        raise ReportDataError(f"No W&B media/table directory found under {run_dir}")

    labels_path = args.labels or _find_labels_csv(run_dir)
    label_mapping = (
        explicit_labels
        if args.labels is not None
        else (_load_labels_csv(labels_path) if labels_path else None)
    )
    summary_path = files_dir / "wandb-summary.json"
    wandb_summary = _load_summary(summary_path)
    transaction = _load_run_transaction(run_dir, files_dir)
    best_metrics = (
        _load_best_metrics(best_metrics_path) if best_metrics_path else None
    )
    summary = _merge_summary_metadata(
        wandb_summary,
        transaction.summary if transaction is not None else None,
    )
    summary = _merge_summary_metadata(summary, best_metrics)
    tables = (
        transaction.tables
        if transaction is not None and transaction.tables
        else _collect_tables(table_dir)
    )
    if not tables:
        raise ReportDataError(f"No confusion matrix tables found under {table_dir}")

    splits = [args.split] if args.split != "all" else ["train", "devel", "test"]
    cache: dict[Path, ConfusionTable] = {}

    print(f"Run: {run_dir.name}")
    print(f"  path: {run_dir}")
    if labels_path:
        print(f"  labels: {labels_path}")
    else:
        print("  labels: not found (table class names will be used)")
    if wandb_summary is None:
        print("  wandb-summary.json: not found")
    if transaction is not None:
        print(f"  transaction log: {transaction.path}")
    if best_metrics_path is not None:
        print(f"  best metrics: {best_metrics_path}")

    selected_step: int | None = None
    if args.selection == "best":
        recorded_best_step = (
            _summary_step(summary, "best_epoch")
            if summary is not None and args.best_metric == "accuracy"
            else None
        )
        if recorded_best_step is not None:
            recorded_table_path = tables.get("devel", {}).get(recorded_best_step)
            if recorded_table_path is None:
                raise ReportDataError(
                    f"Recorded metadata selects best_epoch={recorded_best_step}, "
                    f"but its devel confusion table is missing for {run_dir.name}"
                )
            recorded_table = _load_cached_table(
                recorded_table_path, label_mapping, cache
            )
            recorded_metrics = _compute_metrics(
                "devel",
                recorded_best_step,
                recorded_table.matrix,
                recorded_table.labels,
            )
            recorded_score = recorded_metrics.accuracy
            summary_score = summary.get("best_devel_accuracy") if summary else None
            if summary_score is not None and not math.isclose(
                recorded_score, summary_score, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ReportDataError(
                    "Recomputed devel accuracy does not match "
                    f"best_devel_accuracy at step {recorded_best_step}: "
                    f"{recorded_score:.12g} != {summary_score:.12g}"
                )
            best = (recorded_best_step, recorded_score)
            selection_source = "recorded best_epoch"
        else:
            best = _select_best_step(
                tables, label_mapping, args.best_metric, cache=cache
            )
            selection_source = "confusion tables"
        if best is None:
            raise ReportDataError(
                f"Cannot select a best step for {run_dir.name}: no devel tables found"
            )
        selected_step, best_score = best
        print(
            "  selection: best by devel "
            f"{args.best_metric} (step {selected_step}, score={best_score:.4f}; "
            f"source={selection_source})"
        )
    else:
        print("  selection: latest per split")

    reported_splits = 0
    for split in splits:
        if selected_step is not None:
            table_path = tables.get(split, {}).get(selected_step)
            if table_path is None:
                print(
                    f"  Split: {split} (no confusion matrix at selected step "
                    f"{selected_step}; no fallback used)"
                )
                continue
            step = selected_step
        else:
            latest = _select_latest_table_for_split(tables, split)
            if latest is None:
                print(f"  Split: {split} (no confusion matrix table found)")
                continue
            step, table_path = latest

        table = _load_cached_table(table_path, label_mapping, cache)
        metrics = _compute_metrics(split, step, table.matrix, table.labels)
        _print_split_metrics(metrics, summary)
        reported_splits += 1

    if reported_splits == 0:
        raise ReportDataError(
            f"No requested split could be reported for run {run_dir.name}"
        )
    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report training results from local W&B confusion matrices."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "W&B run directory or a parent containing wandb/run-* or "
            "wandb/offline-run-* directories."
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional labels.csv path. If omitted, parent folders are searched.",
    )
    parser.add_argument(
        "--best-metrics",
        type=Path,
        help=(
            "Optional best_metrics.json path for a single run. If omitted, unique "
            "files in parent folders are discovered automatically."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "devel", "test", "all"],
        default="all",
        help="Which split(s) to report.",
    )
    parser.add_argument(
        "--selection",
        choices=["latest", "best"],
        default="best",
        help=(
            "Use the latest step per split or one shared step selected by the "
            "devel metric. Missing splits are never substituted from another step."
        ),
    )
    parser.add_argument(
        "--best-metric",
        choices=["accuracy", "macro_f1", "weighted_f1"],
        default="accuracy",
        help="Metric used to pick the best devel step.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    root = args.run_dir
    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        raise SystemExit(f"No W&B run directories found under: {root}")

    explicit_labels: dict[int, str] | None = None
    if args.labels is not None:
        try:
            explicit_labels = _load_labels_csv(args.labels)
        except ReportDataError as exc:
            raise SystemExit(str(exc)) from exc

    if args.best_metrics is not None and len(run_dirs) != 1:
        raise SystemExit("--best-metrics requires run_dir to resolve to exactly one run")

    if args.best_metrics is not None:
        best_metrics_paths = {run_dirs[0]: args.best_metrics}
    else:
        candidates = {
            run_dir: _find_best_metrics_json(run_dir) for run_dir in run_dirs
        }
        candidate_values = [path for path in candidates.values() if path is not None]
        best_metrics_paths: dict[Path, Path | None] = {}
        for run_dir, candidate in candidates.items():
            if candidate is not None and candidate_values.count(candidate) == 1:
                best_metrics_paths[run_dir] = candidate
            else:
                best_metrics_paths[run_dir] = None
        for candidate in sorted(set(candidate_values)):
            if candidate_values.count(candidate) > 1:
                print(
                    "Ignoring ambiguous best_metrics.json shared by multiple runs: "
                    f"{candidate}",
                    file=sys.stderr,
                )

    reported = 0
    failures: list[tuple[Path, str]] = []
    for run_dir in run_dirs:
        try:
            _report_run(
                run_dir,
                args,
                explicit_labels,
                best_metrics_path=best_metrics_paths.get(run_dir),
            )
            reported += 1
        except ReportDataError as exc:
            failures.append((run_dir, str(exc)))
            print(f"Run failed: {run_dir}: {exc}", file=sys.stderr)

    print(f"Summary: reported={reported}, failed={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
