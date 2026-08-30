"""DRAGON model inference with training-time asinh normalization."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from cnn import (
    DRAGON,
    DRAGON_CUTOUT_SIZE,
    DRAGON_MIN_SIZE,
    DRAGON_SIZE_DIVISOR,
)
from data_preprocessing import HDF5Dataset, get_data_loader
from utils import (
    asinh_normalize,
    discover_devices,
    load_label_mapping,
    load_model_state,
    normalization_kwargs_from_stats,
)

PREDICTION_FLOAT_FORMAT = "%.17g"


def predict_probabilities(
    model_path,
    dataset,
    channels,
    batch_size=256,
    n_workers=1,
    num_classes=6,
    normalization_kwargs=None,
):
    """Return the complete softmax probability matrix for ``dataset``."""
    if not normalization_kwargs or not {
        "vmin",
        "vmax",
        "softening",
    }.issubset(normalization_kwargs):
        raise ValueError(
            "Inference requires vmin/vmax/softening loaded from "
            "normalization_stats.json"
        )

    device = discover_devices()
    model_args = {
        "channels": channels,
        "num_classes": num_classes,
    }
    model = DRAGON(**model_args)
    logging.info("Loading model from %s", model_path)
    load_model_state(model, model_path, device=device)
    model = model.to(device)

    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
    )
    model.eval()

    probabilities = []
    with torch.no_grad():
        for images, _labels in tqdm(loader, desc="Inference"):
            images = images.to(device)
            images = asinh_normalize(images, **normalization_kwargs)
            logits = model(images)
            probabilities.append(nn.functional.softmax(logits, dim=1))

    if not probabilities:
        raise ValueError("Inference dataset contains no rows")
    probabilities = torch.cat(probabilities).cpu()
    if probabilities.shape[1] < 2:
        raise ValueError("Inference requires a model with at least two classes")
    if probabilities.shape[1] != num_classes:
        raise ValueError(
            "Model output class count does not match the inference "
            f"configuration: {probabilities.shape[1]} != {num_classes}"
        )
    return probabilities.numpy()


def top_two_predictions(probabilities):
    """Return top-two indices and values from an ``(N, C)`` probability array."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Top-two prediction requires a two-dimensional probability array "
            "with at least two classes"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probability array contains a non-finite value")

    values, indices = torch.topk(torch.from_numpy(probabilities), 2, dim=1)
    return (
        indices[:, 0].cpu().numpy(),
        values[:, 0].cpu().numpy(),
        indices[:, 1].cpu().numpy(),
        values[:, 1].cpu().numpy(),
    )


def write_predictions(frame, path):
    """Write probabilities without changing float32 threshold comparisons."""
    frame.to_csv(path, index=False, float_format=PREDICTION_FLOAT_FORMAT)


def predict(
    model_path,
    dataset,
    channels,
    batch_size=256,
    n_workers=1,
    num_classes=6,
    normalization_kwargs=None,
):
    """Return top-two labels and confidences for ``dataset``."""
    probabilities = predict_probabilities(
        model_path,
        dataset,
        channels,
        batch_size=batch_size,
        n_workers=n_workers,
        num_classes=num_classes,
        normalization_kwargs=normalization_kwargs,
    )
    return top_two_predictions(probabilities)


@click.command()
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option("--output-dir", type=click.Path(file_okay=False), required=True)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--cutout-size",
    type=int,
    default=DRAGON_CUTOUT_SIZE,
    show_default=True,
    help=(
        f"Cutout side length; must be a multiple of {DRAGON_SIZE_DIVISOR} "
        f"and at least {DRAGON_MIN_SIZE}."
    ),
)
@click.option(
    "--channels",
    type=int,
    required=True,
    help="Input band count; must match the checkpoint and the cutouts.",
)
@click.option(
    "--normalization-stats",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--labels-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Dataset-level labels.csv; required unless --no-labels is used.",
)
@click.option("--batch-size", type=int, default=256)
@click.option("--n-workers", type=int, default=4)
@click.option(
    "--n-classes",
    type=int,
    required=True,
    help="Output class count; must match the checkpoint and labels.csv.",
)
@click.option("--labels/--no-labels", default=True)
@click.option(
    "--all-probabilities/--top-two-only",
    default=False,
    show_default=True,
    help=(
        "Also write probability__<class-index> columns for every class. "
        "The default preserves the compact top-two output."
    ),
)
def main(
    model_path,
    output_dir,
    data_dir,
    cutout_size,
    channels,
    normalization_stats,
    labels_path,
    batch_size,
    n_workers,
    n_classes,
    labels,
    all_probabilities,
):
    """Run label-free inference against DATA_DIR/info.csv."""
    if channels <= 0 or n_classes < 2 or batch_size <= 0 or n_workers < 0:
        raise click.UsageError("Invalid channels/classes/batch-size/worker count")
    if cutout_size % DRAGON_SIZE_DIVISOR or cutout_size < DRAGON_MIN_SIZE:
        raise click.BadParameter(
            "DRAGON halves the map four times, so the cutout side length must "
            f"be a multiple of {DRAGON_SIZE_DIVISOR} and at least "
            f"{DRAGON_MIN_SIZE}",
            param_hint="--cutout-size",
        )
    if labels and labels_path is None:
        raise click.UsageError("Pass dataset-level --labels-path or use --no-labels.")

    stats_path = Path(normalization_stats)
    try:
        normalization_kwargs = normalization_kwargs_from_stats(
            stats_path,
            channels=channels,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    dataset = HDF5Dataset(
        data_dir,
        load_labels=False,
    )
    expected_image_shape = (channels, cutout_size, cutout_size)
    stored_image_shape = dataset.h5_image_shape[1:]
    if stored_image_shape != expected_image_shape:
        raise click.ClickException(
            "Stored HDF5 image shape does not match the inference "
            f"configuration: {stored_image_shape} != {expected_image_shape}"
        )
    catalog_path = Path(data_dir) / "info.csv"
    catalog = pd.read_csv(
        catalog_path,
        dtype={"object_id": "string"},
        low_memory=False,
    )
    label_names = None
    if labels:
        labels_path = Path(labels_path)
        try:
            label_names = load_label_mapping(
                labels_path,
                expected_classes=n_classes,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    probabilities = predict_probabilities(
        model_path,
        dataset,
        channels,
        batch_size=batch_size,
        n_workers=n_workers,
        num_classes=n_classes,
        normalization_kwargs=normalization_kwargs,
    )
    predicted, confidence, second, second_confidence = top_two_predictions(
        probabilities
    )

    result = catalog.copy()
    result["predicted_labels"] = predicted
    result["predicted_confidence"] = confidence
    result["second_predicted_labels"] = second
    result["second_predicted_confidence"] = second_confidence
    if all_probabilities:
        for class_index in range(n_classes):
            result[f"probability__{class_index}"] = probabilities[:, class_index]
    if label_names is not None:
        result["predicted_class"] = [label_names[int(index)] for index in predicted]
        result["second_predicted_class"] = [
            label_names[int(index)] for index in second
        ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    # Preserve float32 softmax values exactly when pandas reads the CSV back as
    # float64. Calibrated thresholds may sit one float64 ULP above a tied score,
    # so the usual shortened decimal representation can change a >= decision.
    write_predictions(result, prediction_path)
    logging.info("Saved predictions to %s", prediction_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
