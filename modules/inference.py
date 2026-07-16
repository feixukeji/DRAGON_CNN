"""DRAGON model inference with training-time asinh normalization."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import kornia.augmentation as K
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from cnn import model_factory
from data_preprocessing import FITSDataset, get_data_loader
from utils import (
    DEFAULT_ASINH_SOFTENING,
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    asinh_normalize,
    discover_devices,
    enable_dropout,
    load_label_mapping,
    load_model_state,
    normalization_kwargs_from_stats,
    specify_dropout_rate,
)


def predict(
    model_path,
    dataset,
    cutout_size,
    channels,
    parallel=False,
    batch_size=256,
    n_workers=1,
    num_classes=6,
    model_type="dragon",
    mc_dropout=False,
    dropout_rate=None,
    apply_softmax=True,
    normalize=True,
    normalization_kwargs=None,
):
    """Return top-two labels and confidences for ``dataset``."""
    if not normalize:
        raise ValueError(
            "Inference must use asinh normalization; normalize=False is unsupported"
        )
    if (
        not normalization_kwargs
        or "vmin" not in normalization_kwargs
        or "vmax" not in normalization_kwargs
    ):
        raise ValueError(
            "Normalized inference requires vmin/vmax loaded from "
            "normalization_stats.json"
        )

    device = discover_devices()
    model_cls = model_factory(model_type)
    model_args = {
        "cutout_size": cutout_size,
        "channels": channels,
        "num_classes": num_classes,
    }
    if "drp" in model_type.split("_"):
        model_args["dropout"] = "True"

    model = model_cls(**model_args)
    logging.info("Loading model from %s", model_path)
    load_model_state(model, model_path, device=device)
    if parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    if dropout_rate is not None:
        specify_dropout_rate(model, dropout_rate)

    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
    )
    model.eval()
    if mc_dropout:
        logging.info("Activating Monte Carlo dropout")
        enable_dropout(model)

    outputs = []
    with torch.no_grad():
        for images, _labels in tqdm(loader, desc="Inference"):
            images = images.to(device)
            if dataset.transform is not None:
                images = dataset.transform(images)
            images = asinh_normalize(images, **normalization_kwargs)
            logits = model(images)
            outputs.append(
                nn.functional.softmax(logits, dim=1)
                if apply_softmax
                else logits
            )

    if not outputs:
        raise ValueError("Inference dataset contains no rows")
    probabilities = torch.cat(outputs)
    if probabilities.shape[1] < 2:
        raise ValueError("Inference requires a model with at least two classes")
    values, indices = torch.topk(probabilities, 2, dim=1)
    return (
        indices[:, 0].cpu().numpy(),
        values[:, 0].cpu().numpy(),
        indices[:, 1].cpu().numpy(),
        values[:, 1].cpu().numpy(),
    )


def _output_file(output_dir, output_path, run_num, stem):
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{stem}_{run_num}.csv"
    target = Path(f"{output_path}{stem}_{run_num}.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _summary_file(output_dir, output_path, run_num, n_runs):
    suffix = "" if n_runs == 1 else f"_{run_num}"
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"summary_counts{suffix}.csv"
    target = Path(f"{output_path}summary_counts{suffix}.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@click.command()
@click.option("--model-path", "--model_path", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(file_okay=False), default=None)
@click.option(
    "--output-path",
    "--output_path",
    type=click.Path(),
    default=None,
    help="Legacy filename prefix; prefer --output-dir.",
)
@click.option("--data-dir", "--data_dir", type=click.Path(exists=True), required=True)
@click.option(
    "--model-type",
    "--model_type",
    type=click.Choice(["dragon"], case_sensitive=False),
    default="dragon",
)
@click.option("--cutout-size", "--cutout_size", type=int, default=167)
@click.option("--channels", type=int, default=3)
@click.option("--slug", type=str, default=None)
@click.option("--split", type=str, default=None)
@click.option("--normalize/--no-normalize", default=True)
@click.option(
    "--normalize-low-pct",
    type=float,
    default=DEFAULT_LOW_PERCENTILE,
    show_default=True,
)
@click.option(
    "--normalize-high-pct",
    type=float,
    default=DEFAULT_HIGH_PERCENTILE,
    show_default=True,
)
@click.option(
    "--asinh-softening",
    type=float,
    default=DEFAULT_ASINH_SOFTENING,
    show_default=True,
)
@click.option(
    "--normalization-stats",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
)
@click.option("--batch-size", "--batch_size", type=int, default=256)
@click.option("--n-workers", "--n_workers", type=int, default=4)
@click.option("--parallel/--no-parallel", default=True)
@click.option("--label-col", "--label_col", type=str, default="classes")
@click.option("--mc_dropout/--no_mc_dropout", default=True)
@click.option("--n-runs", "--n_runs", type=int, default=1)
@click.option("--n-classes", "--n_classes", type=int, default=6)
@click.option("--ini-run-num", "--ini_run_num", type=int, default=1)
@click.option("--dropout-rate", "--dropout_rate", type=float, default=None)
@click.option("--crop/--no-crop", default=True)
@click.option("--labels/--no-labels", default=True)
def main(
    model_path,
    output_dir,
    output_path,
    data_dir,
    model_type,
    cutout_size,
    channels,
    slug,
    split,
    normalize,
    normalize_low_pct,
    normalize_high_pct,
    asinh_softening,
    normalization_stats,
    batch_size,
    n_workers,
    parallel,
    label_col,
    mc_dropout,
    n_runs,
    n_classes,
    ini_run_num,
    dropout_rate,
    crop,
    labels,
):
    """Run label-free inference against DATA_DIR/info.csv."""
    del slug, split, label_col  # Pure inference always uses info.csv without labels.
    if (output_dir is None) == (output_path is None):
        raise click.UsageError("Specify exactly one of --output-dir or --output-path")
    if not normalize:
        raise click.UsageError(
            "Inference must use normalization_stats.json; --no-normalize is unsupported"
        )
    if channels <= 0 or n_classes < 2 or batch_size <= 0 or n_workers < 0:
        raise click.UsageError("Invalid channels/classes/batch-size/worker count")
    if n_runs <= 0 or ini_run_num <= 0:
        raise click.UsageError("--n-runs and --ini-run-num must be positive")

    stats_path = (
        Path(normalization_stats)
        if normalization_stats
        else Path(data_dir) / "normalization_stats.json"
    )
    if not stats_path.is_file():
        raise click.ClickException(
            f"Training-time normalization statistics not found: {stats_path}"
        )
    try:
        normalization_kwargs = normalization_kwargs_from_stats(
            stats_path,
            channels=channels,
            low_pct=normalize_low_pct,
            high_pct=normalize_high_pct,
            softening=asinh_softening,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    transform = K.CenterCrop(cutout_size) if crop else None
    dataset = FITSDataset(
        data_dir,
        slug=None,
        split=None,
        cutout_size=cutout_size,
        channels=channels,
        transforms=transform,
        load_labels=False,
    )
    catalog_path = Path(data_dir) / "info.csv"
    catalog = pd.read_csv(catalog_path, dtype={"object_id": str})
    label_names = None
    labels_path = Path(data_dir) / "labels.csv"
    if labels and labels_path.is_file():
        try:
            label_names = load_label_mapping(
                labels_path,
                expected_classes=n_classes,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    for run_num in range(ini_run_num, ini_run_num + n_runs):
        predicted, confidence, second, second_confidence = predict(
            model_path,
            dataset,
            cutout_size,
            channels,
            parallel=parallel,
            batch_size=batch_size,
            n_workers=n_workers,
            num_classes=n_classes,
            model_type=model_type,
            mc_dropout=mc_dropout,
            dropout_rate=dropout_rate,
            normalize=True,
            normalization_kwargs=normalization_kwargs,
        )

        result = catalog.copy()
        result["predicted_labels"] = predicted
        result["predicted_confidence"] = confidence
        result["second_predicted_labels"] = second
        result["second_predicted_confidence"] = second_confidence
        if label_names is not None:
            result["predicted_class"] = [label_names[int(index)] for index in predicted]
            result["second_predicted_class"] = [
                label_names[int(index)] for index in second
            ]

        prediction_path = _output_file(
            output_dir,
            output_path,
            run_num,
            "inf",
        )
        result.to_csv(prediction_path, index=False)
        logging.info("Saved predictions to %s", prediction_path)

        if label_names is not None:
            summary_path = _summary_file(
                output_dir,
                output_path,
                run_num,
                n_runs,
            )
            (
                result["predicted_class"]
                .value_counts()
                .rename_axis("predicted_class")
                .reset_index(name="count")
                .to_csv(summary_path, index=False)
            )
            logging.info("Saved prediction summary to %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
