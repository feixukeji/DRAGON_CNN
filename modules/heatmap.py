# -*- coding: utf-8 -*-
import click
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

import torch
import torch.nn as nn

from tqdm import tqdm

from data_preprocessing import FITSDataset, get_data_loader
from pytorch_grad_cam.utils.image import show_cam_on_image

import kornia.augmentation as K

from cnn import model_factory
from modules.batched_eigen_grad_cam import BatchedEigenGradCAM
from utils import (
    DEFAULT_ASINH_SOFTENING,
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    asinh_normalize,
    discover_devices,
    enable_dropout,
    load_asinh_stats,
    specify_dropout_rate,
)

import cv2


def _object_id_output_paths(dataset, output_path):
    """Return one validated ``<object_id>.png`` path per dataset row."""
    if not hasattr(dataset, "data_info"):
        raise ValueError("Dataset does not expose the metadata needed for object_id filenames.")
    if "object_id" not in dataset.data_info.columns:
        raise ValueError("Inference metadata must contain an object_id column.")

    object_ids = dataset.data_info["object_id"]
    if object_ids.isna().any():
        first_missing = int(object_ids.isna().to_numpy().nonzero()[0][0]) + 2
        raise ValueError(f"info.csv row {first_missing} has an empty object_id.")

    names = []
    seen_rows = {}
    for index, raw_object_id in enumerate(object_ids.tolist()):
        row_number = index + 2
        object_id = str(raw_object_id).strip()
        if not object_id:
            raise ValueError(f"info.csv row {row_number} has an empty object_id.")
        if (
            object_id in {".", ".."}
            or "/" in object_id
            or "\\" in object_id
            or "\x00" in object_id
        ):
            raise ValueError(
                f"info.csv row {row_number} has an object_id that cannot be used "
                f"as a filename: {object_id!r}"
            )
        if object_id in seen_rows:
            raise ValueError(
                f"Duplicate object_id {object_id!r} in info.csv rows "
                f"{seen_rows[object_id]} and {row_number}."
            )
        seen_rows[object_id] = row_number
        names.append(object_id)

    if len(names) != len(dataset):
        raise ValueError(
            f"Metadata contains {len(names)} object_id values, but the dataset "
            f"contains {len(dataset)} samples."
        )

    output_dir = Path(output_path)
    return [output_dir / f"{object_id}.png" for object_id in names]


def _render_and_save_heatmap(img_tensor, grayscale_cam, save_file, channels):
    """Create and save one overlay; safe to run in a worker thread."""
    # Handle 3-channel i/r/g input as RGB vs 1-channel fallback.
    if channels == 3:
        # Channels are already ordered as i/r/g -> R/G/B.
        img_bg = img_tensor.transpose(1, 2, 0)
        img_normalized = (
            (img_bg - img_bg.min())
            / (img_bg.max() - img_bg.min() + 1e-8)
        )
    else:
        img_bg = img_tensor[0, :, :]
        img_normalized = (
            (img_bg - img_bg.min())
            / (img_bg.max() - img_bg.min() + 1e-8)
        )
        img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)

    cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
    encoded_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(save_file), encoded_image):
        raise OSError(f"Failed to write heatmap: {save_file}")


def _save_heatmap_batch(
        output_pool,
        image_batch,
        grayscale_cam,
        output_paths,
        channels,
):
    """Render and write a batch concurrently, propagating worker failures."""
    if len(image_batch) != len(grayscale_cam) or len(image_batch) != len(output_paths):
        raise ValueError("Image, CAM, and output-path batch sizes must match.")

    futures = [
        output_pool.submit(
            _render_and_save_heatmap,
            image_batch[index],
            grayscale_cam[index],
            output_paths[index],
            channels,
        )
        for index in range(len(image_batch))
    ]
    for future in futures:
        future.result()


def heatmap(
        model_path,
        output_path,
        dataset,
        cutout_size,
        channels,
        parallel=False,
        batch_size=256,
        n_workers=1,
        output_workers=4,
        num_classes=6,
        model_type="dragon",
        mc_dropout=False,
        dropout_rate=None,
        apply_softmax=True,
        normalize=False,
        normalization_kwargs=None,
):
    """Using the model defined in model path, return the output values for
    the given set of images"""

    if output_workers < 1:
        raise ValueError("output_workers must be at least 1.")

    # Discover devices
    device = discover_devices()

    # Declare the model given model_type
    cls = model_factory(model_type)
    model_args = {
        "cutout_size": cutout_size,
        "channels": channels,
        "num_classes": num_classes
    }

    if "drp" in model_type.split("_"):
        logging.info(
            "Using dropout rate of {} in the model".format(dropout_rate)
        )
        model_args["dropout"] = "True"

    model = cls(**model_args)

    # Load the model
    logging.info("Loading model...")
    if device == "cpu":
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(model_path))


    model = nn.DataParallel(model) if parallel else model
    model = model.to(device)

    # Changing the dropout rate if specified
    if dropout_rate is not None:
        specify_dropout_rate(model, dropout_rate)

    # Set to evaluation mode
    model.eval()

    # If using Monte Carlo dropout, re-enable dropout layers
    if mc_dropout:
        enable_dropout(model)

    # Create a data_preprocessing loader
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
    )

    # Acquiring GradCAM layer
    target_layer = model.module.layer4 if parallel else model.layer4

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _object_id_output_paths(dataset, output_dir)
    global_img_idx = 0
    logging.info("Performing heatmap creation...")
    logging.info("Computing EigenGradCAM with batched Torch SVD on the model device.")
    logging.info(f"Saving PNG overlays with {output_workers} worker threads.")

    with BatchedEigenGradCAM(model=model, target_layer=target_layer) as cam, \
            ThreadPoolExecutor(
                max_workers=output_workers,
                thread_name_prefix="heatmap-png",
            ) as output_pool:
        for data in tqdm(loader):
            X, _ = data
            X = X.to(device, non_blocking=True)
            if dataset.transform is not None:
                X = dataset.transform(X)
            if normalize:
                X = asinh_normalize(X, **(normalization_kwargs or {}))

            grayscale_cam = cam(input_tensor=X)

            # Move the whole batch to CPU once, then render/write each PNG in
            # parallel. Waiting per batch bounds memory use and surfaces errors
            # before more GPU work is started.
            image_batch = X.detach().cpu().numpy()
            grayscale_cam = grayscale_cam.cpu().numpy()
            batch_end = global_img_idx + len(image_batch)
            _save_heatmap_batch(
                output_pool,
                image_batch,
                grayscale_cam,
                output_paths[global_img_idx:batch_end],
                channels,
            )
            global_img_idx = batch_end

    if global_img_idx != len(output_paths):
        raise RuntimeError(
            f"Created {global_img_idx} heatmaps for {len(output_paths)} metadata rows."
        )




@click.command()
@click.option(
    "--model_type",
    type=click.Choice(
        [
            "dragon"
        ],
        case_sensitive=False,
    ),
    default="dragon",
)
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--output_path", type=click.Path(writable=True), required=True)
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--cutout_size", type=int, default=167)
@click.option("--channels", type=int, default=3)
@click.option(
    "--slug",
    type=str,
    required=True,
    help="""This specifies which slug (balanced/unbalanced
              xs, sm, lg, dev) is used to perform predictions on.""",
)
@click.option("--split", type=str, required=True, default="test")
@click.option(
    "--normalize/--no-normalize",
    default=True,
    help="Apply percentile clipping followed by a normalized asinh stretch.",
)
@click.option("--normalize-low-pct", type=float, default=DEFAULT_LOW_PERCENTILE, show_default=True)
@click.option("--normalize-high-pct", type=float, default=DEFAULT_HIGH_PERCENTILE, show_default=True)
@click.option("--asinh-softening", type=float, default=DEFAULT_ASINH_SOFTENING, show_default=True)
@click.option(
    "--normalization-stats",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="JSON with the same per-channel vmin/vmax used during training.",
)
@click.option("--batch_size", type=int, default=256)
@click.option(
    "--n_workers",
    type=int,
    default=4,
    help="""The number of workers to be used during the
              data_preprocessing loading process.""",
)
@click.option(
    "--output_workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of threads used to render and write PNG overlays.",
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    help="""The parallel argument controls whether or not
              to use multiple GPUs when they are available""",
)
@click.option(
    "--label_col",
    type=str,
    default="classes",
    help="""Enter the label column(s) separated by commas. Note
    that you should pass the exactly same argument for label_col
    as was used during the training phase (of the model being used
    for inference). """,
)
@click.option(
    "--mc_dropout/--no-mc_dropout",
    default=True,
    help="""Turn on Monte Carlo dropout during inference.""",
)
@click.option(
    "--n_runs",
    type=int,
    default=1,
    help="""The number of times to run inference. This is helpful
    when usng mc_dropout""",
)
@click.option("--n_classes", type=int, default=6)
@click.option(
    "--ini_run_num",
    type=int,
    default=1,
    help="""The number of the first run. i.e. the output csv files
    are named as (inf_run_num+iteration_number).csv""",
)
@click.option(
    "--dropout_rate",
    type=float,
    default=None,
    help="""The dropout rate to use for all the layers in the
    model. If this is set to None, then the default dropout rate
    in the specific model is used. This option should only be
    used when you have used a non-default dropout rate during
    training and have set --mc_dropout to True. The rate should
    be set equal to the rate used during training.""",
)
@click.option(
    "--crop /--no-crop",
    default=True,
    help="""If True, the images are passed through a cropping transformation
to ensure proper cutout size""",
)
@click.option(
    "--labels/--no-labels",
    default=True,
    help="""If True, this means you have labels available for the dataset.
    If False, this means that you have no labels available and want to do
    pure inference using a pre-trained model.""",
)
def main(
    model_path,
    output_path,
    data_dir,
    cutout_size,
    channels,
    parallel,
    slug,
    split,
    normalize,
    normalize_low_pct,
    normalize_high_pct,
    asinh_softening,
    normalization_stats,
    batch_size,
    n_workers,
    output_workers,
    label_col,
    model_type,
    mc_dropout,
    dropout_rate,
    crop,
    n_runs,
    n_classes,
    ini_run_num,
    labels,
):

    if not 0.0 <= normalize_low_pct < normalize_high_pct <= 100.0:
        raise click.BadParameter(
            "must satisfy 0 <= low < high <= 100",
            param_hint="--normalize-low-pct/--normalize-high-pct",
        )
    if asinh_softening <= 0:
        raise click.BadParameter("must be greater than zero", param_hint="--asinh-softening")

    normalization_kwargs = {
        "low_pct": normalize_low_pct,
        "high_pct": normalize_high_pct,
        "softening": asinh_softening,
    }
    if normalization_stats:
        stats = load_asinh_stats(normalization_stats, channels=channels)
        normalization_kwargs.update(vmin=stats["vmin"], vmax=stats["vmax"])

    logging.info(
        """Creating full heatmaps. Using
            column names to infer number of expected outputs.
            Split and Slug values entered will be ignored and
            info.csv will be used."""
    )
    split = None
    slug = None

    # Create label cols array
    label_col_arr = label_col.split(",")

    # Transforming the dataset to the proper cutout size
    T = None
    if crop:
        T = K.CenterCrop(cutout_size)

    # Test

    # Load the data_preprocessing and create a data_preprocessing loader
    logging.info("Loading images to device...")
    dataset = FITSDataset(
        data_dir,
        slug=slug,
        split=split,
        label_col=label_col_arr,
        transforms=T,
        load_labels=False
    )

    for run_num in range(ini_run_num, n_runs + ini_run_num):

        logging.info(f"Running heatmap run {run_num}")

        # Make predictions
        heatmap(
            model_path,
            output_path,
            dataset,
            cutout_size,
            channels,
            parallel=parallel,
            batch_size=batch_size,
            n_workers=n_workers,
            output_workers=output_workers,
            num_classes=n_classes,
            model_type=model_type,
            mc_dropout=mc_dropout,
            dropout_rate=dropout_rate,
            normalize=normalize,
            normalization_kwargs=normalization_kwargs,
        )



if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
