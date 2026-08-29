import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import cv2
import torch
import torch.nn as nn
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

from cnn import (
    DRAGON,
    DRAGON_CUTOUT_SIZE,
    DRAGON_MIN_SIZE,
    DRAGON_SIZE_DIVISOR,
)
from data_preprocessing import HDF5Dataset, get_data_loader
from modules.batched_eigen_grad_cam import BatchedEigenGradCAM
from utils import (
    asinh_normalize,
    discover_devices,
    load_model_state,
    normalization_kwargs_from_stats,
)


def _object_id_output_paths(dataset, output_dir):
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

    output_dir = Path(output_dir)
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
        output_dir,
        dataset,
        channels,
        parallel=False,
        batch_size=256,
        n_workers=1,
        output_workers=4,
        num_classes=6,
        normalization_kwargs=None,
):
    """Using the model defined in model path, return the output values for
    the given set of images"""

    if output_workers < 1:
        raise ValueError("output_workers must be at least 1.")
    if not normalization_kwargs or not {
        "vmin",
        "vmax",
        "softening",
    }.issubset(normalization_kwargs):
        raise ValueError(
            "Heatmap generation requires vmin/vmax/softening loaded from "
            "normalization_stats.json"
        )

    # Discover devices
    device = discover_devices()

    model_args = {
        "channels": channels,
        "num_classes": num_classes
    }
    model = DRAGON(**model_args)

    # Load the model
    logging.info("Loading model...")
    load_model_state(model, model_path, device=device)
    if parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Set to evaluation mode
    model.eval()

    # Create a data_preprocessing loader
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
    )

    # Acquiring GradCAM layer
    target_layer = model.module.layer4 if isinstance(model, nn.DataParallel) else model.layer4

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_heatmap in output_dir.glob("*.png"):
        stale_heatmap.unlink()
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
            X = asinh_normalize(X, **normalization_kwargs)

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
    help="Dataset-level JSON containing training vmin/vmax and softening.",
)
@click.option("--batch-size", type=int, default=256)
@click.option(
    "--n-workers",
    type=int,
    default=4,
    help="""The number of workers to be used during the
              data_preprocessing loading process.""",
)
@click.option(
    "--output-workers",
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
    "--n-classes",
    type=int,
    required=True,
    help="Output class count; must match the checkpoint and labels.csv.",
)
def main(
    model_path,
    output_dir,
    data_dir,
    cutout_size,
    channels,
    parallel,
    normalization_stats,
    batch_size,
    n_workers,
    output_workers,
    n_classes,
):

    if cutout_size % DRAGON_SIZE_DIVISOR or cutout_size < DRAGON_MIN_SIZE:
        raise click.BadParameter(
            "DRAGON halves the map four times, so the cutout side length must "
            f"be a multiple of {DRAGON_SIZE_DIVISOR} and at least "
            f"{DRAGON_MIN_SIZE}",
            param_hint="--cutout-size",
        )

    stats_path = Path(normalization_stats)
    try:
        normalization_kwargs = normalization_kwargs_from_stats(
            stats_path,
            channels=channels,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # Load the data and create a loader.
    logging.info("Loading images to device...")
    dataset = HDF5Dataset(
        data_dir,
        load_labels=False
    )
    expected_image_shape = (channels, cutout_size, cutout_size)
    stored_image_shape = dataset.h5_image_shape[1:]
    if stored_image_shape != expected_image_shape:
        raise click.ClickException(
            "Stored HDF5 image shape does not match the heatmap "
            f"configuration: {stored_image_shape} != {expected_image_shape}"
        )

    heatmap(
        model_path,
        output_dir,
        dataset,
        channels,
        parallel=parallel,
        batch_size=batch_size,
        n_workers=n_workers,
        output_workers=output_workers,
        num_classes=n_classes,
        normalization_kwargs=normalization_kwargs,
    )



if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
