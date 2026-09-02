"""Training-split statistics for Euclid-style asinh normalization."""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (
    DEFAULT_ASINH_SOFTENING,
    validate_asinh_softening,
    validate_asinh_stats,
)

PIXEL_HIGH_STATISTIC = "pixel"
PEAK_HIGH_STATISTIC = "peak"
HIGH_STATISTICS = (PIXEL_HIGH_STATISTIC, PEAK_HIGH_STATISTIC)


def _sample_h5_in_chunk_order(
    dataset,
    channels,
    num_images,
    per_image_limit,
    rng,
    show_progress,
):
    """Sample an HDF5-backed dataset while reading each row chunk once."""
    h5_path = getattr(dataset, "h5_path", None)
    h5_indices = getattr(dataset, "h5_indices", None)
    if h5_path is None or h5_indices is None:
        return None

    try:
        h5_indices = np.asarray(h5_indices, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return None
    if h5_indices.shape != (num_images,):
        return None

    import h5py

    with h5py.File(
        h5_path,
        "r",
        swmr=True,
        rdcc_nbytes=16 * 1024**2,
        rdcc_w0=1.0,
    ) as h5_file:
        if "images" not in h5_file:
            return None
        images = h5_file["images"]
        if (
            images.ndim != 4
            or images.shape[1] != channels
            or images.shape[2] == 1
        ):
            return None

        pixels_per_image = int(np.prod(images.shape[2:]))
        if not per_image_limit or pixels_per_image <= per_image_limit:
            return None
        if num_images and (
            h5_indices.min() < 0 or h5_indices.max() >= images.shape[0]
        ):
            return None

        # Consume RNG in the same logical image/channel order as the generic
        # path. Disk reads may then be reordered without changing the samples.
        pixel_indices = np.empty(
            (num_images, channels, per_image_limit), dtype=np.int32
        )
        for image_index in range(num_images):
            for channel in range(channels):
                pixel_indices[image_index, channel] = rng.choice(
                    pixels_per_image,
                    size=per_image_limit,
                    replace=False,
                )

        sampled = np.empty(
            (channels, num_images, per_image_limit), dtype=np.float32
        )
        # Peaks come from every pixel of the cutout, not from the subsample; a
        # random 1000-pixel draw almost never contains the source core.
        peaks = np.empty((channels, num_images), dtype=np.float32)
        row_bytes = pixels_per_image * channels * images.dtype.itemsize
        chunk_rows = (
            int(images.chunks[0])
            if images.chunks is not None
            else max(1, (8 * 1024**2) // row_bytes)
        )

        logical_order = np.argsort(h5_indices, kind="stable")
        sorted_chunk_ids = h5_indices[logical_order] // chunk_rows
        group_starts = np.r_[
            0,
            np.flatnonzero(np.diff(sorted_chunk_ids)) + 1,
            num_images,
        ]
        progress = tqdm(
            total=num_images,
            desc="Sampling normalization stats",
            unit="cutout",
            disable=not show_progress,
        )
        try:
            for group_start, group_stop in zip(group_starts[:-1], group_starts[1:]):
                logical_indices = logical_order[group_start:group_stop]
                chunk_start = int(sorted_chunk_ids[group_start]) * chunk_rows
                chunk_stop = min(chunk_start + chunk_rows, images.shape[0])

                block = np.asarray(
                    images[chunk_start:chunk_stop], dtype=np.float32
                )
                local_indices = h5_indices[logical_indices] - chunk_start
                flat = block[local_indices].reshape(
                    len(logical_indices), channels, pixels_per_image
                )
                selected = np.take_along_axis(
                    flat,
                    pixel_indices[logical_indices],
                    axis=2,
                )
                selected = np.nan_to_num(
                    selected,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                    copy=False,
                )
                image_peaks = np.nan_to_num(
                    flat, nan=-np.inf, posinf=-np.inf, neginf=-np.inf, copy=True
                ).max(axis=2)
                for channel in range(channels):
                    sampled[channel, logical_indices] = selected[:, channel]
                    peaks[channel, logical_indices] = image_peaks[:, channel]
                progress.update(len(logical_indices))
        finally:
            progress.close()

    # Keep the same nested shape consumed by the generic aggregation below.
    return (
        [[sampled[channel].reshape(-1)] for channel in range(channels)],
        [peaks[channel] for channel in range(channels)],
    )


def compute_asinh_stats(
    dataset,
    channels,
    low_pct=0.5,
    high_pct=99.5,
    softening=DEFAULT_ASINH_SOFTENING,
    sample_per_image=1000,
    max_samples_per_channel=2_000_000,
    seed=42,
    show_progress=True,
    high_statistic=PIXEL_HIGH_STATISTIC,
):
    """Estimate per-channel percentiles with bounded, per-cutout sampling.

    Set ``max_samples_per_channel=0`` to disable the global cap.

    ``high_statistic`` selects the population ``high_pct`` is taken over when
    setting ``vmax``. ``"pixel"`` uses the sampled pixels, which are dominated
    by sky and therefore place ``vmax`` far below any source core, clipping
    bright sources into a flat plateau and discarding their colour. ``"peak"``
    uses one whole-cutout maximum per image and channel instead, so ``vmax``
    tracks the source brightness distribution. ``vmin`` always comes from the
    sampled pixels.
    """
    if not 0.0 <= low_pct < high_pct <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= low_pct < high_pct <= 100.")
    if high_statistic not in HIGH_STATISTICS:
        raise ValueError(
            f"high_statistic must be one of {HIGH_STATISTICS}; "
            f"received {high_statistic!r}."
        )
    if channels <= 0:
        raise ValueError("channels must be greater than zero.")
    softening = validate_asinh_softening(softening)
    if sample_per_image < 0 or max_samples_per_channel < 0:
        raise ValueError("Sampling limits must be non-negative.")

    # Prefer the dataset's base row count when it exposes aligned labels.
    num_images = len(dataset.labels) if hasattr(dataset, "labels") else len(dataset)
    if num_images == 0:
        raise ValueError("No pixels were found in the requested dataset split.")

    per_image_limit = sample_per_image
    if max_samples_per_channel and num_images:
        global_quota = max(1, int(np.ceil(max_samples_per_channel / num_images)))
        per_image_limit = (
            min(sample_per_image, global_quota) if sample_per_image else global_quota
        )

    rng = np.random.default_rng(seed)
    sampled = _sample_h5_in_chunk_order(
        dataset=dataset,
        channels=channels,
        num_images=num_images,
        per_image_limit=per_image_limit,
        rng=rng,
        show_progress=show_progress,
    )
    samples_by_channel, peaks_by_channel = (None, None) if sampled is None else sampled
    if samples_by_channel is None:
        samples_by_channel = [[] for _ in range(channels)]
        peaks_by_channel = [[] for _ in range(channels)]
        iterator = tqdm(
            range(num_images),
            desc="Sampling normalization stats",
            unit="cutout",
            disable=not show_progress,
        )
        for index in iterator:
            image, _ = dataset[index]
            image_np = image.detach().cpu().numpy()
            if image_np.ndim == 2:
                image_np = image_np[None, ...]
            if image_np.ndim != 3 or image_np.shape[0] != channels:
                raise ValueError(
                    f"Dataset returned shape {image_np.shape}, "
                    f"expected ({channels}, H, W)."
                )

            for channel in range(channels):
                flat = image_np[channel].reshape(-1).astype(
                    np.float32, copy=False
                )
                flat = np.nan_to_num(
                    flat, nan=0.0, posinf=0.0, neginf=0.0
                )
                # Take the peak before subsampling, for the same reason as the
                # HDF5 path: a subsample rarely contains the source core.
                peaks_by_channel[channel].append(flat.max())
                if per_image_limit and flat.size > per_image_limit:
                    flat = flat[
                        rng.choice(
                            flat.size,
                            size=per_image_limit,
                            replace=False,
                        )
                    ]
                samples_by_channel[channel].append(flat)

    if any(not samples for samples in samples_by_channel):
        raise ValueError("No pixels were found in the requested dataset split.")

    values_by_channel = []
    for samples in samples_by_channel:
        values = np.concatenate(samples).astype(np.float32, copy=False)
        if max_samples_per_channel and values.size > max_samples_per_channel:
            values = values[
                rng.choice(values.size, size=max_samples_per_channel, replace=False)
            ]
        values_by_channel.append(values)

    vmin = [float(np.percentile(values, low_pct)) for values in values_by_channel]
    if high_statistic == PEAK_HIGH_STATISTIC:
        peak_values = [
            np.asarray(peaks, dtype=np.float32).reshape(-1)
            for peaks in peaks_by_channel
        ]
        if any(not values.size for values in peak_values):
            raise ValueError("No cutout peaks were found in the requested split.")
        peak_values = [values[np.isfinite(values)] for values in peak_values]
        if any(not values.size for values in peak_values):
            raise ValueError("Every cutout peak in the requested split is non-finite.")
        vmax = [float(np.percentile(values, high_pct)) for values in peak_values]
    else:
        vmax = [float(np.percentile(values, high_pct)) for values in values_by_channel]
    if any(high <= low for low, high in zip(vmin, vmax)):
        raise ValueError("Computed vmax must be greater than vmin for every channel.")

    return {
        "low_pct": low_pct,
        "high_pct": high_pct,
        "high_statistic": high_statistic,
        "softening": softening,
        "vmin": vmin,
        "vmax": vmax,
        "num_images": num_images,
        "sample_per_image": sample_per_image,
        "max_samples_per_channel": max_samples_per_channel,
        "seed": seed,
        "channels": channels,
    }


def save_asinh_stats(stats, path):
    """Validate and write fixed asinh normalization statistics."""
    validated_stats = validate_asinh_stats(stats)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(validated_stats, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
