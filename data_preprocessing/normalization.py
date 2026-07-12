"""Training-split statistics for Euclid-style asinh normalization."""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import load_asinh_stats


def compute_asinh_stats(
    dataset,
    channels,
    low_pct=0.5,
    high_pct=99.5,
    sample_per_image=1000,
    max_samples_per_channel=2_000_000,
    seed=42,
    show_progress=True,
):
    """Estimate per-channel percentiles from a dataset using bounded sampling.

    Sampling is balanced across cutouts, like Euclid's per-image sampling, and
    the global cap prevents large training sets from exhausting host memory.
    ``max_samples_per_channel=0`` disables the global cap.
    """
    if not 0.0 <= low_pct < high_pct <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= low_pct < high_pct <= 100.")
    if channels <= 0:
        raise ValueError("channels must be greater than zero.")
    if sample_per_image < 0 or max_samples_per_channel < 0:
        raise ValueError("Sampling limits must be non-negative.")

    rng = np.random.default_rng(seed)
    samples_by_channel = [[] for _ in range(channels)]

    # Avoid counting expand_factor replicas when the dataset exposes base labels.
    num_images = len(dataset.labels) if hasattr(dataset, "labels") else len(dataset)
    per_image_limit = sample_per_image
    if max_samples_per_channel and num_images:
        global_quota = max(1, int(np.ceil(max_samples_per_channel / num_images)))
        per_image_limit = (
            min(sample_per_image, global_quota) if sample_per_image else global_quota
        )
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
                f"Dataset returned shape {image_np.shape}, expected ({channels}, H, W)."
            )

        for channel in range(channels):
            flat = image_np[channel].reshape(-1).astype(np.float32, copy=False)
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            if per_image_limit and flat.size > per_image_limit:
                flat = flat[rng.choice(flat.size, size=per_image_limit, replace=False)]
            samples_by_channel[channel].append(flat)

    if num_images == 0 or any(not samples for samples in samples_by_channel):
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
    vmax = [float(np.percentile(values, high_pct)) for values in values_by_channel]
    if any(high <= low for low, high in zip(vmin, vmax)):
        raise ValueError("Computed vmax must be greater than vmin for every channel.")

    return {
        "low_pct": low_pct,
        "high_pct": high_pct,
        "vmin": vmin,
        "vmax": vmax,
        "num_images": num_images,
        "sample_per_image": sample_per_image,
        "max_samples_per_channel": max_samples_per_channel,
        "seed": seed,
        "channels": channels,
    }


def save_asinh_stats(stats, path):
    """Write normalization statistics as Euclid-compatible JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_path


def load_or_compute_asinh_stats(path, dataset, channels, **compute_kwargs):
    """Load an existing stats JSON or compute and persist it from a split."""
    stats_path = Path(path)
    if stats_path.is_file():
        return load_asinh_stats(stats_path, channels=channels), False

    stats = compute_asinh_stats(dataset, channels=channels, **compute_kwargs)
    save_asinh_stats(stats, stats_path)
    return stats, True
