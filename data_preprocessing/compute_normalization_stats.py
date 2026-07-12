#!/usr/bin/env python
"""Compute Euclid-style global percentile limits from a DRAGON split."""

from pathlib import Path

import click

from data_preprocessing import FITSDataset, compute_asinh_stats, save_asinh_stats
from utils import DEFAULT_HIGH_PERCENTILE, DEFAULT_LOW_PERCENTILE


@click.command()
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--split-slug", type=str, required=True)
@click.option("--split", type=str, default="train", show_default=True)
@click.option("--channels", type=int, default=1, show_default=True)
@click.option("--cutout-size", type=int, default=94, show_default=True)
@click.option("--low-pct", type=float, default=DEFAULT_LOW_PERCENTILE, show_default=True)
@click.option("--high-pct", type=float, default=DEFAULT_HIGH_PERCENTILE, show_default=True)
@click.option("--sample-per-image", type=int, default=1000, show_default=True)
@click.option("--max-samples-per-channel", type=int, default=2000000, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output JSON path (default: DATA_DIR/normalization_stats.json).",
)
def main(
    data_dir,
    split_slug,
    split,
    channels,
    cutout_size,
    low_pct,
    high_pct,
    sample_per_image,
    max_samples_per_channel,
    seed,
    output,
):
    """Sample pixels and save fixed per-channel vmin/vmax values."""
    if not 0.0 <= low_pct < high_pct <= 100.0:
        raise click.BadParameter(
            "must satisfy 0 <= low < high <= 100",
            param_hint="--low-pct/--high-pct",
        )
    if sample_per_image < 0 or max_samples_per_channel < 0:
        raise click.BadParameter(
            "must be non-negative",
            param_hint="--sample-per-image/--max-samples-per-channel",
        )

    dataset = FITSDataset(
        data_dir=data_dir,
        slug=split_slug,
        split=split,
        channels=channels,
        cutout_size=cutout_size,
        load_labels=False,
    )
    stats = compute_asinh_stats(
        dataset,
        channels=channels,
        low_pct=low_pct,
        high_pct=high_pct,
        sample_per_image=sample_per_image,
        max_samples_per_channel=max_samples_per_channel,
        seed=seed,
    )
    stats.update(split_slug=split_slug, split=split)
    output_path = Path(output) if output else Path(data_dir) / "normalization_stats.json"
    save_asinh_stats(stats, output_path)
    click.echo(f"Saved normalization stats to {output_path}")


if __name__ == "__main__":
    main()
