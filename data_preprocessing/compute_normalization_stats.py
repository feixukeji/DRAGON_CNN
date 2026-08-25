"""Compute Euclid-style global percentile limits from a DRAGON split."""

from pathlib import Path

import click

from utils import (
    DEFAULT_ASINH_SOFTENING,
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
)

from .dataset import HDF5Dataset
from .normalization import compute_asinh_stats, save_asinh_stats


@click.command()
@click.option(
    "--data-dir", type=click.Path(exists=True, file_okay=False), required=True
)
@click.option("--split-slug", type=str, required=True)
@click.option("--split", type=str, default="train", show_default=True)
@click.option("--channels", type=int, default=1, show_default=True)
@click.option(
    "--low-pct", type=float, default=DEFAULT_LOW_PERCENTILE, show_default=True
)
@click.option(
    "--high-pct", type=float, default=DEFAULT_HIGH_PERCENTILE, show_default=True
)
@click.option(
    "--asinh-softening",
    type=float,
    default=DEFAULT_ASINH_SOFTENING,
    show_default=True,
)
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
    low_pct,
    high_pct,
    asinh_softening,
    sample_per_image,
    max_samples_per_channel,
    seed,
    output,
):
    """Sample pixels and save fixed per-channel vmin/vmax values."""
    dataset = HDF5Dataset(
        data_dir=data_dir,
        slug=split_slug,
        split=split,
        load_labels=False,
    )
    try:
        stats = compute_asinh_stats(
            dataset,
            channels=channels,
            low_pct=low_pct,
            high_pct=high_pct,
            softening=asinh_softening,
            sample_per_image=sample_per_image,
            max_samples_per_channel=max_samples_per_channel,
            seed=seed,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    stats.update(split_slug=split_slug, split=split)
    output_path = (
        Path(output) if output else Path(data_dir) / "normalization_stats.json"
    )
    save_asinh_stats(stats, output_path)
    click.echo(f"Saved normalization stats to {output_path}")


if __name__ == "__main__":
    main()
