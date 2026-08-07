# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import numpy as np
import pandas as pd


def make_splits(x, weights, label_col, seed=0):
    """Create one deterministic, class-stratified set of data splits."""
    split_items = list(weights.items())
    split_parts = {k: [] for k in weights}

    for _label, group in x.groupby(label_col, sort=False):
        group = group.sample(frac=1, random_state=seed)
        total_size = len(group)
        exact_counts = np.asarray(
            [total_size * weight for _, weight in split_items],
            dtype=float,
        )
        counts = np.floor(exact_counts).astype(int)
        remainder = total_size - int(counts.sum())
        remainder_order = np.argsort(-(exact_counts - counts), kind="stable")
        counts[remainder_order[:remainder]] += 1

        prev_index = 0
        for (k, _weight), count in zip(split_items, counts):
            next_index = prev_index + int(count)
            split_parts[k].append(group.iloc[prev_index:next_index])
            prev_index = next_index

    return {
        k: pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        if parts else pd.DataFrame(columns=x.columns)
        for k, parts in split_parts.items()
    }


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--label-col", default="class", show_default=True)
@click.option("--info-name", default="info.csv", show_default=True)
@click.option("--split-slug", default="stratified", show_default=True)
@click.option(
    "--train-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=0.70,
    show_default=True,
)
@click.option(
    "--devel-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=0.15,
    show_default=True,
)
@click.option(
    "--test-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=0.15,
    show_default=True,
)
@click.option("--seed", type=int, default=0, show_default=True)
def main(
    data_dir,
    label_col,
    info_name,
    split_slug,
    train_fraction,
    devel_fraction,
    test_fraction,
    seed,
):
    """Generate one train/devel/test split set from aligned metadata."""
    weights = {
        "train": train_fraction,
        "devel": devel_fraction,
        "test": test_fraction,
    }
    if not np.isclose(sum(weights.values()), 1.0):
        raise click.UsageError(
            "--train-fraction, --devel-fraction, and --test-fraction must sum to 1"
        )
    if (
        not split_slug
        or split_slug in {".", ".."}
        or Path(split_slug).name != split_slug
    ):
        raise click.BadParameter(
            "must be a non-empty file-name slug",
            param_hint="--split-slug",
        )

    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    info_path = data_dir / info_name
    if not info_path.is_file():
        raise click.ClickException(f"Metadata CSV not found: {info_path}")
    df = pd.read_csv(info_path)
    if df.empty:
        raise click.ClickException(f"Metadata CSV contains no rows: {info_path}")
    if label_col not in df.columns:
        raise click.ClickException(
            f"Metadata CSV is missing label column '{label_col}': {info_path}"
        )
    if df[label_col].isna().any():
        raise click.ClickException(
            f"Metadata label column '{label_col}' contains missing values: {info_path}"
        )
    if "h5_index" not in df.columns:
        raise click.ClickException(
            f"Metadata CSV is missing required 'h5_index' column: {info_path}"
        )
    if df["h5_index"].duplicated().any():
        raise click.ClickException(
            f"Metadata CSV contains duplicate 'h5_index' values: {info_path}"
        )

    splits = make_splits(df, weights, label_col, seed=seed)
    for split_name, split_df in splits.items():
        output_path = splits_dir / f"{split_slug}-{split_name}.csv"
        split_df.to_csv(output_path, index=False)
        click.echo(f"Saved {len(split_df)} rows to {output_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
