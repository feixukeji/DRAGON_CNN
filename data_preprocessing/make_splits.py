# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import numpy as np
import pandas as pd

split_types = dict(
    xs=dict(train=0.027, devel=0.003, test=0.970),
    sm=dict(train=0.045, devel=0.005, test=0.950),
    md=dict(train=0.090, devel=0.010, test=0.900),
    lg=dict(train=0.200, devel=0.050, test=0.750),
    xl=dict(train=0.450, devel=0.050, test=0.500),
    dev=dict(train=0.700, devel=0.150, test=0.150),
    dev2=dict(train=0.700, devel=0.050, test=0.250),
)


def make_splits(x, weights, label_col):
    split_items = list(weights.items())
    split_parts = {k: [] for k in weights}

    for _label, group in x.groupby(label_col, sort=False):
        total_size = len(group)
        prev_index = 0
        for i, (k, v) in enumerate(split_items):
            next_index = total_size if i == len(split_items) - 1 else prev_index + int(total_size * v)
            split_parts[k].append(group.iloc[prev_index:next_index])
            prev_index = next_index

    return {
        k: pd.concat(parts).sample(frac=1, random_state=0).reset_index(drop=True)
        if parts else pd.DataFrame(columns=x.columns)
        for k, parts in split_parts.items()
    }


@click.command()
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--target_metric", type=str, default="classes")
@click.option("--info_name", type=str, default="info.csv")
def main(data_dir, target_metric, info_name):
    """Generate train/devel/test splits from the dataset provided."""
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / info_name)
    df['h5_index'] = np.arange(len(df))

    # Keep original class frequencies; class imbalance is handled by the loss.
    df = df.sample(frac=1, random_state=0)

    for split_type in split_types.keys():
        splits = make_splits(df, split_types[split_type], target_metric)
        split_slug = f"unbalanced-{split_type}"
        for k, v in splits.items():
            v.to_csv(splits_dir / f"{split_slug}-{k}.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
