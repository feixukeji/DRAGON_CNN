# -*- coding: utf-8 -*-
from pathlib import Path

import click
import logging
from functools import partial

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia.augmentation as K

from data_preprocessing import (
    FITSDataset,
    get_data_loader,
    load_or_compute_asinh_stats,
)
from cnn import model_factory, model_stats, save_trained_model
from create_trainer import create_trainer, create_transfer_learner
from utils import (
    DEFAULT_ASINH_SOFTENING,
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    build_optimizer,
    discover_devices,
    specify_dropout_rate,
)


import random
import numpy as np


def _compute_balanced_class_weights(labels, num_classes, device):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = len(labels) / (num_classes * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


class ClassWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight.detach().float())

    def forward(self, input, target):
        weight = self._weight_for(input)
        return F.cross_entropy(input, target, weight=weight)

    def _weight_for(self, input):
        if self.weight is None:
            return None
        return self.weight.to(device=input.device, dtype=input.dtype)


class ClassWeightedNLLLoss(ClassWeightedCrossEntropyLoss):
    def forward(self, input, target):
        weight = self._weight_for(input)
        return F.nll_loss(input, target, weight=weight)


class RandomDihedralAugmentation(nn.Module):
    """Apply one of four right-angle rotations and an optional horizontal flip."""

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError(
                f"Expected a (batch, channels, height, width) tensor, got {images.shape}"
            )

        transform_ids = torch.randint(0, 8, (images.shape[0],), device=images.device)
        augmented = torch.empty_like(images)

        for transform_id in range(8):
            mask = transform_ids == transform_id
            if not torch.any(mask):
                continue

            transformed = images[mask]
            if transform_id >= 4:
                transformed = torch.flip(transformed, dims=(-1,))
            augmented[mask] = torch.rot90(
                transformed,
                k=transform_id % 4,
                dims=(-2, -1),
            )

        return augmented


@click.command()
@click.option("--experiment_name", type=str, default="demo")
@click.option(
    "--run_id",
    type=str,
    default=None,
    help="""The run id. Practically this only needs to be used
if you are resuming a previosuly run experiment""",
)
@click.option(
    "--run_name",
    type=str,
    default=None,
    help="""A run is supposed to be a sub-class of an experiment.
So this variable should be specified accordingly""",
)
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
@click.option("--model_state", type=click.Path(exists=True), default=None)
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option(
    "--run_dir",
    type=click.Path(),
    default=None,
    help="Output directory for checkpoints/models (defaults to data_dir).",
)
@click.option(
    "--split_slug",
    type=str,
    required=True,
    help="""This specifies how the data_preprocessing is split into train/
devel/test sets. Balanced/Unbalanced refer to whether selecting
equal number of images from each class. xs, sm, lg, dev all refer
to what fraction is picked for train/devel/test.""",
)
@click.option("--cutout_size", type=int, default=94)
@click.option("--channels", type=int, default=1)
@click.option("--n_classes", type=int, default=6)
@click.option(
    "--n_workers",
    type=int,
    default=4,
    help="""The number of workers to be used during the
data_preprocessing loading process.""",
)
@click.option(
    "--loss",
    type=click.Choice(
        [
            "nll",
            "ce",
        ],
        case_sensitive=False,
    ),
    default="ce",
    help="""The loss function to use""",
)
@click.option("--batch_size", type=int, default=16)
@click.option("--epochs", type=int, default=40)
@click.option(
    "--lr0",
    "--lr",
    "lr0",
    type=float,
    default=5e-7,
    show_default=True,
    help="Initial learning rate; --lr is retained as a compatibility alias.",
)
@click.option("--momentum", type=float, default=0.9)
@click.option("--weight_decay", type=float, default=0)
@click.option(
    "--optimizer",
    type=click.Choice(["sgd", "adamw"], case_sensitive=False),
    default="sgd",
    show_default=True,
)
@click.option("--adamw-beta1", type=click.FloatRange(0.0, 1.0, max_open=True), default=0.9, show_default=True)
@click.option("--adamw-beta2", type=click.FloatRange(0.0, 1.0, max_open=True), default=0.999, show_default=True)
@click.option("--adamw-eps", type=click.FloatRange(min=0.0, min_open=True), default=1e-8, show_default=True)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    help="""The parallel argument controls whether or not
to use multiple GPUs when they are available""",
)
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
    type=click.Path(dir_okay=False),
    default=None,
    help=(
        "Euclid-style JSON containing per-channel vmin/vmax. If the path does "
        "not exist, statistics are computed from the training split and saved. "
        "If omitted, percentiles are computed per cutout/channel."
    ),
)
@click.option("--normalization-sample-per-image", type=int, default=1000, show_default=True)
@click.option("--normalization-max-samples", type=int, default=2000000, show_default=True)
@click.option("--normalization-seed", type=int, default=42, show_default=True)
@click.option(
    "--crop/--no-crop",
    default=True,
    help="""If True, all images are passed through a cropping
operation before being fed into the network. Images are cropped
to the cutout_size parameter""",
)
@click.option(
    "--nesterov/--no-nesterov",
    default=False,
    help="""Whether to use Nesterov momentum or not""",
)
@click.option(
    "--dropout_rate",
    type=float,
    default=None,
    help="""The dropout rate to use for all the layers in the
    model. If this is set to None, then the default dropout rate
    in the specific model is used.""",
)
@click.option(
    "--force_reload/--no_force_reload",
    default=False,
)
@click.option(
    "--expand_data",
    type=int,
    default=1,
    help="""This controls the factor by which the training
data is augmented""",
)
@click.option(
    "--augment/--no-augment",
    default=True,
    help="Randomly apply one of the eight right-angle rotation/horizontal-flip transforms to each training sample.",
)
@click.option(
    "--train/--transfer_learn",
    default=True,
    help="""Specifies whether you wish to do transfer learning. If transfer learning,
    you must specify model path in the model_state argument."""
)
@click.option(
    "--unfreeze-warmup-epochs",
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
    help="Head-only epochs before unfreezing the first backbone block.",
)
@click.option(
    "--unfreeze-blocks-per-epoch",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Complete layerN backbone blocks to unfreeze after each epoch.",
)
@click.option(
    "--scheduler/--no_scheduler",
    default=True
)
@click.option(
    "--class_weight",
    type=click.Choice(["none", "balanced"], case_sensitive=False),
    default="none",
    help="Use class weights in the loss. 'balanced' uses n_samples / (n_classes * class_count) from the train split.",
)
def train(**kwargs):
    """Runs the training procedure using MLFlow."""

    # Copy and log args
    args = {k: v for k, v in kwargs.items()}

    if not 0.0 <= args["normalize_low_pct"] < args["normalize_high_pct"] <= 100.0:
        raise click.BadParameter(
            "must satisfy 0 <= low < high <= 100",
            param_hint="--normalize-low-pct/--normalize-high-pct",
        )
    if args["asinh_softening"] <= 0:
        raise click.BadParameter("must be greater than zero", param_hint="--asinh-softening")
    if args["normalization_sample_per_image"] < 0 or args["normalization_max_samples"] < 0:
        raise click.BadParameter(
            "must be non-negative",
            param_hint="--normalization-sample-per-image/--normalization-max-samples",
        )

    normalization_kwargs = {
        "low_pct": args["normalize_low_pct"],
        "high_pct": args["normalize_high_pct"],
        "softening": args["asinh_softening"],
    }
    if not args["normalize"]:
        args["normalization_mode"] = "disabled"
    elif args["normalization_stats"]:
        args["normalization_mode"] = "global"
    else:
        args["normalization_mode"] = "per_cutout"

    # Discover devices
    args["device"] = discover_devices()

    # Resolve run directory
    run_dir = Path(args["run_dir"]) if args.get("run_dir") else Path(args["data_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    args["run_dir"] = str(run_dir)

    # Create the model given model_type
    cls = model_factory(args["model_type"])
    model_args = {
        "cutout_size": args["cutout_size"],
        "channels": args["channels"],
        "num_classes": args["n_classes"]
    }

    if "drp" in args["model_type"].split("_"):
        logging.info(
            "Using dropout rate of {} in the model".format(
                args["dropout_rate"]
            )
        )
        model_args["dropout"] = "True"

    model = cls(**model_args)
    if args["parallel"] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        args["device"] = "cuda"

    model = model.to(args["device"])

    # Chnaging the default dropout rate if specified
    if args["dropout_rate"] is not None:
        specify_dropout_rate(model, args["dropout_rate"])

    # Load the model from a saved state if provided
    if args["model_state"]:
        logging.info(f'Loading model from {args["model_state"]}...')
        if args["device"] == "cpu":
            model.load_state_dict(torch.load(args["model_state"], map_location="cpu"))
        else:
            model.load_state_dict(torch.load(args["model_state"]))

    optimizer = build_optimizer(
        model,
        optimizer_name=args["optimizer"],
        lr=args["lr0"],
        weight_decay=args["weight_decay"],
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        adamw_beta1=args["adamw_beta1"],
        adamw_beta2=args["adamw_beta2"],
        adamw_eps=args["adamw_eps"],
    )
    logging.info("Using %s optimizer with lr=%g.", args["optimizer"].upper(), args["lr0"])

    # Create a DataLoader factory based on command-line args
    loader_factory = partial(
        get_data_loader,
        batch_size=args["batch_size"],
        n_workers=args["n_workers"],
    )

    # Keep deterministic preprocessing separate from train-only random augmentation.
    eval_transforms = []
    if args["crop"]:
        eval_transforms.append(K.CenterCrop(args["cutout_size"]))
    train_transforms = list(eval_transforms)
    if args["augment"]:
        train_transforms.append(RandomDihedralAugmentation())

    train_transforms = train_transforms or None
    eval_transforms = eval_transforms or None

    # Generate the DataLoaders and log the train/devel/test split sizes
    splits = ("train", "devel", "test")
    datasets = {
        k: FITSDataset(
            data_dir=args["data_dir"],
            slug=args["split_slug"],
            cutout_size=args["cutout_size"],
            channels=args["channels"],
            normalize=args["normalize"],
            transforms=None,
            split=k,
            num_classes=args["n_classes"],
            expand_factor=args["expand_data"] if k == "train" else 1,
            force_reload=args["force_reload"]
        )
        for k in splits
    }

    if args["normalize"] and args["normalization_stats"]:
        stats, computed = load_or_compute_asinh_stats(
            args["normalization_stats"],
            datasets["train"],
            channels=args["channels"],
            low_pct=args["normalize_low_pct"],
            high_pct=args["normalize_high_pct"],
            sample_per_image=args["normalization_sample_per_image"],
            max_samples_per_channel=args["normalization_max_samples"],
            seed=args["normalization_seed"],
        )
        action = "Computed and saved" if computed else "Loaded"
        logging.info(f'{action} normalization stats: {args["normalization_stats"]}')
        normalization_kwargs.update(vmin=stats["vmin"], vmax=stats["vmax"])
        args["normalization_vmin"] = stats["vmin"]
        args["normalization_vmax"] = stats["vmax"]

    loaders = {k: loader_factory(v, shuffle=(k == 'train')) for k, v in datasets.items()}
    args["splits"] = {k: len(v.dataset) for k, v in loaders.items()}

    class_weights = None
    if args["class_weight"].lower() == "balanced":
        class_weights, class_counts = _compute_balanced_class_weights(
            datasets["train"].labels,
            args["n_classes"],
            args["device"],
        )
        args["class_counts"] = class_counts.astype(int).tolist()
        args["class_weights"] = class_weights.detach().cpu().tolist()
        logging.info(f"Using balanced class weights: {args['class_weights']}")

    # Define the criterion
    loss_dict = {
        "nll": ClassWeightedNLLLoss(weight=class_weights),
        "ce": ClassWeightedCrossEntropyLoss(weight=class_weights),
    }
    criterion = loss_dict[args["loss"]]

    # Log into W&B
    wandb.login()

    # Initializing W&B run
    with wandb.init(
        project=args["experiment_name"],
        id=args["run_id"],
        resume="allow",

        # track hyperparameters and run metadata
        config={
            "num_classes": args["n_classes"],
            "architecture": "CNN",
            "parameters": {
                "initial_learning_rate": args["lr0"],
                "optimizer": args["optimizer"],
                "momentum": args["momentum"],
                "nesterov": args["nesterov"],
                "weight_decay": args["weight_decay"],
                "adamw_beta1": args["adamw_beta1"],
                "adamw_beta2": args["adamw_beta2"],
                "adamw_eps": args["adamw_eps"],
                "epochs": args["epochs"],
                "batch_size": args["batch_size"]
            }
        }
    ) as run:
        # Write the parameters and model stats to W&B
        args = {**args, **model_stats(model)}
        run.log(args)

        # Making an output directory for checkpoints
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set up trainer
        if args["train"]:
            logging.info("Creating trainer...")
            # Register a hook to automatically clamp gradients during the backward pass
            for param in model.parameters():
                param.register_hook(lambda grad: torch.clamp(grad, -1, 1))

            trainer = create_trainer(
                model,
                optimizer,
                criterion,
                loaders,
                args["device"],
                args["scheduler"],
                gpu_transforms=train_transforms,
                eval_gpu_transforms=eval_transforms,
                normalize=args["normalize"],
                normalization_kwargs=normalization_kwargs,
                checkpoint_dir=checkpoint_dir,
                run_id=run.id,
                num_epochs=args["epochs"],
                run_dir=run_dir,
            )
        else:
            logging.info("Creating trainer and freezing layers for transfer learning...")
            trainer = create_transfer_learner(
                model,
                optimizer,
                criterion,
                loaders,
                args["device"],
                args["scheduler"],
                gpu_transforms=train_transforms,
                eval_gpu_transforms=eval_transforms,
                normalize=args["normalize"],
                normalization_kwargs=normalization_kwargs,
                unfreeze_warmup_epochs=args["unfreeze_warmup_epochs"],
                unfreeze_blocks_per_epoch=args["unfreeze_blocks_per_epoch"],
                checkpoint_dir=checkpoint_dir,
                run_id=run.id,
                num_epochs=args["epochs"],
                run_dir=run_dir,
            )

        # Run trainer and save model state
        trainer.run(loaders["train"], max_epochs=args["epochs"])
        slug = (
            f"{args['experiment_name']}-{args['split_slug']}-"
            f"{run.id}"
        )

        model_path = save_trained_model(model, slug, output_dir=run_dir / "models")

        # Log model as an artifact
        logging.info(f"Saved model to {model_path}")
        run.log_artifact(model_path)

        # Finish the W&B run!
        wandb.finish()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train()
