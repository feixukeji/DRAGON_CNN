# -*- coding: utf-8 -*-
from pathlib import Path
import re
import shutil

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
)
from cnn import DRAGON, DRAGON_CUTOUT_SIZE, model_stats
from .create_trainer import create_trainer, create_transfer_learner
from utils import (
    build_optimizer,
    discover_devices,
    load_model_state,
    normalization_kwargs_from_stats,
)


import random
import numpy as np


def _seed_training(seed):
    """Seed every random source used by the main training process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_data_loader_worker(_worker_id):
    """Give each DataLoader worker reproducible Python and NumPy RNGs."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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
    "--model_state",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--run_dir",
    type=click.Path(),
    default=None,
    help=(
        "Root directory for training runs. Outputs are written below "
        "RUN_DIR/EXPERIMENT_NAME (default: DATA_DIR/dragon_runs)."
    ),
)
@click.option(
    "--split_slug",
    type=str,
    required=True,
    help=(
        "Shared basename for splits/<slug>-train.csv, "
        "splits/<slug>-devel.csv, and splits/<slug>-test.csv."
    ),
)
@click.option("--cutout_size", type=int, default=96, show_default=True)
@click.option("--channels", type=int, default=1)
@click.option("--n_classes", type=int, default=6)
@click.option(
    "--n_workers",
    type=int,
    default=4,
    help="Number of DataLoader worker processes.",
)
@click.option(
    "--seed",
    type=click.IntRange(min=0, max=(2 ** 32) - 1),
    default=42,
    show_default=True,
    help="Seed model initialization, data shuffling, augmentation, and dropout.",
)
@click.option("--batch_size", type=int, default=16)
@click.option("--epochs", type=click.IntRange(min=1), default=40, show_default=True)
@click.option(
    "--lr0",
    type=float,
    default=5e-7,
    show_default=True,
    help="Initial learning rate.",
)
@click.option("--momentum", type=float, default=0.9)
@click.option("--weight_decay", type=float, default=0)
@click.option(
    "--max-grad-norm",
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help="Clip the global gradient L2 norm before each optimizer step; 0 disables clipping.",
)
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
    "--scheduler/--no-scheduler",
    default=True,
    show_default=True,
)
@click.option(
    "--class_weight",
    type=click.Choice(["none", "balanced"], case_sensitive=False),
    default="none",
    help="Use class weights in the loss. 'balanced' uses n_samples / (n_classes * class_count) from the train split.",
)
def train(**kwargs):
    """Train a DRAGON model and log the run to W&B."""

    # Copy and log args
    args = {k: v for k, v in kwargs.items()}

    _seed_training(args["seed"])
    logging.info("Using training seed %d.", args["seed"])

    if not args["train"] and not args["model_state"]:
        raise click.UsageError("Transfer learning requires --model_state.")
    if args["cutout_size"] != DRAGON_CUTOUT_SIZE:
        raise click.BadParameter(
            f"DRAGON requires {DRAGON_CUTOUT_SIZE}x{DRAGON_CUTOUT_SIZE} inputs",
            param_hint="--cutout_size",
        )

    experiment_name = args["experiment_name"].strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_name):
        raise click.BadParameter(
            "must be a single safe slug using letters, digits, '.', '_', or '-'",
            param_hint="--experiment_name",
        )
    args["experiment_name"] = experiment_name

    # Discover devices
    args["device"] = discover_devices()

    # Resolve the shared run root and this experiment's isolated output directory.
    run_root = (
        Path(args["run_dir"])
        if args.get("run_dir")
        else Path(args["data_dir"]) / "dragon_runs"
    )
    run_root = run_root.expanduser()
    run_root.mkdir(parents=True, exist_ok=True)
    run_root = run_root.resolve()
    experiment_dir = run_root / experiment_name
    if experiment_dir.parent != run_root:
        raise click.ClickException(
            f"Experiment directory escapes the run root: {experiment_dir}"
        )
    args["run_dir"] = str(run_root)
    args["experiment_dir"] = str(experiment_dir)

    model_args = {
        "channels": args["channels"],
        "num_classes": args["n_classes"]
    }
    model = DRAGON(**model_args).to(args["device"])

    # Load the model from a saved state if provided
    if args["model_state"]:
        logging.info(f'Loading model from {args["model_state"]}...')
        load_model_state(model, args["model_state"], device=args["device"])

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
            split=k,
        )
        for k in splits
    }

    stats_path = Path(args["data_dir"]) / "normalization_stats.json"
    if not stats_path.is_file():
        raise click.ClickException(
            "Training normalization statistics not found: "
            f"{stats_path}. Generate them before training."
        )
    try:
        normalization_kwargs = normalization_kwargs_from_stats(
            stats_path,
            channels=args["channels"],
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"Invalid normalization statistics at {stats_path}: {exc}"
        ) from exc

    logging.info("Loaded normalization stats: %s", stats_path)
    args["normalization_stats"] = str(stats_path)
    args["normalization_mode"] = "global"
    args["normalization_vmin"] = normalization_kwargs["vmin"]
    args["normalization_vmax"] = normalization_kwargs["vmax"]
    args["normalization_softening"] = normalization_kwargs["softening"]

    # Keep each split's RNG independent so devel/test iteration cannot consume
    # the training shuffle stream. The generator also supplies reproducible
    # worker base seeds to _seed_data_loader_worker.
    loaders = {}
    for split_index, (split_name, dataset) in enumerate(datasets.items()):
        generator = torch.Generator()
        generator.manual_seed(args["seed"] + split_index)
        loaders[split_name] = loader_factory(
            dataset,
            shuffle=(split_name == "train"),
            generator=generator,
            worker_init_fn=_seed_data_loader_worker,
        )
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

    criterion = ClassWeightedCrossEntropyLoss(weight=class_weights)

    # Inputs are now validated and loaded. A same-named experiment is an
    # explicit overwrite, so remove only the verified RUN_ROOT/EXPERIMENT_NAME
    # target before W&B or training can create new output files.
    if experiment_dir.is_symlink() or experiment_dir.is_file():
        experiment_dir.unlink()
    elif experiment_dir.is_dir():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=False)

    # Log into W&B
    wandb.login()

    # Initializing W&B run
    with wandb.init(
        project=args["experiment_name"],
        dir=str(experiment_dir),

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
                "max_grad_norm": args["max_grad_norm"],
                "seed": args["seed"],
                "epochs": args["epochs"],
                "batch_size": args["batch_size"]
            }
        }
    ) as run:
        # Write the parameters and model stats to W&B
        args = {**args, **model_stats(model)}
        run.log(args)

        model_path = experiment_dir / "model.pt"

        # Set up trainer
        if args["train"]:
            logging.info("Creating trainer...")
            trainer = create_trainer(
                model,
                optimizer,
                criterion,
                loaders,
                args["device"],
                model_path=model_path,
                normalization_kwargs=normalization_kwargs,
                use_scheduler=args["scheduler"],
                gpu_transforms=train_transforms,
                eval_gpu_transforms=eval_transforms,
                num_epochs=args["epochs"],
                run_dir=experiment_dir,
                max_grad_norm=args["max_grad_norm"],
            )
        else:
            logging.info("Creating trainer and freezing layers for transfer learning...")
            trainer = create_transfer_learner(
                model,
                optimizer,
                criterion,
                loaders,
                args["device"],
                model_path=model_path,
                normalization_kwargs=normalization_kwargs,
                use_scheduler=args["scheduler"],
                gpu_transforms=train_transforms,
                eval_gpu_transforms=eval_transforms,
                unfreeze_warmup_epochs=args["unfreeze_warmup_epochs"],
                unfreeze_blocks_per_epoch=args["unfreeze_blocks_per_epoch"],
                num_epochs=args["epochs"],
                run_dir=experiment_dir,
                max_grad_norm=args["max_grad_norm"],
            )

        # Train and publish the devel macro-F1 best model selected by the trainer.
        trainer.run(loaders["train"], max_epochs=args["epochs"])
        best_model_path = trainer.best_model_path
        if best_model_path != model_path or not model_path.is_file():
            raise RuntimeError("Training completed without producing a best model.")

        # Log model as an artifact
        logging.info(f"Uploading best model from {best_model_path}")
        run.log_artifact(best_model_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train()
