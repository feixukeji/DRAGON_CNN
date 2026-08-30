import logging
import random
import re
import shutil
from functools import partial
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from cnn import (
    DRAGON,
    DRAGON_CUTOUT_SIZE,
    DRAGON_MIN_SIZE,
    DRAGON_SIZE_DIVISOR,
    model_stats,
)
from data_preprocessing import (
    HDF5Dataset,
    get_data_loader,
    preload_h5_images,
)
from utils import (
    build_optimizer,
    discover_devices,
    load_model_state,
    normalization_kwargs_from_stats,
)

from .create_trainer import create_trainer, create_transfer_learner


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
@click.option(
    "--project",
    type=str,
    default="dragon",
    show_default=True,
    help="W&B project name.",
)
@click.option(
    "--experiment",
    type=str,
    default="demo",
    show_default=True,
    help="Experiment name used for the W&B run and local output directory.",
)
@click.option(
    "--model-state",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--run-dir",
    type=click.Path(),
    default=None,
    help=(
        "Root directory for training runs. Outputs are written below "
        "RUN_DIR/EXPERIMENT (default: DATA_DIR/dragon_runs)."
    ),
)
@click.option(
    "--split-slug",
    type=str,
    required=True,
    help=(
        "Shared basename for splits/<slug>-train.csv, "
        "splits/<slug>-devel.csv, and splits/<slug>-test.csv."
    ),
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
    help="Input band count; must match the stored cutouts.",
)
@click.option(
    "--n-classes",
    type=int,
    required=True,
    help="Output class count; must match labels.csv.",
)
@click.option(
    "--n-workers",
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
@click.option("--batch-size", type=int, default=16)
@click.option("--epochs", type=click.IntRange(min=1), default=40, show_default=True)
@click.option(
    "--lr0",
    type=float,
    default=5e-7,
    show_default=True,
    help="Initial learning rate.",
)
@click.option("--momentum", type=float, default=0.9)
@click.option("--weight-decay", type=float, default=0)
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
    "--train/--transfer-learn",
    default=True,
    help="""Specifies whether you wish to do transfer learning. If transfer learning,
    you must specify a model path with --model-state."""
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
    "--class-weight",
    type=click.Choice(["none", "balanced"], case_sensitive=False),
    default="none",
    help="Use class weights in the loss. 'balanced' uses n_samples / (n_classes * class_count) from the train split.",
)
def train(**kwargs):
    """Train a DRAGON model and log the run to W&B."""

    args = {k: v for k, v in kwargs.items()}

    _seed_training(args["seed"])
    logging.info("Using training seed %d.", args["seed"])

    if not args["train"] and not args["model_state"]:
        raise click.UsageError("Transfer learning requires --model-state.")
    cutout_size = args["cutout_size"]
    if cutout_size % DRAGON_SIZE_DIVISOR or cutout_size < DRAGON_MIN_SIZE:
        raise click.BadParameter(
            "DRAGON halves the map four times, so the cutout side length must "
            f"be a multiple of {DRAGON_SIZE_DIVISOR} and at least "
            f"{DRAGON_MIN_SIZE}",
            param_hint="--cutout-size",
        )

    for parameter in ("project", "experiment"):
        value = args[parameter].strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise click.BadParameter(
                "must be a single safe slug using letters, digits, '.', '_', or '-'",
                param_hint=f"--{parameter}",
            )
        args[parameter] = value

    args["device"] = discover_devices()

    # Keep each experiment inside the resolved run root.
    run_root = (
        Path(args["run_dir"])
        if args.get("run_dir")
        else Path(args["data_dir"]) / "dragon_runs"
    )
    run_root = run_root.expanduser()
    run_root.mkdir(parents=True, exist_ok=True)
    run_root = run_root.resolve()
    experiment_dir = run_root / args["experiment"]
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

    loader_factory = partial(
        get_data_loader,
        batch_size=args["batch_size"],
        n_workers=args["n_workers"],
    )

    train_transforms = (
        [RandomDihedralAugmentation()] if args["augment"] else None
    )

    splits = ("train", "devel", "test")
    datasets = {
        k: HDF5Dataset(
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

    expected_image_shape = (
        args["channels"],
        args["cutout_size"],
        args["cutout_size"],
    )
    stored_image_shape = datasets["train"].h5_image_shape[1:]
    if stored_image_shape != expected_image_shape:
        raise click.ClickException(
            "Stored HDF5 image shape does not match the training configuration: "
            f"{stored_image_shape} != {expected_image_shape}"
        )

    # Splits and forked workers share one normalized in-memory tensor store.
    preloaded_images = preload_h5_images(
        args["data_dir"],
        normalization_kwargs=normalization_kwargs,
        device=args["device"],
    )
    for dataset in datasets.values():
        dataset.attach_preloaded_h5_images(preloaded_images)
    args["hdf5_preloaded"] = True
    args["hdf5_preloaded_gib"] = preloaded_images.nbytes / 2**30
    args["normalization_stage"] = "hdf5_preload"

    # Give each split an independent, reproducible RNG stream.
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

    # Reusing a validated experiment name replaces only that experiment.
    if experiment_dir.is_symlink() or experiment_dir.is_file():
        experiment_dir.unlink()
    elif experiment_dir.is_dir():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=False)

    wandb.login()

    args = {**args, **model_stats(model)}

    with wandb.init(
        project=args["project"],
        name=args["experiment"],
        dir=str(experiment_dir),
        config={**args, "architecture": "CNN"},
    ) as run:
        model_path = experiment_dir / "model.pt"

        if args["train"]:
            logging.info("Creating trainer...")
            trainer = create_trainer(
                model,
                optimizer,
                criterion,
                loaders,
                args["device"],
                model_path=model_path,
                num_classes=args["n_classes"],
                wandb_run=run,
                use_scheduler=args["scheduler"],
                train_transforms=train_transforms,
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
                num_classes=args["n_classes"],
                wandb_run=run,
                use_scheduler=args["scheduler"],
                train_transforms=train_transforms,
                unfreeze_warmup_epochs=args["unfreeze_warmup_epochs"],
                unfreeze_blocks_per_epoch=args["unfreeze_blocks_per_epoch"],
                num_epochs=args["epochs"],
                run_dir=experiment_dir,
                max_grad_norm=args["max_grad_norm"],
            )

        trainer.run(loaders["train"], max_epochs=args["epochs"])
        best_model_path = trainer.best_model_path
        if best_model_path != model_path or not model_path.is_file():
            raise RuntimeError("Training completed without producing a best model.")

        logging.info(f"Uploading best model from {best_model_path}")
        run.log_artifact(best_model_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train()
