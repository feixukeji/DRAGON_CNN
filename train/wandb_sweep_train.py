# -*- coding: utf-8 -*-
import click
import logging
from functools import partial

import wandb
import os
import subprocess

import torch
import torch.nn as nn

import kornia.augmentation as K
import torch.multiprocessing as mp

from data_preprocessing import (
    FITSDataset,
    get_data_loader,
    load_or_compute_asinh_stats,
)
from cnn import model_factory, model_stats, save_trained_model
from train import create_trainer, create_transfer_learner
from utils import (
    DEFAULT_ASINH_SOFTENING,
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    build_optimizer,
    discover_devices,
    load_model_state,
    specify_dropout_rate,
    validate_transfer_learning_options,
)

# Global Sweep Configuration. This also effects early stopping
# for bad runs!
sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "devel_accuracy"},
    "parameters": {
        "learning_rate": {"values": [0.001, 0.0001, 0.0005]},
        "momentum": {"values": [1e-4, 1e-5, 1e-6]},
        "nesterov": {"values": [True, False]},
        "weight_decay": {"values": [1e-1, 1e-2, 1e-3]},
        "epochs": {"values": [10, 15, 20]},
        "batch_size": {"values": [16, 32, 64]},
        "dropout_rate": {"values": [0, 0.2, 0.3, 0.4, 0.5]},
        "scheduler": {"values": [True, False]}
    },
    "early_terminate": {
        "type": "hyperband",
        "eta": 2,
        "min_iter": 3,
        "strict": True  # Corrected
    }
}


def initialize_and_run_agent(device, p_args):
    if device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    reset_wandb_env()  # Reset W&B environment
    wandb.agent(**p_args)


def free_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_model_and_optimizer(model, optimizer):
    del model
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_wandb_env():
    logging.info("Resetting W&B environment to ensure separation.")
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

@click.command()
@click.option("--experiment_name", type=str, default="demo")
@click.option("--entity", type=str, default="dragon_merger_agn")
@click.option("--n_sweeps", type=int, default=12)
@click.option(
    "--run_id",
    type=str,
    default=None,
    help="""The run id. Practically this only needs to be used
if you are resuming a previously run experiment""",
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
            "dragon",
            "resnet"
        ],
        case_sensitive=False,
    ),
    default="dragon",
)
@click.option("--model_state", type=click.Path(exists=True), default=None)
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option(
    "--split_slug",
    type=str,
    required=True,
    help="""This specifies how the data_preprocessing is split into train/
devel/test sets. Balanced/Unbalanced refer to whether selecting
equal number of images from each class. xs, sm, lg, dev all refer
to what fraction is picked for train/devel/test.""",
)
@click.option("--cutout_size", type=int, default=167)
@click.option("--channels", type=int, default=1)
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
    "--n_workers",
    type=int,
    default=4,
    help="""The number of workers to be used during the
data_preprocessing loading process.""",
)
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
        "Euclid-style JSON with fixed per-channel vmin/vmax. A missing file "
        "is computed from the training split and saved. Transfer learning "
        "requires this option."
    ),
)
@click.option("--normalization-sample-per-image", type=int, default=1000, show_default=True)
@click.option("--normalization-max-samples", type=int, default=2000000, show_default=True)
@click.option("--normalization-seed", type=int, default=42, show_default=True)
@click.option("--n_classes", type=int, default=6)
@click.option(
    "--loss",
    type=click.Choice(
        [
            "nll",
            "ce"
        ],
        case_sensitive=False,
    ),
    default="ce",
    help="""The loss function to use""",
)
@click.option(
    "--crop/--no-crop",
    default=True,
    help="""If True, all images are passed through a cropping
operation before being fed into the network. Images are cropped
to the cutout_size parameter""",
)
@click.option(
    "--force_reload/--no_force_reload",
    default=False,
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
)
@click.option(
    "--unfreeze-blocks-per-epoch",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
)
def sweep_init(**kwargs):
    # Copy and log args
    args = {k: v for k, v in kwargs.items()}

    try:
        validate_transfer_learning_options(
            is_training=args["train"],
            model_state=args["model_state"],
            normalize=args["normalize"],
            normalization_stats=args["normalization_stats"],
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

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
    args["normalization_kwargs"] = normalization_kwargs

    # Discover devices
    args["device"] = discover_devices()

    # Create the model given model_type
    cls = model_factory(args["model_type"])

    # Select the desired transforms
    T = None
    if args["crop"]:
        T = K.CenterCrop(args["cutout_size"])

    # Generate the DataLoaders and log the train/devel/test split sizes
    splits = ("train", "devel", "test")
    datasets = {
        k: FITSDataset(
            data_dir=args["data_dir"],
            slug=args["split_slug"],
            cutout_size=args["cutout_size"],
            channels=args["channels"],
            normalize=args["normalize"],
            transforms=T,
            split=k,
            force_reload=args["force_reload"],
            num_classes=args["n_classes"]
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

    # Select the desired transforms
    T = None
    if args["crop"]:
        T = K.CenterCrop(args["cutout_size"])

    # Define the criterion
    loss_dict = {
        "nll": nn.NLLLoss(),
        "ce": nn.CrossEntropyLoss()
    }
    criterion = loss_dict[args["loss"]]

    # Log into W&B
    reset_wandb_env()  # Initial reset of W&B environment.
    wandb.login()
    wandb.require("service")

    # Initializing the Sweep
    trainer_func = partial(train, model_cls=cls, datasets=datasets, criterion=criterion, args=args)
    sweep_id = wandb.sweep(sweep=sweep_config, project=args["experiment_name"])
    logging.info(f"The W&B sweep ID for this run is {sweep_id}.")

    # Multiplexing capability.
    p_args = {
        "sweep_id": sweep_id,
        "function": trainer_func,
        "project": args["experiment_name"],
        "entity": args["entity"],
        "count": (args["n_sweeps"] / args["n_workers"])
    }
    processes = []
    if args["device"] == "cpu" and args["parallel"]:  # Multiplex given N cpus
        num_agents = min(mp.cpu_count(), args["n_workers"])
        logging.info(f"Parallelizing sweeps over {num_agents} CPUs.")
        for _ in range(num_agents):
            p = mp.Process(target=wandb.agent, kwargs=p_args)
            p.start()  # Start the new child process
            processes.append(p)

        for p in processes:
            p.join()  # Thread join to wait for each to finish execution.

    elif args["device"] == "cuda" and args["parallel"]:  # Multiplexing using GPUs.
        num_agents = 1 # torch.cuda.device_count()
        logging.info(f"Parallelizing sweeps over {num_agents} agents.")

        for i in range(num_agents):
            p = mp.Process(target=initialize_and_run_agent, args=(0, p_args))
            p.start()  # Start the new child process
            processes.append(p)

        for p in processes:
            p.join()  # Thread join to wait for each to finish execution.

    # Housekeeping
    sweep_path = f'{args["entity"]}/{args["experiment_name"]}/{sweep_id}'
    sweep_list = ['wandb', 'sweep', '--cancel', sweep_path]
    try:
        result = subprocess.run(" ".join(sweep_list), shell=True)
        logging.info(f"All runs on sweep ID {sweep_id} have terminated and sweep is now canceled.")
        logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"ERROR: Failed to cancel sweep {sweep_id}: {e}")

    return


def train(model_cls, datasets, criterion, args):
    # Initializing W&B run
    with wandb.init(
        id=args["run_id"],
        resume="allow",
        group="DDP",
        entity=args["entity"],
        config={
            "num_classes": args["n_classes"],
            "architecture": "CNN"
        },
        reinit=True
    ) as run:
        # Overriding run name if it is specified.
        name_str = "_".join(
            [f"{key}_{wandb.config[key]}" for key in wandb.config.keys()[2:]]
        )
        if args["run_name"] is not None:
            run.name = args["run_name"] + "_" + name_str
        else:
            run.name = name_str

        model_args = {
            "cutout_size": args["cutout_size"],
            "channels": args["channels"],
            "num_classes": args["n_classes"]
        }

        logging.info("Reinitializing model.")
        model = model_cls(**model_args)
        model = nn.DataParallel(model) if args["parallel"] else model
        model = model.to(args["device"])

        # Chnaging the default dropout rate if specified
        specify_dropout_rate(model, wandb.config.dropout_rate)

        if args["model_state"]:
            logging.info(f'Loading model from {args["model_state"]}...')
            load_model_state(model, args["model_state"], device=args["device"])

        optimizer = build_optimizer(
            model,
            optimizer_name=args["optimizer"],
            lr=wandb.config.learning_rate,
            weight_decay=wandb.config.weight_decay,
            momentum=wandb.config.momentum,
            nesterov=wandb.config.nesterov,
            adamw_beta1=args["adamw_beta1"],
            adamw_beta2=args["adamw_beta2"],
            adamw_eps=args["adamw_eps"],
        )

        # Create a DataLoader factory based on command-line args
        loader_factory = partial(
            get_data_loader,
            batch_size=wandb.config.batch_size,
            n_workers=args["n_workers"],
        )

        loaders = {k: loader_factory(v, shuffle=(k == 'train')) for k, v in datasets.items()}
        args["splits"] = {k: len(v.dataset) for k, v in loaders.items()}

        # Write the parameters and model stats to W&B
        args = {**args, **model_stats(model)}
        wandb.log(args)
        wandb.watch(model, log_freq=1)

        # Set up trainer
        if args["train"]:
            logging.info("Creating trainer...")
            trainer = create_trainer(
                model,
                optimizer,
                criterion,
                loaders,
                args["device"],
                wandb.config.scheduler,
                gpu_transforms=datasets["train"].transform,
                eval_gpu_transforms=datasets["devel"].transform,
                normalize=args["normalize"],
                normalization_kwargs=args["normalization_kwargs"],
            )
        else:
            logging.info("Creating trainer and freezing layers for transfer learning...")
            trainer = create_transfer_learner(
                model,
                optimizer,
                criterion,
                loaders,
                args["device"],
                wandb.config.scheduler,
                gpu_transforms=datasets["train"].transform,
                eval_gpu_transforms=datasets["devel"].transform,
                normalize=args["normalize"],
                normalization_kwargs=args["normalization_kwargs"],
                unfreeze_warmup_epochs=args["unfreeze_warmup_epochs"],
                unfreeze_blocks_per_epoch=args["unfreeze_blocks_per_epoch"],
            )

        # Run trainer and save model state
        trainer.run(loaders["train"], max_epochs=wandb.config.epochs)
        slug = (
            f"{args['experiment_name']}-{args['split_slug']}-"
            f"{run.id}"
        )

        model_path = save_trained_model(model, slug)

        # Log model as an artifact
        logging.info(f"Saved model to {model_path}")
        run.log_artifact(model_path)

        # Resetting model.
        reset_model_and_optimizer(model, optimizer)

        # Finish the W&B run!
        wandb.finish()
        free_gpu_memory()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Setting multiprocess spawn method.
    if torch.cuda.is_available():
        mp.set_start_method('spawn')
        logging.info("Setting multiprocessing start method to 'spawn'.")

    sweep_init()
