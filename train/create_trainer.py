import json
from pathlib import Path

import wandb

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Loss, Accuracy, Precision, ConfusionMatrix, Recall, Fbeta
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
import logging

from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import arsinh_normalize


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def create_trainer(model, optimizer, criterion, loaders, device, use_scheduler=True, gpu_transforms=None,
                   normalize=False, checkpoint_dir=None, run_id=None, num_epochs=32, run_dir=None):
    """Set up Ignite trainer and evaluator with GPU transforms."""


    # 1. Define the custom batch preparation function
    def custom_prepare_batch(batch, device, non_blocking):
        x, y = batch

        # Move the raw batch to the GPU first
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

        # Apply transformations on the GPU to the whole batch (B, C, H, W)
        if gpu_transforms is not None:
            if hasattr(gpu_transforms, "__len__"):
                for transform in gpu_transforms:
                    x = transform(x)
            else:
                x = gpu_transforms(x)

        # Apply normalization on the GPU
        if normalize:
            x = arsinh_normalize(x)  # Ensure this function supports batched tensors!

        return x, y

    # Define a score function.
    # If using Accuracy (higher is better):
    def score_function(engine):
        return engine.state.metrics['accuracy']

    # 2. Pass the custom function to the trainer
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device,
        prepare_batch=custom_prepare_batch,
        amp_mode="amp"
    )

    pbar = ProgressBar(persist=False)

    # Attach it to the trainer.
    pbar.attach(trainer, output_transform=lambda x: {'batch_loss': x})

    if use_scheduler:
        num_training_steps = len(loaders['train']) * num_epochs

        torch_lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
        scheduler = LRScheduler(torch_lr_scheduler)

    metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average="weighted"),
        "recall": Recall(average="weighted"),
        "loss": Loss(criterion),
        "cm": ConfusionMatrix(num_classes=wandb.config["num_classes"], output_transform=lambda x: x),
        "f1": Fbeta(beta=1)
    }

    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device,
        prepare_batch=custom_prepare_batch,
        amp_mode="amp"
    )

    # Define the checkpoint handler
    checkpoint_handler = ModelCheckpoint(
        dirname=checkpoint_dir,
        filename_prefix=f'best_{run_id}',
        n_saved=1,
        require_empty=False,
        score_function=score_function,
        score_name="accuracy",
        global_step_transform=lambda engine, event: engine.state.epoch
    )

    pbar_eval = ProgressBar(persist=False, desc="Evaluating")
    pbar_eval.attach(evaluator)

    best_state = {"epoch": None, "score": None, "metrics": {}}

    # Function to log metrics to wandb
    def log_metrics(trainer, loader, log_prefix=""):
        logging.info(f"Logging metrics for {log_prefix}")

        # Reset evaluator state before running evaluation
        evaluator.state.metrics = {}

        evaluator.run(loader)
        metrics = evaluator.state.metrics
        log_dict = {f"{log_prefix}{k}": v for k, v in metrics.items() if k != "cm"}

        # Handle the confusion matrix separately
        cm = metrics["cm"].cpu().numpy()
        class_names = [str(i) for i in range(wandb.config["num_classes"])]

        # Calculate true and predicted labels from the confusion matrix
        y_true, y_pred = [], []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                y_true.extend([i] * int(cm[i, j]))
                y_pred.extend([j] * int(cm[i, j]))

        cm_plot = wandb.plot.confusion_matrix(probs=None,
                                              y_true=y_true,
                                              preds=y_pred,
                                              class_names=class_names)

        # Log other metrics and the confusion matrix plot
        log_dict[f"{log_prefix}confusion_matrix"] = cm_plot
        wandb.log(log_dict, step=trainer.state.epoch)

        # Only save if this is the validation set
        if log_prefix == "devel_" and checkpoint_handler is not None:
            checkpoint_handler(evaluator, to_save={'model': model})

        scalar_metrics = {}
        for key, value in metrics.items():
            if key == "cm":
                continue
            scalar = _to_float(value)
            if scalar is not None:
                scalar_metrics[key] = scalar
        return scalar_metrics

    def get_current_lr(optimizer):
        return optimizer.param_groups[0]['lr']

    # Define training hooks
    if use_scheduler:
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    @trainer.on(Events.STARTED)
    def log_results_start(trainer):
        logging.info("Log results started.")
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_devel_results(trainer):
        epoch_metrics = {}
        for L, loader in loaders.items():
            epoch_metrics[L] = log_metrics(trainer, loader, log_prefix=f"{L}_")
        wandb.log({"lr": get_current_lr(optimizer)}, step=trainer.state.epoch)

        devel_metrics = epoch_metrics.get("devel", {})
        devel_accuracy = devel_metrics.get("accuracy")
        if devel_accuracy is not None:
            if best_state["score"] is None or devel_accuracy > best_state["score"]:
                best_state["score"] = devel_accuracy
                best_state["epoch"] = trainer.state.epoch
                best_state["metrics"] = {
                    f"{split}_{key}": value
                    for split, metrics in epoch_metrics.items()
                    for key, value in metrics.items()
                }

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

        if best_state["metrics"]:
            payload = {
                "best_epoch": best_state["epoch"],
                "best_devel_accuracy": best_state["score"],
                "metrics": best_state["metrics"],
            }
            if wandb.run is not None:
                wandb.run.summary["best_epoch"] = best_state["epoch"]
                wandb.run.summary["best_devel_accuracy"] = best_state["score"]
                for key, value in best_state["metrics"].items():
                    wandb.run.summary[f"best_{key}"] = value

            target_dir = None
            if run_dir is not None:
                target_dir = Path(run_dir)
            elif checkpoint_dir is not None:
                target_dir = Path(checkpoint_dir)

            if target_dir is not None:
                target_dir.mkdir(parents=True, exist_ok=True)
                best_path = target_dir / "best_metrics.json"
                best_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
                logging.info(f"Saved best metrics to {best_path}")

        logging.info("Terminating run explicitly.")
        trainer.terminate()

    return trainer


def create_transfer_learner(
    model,
    optimizer,
    criterion,
    loaders,
    device,
    use_scheduler=True,
    gpu_transforms=None,
    normalize=False,
    checkpoint_dir=None,
    run_id=None,
    num_epochs=32,
    run_dir=None,
):
    """Method to create a transfer learner trainer."""

    # Initialize a stack that contains all frozen layers.
    frozen_layer_stack = []

    # Initial freezing of the layers.
    logging.info("Freezing non-FC layers for given model...")
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            frozen_layer_stack.append((name, param))

    # Create trainer
    trainer = create_trainer(
        model,
        optimizer,
        criterion,
        loaders,
        device,
        use_scheduler,
        gpu_transforms,
        normalize,
        checkpoint_dir,
        run_id,
        num_epochs,
        run_dir,
    )

    # Gradual unfreezing of layers based on epoch.
    @trainer.on(Events.EPOCH_COMPLETED)
    def unfreeze_layers(engine):
        epoch = engine.state.epoch

        # We unfreeze one entire layer at a time (O(1) complexity).
        wandb.log({"frozen_layers": len(frozen_layer_stack)})
        if frozen_layer_stack:
            top_name, top_param = frozen_layer_stack[-1]
            layer_name = top_name.split('.')[1]
            while frozen_layer_stack and frozen_layer_stack[-1][0].split('.')[1] == layer_name:
                name, param = frozen_layer_stack.pop()
                param.requires_grad = True
                logging.info(f"Epoch[{epoch}]: layer {name} is now trainable.")
        else:
            # All layers unfrozen already!
            logging.info(f"Epoch[{epoch}]: all layers trainable.")

    return trainer
