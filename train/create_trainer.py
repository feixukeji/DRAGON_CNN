import json
import math
from pathlib import Path
import re

import torch
import wandb

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Loss, Accuracy, Precision, ConfusionMatrix, Recall, Fbeta
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
import logging

from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import asinh_normalize


class GradualBackboneUnfreezer:
    """Freeze and unfreeze complete ``layerN`` backbone blocks."""

    @staticmethod
    def _force_eval(module, _inputs):
        module.eval()

    def __init__(self, model):
        self.model = model
        backbone_root = self.model
        blocks = [
            (name, module)
            for name, module in backbone_root.named_children()
            if re.fullmatch(r"layer\d+", name)
        ]
        if not blocks:
            raise ValueError("Transfer learning requires backbone blocks named layerN.")

        # Unfreeze from the task-specific end of the backbone toward the input.
        self._frozen_blocks = sorted(
            blocks, key=lambda item: int(item[0].removeprefix("layer")), reverse=True
        )
        self._eval_hooks = {}
        for name, module in self._frozen_blocks:
            for parameter in module.parameters():
                parameter.requires_grad = False
            # model.train() would otherwise keep updating frozen BatchNorm buffers.
            self._eval_hooks[name] = module.register_forward_pre_hook(
                self._force_eval
            )

    @property
    def frozen_block_names(self):
        return [name for name, _ in self._frozen_blocks]

    def unfreeze_next(self, count=1):
        """Unfreeze up to ``count`` complete blocks and return their names."""
        unfrozen = []
        for _ in range(min(count, len(self._frozen_blocks))):
            name, module = self._frozen_blocks.pop(0)
            self._eval_hooks.pop(name).remove()
            for parameter in module.parameters():
                parameter.requires_grad = True
            module.train()
            unfrozen.append(name)
        return unfrozen


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def _register_global_grad_norm_clipping(optimizer, max_grad_norm):
    """Clip the total gradient L2 norm immediately before optimizer updates."""
    if max_grad_norm is None:
        return None

    max_grad_norm = float(max_grad_norm)
    if not math.isfinite(max_grad_norm) or max_grad_norm < 0:
        raise ValueError("max_grad_norm must be a finite non-negative value.")
    if max_grad_norm == 0:
        return None

    def clip_grad_norm_before_step(optimizer, _args, _kwargs):
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)

    return optimizer.register_step_pre_hook(clip_grad_norm_before_step)


def create_trainer(
    model,
    optimizer,
    criterion,
    loaders,
    device,
    *,
    model_path,
    normalization_kwargs,
    use_scheduler=True,
    gpu_transforms=None,
    num_epochs=32,
    run_dir=None,
    eval_gpu_transforms=None,
    max_grad_norm=1.0,
):
    """Set up Ignite trainer with train-only augmentation and deterministic evaluation."""

    if not normalization_kwargs or not {
        "vmin",
        "vmax",
        "softening",
    }.issubset(normalization_kwargs):
        raise ValueError(
            "Training requires fixed vmin/vmax/softening loaded from "
            "normalization_stats.json."
        )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    amp_mode = None
    if torch.device(device).type == "cuda":
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        amp_mode = "amp"
        logging.info("Using CUDA BF16 automatic mixed precision.")

    # 1. Define the custom batch preparation function
    def custom_prepare_batch(batch, device, non_blocking, transforms):
        x, y = batch

        # Move the raw batch to the GPU first
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

        # Apply transformations on the GPU to the whole batch (B, C, H, W)
        if transforms is not None:
            if hasattr(transforms, "__len__"):
                for transform in transforms:
                    x = transform(x)
            else:
                x = transforms(x)

        # Training and evaluation always use the same fixed global statistics.
        x = asinh_normalize(x, **normalization_kwargs)

        return x, y

    def prepare_train_batch(batch, device, non_blocking):
        return custom_prepare_batch(batch, device, non_blocking, gpu_transforms)

    def prepare_eval_batch(batch, device, non_blocking):
        return custom_prepare_batch(batch, device, non_blocking, eval_gpu_transforms)

    # 2. Pass the custom function to the trainer
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device,
        prepare_batch=prepare_train_batch,
        amp_mode=amp_mode,
    )

    gradient_clip_handle = _register_global_grad_norm_clipping(
        optimizer,
        max_grad_norm,
    )
    if gradient_clip_handle is not None:
        logging.info(
            "Clipping the global gradient L2 norm to max_norm=%g before each optimizer step.",
            max_grad_norm,
        )
        # Keep the removable hook handle with the trainer that owns it.
        trainer.gradient_clip_handle = gradient_clip_handle

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
        "f1": Fbeta(beta=1, average=True)
    }

    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device,
        prepare_batch=prepare_eval_batch,
        amp_mode=amp_mode,
    )

    pbar_eval = ProgressBar(persist=False, desc="Evaluating")
    pbar_eval.attach(evaluator)

    best_state = {
        "epoch": None,
        "score": None,
        "metrics": {},
        "confusion_matrices": {},
    }
    trainer.best_model_path = None

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

        scalar_metrics = {}
        for key, value in metrics.items():
            if key == "cm":
                continue
            scalar = _to_float(value)
            if scalar is not None:
                scalar_metrics[key] = scalar
        return scalar_metrics, cm.astype(int, copy=False).tolist()

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
        epoch_confusion_matrices = {}
        for L, loader in loaders.items():
            metrics, confusion_matrix = log_metrics(
                trainer,
                loader,
                log_prefix=f"{L}_",
            )
            epoch_metrics[L] = metrics
            epoch_confusion_matrices[L] = confusion_matrix
        wandb.log({"lr": get_current_lr(optimizer)}, step=trainer.state.epoch)

        devel_metrics = epoch_metrics.get("devel", {})
        devel_macro_f1 = devel_metrics.get("f1")
        if devel_macro_f1 is None or not math.isfinite(devel_macro_f1):
            logging.warning(
                "Epoch %d produced no finite devel macro-F1; not saving a model.",
                trainer.state.epoch,
            )
            return

        if best_state["score"] is None or devel_macro_f1 > best_state["score"]:
            torch.save(model.state_dict(), model_path)
            trainer.best_model_path = model_path
            best_state["score"] = devel_macro_f1
            best_state["epoch"] = trainer.state.epoch
            best_state["metrics"] = {
                f"{split}_{key}": value
                for split, metrics in epoch_metrics.items()
                for key, value in metrics.items()
            }
            best_state["confusion_matrices"] = epoch_confusion_matrices
            logging.info(
                "Saved epoch %d as the best model (devel macro-F1 %.6f): %s",
                trainer.state.epoch,
                devel_macro_f1,
                model_path,
            )

    @trainer.on(Events.COMPLETED)
    def save_best_metrics(_trainer):
        if best_state["metrics"]:
            payload = {
                "best_epoch": best_state["epoch"],
                "best_devel_macro_f1": best_state["score"],
                "best_devel_accuracy": best_state["metrics"]["devel_accuracy"],
                "metrics": best_state["metrics"],
                "confusion_matrices": best_state["confusion_matrices"],
            }
            if wandb.run is not None:
                wandb.run.summary["best_epoch"] = best_state["epoch"]
                wandb.run.summary["best_devel_macro_f1"] = best_state["score"]
                wandb.run.summary["best_devel_accuracy"] = best_state["metrics"][
                    "devel_accuracy"
                ]
                for key, value in best_state["metrics"].items():
                    wandb.run.summary[f"best_{key}"] = value

            if run_dir is not None:
                target_dir = Path(run_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
                best_path = target_dir / "best_metrics.json"
                best_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
                logging.info(f"Saved best metrics to {best_path}")
        else:
            raise RuntimeError(
                "Training completed without a devel macro-F1 result; "
                "no best model was saved."
            )

    return trainer


def create_transfer_learner(
    model,
    optimizer,
    criterion,
    loaders,
    device,
    *,
    model_path,
    normalization_kwargs,
    use_scheduler=True,
    gpu_transforms=None,
    num_epochs=32,
    run_dir=None,
    eval_gpu_transforms=None,
    unfreeze_warmup_epochs=3,
    unfreeze_blocks_per_epoch=1,
    max_grad_norm=1.0,
):
    """Create a transfer learner with deterministic block-wise unfreezing."""
    if unfreeze_warmup_epochs < 0:
        raise ValueError("unfreeze_warmup_epochs must be non-negative.")
    if unfreeze_blocks_per_epoch <= 0:
        raise ValueError("unfreeze_blocks_per_epoch must be greater than zero.")

    unfreezer = GradualBackboneUnfreezer(model)
    logging.info(
        "Frozen backbone blocks: %s. Training classifier head for %d epoch(s).",
        ", ".join(unfreezer.frozen_block_names),
        unfreeze_warmup_epochs,
    )

    # Create trainer
    trainer = create_trainer(
        model,
        optimizer,
        criterion,
        loaders,
        device,
        model_path=model_path,
        normalization_kwargs=normalization_kwargs,
        use_scheduler=use_scheduler,
        gpu_transforms=gpu_transforms,
        num_epochs=num_epochs,
        run_dir=run_dir,
        eval_gpu_transforms=eval_gpu_transforms,
        max_grad_norm=max_grad_norm,
    )

    if unfreeze_warmup_epochs == 0:
        for block_name in unfreezer.unfreeze_next(unfreeze_blocks_per_epoch):
            logging.info("Unfroze backbone block %s before epoch 1.", block_name)

    # Unfreeze complete blocks after the configured head-only warmup.
    reported_all_trainable = False

    @trainer.on(Events.EPOCH_COMPLETED)
    def unfreeze_backbone_blocks(engine):
        nonlocal reported_all_trainable
        epoch = engine.state.epoch
        unfrozen = []
        if epoch >= max(1, unfreeze_warmup_epochs):
            unfrozen = unfreezer.unfreeze_next(unfreeze_blocks_per_epoch)
            for block_name in unfrozen:
                logging.info(
                    "Epoch[%d]: backbone block %s is now trainable.",
                    epoch,
                    block_name,
                )

        if wandb.run is not None:
            wandb.log(
                {
                    "frozen_blocks": len(unfreezer.frozen_block_names),
                    "newly_unfrozen_blocks": ",".join(unfrozen),
                },
                step=epoch,
            )

        if not unfreezer.frozen_block_names and not reported_all_trainable:
            logging.info("Epoch[%d]: all backbone blocks are trainable.", epoch)
            reported_all_trainable = True

    return trainer
