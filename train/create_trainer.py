import json
import logging
import math
import re
from pathlib import Path

import torch
import wandb
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.metrics import Accuracy, ConfusionMatrix, Fbeta, Loss, Precision, Recall
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import load_model_state

WANDB_METRIC_NAMES = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "loss": "loss",
    "f1": "f1_macro",
}


def _configure_wandb_metrics(wandb_run):
    """Use epoch only for metrics that genuinely form an epoch series."""
    wandb_run.define_metric("epoch")
    for namespace in ("train", "devel", "optimizer", "transfer"):
        wandb_run.define_metric(
            f"{namespace}/*",
            step_metric="epoch",
            summary="none",
        )


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

        # Unfreeze from the output side toward the input.
        self._frozen_blocks = sorted(
            blocks, key=lambda item: int(item[0].removeprefix("layer")), reverse=True
        )
        self._eval_hooks = {}
        for name, module in self._frozen_blocks:
            for parameter in module.parameters():
                parameter.requires_grad = False
            # Keep frozen BatchNorm buffers fixed when model.train() is called.
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
    num_classes,
    wandb_run,
    use_scheduler=True,
    train_transforms=None,
    num_epochs=32,
    run_dir=None,
    max_grad_norm=1.0,
):
    """Set up Ignite trainer with train-only augmentation and deterministic evaluation."""

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_wandb_metrics(wandb_run)

    amp_mode = None
    if torch.device(device).type == "cuda":
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        amp_mode = "amp"
        logging.info("Using CUDA BF16 automatic mixed precision.")

    def custom_prepare_batch(batch, device, non_blocking, transforms):
        x, y = batch

        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

        if transforms is not None:
            if hasattr(transforms, "__len__"):
                for transform in transforms:
                    x = transform(x)
            else:
                x = transforms(x)

        return x, y

    def prepare_train_batch(batch, device, non_blocking):
        return custom_prepare_batch(batch, device, non_blocking, train_transforms)

    def prepare_eval_batch(batch, device, non_blocking):
        return custom_prepare_batch(batch, device, non_blocking, None)

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
        trainer.gradient_clip_handle = gradient_clip_handle

    pbar = ProgressBar(persist=False)

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
        "cm": ConfusionMatrix(
            num_classes=num_classes,
            output_transform=lambda x: x,
        ),
        "f1": Fbeta(beta=1, average=True),
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

    class_names = [str(index) for index in range(num_classes)]

    def confusion_matrix_plot(confusion_matrix):
        """Build a W&B confusion-matrix chart from aggregated counts."""
        y_true, y_pred = [], []
        for true_class, row in enumerate(confusion_matrix):
            for predicted_class, count in enumerate(row):
                y_true.extend([true_class] * int(count))
                y_pred.extend([predicted_class] * int(count))
        return wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=class_names,
        )

    def evaluate_metrics(loader, split):
        logging.info("Evaluating %s metrics.", split)

        evaluator.state.metrics = {}

        evaluator.run(loader)
        metrics = evaluator.state.metrics
        cm = metrics["cm"].cpu().numpy()

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

    if use_scheduler:
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    epoch_train_loss = {
        "weighted_sum": 0.0,
        "samples": 0,
        "mean": None,
    }

    @trainer.on(Events.EPOCH_STARTED)
    def reset_epoch_train_loss(_trainer):
        epoch_train_loss["weighted_sum"] = 0.0
        epoch_train_loss["samples"] = 0
        epoch_train_loss["mean"] = None

    @trainer.on(Events.ITERATION_COMPLETED)
    def accumulate_epoch_train_loss(trainer):
        batch_loss = _to_float(trainer.state.output)
        if batch_loss is None or not math.isfinite(batch_loss):
            raise RuntimeError(
                f"Training produced a non-finite loss at epoch "
                f"{trainer.state.epoch}, iteration {trainer.state.iteration}."
            )
        targets = trainer.state.batch[1]
        batch_size = int(targets.shape[0])
        epoch_train_loss["weighted_sum"] += batch_loss * batch_size
        epoch_train_loss["samples"] += batch_size

    @trainer.on(Events.EPOCH_COMPLETED)
    def finalize_epoch_train_loss(trainer):
        if not epoch_train_loss["samples"]:
            raise RuntimeError(
                f"Epoch {trainer.state.epoch} completed without training samples."
            )
        epoch_train_loss["mean"] = (
            epoch_train_loss["weighted_sum"] / epoch_train_loss["samples"]
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_devel_and_checkpoint(trainer):
        devel_metrics, devel_confusion_matrix = evaluate_metrics(
            loaders["devel"],
            "devel",
        )
        epoch_log = {
            "epoch": trainer.state.epoch,
            "train/loss": epoch_train_loss["mean"],
            "optimizer/lr": get_current_lr(optimizer),
            "devel/confusion_matrix": confusion_matrix_plot(
                devel_confusion_matrix
            ),
        }
        epoch_log.update(
            {
                f"devel/{WANDB_METRIC_NAMES[key]}": value
                for key, value in devel_metrics.items()
            }
        )
        wandb_run.log(epoch_log)

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
                "train_loss": epoch_train_loss["mean"],
                **{
                    f"devel_{key}": value
                    for key, value in devel_metrics.items()
                },
            }
            best_state["confusion_matrices"] = {
                "devel": devel_confusion_matrix,
            }
            logging.info(
                "Saved epoch %d as the best model (devel macro-F1 %.6f): %s",
                trainer.state.epoch,
                devel_macro_f1,
                model_path,
            )

    @trainer.on(Events.COMPLETED)
    def evaluate_best_model_and_save_metrics(trainer):
        if not best_state["metrics"] or trainer.best_model_path != model_path:
            raise RuntimeError(
                "Training completed without a devel macro-F1 result; "
                "no best model was saved."
            )

        logging.info(
            "Reloading the epoch %d best model for final test evaluation: %s",
            best_state["epoch"],
            model_path,
        )
        load_model_state(model, model_path, device=device)

        # Evaluate test once after restoring the devel-selected model.
        test_metrics, test_confusion_matrix = evaluate_metrics(
            loaders["test"],
            "test",
        )
        best_state["metrics"].update(
            {
                f"test_{key}": value
                for key, value in test_metrics.items()
            }
        )
        best_state["confusion_matrices"]["test"] = test_confusion_matrix

        payload = {
            "best_epoch": best_state["epoch"],
            "best_devel_macro_f1": best_state["score"],
            "best_devel_accuracy": best_state["metrics"]["devel_accuracy"],
            "metrics": best_state["metrics"],
            "confusion_matrices": best_state["confusion_matrices"],
        }
        # Keep final scalars in the summary rather than epoch history.
        wandb_run.log(
            {
                "best/devel/confusion_matrix": confusion_matrix_plot(
                    best_state["confusion_matrices"]["devel"]
                ),
                "best/test/confusion_matrix": confusion_matrix_plot(
                    best_state["confusion_matrices"]["test"]
                ),
            }
        )
        wandb_run.summary["best/epoch"] = best_state["epoch"]
        wandb_run.summary["best/train/loss"] = best_state["metrics"][
            "train_loss"
        ]
        for split in ("devel", "test"):
            for metric_name, wandb_name in WANDB_METRIC_NAMES.items():
                wandb_run.summary[f"best/{split}/{wandb_name}"] = (
                    best_state["metrics"][f"{split}_{metric_name}"]
                )

        if run_dir is not None:
            target_dir = Path(run_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            best_path = target_dir / "best_metrics.json"
            best_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            logging.info(f"Saved best metrics to {best_path}")

    return trainer


def create_transfer_learner(
    model,
    optimizer,
    criterion,
    loaders,
    device,
    *,
    model_path,
    num_classes,
    wandb_run,
    use_scheduler=True,
    train_transforms=None,
    num_epochs=32,
    run_dir=None,
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

    trainer = create_trainer(
        model,
        optimizer,
        criterion,
        loaders,
        device,
        model_path=model_path,
        num_classes=num_classes,
        wandb_run=wandb_run,
        use_scheduler=use_scheduler,
        train_transforms=train_transforms,
        num_epochs=num_epochs,
        run_dir=run_dir,
        max_grad_norm=max_grad_norm,
    )

    if unfreeze_warmup_epochs == 0:
        for block_name in unfreezer.unfreeze_next(unfreeze_blocks_per_epoch):
            logging.info("Unfroze backbone block %s before epoch 1.", block_name)

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

        wandb_run.log(
            {
                "epoch": epoch,
                "transfer/frozen_blocks": len(unfreezer.frozen_block_names),
                "transfer/newly_unfrozen_blocks": ",".join(unfrozen),
            }
        )

        if not unfreezer.frozen_block_names and not reported_all_trainable:
            logging.info("Epoch[%d]: all backbone blocks are trainable.", epoch)
            reported_all_trainable = True

    return trainer
