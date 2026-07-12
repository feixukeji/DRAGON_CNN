"""Optimizer construction shared by DRAGON training entrypoints."""

import torch


def build_optimizer(
    model,
    optimizer_name,
    lr,
    weight_decay=0.0,
    momentum=0.9,
    nesterov=False,
    adamw_beta1=0.9,
    adamw_beta2=0.999,
    adamw_eps=1e-8,
):
    """Build SGD or AdamW with consistent validation and parameter handling."""
    name = optimizer_name.lower()
    if lr <= 0:
        raise ValueError("lr must be greater than zero.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )

    if name == "adamw":
        if not 0.0 <= adamw_beta1 < 1.0 or not 0.0 <= adamw_beta2 < 1.0:
            raise ValueError("AdamW beta values must be in [0, 1).")
        if adamw_eps <= 0:
            raise ValueError("adamw_eps must be greater than zero.")

        # Biases and normalization scales/shifts are conventionally excluded
        # from decoupled weight decay.
        decay = []
        no_decay = []
        for parameter_name, parameter in model.named_parameters():
            if parameter.ndim <= 1 or parameter_name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)

        parameter_groups = []
        if decay:
            parameter_groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            parameter_groups.append({"params": no_decay, "weight_decay": 0.0})

        return torch.optim.AdamW(
            parameter_groups,
            lr=lr,
            betas=(adamw_beta1, adamw_beta2),
            eps=adamw_eps,
            weight_decay=0.0,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
