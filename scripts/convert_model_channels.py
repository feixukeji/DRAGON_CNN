from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from cnn import (
    DRAGON_CLASSIFIER_BIAS_KEY,
    DRAGON_CLASSIFIER_WEIGHT_KEY,
    DRAGON_FIRST_CONV_WEIGHT_KEY,
)
from utils.model_utils import load_plain_state_dict


def _find_conv_key(state_dict: dict[str, torch.Tensor]) -> str:
    key = DRAGON_FIRST_CONV_WEIGHT_KEY
    if key not in state_dict:
        raise KeyError(f"DRAGON checkpoint is missing {key!r}")
    return key


def _adapt_conv_weight(weight: torch.Tensor, target_channels: int) -> torch.Tensor:
    if target_channels <= 0:
        raise ValueError("target_channels must be positive")
    if weight.ndim != 4:
        raise ValueError("Expected a 4-D conv weight tensor")

    _, in_channels, _, _ = weight.shape
    if in_channels == target_channels:
        return weight

    if target_channels > in_channels:
        if in_channels == 1:
            expanded = weight.repeat(1, target_channels, 1, 1)
            return expanded / float(target_channels)
        mean = weight.mean(dim=1, keepdim=True)
        pad = mean.repeat(1, target_channels - in_channels, 1, 1)
        expanded = torch.cat([weight, pad], dim=1)
        return expanded * (float(in_channels) / float(target_channels))

    return weight[:, :target_channels, :, :]


def _find_classifier_pairs(
    state_dict: dict[str, torch.Tensor],
) -> list[tuple[str, str]]:
    weight_key = DRAGON_CLASSIFIER_WEIGHT_KEY
    bias_key = DRAGON_CLASSIFIER_BIAS_KEY
    if weight_key not in state_dict or bias_key not in state_dict:
        raise KeyError(
            f"DRAGON checkpoint must contain {weight_key!r} and {bias_key!r}"
        )
    return [(weight_key, bias_key)]


def _adapt_classifier(
    state_dict: dict[str, torch.Tensor],
    target_classes: int | None,
) -> list[str]:
    if target_classes is None:
        return []
    if target_classes <= 0:
        raise ValueError("target_classes must be positive")

    updated: list[str] = []
    for weight_key, bias_key in _find_classifier_pairs(state_dict):
        weight = state_dict[weight_key]
        bias = state_dict[bias_key]

        if weight.ndim != 2 or bias.ndim != 1:
            continue
        if weight.shape[0] == target_classes and bias.shape[0] == target_classes:
            continue

        new_weight = torch.empty(
            (target_classes, weight.shape[1]), dtype=weight.dtype
        )
        torch.nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
        new_bias = torch.zeros((target_classes,), dtype=bias.dtype)

        state_dict[weight_key] = new_weight
        state_dict[bias_key] = new_bias
        updated.append(weight_key)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DRAGON_CNN model weights to a different input channel count."
    )
    parser.add_argument("--in-model", type=Path, required=True, help="Input .pt file")
    parser.add_argument("--out-model", type=Path, required=True, help="Output .pt file")
    parser.add_argument(
        "--target-channels",
        type=int,
        required=True,
        help="Target input channel count",
    )
    parser.add_argument(
        "--target-classes",
        type=int,
        default=None,
        help="Optional target output class count to reset classifier",
    )
    args = parser.parse_args()

    if args.target_channels <= 0:
        parser.error("--target-channels must be positive")
    if args.target_classes is not None and args.target_classes <= 0:
        parser.error("--target-classes must be positive")

    state_dict = load_plain_state_dict(args.in_model)

    conv_key = _find_conv_key(state_dict)
    weight = state_dict[conv_key]
    new_weight = _adapt_conv_weight(weight, args.target_channels)

    state_dict[conv_key] = new_weight
    updated_heads = _adapt_classifier(state_dict, args.target_classes)
    torch.save(state_dict, args.out_model)

    print(
        f"Saved {args.out_model} (conv key {conv_key}: "
        f"{tuple(weight.shape)} -> {tuple(new_weight.shape)})"
    )
    if updated_heads:
        print(f"Reset classifier weights: {', '.join(updated_heads)}")


if __name__ == "__main__":
    main()
