from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cnn import (
    DRAGON,
    DRAGON_CLASSIFIER_BIAS_KEY,
    DRAGON_FIRST_CONV_WEIGHT_KEY,
    DRAGON_HEAD_PREFIXES,
)
from utils.model_utils import load_plain_state_dict

DEFAULT_SEED = 42


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


def _checkpoint_classes(state_dict: dict[str, torch.Tensor]) -> int:
    """Read the checkpoint's own class count from the classifier bias."""
    if DRAGON_CLASSIFIER_BIAS_KEY not in state_dict:
        raise KeyError(f"DRAGON checkpoint must contain {DRAGON_CLASSIFIER_BIAS_KEY!r}")
    return int(state_dict[DRAGON_CLASSIFIER_BIAS_KEY].shape[0])


def _adapt_head(
    state_dict: dict[str, torch.Tensor],
    target_channels: int,
    target_classes: int | None,
    seed: int,
) -> list[str]:
    """Rebuild the head when the checkpoint's no longer fits the architecture.

    The backbone transfers whenever its shapes still match, but the head does
    not survive a change in class count, in the statistics the taps emit, or in
    how the head is built -- and only the first of those shows up in the output
    layer. Patching that layer in place, which is what this script used to do,
    silently keeps a stale feature width and fails much later at load. So the
    unit of replacement is every parameter under ``DRAGON_HEAD_PREFIXES``, and
    it is replaced only when at least one of them actually mismatches, so an
    unchanged head keeps its trained weights.

    The replacements come from ``DRAGON._initialize_weights`` rather than from a
    rule restated here, and ``seed`` is set first, so the converted head is
    reproducible from the same source checkpoint.
    """
    if target_classes is not None and target_classes <= 0:
        raise ValueError("target_classes must be positive")
    classes = target_classes or _checkpoint_classes(state_dict)

    torch.manual_seed(seed)
    reference = DRAGON(channels=target_channels, num_classes=classes).state_dict()
    head_keys = [key for key in reference if key.startswith(DRAGON_HEAD_PREFIXES)]

    mismatched = [
        key
        for key in head_keys
        if key not in state_dict
        or tuple(state_dict[key].shape) != tuple(reference[key].shape)
    ]
    if not mismatched:
        return []

    for key in [key for key in state_dict if key.startswith(DRAGON_HEAD_PREFIXES)]:
        del state_dict[key]
    for key in head_keys:
        value = reference[key]
        state_dict[key] = value.clone()
    return head_keys


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
        help=(
            "Target output class count; defaults to the checkpoint's own. The "
            "head is rebuilt whenever it no longer fits, class count or not."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for any head parameters this conversion has to initialise",
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
    updated_heads = _adapt_head(
        state_dict, args.target_channels, args.target_classes, args.seed
    )
    torch.save(state_dict, args.out_model)

    print(
        f"Saved {args.out_model} (conv key {conv_key}: "
        f"{tuple(weight.shape)} -> {tuple(new_weight.shape)})"
    )
    if updated_heads:
        print(f"Reinitialised head ({len(updated_heads)} tensors, seed {args.seed})")
    else:
        print("Head kept: every head tensor already matches the architecture")


if __name__ == "__main__":
    main()
