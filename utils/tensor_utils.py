import json
import logging

import numpy as np
import torch

def tensor_to_numpy(x):
    """Convert a torch tensor to NumPy for plotting."""
    return np.clip(x.numpy().transpose((1, 2, 0)), 0, 1)


DEFAULT_LOW_PERCENTILE = 0.5
DEFAULT_HIGH_PERCENTILE = 99.5
DEFAULT_ASINH_SOFTENING = 0.1


def load_asinh_stats(path, channels=None):
    """Load Euclid-style per-channel percentile limits from JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        stats = json.load(handle)

    if "vmin" not in stats or "vmax" not in stats:
        raise ValueError("Normalization stats JSON must contain 'vmin' and 'vmax'.")

    vmin = [float(value) for value in stats["vmin"]]
    vmax = [float(value) for value in stats["vmax"]]
    if len(vmin) != len(vmax):
        raise ValueError("Normalization stats 'vmin' and 'vmax' must have equal lengths.")
    if channels is not None and len(vmin) != channels:
        raise ValueError(
            f"Normalization stats contain {len(vmin)} channels, expected {channels}."
        )
    if any(high <= low for low, high in zip(vmin, vmax)):
        raise ValueError("Every normalization vmax must be greater than vmin.")

    return {**stats, "vmin": vmin, "vmax": vmax}


def normalization_kwargs_from_stats(
    path,
    channels,
    low_pct=DEFAULT_LOW_PERCENTILE,
    high_pct=DEFAULT_HIGH_PERCENTILE,
    softening=DEFAULT_ASINH_SOFTENING,
):
    """Build validated ``asinh_normalize`` arguments from a statistics file."""
    if not 0.0 <= low_pct < high_pct <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= low_pct < high_pct <= 100.")
    if softening <= 0:
        raise ValueError("softening must be greater than zero.")
    stats = load_asinh_stats(path, channels=channels)
    return {
        "low_pct": low_pct,
        "high_pct": high_pct,
        "softening": softening,
        "vmin": stats["vmin"],
        "vmax": stats["vmax"],
    }


def _channel_limits(values, channels, name):
    limits = torch.as_tensor(values, dtype=torch.float32)
    if limits.ndim == 0:
        limits = limits.reshape(1)
    if limits.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional sequence.")
    if limits.numel() not in (1, channels):
        raise ValueError(
            f"{name} contains {limits.numel()} values, expected 1 or {channels}."
        )
    return limits


def asinh_normalize(
    X,
    low_pct=DEFAULT_LOW_PERCENTILE,
    high_pct=DEFAULT_HIGH_PERCENTILE,
    softening=DEFAULT_ASINH_SOFTENING,
    vmin=None,
    vmax=None,
):
    """Apply the Euclid YOLO percentile-clipped, normalized asinh stretch.

    Without ``vmin``/``vmax``, percentiles are estimated independently for
    every image and channel over the two spatial dimensions. Supplying limits
    applies fixed per-channel statistics, normally computed from the training
    split. Inputs may have shape ``(H, W)``, ``(C, H, W)``, or
    ``(B, C, H, W)`` and outputs are finite floating-point values in ``[0, 1]``.
    """
    if not torch.is_tensor(X):
        raise TypeError("X must be a torch.Tensor.")
    if X.ndim not in (2, 3, 4):
        raise ValueError(
            f"Expected a 2-D, 3-D, or 4-D image tensor, got shape {tuple(X.shape)}."
        )
    if not 0.0 <= low_pct < high_pct <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= low_pct < high_pct <= 100.")
    if softening <= 0:
        raise ValueError("softening must be greater than zero.")
    if (vmin is None) != (vmax is None):
        raise ValueError("vmin and vmax must either both be supplied or both be omitted.")

    x = X if X.dtype in (torch.float32, torch.float64) else X.float()
    channels = x.shape[-3] if x.ndim >= 3 else 1

    if vmin is None:
        # Match Euclid's stats behavior: non-finite samples contribute zero.
        stats_x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        flat = stats_x.flatten(start_dim=-2)
        low = torch.quantile(flat, low_pct / 100.0, dim=-1, keepdim=True)
        high = torch.quantile(flat, high_pct / 100.0, dim=-1, keepdim=True)
        low = low.unsqueeze(-1)
        high = high.unsqueeze(-1)
    else:
        low = _channel_limits(vmin, channels, "vmin").to(device=x.device, dtype=x.dtype)
        high = _channel_limits(vmax, channels, "vmax").to(device=x.device, dtype=x.dtype)
        shape = (
            (1, channels, 1, 1)
            if x.ndim == 4
            else ((channels, 1, 1) if x.ndim == 3 else (1, 1))
        )
        low = low.reshape(shape)
        high = high.reshape(shape)

    if vmin is not None and torch.any(high <= low):
        raise ValueError("Every normalization vmax must be greater than vmin.")

    # Match Euclid's stretch behavior for non-finite input values.
    x = torch.where(torch.isnan(x) | torch.isneginf(x), low, x)
    x = torch.where(torch.isposinf(x), high, x)
    x = torch.minimum(torch.maximum(x, low), high)
    x = (x - low) / (high - low + 1e-6)
    stretched = torch.asinh(x / softening) / torch.asinh(
        x.new_tensor(1.0 / softening)
    )
    return stretched.clamp_(0.0, 1.0)


# Backward-compatible alias for the historical misspelling used by DRAGON.
arsinh_normalize = asinh_normalize

def load_tensor(filename, tensors_path, device="cpu", as_numpy=False):
    """Load a Torch tensor from disk."""
    try:
        filename = filename + ".pt" if ".pt" not in filename else filename
        tensor = torch.load(tensors_path / filename, map_location=device)
        if not as_numpy:
            return tensor
        return tensor.numpy()
    except Exception as e:
        logging.error(f"ERROR: Failed to load tensor from {filename}: {e}")
        raise

def load_tensor_to_gpu(filename, tensors_path, device, as_numpy=False):
    tensor = load_tensor(filename, tensors_path, as_numpy=False)
    return tensor.to(device)
