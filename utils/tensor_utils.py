import json
import math

import torch


DEFAULT_LOW_PERCENTILE = 0.5
DEFAULT_HIGH_PERCENTILE = 99.5
DEFAULT_ASINH_SOFTENING = 0.1


def validate_asinh_softening(value):
    """Return ``value`` as a finite positive floating-point softening."""
    if isinstance(value, bool):
        raise ValueError("Normalization softening must be a finite positive number.")
    try:
        softening = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Normalization softening must be a finite positive number."
        ) from exc
    if not math.isfinite(softening) or softening <= 0:
        raise ValueError("Normalization softening must be a finite positive number.")
    return softening


def validate_asinh_stats(stats, channels=None):
    """Validate and canonicalize fixed asinh normalization statistics."""
    if not isinstance(stats, dict):
        raise ValueError("Normalization stats JSON must contain an object.")

    missing = {"vmin", "vmax", "softening"} - set(stats)
    if missing:
        raise ValueError(
            "Normalization stats JSON is missing required field(s): "
            + ", ".join(sorted(missing))
        )

    try:
        vmin = [float(value) for value in stats["vmin"]]
        vmax = [float(value) for value in stats["vmax"]]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Normalization stats 'vmin' and 'vmax' must be numeric arrays."
        ) from exc
    if not vmin or not vmax:
        raise ValueError("Normalization stats 'vmin' and 'vmax' cannot be empty.")
    if len(vmin) != len(vmax):
        raise ValueError("Normalization stats 'vmin' and 'vmax' must have equal lengths.")
    if channels is not None and len(vmin) != channels:
        raise ValueError(
            f"Normalization stats contain {len(vmin)} channels, expected {channels}."
        )
    if any(not math.isfinite(value) for value in (*vmin, *vmax)):
        raise ValueError("Normalization stats 'vmin' and 'vmax' must be finite.")
    if any(high <= low for low, high in zip(vmin, vmax)):
        raise ValueError("Every normalization vmax must be greater than vmin.")

    softening = validate_asinh_softening(stats["softening"])

    return {
        **stats,
        "vmin": vmin,
        "vmax": vmax,
        "softening": softening,
    }


def load_asinh_stats(path, channels=None):
    """Load fixed per-channel limits and softening from JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    return validate_asinh_stats(stats, channels=channels)


def normalization_kwargs_from_stats(
    path,
    channels,
):
    """Build fixed ``asinh_normalize`` arguments from training statistics."""
    stats = load_asinh_stats(path, channels=channels)
    return {
        "softening": stats["softening"],
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


def asinh_normalize(X, *, vmin, vmax, softening):
    """Apply the fixed, training-statistics asinh normalization.

    ``vmin``, ``vmax``, and ``softening`` must come from the dataset-level
    ``normalization_stats.json``. Inputs may have shape ``(H, W)``,
    ``(C, H, W)``, or ``(B, C, H, W)`` and outputs are finite floating-point
    values in ``[0, 1]``.
    """
    if not torch.is_tensor(X):
        raise TypeError("X must be a torch.Tensor.")
    if X.ndim not in (2, 3, 4):
        raise ValueError(
            f"Expected a 2-D, 3-D, or 4-D image tensor, got shape {tuple(X.shape)}."
        )
    softening = validate_asinh_softening(softening)

    x = X if X.dtype in (torch.float32, torch.float64) else X.float()
    channels = x.shape[-3] if x.ndim >= 3 else 1

    low = _channel_limits(vmin, channels, "vmin").to(device=x.device, dtype=x.dtype)
    high = _channel_limits(vmax, channels, "vmax").to(device=x.device, dtype=x.dtype)
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise ValueError("Every normalization vmin/vmax must be finite.")
    shape = (
        (1, channels, 1, 1)
        if x.ndim == 4
        else ((channels, 1, 1) if x.ndim == 3 else (1, 1))
    )
    low = low.reshape(shape)
    high = high.reshape(shape)

    if torch.any(high <= low):
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
