from .dragon import (
    DRAGON,
    DRAGON_CLASSIFIER_BIAS_KEY,
    DRAGON_CUTOUT_SIZE,
    DRAGON_FIRST_CONV_WEIGHT_KEY,
    DRAGON_HEAD_PREFIXES,
    DRAGON_MIN_SIZE,
    DRAGON_SIZE_DIVISOR,
)


def model_stats(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable_params": n_params}


__all__ = [
    "DRAGON",
    "DRAGON_CLASSIFIER_BIAS_KEY",
    "DRAGON_CUTOUT_SIZE",
    "DRAGON_FIRST_CONV_WEIGHT_KEY",
    "DRAGON_HEAD_PREFIXES",
    "DRAGON_MIN_SIZE",
    "DRAGON_SIZE_DIVISOR",
    "model_stats",
]
