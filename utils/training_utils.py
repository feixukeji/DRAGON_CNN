"""Validation shared by standard and sweep training entry points."""


def validate_transfer_learning_options(
    *,
    is_training,
    model_state,
    normalize,
    normalization_stats,
):
    """Require a checkpoint and fixed asinh statistics for transfer learning."""
    if is_training:
        return
    if not model_state:
        raise ValueError("Transfer learning requires --model_state.")
    if not normalize:
        raise ValueError(
            "Transfer learning must use asinh normalization; "
            "--no-normalize is unsupported."
        )
    if not normalization_stats:
        raise ValueError(
            "Transfer learning requires --normalization-stats. The JSON is "
            "loaded when present or computed from the training split."
        )
