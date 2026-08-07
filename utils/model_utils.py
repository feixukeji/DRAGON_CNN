import torch


def load_plain_state_dict(model_path, device="cpu"):
    """Load the exact plain ``state_dict`` format written by the trainer."""
    state = torch.load(model_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(
            "Model file must contain the non-empty plain state_dict written by "
            "the current DRAGON trainer."
        )

    invalid_keys = [
        key
        for key, value in state.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid_keys:
        raise ValueError(
            "Wrapped checkpoint dictionaries are unsupported; expected a plain "
            "mapping from parameter names to tensors."
        )
    if any(key.startswith("module.") for key in state):
        raise ValueError(
            "DataParallel 'module.' parameter prefixes are unsupported; use "
            "model.pt written by the current DRAGON trainer."
        )
    return state


def load_model_state(model, model_path, device="cpu"):
    """Load current-trainer weights into ``model`` on the requested device."""
    model.load_state_dict(load_plain_state_dict(model_path, device=device))
    return model
