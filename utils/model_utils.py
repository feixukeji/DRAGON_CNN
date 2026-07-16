import torch


def unwrap_state_dict(state):
    """Return a plain model state dict from common checkpoint wrappers."""
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    elif isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dict")
    if any(str(key).startswith("module.") for key in state):
        state = {
            str(key).removeprefix("module."): value
            for key, value in state.items()
        }
    return state


def load_model_state(model, model_path, device="cpu"):
    """Load a checkpoint into ``model`` on the requested device."""
    state = torch.load(model_path, map_location=device)
    parallel_types = (
        torch.nn.DataParallel,
        torch.nn.parallel.DistributedDataParallel,
    )
    target_model = model.module if isinstance(model, parallel_types) else model
    target_model.load_state_dict(unwrap_state_dict(state))
    return model


def get_output_shape(model, image_dim):
    """Get output shape of a PyTorch model or layer"""
    return model(torch.rand(*(image_dim))).data.shape


def enable_dropout(model):
    """Enable random dropout during inference. From StackOverflow #63397197"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def specify_dropout_rate(model, rate):
    """Specify the dropout rate of all layers"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.p = rate
