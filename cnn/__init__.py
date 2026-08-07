from .DRAGON_cnn import DRAGON, DRAGON_CUTOUT_SIZE


def model_stats(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(trainable_params=n_params)


__all__ = ["DRAGON", "DRAGON_CUTOUT_SIZE", "model_stats"]
