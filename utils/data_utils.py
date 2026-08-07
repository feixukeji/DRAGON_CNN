import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F


def load_data_dir(data_dir, slug=None, split=None):
    """Loads and returns pandas dataframe"""

    data_dir = Path(data_dir)

    if split:
        catalog = data_dir / f"splits/{slug}-{split}.csv"
    else:
        catalog = data_dir / "info.csv"

    return pd.read_csv(catalog)


def center_crop_or_pad_torch(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Crops or pads a 2D PyTorch tensor to the specified square size from the center."""
    h, w = tensor.shape

    # Pad if the image is smaller than the requested size
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
        h, w = tensor.shape

    # Crop if the image is larger
    start_y = h // 2 - size // 2
    start_x = w // 2 - size // 2
    return tensor[start_y:start_y + size, start_x:start_x + size]
