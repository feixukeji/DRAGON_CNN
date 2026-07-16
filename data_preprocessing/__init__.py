import torch
from torch.utils.data import DataLoader

from .dataset import FITSDataset
from .create_cutouts import create_cutout_tensors
from .normalization import (
    compute_asinh_stats,
    load_or_compute_asinh_stats,
    save_asinh_stats,
)


def get_data_loader(dataset, batch_size, n_workers, shuffle=True, sampler=None):
    # IMPORTANT: If a sampler is provided, shuffle MUST be False.
    # The DistributedSampler handles the shuffling internally.
    if sampler is not None:
        shuffle = False

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": n_workers,
        "sampler": sampler,
        "pin_memory": torch.cuda.is_available(),
    }
    if n_workers > 0:
        loader_kwargs.update(
            prefetch_factor=8,
            persistent_workers=True,
        )

    return DataLoader(
        **loader_kwargs,
    )


__all__ = [
    "FITSDataset",
    "create_cutout_tensors",
    "get_data_loader",
    "compute_asinh_stats",
    "load_or_compute_asinh_stats",
    "save_asinh_stats",
]
