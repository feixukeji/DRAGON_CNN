import torch
from torch.utils.data import DataLoader

from .dataset import FITSDataset, preload_h5_images
from .normalization import (
    compute_asinh_stats,
    save_asinh_stats,
)


def get_data_loader(
    dataset,
    batch_size,
    n_workers,
    shuffle=True,
    generator=None,
    worker_init_fn=None,
):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": n_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
        "worker_init_fn": worker_init_fn,
    }
    if n_workers > 0:
        loader_kwargs.update(
            prefetch_factor=2,
            persistent_workers=True,
        )
        if getattr(dataset, "h5_preloaded", False):
            # Linux fork workers inherit the large NumPy allocation through
            # copy-on-write. Spawn would serialize one complete copy per worker.
            loader_kwargs["multiprocessing_context"] = "fork"

    return DataLoader(
        **loader_kwargs,
    )


__all__ = [
    "FITSDataset",
    "preload_h5_images",
    "get_data_loader",
    "compute_asinh_stats",
    "save_asinh_stats",
]
