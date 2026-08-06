import torch
import logging


def discover_devices():
    """Check for available devices."""
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        logging.info("Detected %d CUDA GPU(s).", n_devices)
        return "cuda"
    return "cpu"
