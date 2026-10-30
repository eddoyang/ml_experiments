import torch


def get_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available")
    return torch.device("mps")