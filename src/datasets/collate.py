import torch
from torch.utils.data import DataLoader

def create_loader(ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=None):
    """Create a DataLoader with sensible CUDA defaults.

    If pin_memory is None, it will be set to True when CUDA is available to speed up
    host-to-device transfers.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
