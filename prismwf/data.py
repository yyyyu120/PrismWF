"""Dataset loading for precomputed PrismWF features."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .reproducibility import seed_worker


def align_length(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        widths = [(0, 0)] * x.ndim
        widths[-1] = (0, length - x.shape[-1])
        return np.pad(x, widths, mode="constant")
    return x


FEATURE_GROUPS = {
    "all": (0, 1, 2, 3, 4, 5),
    "packet-count": (0, 1),
    "transition-count": (2, 3),
    "transition-interval": (4, 5),
}


def load_feature_file(
    path: str | Path,
    length: int = 8000,
    feature_group: str = "all",
) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(Path(path))
    x = align_length(data["X"], length).astype(np.float32, copy=False)
    y = data["y"].astype(np.float32, copy=False)
    if x.ndim != 3 or x.shape[1] != 6:
        raise ValueError(f"Expected features with shape (N, 6, L), got {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"Expected multi-hot labels with shape (N, C), got {y.shape}")
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {feature_group}")
    if feature_group != "all":
        selected = FEATURE_GROUPS[feature_group]
        masked = np.zeros_like(x)
        masked[:, selected] = x[:, selected]
        x = masked
    return torch.from_numpy(x), torch.from_numpy(y)


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    seed: int,
    train: bool,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )
