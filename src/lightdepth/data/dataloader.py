# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: DataLoader utilities for LightDepth model

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from lightdepth.data.dataset import NYUDepthV2Dataset
from lightdepth.data.transforms import TrainTransform, ValTransform


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable cudnn benchmark for faster training (slight non-determinism)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: tuple[int, int] = (480, 640),
    train_size: int | None = None,
    val_size: int | None = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_root: Path to dataset root directory
        batch_size: Batch size for both dataloaders
        num_workers: Number of parallel data loading workers
        img_size: Target image size as (height, width)
        train_size: Optional limit on training set size
        val_size: Optional limit on validation set size
        shuffle: Whether to shuffle training data
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for reproducibility

    Returns:
        train_loader: DataLoader with augmentation
        val_loader: DataLoader without augmentation
    """
    # Validate arguments
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")

    # Set random seed for reproducibility
    set_seed(seed)

    # Get transforms
    train_transforms = TrainTransform(img_size=img_size)
    val_transforms = ValTransform(img_size=img_size)

    # Create datasets
    try:
        train_dataset = NYUDepthV2Dataset(
            root_dir=data_root,
            split="train",
            transform=train_transforms,
            subset_size=train_size,
        )

        val_dataset = NYUDepthV2Dataset(
            root_dir=data_root,
            split="test",  # NYU uses "test" for validation
            transform=val_transforms,
            subset_size=val_size,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find dataset at {data_root}. "
            f"Please check the path and ensure data is downloaded. Error: {e}"
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # Drop incomplete batch for consistent batch size
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,  # Keep all validation samples
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
