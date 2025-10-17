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
        seed (int): Random seed value
    
    Example:
        >>> set_seed(42)
        >>> # All random operations will now be reproducible
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
    seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with appropriate transforms.
    
    This function handles:
    - Loading training and validation datasets
    - Applying appropriate transforms (augmentation for train, none for val)
    - Creating dataloaders with optimal settings
    - Setting random seeds for reproducibility
    - Supporting subset sampling for quick experiments
    
    Args:
        data_root (str): Path to dataset root directory (e.g., "data/nyu")
        batch_size (int): Batch size for both dataloaders (default: 16)
        num_workers (int): Number of dataloader workers for parallel loading (default: 4)
        img_size (tuple): Target image size as (height, width) (default: (480, 640))
        train_size (int, optional): Limit training set size (None = use all)
        val_size (int, optional): Limit validation set size (None = use all)
        shuffle (bool): Whether to shuffle training data (default: True)
        pin_memory (bool): Pin memory for faster GPU transfer (default: True)
        seed (int): Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
            - train_loader: DataLoader for training with augmentation
            - val_loader: DataLoader for validation without augmentation
    
    Raises:
        FileNotFoundError: If data_root directory does not exist
        ValueError: If batch_size <= 0 or num_workers < 0
    
    Example:
        >>> # Standard usage
        >>> train_loader, val_loader = create_dataloaders(
        ...     data_root="data/nyu",
        ...     batch_size=16,
        ...     num_workers=4
        ... )
        >>> print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        >>> # Quick experiment with subset
        >>> train_loader, val_loader = create_dataloaders(
        ...     data_root="data/nyu",
        ...     batch_size=8,
        ...     train_size=100,
        ...     val_size=50
        ... )
        
        >>> # Custom image size
        >>> train_loader, val_loader = create_dataloaders(
        ...     data_root="data/nyu",
        ...     img_size=(384, 512),
        ...     batch_size=32
        ... )
    
    Note:
        - Training data is shuffled by default for better generalization
        - Validation data is not shuffled to ensure consistent evaluation
        - pin_memory=True speeds up CPU->GPU transfer but uses more RAM
        - num_workers=0 uses main process (useful for debugging)
        - Higher num_workers can speed up data loading but uses more CPU/RAM
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
            subset_size=train_size
        )
        
        val_dataset = NYUDepthV2Dataset(
            root_dir=data_root,
            split="test",  # NYU uses "test" for validation
            transform=val_transforms,
            subset_size=val_size
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
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,  # Keep all validation samples
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader
