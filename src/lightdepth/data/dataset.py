# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: NYU-Depth-v2 dataset for indoor depth estimation

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class NYUDepthV2Dataset(Dataset):
    """
    NYU-Depth-v2 indoor depth estimation dataset.

    Contains 1449 RGB-D pairs from 464 indoor scenes at 640x480 resolution.
    Loads from CSV files or directory structure (nyu2_train/, nyu2_test/).

    Args:
        root_dir: Path to dataset root directory
        split: 'train' or 'test'
        transform: Optional transform for RGB images and depth maps
        subset_size: Optional limit on dataset size

    Returns (via __getitem__):
        image: RGB image tensor (after transform)
        depth: Depth map tensor in meters
        mask: Valid depth mask (True = valid pixel)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: (
            Callable[[Image.Image, np.ndarray], tuple[torch.Tensor, torch.Tensor]]
            | None
        ) = None,
        subset_size: int | None = None,
    ) -> None:
        """
        Initialize NYU-Depth-v2 dataset.
        Args:
            root_dir: Path to dataset root directory
            split: 'train' or 'test'. Default: 'train'
            transform: Optional transform for RGB images and depth maps
            subset_size: Optional limit on dataset size
        """
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transform = transform

        # Validate split
        if self.split not in ["train", "test"]:
            raise ValueError(f"Split must be 'train' or 'test', got '{split}'")

        # Check if root directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root directory not found: {self.root_dir}"
            )

        # Load file paths
        self.image_paths = []
        self.depth_paths = []
        self._load_file_paths()

        # Apply subset if requested
        if subset_size is not None and subset_size > 0:
            subset_size = min(subset_size, len(self.image_paths))
            self.image_paths = self.image_paths[:subset_size]
            self.depth_paths = self.depth_paths[:subset_size]

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.root_dir} for split '{self.split}'"
            )

    def _load_file_paths(self) -> None:
        """Load image and depth paths from CSV or directory."""
        csv_path = self.root_dir / f"nyu2_{self.split}.csv"

        if csv_path.exists():
            # Load from CSV file (no headers, just image,depth paths)
            df = pd.read_csv(csv_path, header=None, names=["image", "depth"])

            # Remove data/ prefix if present for image and depth paths
            df["image"] = df["image"].apply(
                lambda x: x[5:] if x.startswith("data/") else x
            )
            df["depth"] = df["depth"].apply(
                lambda x: x[5:] if x.startswith("data/") else x
            )

            # Construct full paths
            self.image_paths = [self.root_dir / p for p in df["image"].tolist()]
            self.depth_paths = [self.root_dir / p for p in df["depth"].tolist()]
        else:
            # Fallback: scan directory structure
            img_dir = self.root_dir / f"nyu2_{self.split}"
            if not img_dir.exists():
                raise FileNotFoundError(
                    f"Neither CSV file ({csv_path}) nor image directory ({img_dir}) found"
                )

            # Assuming paired files: image_XXX.jpg and depth_XXX.png
            for img_path in sorted(img_dir.glob("*_rgb.jpg")):
                depth_path = img_path.parent / img_path.name.replace(
                    "_rgb.jpg", "_depth.png"
                )
                if depth_path.exists():
                    self.image_paths.append(img_path)
                    self.depth_paths.append(depth_path)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get sample at index.

        Args:
            idx: Sample index

        Returns:
            image: RGB image tensor
            depth: Depth map tensor in meters
            mask: Valid depth mask tensor
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Load RGB image
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image {self.image_paths[idx]}: {e}")

        # Load depth map
        try:
            depth = Image.open(self.depth_paths[idx])
            depth = np.array(depth, dtype=np.float32)

            # Convert from millimeters to meters if needed
            if depth.max() > 100:  # Likely in millimeters
                depth = depth / 1000.0
        except Exception as e:
            raise IOError(f"Failed to load depth {self.depth_paths[idx]}: {e}")

        # Create validity mask (mask invalid depth values)
        mask = (depth > 0.01) & (depth < 10.0)
        mask = mask.astype(np.float32)

        # Apply transforms (handles both image and depth)
        transformed_image: torch.Tensor = image  # type: ignore
        transformed_depth: torch.Tensor = depth  # type: ignore

        if self.transform is not None:
            transformed_image, transformed_depth = self.transform(image, depth)

        # Convert mask to tensor
        mask = torch.from_numpy(mask).unsqueeze(0)

        return transformed_image, transformed_depth, mask

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NYUDepthV2Dataset(\\n"
            f"  split='{self.split}',\\n"
            f"  num_samples={len(self)},\\n"
            f"  root_dir='{self.root_dir}'\\n"
            f")"
        )
