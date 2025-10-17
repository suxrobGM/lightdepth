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
    NYU-Depth-v2 dataset for indoor depth estimation.

    Supports directory format with preprocessed RGB-D pairs organized as:
    - nyu2_train/ and nyu2_test/ folders with images
    - nyu2_train.csv and nyu2_test.csv with file paths

    Dataset contains 1449 RGB-D pairs from 464 indoor scenes.
    Resolution: 640x480 pixels
    Depth range: 0.5m to 10m (typical indoor scenes)

    Args:
        root_dir (str): Path to dataset root directory (e.g., 'data/nyu')
        split (str): Dataset split - 'train' or 'test' (default: 'train')
        transform (callable, optional): Transform to apply to RGB images
        target_transform (callable, optional): Transform to apply to depth maps
        subset_size (int, optional): Limit dataset size for quick experiments

    Attributes:
        root_dir (Path): Root directory path
        split (str): Current data split
        image_paths (list): List of RGB image file paths
        depth_paths (list): List of depth map file paths

    Methods:
        __len__(): Return total number of samples in dataset
        __getitem__(idx): Return (image, depth, mask) tuple for given index
        get_statistics(): Compute and return dataset depth statistics

    Example:
        >>> dataset = NYUDepthV2Dataset(root_dir='data/nyu', split='train')
        >>> print(f"Dataset size: {len(dataset)}")
        >>> image, depth, mask = dataset[0]
        >>> print(f"Image shape: {image.shape}, Depth range: [{depth[mask].min():.2f}, {depth[mask].max():.2f}]")

    Note:
        - Images are returned as PIL Images (can be tensors after transform)
        - Depth maps are numpy arrays in meters
        - Masks indicate valid depth pixels (True = valid, False = invalid)
        - Invalid depth values (0 or > 10m) are automatically masked
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
        """Initialize NYU-Depth-v2 dataset."""
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
        """Load image and depth file paths from CSV or directory structure."""
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
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get dataset sample at given index.

        Args:
            idx (int): Sample index

        Returns:
            Tuple containing:
                - image (Tensor): RGB image, shape depends on transforms
                - depth (Tensor): Depth map, shape depends on transforms
                - mask (Tensor): Valid depth mask, same shape as depth

        Raises:
            IndexError: If idx is out of range
            IOError: If image or depth file cannot be loaded
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

    def get_statistics(self) -> dict:
        """
        Compute dataset statistics.

        Returns:
            dict: Dictionary containing depth statistics:
                - min_depth: Minimum valid depth value
                - max_depth: Maximum valid depth value
                - mean_depth: Mean valid depth value
                - num_samples: Total number of samples

        Example:
            >>> dataset = NYUDepthV2Dataset('data/nyu', 'train')
            >>> stats = dataset.get_statistics()
            >>> print(f"Depth range: [{stats['min_depth']:.2f}, {stats['max_depth']:.2f}]m")
        """
        depths = []

        for idx in range(min(len(self), 100)):  # Sample first 100 for speed
            try:
                depth = np.array(Image.open(self.depth_paths[idx]), dtype=np.float32)
                if depth.max() > 100:
                    depth = depth / 1000.0
                valid_depth = depth[(depth > 0.01) & (depth < 10.0)]
                if len(valid_depth) > 0:
                    depths.append(valid_depth)
            except:
                continue

        if len(depths) == 0:
            return {
                "min_depth": 0.0,
                "max_depth": 0.0,
                "mean_depth": 0.0,
                "num_samples": len(self),
            }

        all_depths = np.concatenate(depths)

        return {
            "min_depth": float(np.min(all_depths)),
            "max_depth": float(np.max(all_depths)),
            "mean_depth": float(np.mean(all_depths)),
            "num_samples": len(self),
        }

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"NYUDepthV2Dataset(\\n"
            f"  split='{self.split}',\\n"
            f"  num_samples={len(self)},\\n"
            f"  root_dir='{self.root_dir}'\\n"
            f")"
        )
