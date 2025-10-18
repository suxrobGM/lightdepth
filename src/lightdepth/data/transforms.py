# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Data transforms for depth estimation

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class TrainTransform:
    """Training transforms with augmentation (resize, flip, color jitter, normalize)."""

    def __init__(self, img_size=(480, 640)) -> None:
        """
        Args:
            img_size: Target size as (height, width). Default: (480, 640)
        """
        # Convert (height, width) to (width, height) for PIL
        self.img_size_pil = (img_size[1], img_size[0])

    def __call__(
        self, image: Image.Image, depth: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Resize (PIL expects width, height)
        image = image.resize(self.img_size_pil, Image.Resampling.BILINEAR)
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize(self.img_size_pil, Image.Resampling.NEAREST)
        depth = np.array(depth_img)

        # Random horizontal flip (using PIL methods to avoid type issues)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            depth = np.fliplr(depth).copy()

        # To tensor
        image_tensor = TF.to_tensor(image)
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)

        # Color jitter on tensor (image only)
        brightness_factor = 1 + random.uniform(-0.2, 0.2)
        contrast_factor = 1 + random.uniform(-0.2, 0.2)
        saturation_factor = 1 + random.uniform(-0.2, 0.2)

        image_tensor = TF.adjust_brightness(image_tensor, brightness_factor)
        image_tensor = TF.adjust_contrast(image_tensor, contrast_factor)
        image_tensor = TF.adjust_saturation(image_tensor, saturation_factor)

        # Normalize image (ImageNet stats)
        image_tensor = TF.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        return image_tensor, depth_tensor


class ValTransform:
    """Validation transforms without augmentation (resize and normalize only)."""

    def __init__(self, img_size=(480, 640)) -> None:
        """
        Args:
            img_size: Target size as (height, width). Default: (480, 640)
        """
        # Convert (height, width) to (width, height) for PIL
        self.img_size_pil = (img_size[1], img_size[0])

    def __call__(
        self, image: Image.Image, depth: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Resize (PIL expects width, height)
        image = image.resize(self.img_size_pil, Image.Resampling.BILINEAR)
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize(self.img_size_pil, Image.Resampling.NEAREST)
        depth = np.array(depth_img)

        # To tensor
        tensor_image = TF.to_tensor(image)
        tensor_depth = torch.from_numpy(depth).float().unsqueeze(0)

        # Normalize image, using ImageNet stats such as mean and std
        tensor_image = TF.normalize(
            tensor_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        return tensor_image, tensor_depth
