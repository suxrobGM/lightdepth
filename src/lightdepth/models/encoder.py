# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Encoder network for depth estimation

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetEncoder(nn.Module):
    """
    ResNet18 encoder for multi-scale feature extraction.

    Extracts features at 5 scales from pretrained ResNet18 for decoder skip connections.
    """

    layer0: nn.Sequential
    """Initial conv layer (conv1 + bn1 + relu)."""

    layer1: nn.Sequential
    """First residual block with maxpool."""

    layer2: nn.Sequential
    """Second residual block."""

    layer3: nn.Sequential
    """Third residual block."""

    layer4: nn.Sequential
    """Fourth residual block."""

    feature_channels: list[int]
    """Output channels at each layer: [64, 64, 128, 256, 512]."""

    def __init__(self, pretrained=True) -> None:
        """
        Initialize ResNet18 encoder.
        Args:
            pretrained: Whether to use ImageNet pretrained weights. Default: True
        """
        super().__init__()

        # Load pretrained ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)

        # Extract layers for feature extraction
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Feature channels at each level
        self.feature_channels = [64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input RGB image (B, 3, H, W)

        Returns:
            List of 5 feature maps at different scales [64, 64, 128, 256, 512] channels
        """
        features = []
        x = self.layer0(x)
        features.append(x)  # 64 channels
        x = self.layer1(x)
        features.append(x)  # 64 channels
        x = self.layer2(x)
        features.append(x)  # 128 channels
        x = self.layer3(x)
        features.append(x)  # 256 channels
        x = self.layer4(x)
        features.append(x)  # 512 channels
        return features
