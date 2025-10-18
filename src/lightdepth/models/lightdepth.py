# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Complete lightweight depth estimation network with encoder-decoder architecture

import torch
import torch.nn as nn

from lightdepth.models.decoder import DepthDecoder
from lightdepth.models.encoder import ResNetEncoder


class LightDepth(nn.Module):
    """
    Lightweight depth estimation network with encoder-decoder architecture.

    Takes RGB image (B, 3, H, W) and outputs depth map (B, 1, H, W).
    Uses ResNet18 encoder with pretrained ImageNet weights and multi-scale decoder.

    Args:
        pretrained: Use pretrained encoder weights
        decoder_channels: Channel configuration for decoder stages
    """

    pretrained: bool
    """Whether to use ImageNet pretrained encoder weights. Default: True."""

    decoder_channels: list[int]
    """Decoder channel config (default: [256, 128, 64, 32])."""

    encoder: ResNetEncoder
    """ResNet18 encoder for multi-scale feature extraction."""

    decoder: DepthDecoder
    """Decoder with skip connections and upsampling."""

    def __init__(
        self,
        pretrained: bool = True,
        decoder_channels: list = [256, 128, 64, 32],
    ) -> None:
        """
        Initialize depth estimation network.
        Args:
            pretrained: Whether to use ImageNet pretrained encoder weights. Default: True.
            decoder_channels: Channel configuration for decoder stages. Default: [256, 128, 64, 32].
        """
        super().__init__()

        self.pretrained = pretrained

        # Decoder channels
        self.decoder_channels = decoder_channels

        # Create encoder (ResNet18)
        self.encoder = ResNetEncoder(pretrained=pretrained)

        # Get encoder output channels
        encoder_channels = self.encoder.feature_channels

        # Create decoder
        self.decoder = DepthDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input RGB image (B, 3, H, W), normalized with ImageNet stats

        Returns:
            Predicted depth map (B, 1, H, W), positive values in meters
        """
        # Extract multi-scale features from encoder
        features = self.encoder(x)

        # Decode features to depth map
        depth = self.decoder(features)

        return depth

    def count_parameters(self) -> tuple[int, int]:
        """
        Count model parameters.

        Returns:
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
