# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Complete lightweight depth estimation network

import torch
import torch.nn as nn

from lightdepth.models.decoder import DepthDecoder
from lightdepth.models.encoder import ResNetEncoder


class LightDepthNet(nn.Module):
    """
    Complete lightweight depth estimation network.

    Architecture consists of:
    1. Encoder: Pretrained backbone (ResNet/MobileNet/EfficientNet)
    2. Decoder: Upsampling path with skip connections
    3. Head: Final prediction layer

    The model takes an RGB image as input and outputs a single-channel
    depth map at the same resolution.

    Args:
        encoder_name (str): Encoder architecture name (default: 'resnet18')
        pretrained (bool): Use pretrained encoder weights (default: True)
        freeze_encoder (bool): Freeze encoder during training (default: False)
        decoder_channels (list): Decoder channel configuration
        upsample_mode (str): Upsampling method (default: 'bilinear')
        skip_connections (bool): Use skip connections (default: True)
        use_attention (bool): Use attention in decoder (default: False)

    Attributes:
        encoder (EncoderBackbone): Encoder backbone
        decoder (DepthDecoder): Decoder network
        total_params (int): Total number of parameters
        trainable_params (int): Number of trainable parameters

    Methods:
        forward(x): Predict depth map from input RGB image
        get_model_info(): Return model architecture information
        load_pretrained(checkpoint_path): Load pretrained weights
        count_parameters(): Count total and trainable parameters

    Example:
        >>> model = LightweightDepthNet(encoder_name='resnet18', pretrained=True)
        >>> image = torch.randn(2, 3, 480, 640)
        >>> depth = model(image)
        >>> depth.shape
        torch.Size([2, 1, 480, 640])
        >>> info = model.get_model_info()
        >>> print(f"Total params: {info['total_params']:,}")
    """

    def __init__(
        self,
        pretrained: bool = True,
        decoder_channels: list | None = None,
    ):
        """Initialize depth estimation network."""
        super().__init__()

        self.pretrained = pretrained

        # Default decoder channels
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]
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
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input RGB image tensor of shape (B, 3, H, W)
                Values should be normalized to [0, 1] or ImageNet stats

        Returns:
            torch.Tensor: Predicted depth map of shape (B, 1, H, W)
                Depth values are positive and typically in range [0, 10]

        Example:
            >>> model = LightweightDepthNet()
            >>> image = torch.randn(4, 3, 480, 640)
            >>> depth = model(image)
            >>> print(f"Depth range: {depth.min():.2f} to {depth.max():.2f}")
        """
        # Extract multi-scale features from encoder
        features = self.encoder(x)

        # Decode features to depth map
        depth = self.decoder(features)

        return depth

    def count_parameters(self) -> tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns:
            tuple: (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
