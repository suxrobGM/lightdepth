# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Decoder network for depth estimation

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    """
    Upsampling block with convolutions and normalization.
    
    This block upsamples the input feature map and applies convolutions
    with batch normalization and ReLU activation. Optionally merges
    skip connections from the encoder.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        skip_channels (int): Number of channels in skip connection (0 if no skip)
        upsample_mode (str): Upsampling method ('bilinear', 'nearest', 'transpose')
        use_attention (bool): Whether to include attention mechanism
    
    Example:
        >>> upblock = UpBlock(512, 256, skip_channels=256, upsample_mode="bilinear")
        >>> x = torch.randn(2, 512, 15, 20)
        >>> skip = torch.randn(2, 256, 30, 40)
        >>> out = upblock(x, skip)
        >>> out.shape
        torch.Size([2, 256, 30, 40])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
    ) -> None:
        """Initialize upsampling block."""
        super().__init__()

        # Convolution after concatenating with skip connection
        conv_in_channels = in_channels + skip_channels
        self.conv1 = nn.Conv2d(
            conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through upsampling block.

        Args:
            x (torch.Tensor): Input tensor (B, C_in, H, W)
            skip (torch.Tensor, optional): Skip connection tensor (B, C_skip, H*2, W*2)

        Returns:
            torch.Tensor: Upsampled and processed tensor (B, C_out, H*2, W*2)
        """
        # Upsample with bilinear interpolation
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Concatenate with skip connection if provided
        if skip is not None:
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        # Convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class DepthDecoder(nn.Module):
    """
    Decoder network for depth map reconstruction.
    
    Takes multi-scale features from encoder and progressively upsamples
    to produce final depth map at original resolution.
    
    Architecture:
    - Progressive upsampling through multiple UpBlocks
    - Skip connections from encoder features
    - Final prediction head
    
    Attributes:
        decoder_channels (list): Channel dimensions for each decoder stage
        encoder_channels (list): Channel dimensions from encoder features
        upsample_mode (str): Upsampling method to use
        skip_connections (bool): Use skip connections from encoder
        num_stages (int): Number of upsampling stages
    
    Methods:
        forward(encoder_features): Decode features to depth map
    
    Example:
        >>> encoder_channels = [64, 64, 128, 256, 512]
        >>> decoder = DepthDecoder(encoder_channels, decoder_channels=[256, 128, 64, 32])
        >>> features = [torch.randn(2, c, 480//(2**i), 640//(2**i)) 
        ...             for i, c in enumerate(encoder_channels)]
        >>> depth = decoder(features)
        >>> depth.shape
        torch.Size([2, 1, 480, 640])
    """
    
    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int] | None = None,
    ) -> None:
        """
        Initialize decoder network.

        Args:
            encoder_channels (list): Channel dimensions from encoder features
            decoder_channels (list): Channel dimensions for decoder stages
        """
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.num_stages = len(decoder_channels)

        # Build decoder stages
        self.stages = nn.ModuleList()

        # Start from the deepest encoder feature
        in_channels = encoder_channels[-1]

        for i, out_channels in enumerate(decoder_channels):
            # Determine skip connection channels
            if i < len(encoder_channels) - 1:
                skip_idx = len(encoder_channels) - 2 - i
                skip_ch = encoder_channels[skip_idx] if skip_idx >= 0 else 0
            else:
                skip_ch = 0

            stage = UpBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                skip_channels=skip_ch,
            )
            self.stages.append(stage)
            in_channels = out_channels
        
        # Final prediction head with one more upsample
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # Ensure positive depth values
        )

        # Final upsample to match input resolution
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Decode encoder features to depth map.
        
        Args:
            encoder_features (List[torch.Tensor]): Multi-scale features from encoder,
                ordered from lowest to highest resolution
        
        Returns:
            torch.Tensor: Predicted depth map (B, 1, H, W)
        
        Raises:
            ValueError: If encoder_features length does not match expected
        """
        if len(encoder_features) != len(self.encoder_channels):
            raise ValueError(
                f"Expected {len(self.encoder_channels)} encoder features, "
                f"got {len(encoder_features)}"
            )
        
        # Start from deepest feature
        x = encoder_features[-1]
        
        # Progressive upsampling with skip connections
        for i, stage in enumerate(self.stages):
            # Get skip connection if available
            if i < len(encoder_features) - 1:
                skip_idx = len(encoder_features) - 2 - i
                skip = encoder_features[skip_idx] if skip_idx >= 0 else None
            else:
                skip = None

            x = stage(x, skip)

        # Final depth prediction
        depth = self.head(x)

        # Final upsample to match original resolution
        depth = self.final_upsample(depth)

        return depth
    
    def __repr__(self) -> str:
        """String representation of decoder."""
        return (
            f"DepthDecoder(\\n"
            f"  encoder_channels={self.encoder_channels},\\n"
            f"  decoder_channels={self.decoder_channels},\\n"
            f"  upsample_mode={self.upsample_mode!r},\\n"
            f"  skip_connections={self.skip_connections},\\n"
            f"  num_stages={self.num_stages}\\n"
            f")"
        )
