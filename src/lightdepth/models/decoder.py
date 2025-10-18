# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Decoder network for depth estimation

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    """
    Upsampling block with skip connections.

    Bilinear upsample (2x) -> concat with skip -> conv -> BN -> ReLU -> conv -> BN -> ReLU

    Args:
        in_channels: Input channels
        out_channels: Output channels
        skip_channels: Skip connection channels (0 if no skip)
    """

    conv1: nn.Conv2d
    """First 3x3 conv (input+skip -> out_channels)."""

    conv2: nn.Conv2d
    """Second 3x3 conv (out_channels -> out_channels)."""

    bn1: nn.BatchNorm2d
    """Batch norm after conv1."""

    bn2: nn.BatchNorm2d
    """Batch norm after conv2."""

    relu1: nn.ReLU
    """ReLU after bn1."""

    relu2: nn.ReLU
    """ReLU after bn2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
    ) -> None:
        """
        Initialize upsampling block.
        Args:
            in_channels: Input channels
            out_channels: Output channels
            skip_channels: Skip connection channels (0 if no skip). Default: 0
        """
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
        Forward pass.

        Args:
            x: Input tensor (B, C_in, H, W)
            skip: Optional skip connection (B, C_skip, H*2, W*2)

        Returns:
            Upsampled tensor (B, C_out, H*2, W*2)
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

    Progressively upsamples encoder features through multiple UpBlocks with skip connections,
    then produces final depth prediction.

    Args:
        encoder_channels: Channel dimensions from encoder features
        decoder_channels: Channel dimensions for decoder stages
    """

    encoder_channels: list[int]
    """Encoder feature channels: [64, 64, 128, 256, 512]."""

    decoder_channels: list[int]
    """Decoder stage channels (default: [256, 128, 64, 32])."""

    num_stages: int
    """Number of UpBlock stages."""

    stages: nn.ModuleList
    """List of UpBlock modules."""

    head: nn.Sequential
    """Final conv layers (32 -> 1 channel depth map)."""

    final_upsample: nn.Upsample
    """Final 2x bilinear upsample to input resolution."""

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int] = [256, 128, 64, 32],
    ) -> None:
        """
        Initialize decoder.
        Args:
            encoder_channels: Channel dimensions from encoder features
            decoder_channels: Channel dimensions for decoder stages (default: [256, 128, 64, 32])
        """
        super().__init__()

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
        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Decode encoder features to depth map.

        Args:
            encoder_features: Multi-scale features from encoder (low to high resolution)

        Returns:
            Predicted depth map (B, 1, H, W)
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
        """String representation."""
        return (
            f"DepthDecoder(\\n"
            f"  encoder_channels={self.encoder_channels},\\n"
            f"  decoder_channels={self.decoder_channels},\\n"
            f"  upsample_mode={self.upsample_mode!r},\\n"
            f"  skip_connections={self.skip_connections},\\n"
            f"  num_stages={self.num_stages}\\n"
            f")"
        )
