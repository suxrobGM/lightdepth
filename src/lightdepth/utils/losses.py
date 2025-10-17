# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Loss functions for depth estimation

import torch
import torch.nn as nn


class DepthLoss(nn.Module):
    """
    Simple L1 loss for depth estimation.

    Computes mean absolute error between predicted and ground truth depth.
    """

    def __init__(self) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute L1 loss on valid depth values.

        Args:
            pred: Predicted depth (B, 1, H, W)
            target: Ground truth depth (B, 1, H, W)
            mask: Valid depth mask (B, 1, H, W), optional

        Returns:
            Loss value
        """
        if mask is not None:
            # Only compute loss on valid pixels
            valid = mask > 0
            if valid.sum() > 0:
                return self.l1_loss(pred[valid], target[valid])
            else:
                return torch.tensor(0.0, device=pred.device)
        else:
            return self.l1_loss(pred, target)
