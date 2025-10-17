# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Metrics for evaluating depth estimation

import torch


def compute_rmse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute Root Mean Squared Error.

    Args:
        pred: Predicted depth (B, 1, H, W)
        target: Ground truth depth (B, 1, H, W)
        mask: Valid depth mask (B, 1, H, W), optional

    Returns:
        RMSE value
    """
    if mask is not None:
        valid = mask > 0
        if valid.sum() > 0:
            return torch.sqrt(torch.mean((pred[valid] - target[valid]) ** 2))
        else:
            return torch.tensor(0.0)
    else:
        return torch.sqrt(torch.mean((pred - target) ** 2))
