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

    return torch.sqrt(torch.mean((pred - target) ** 2))


def compute_mae(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute Mean Absolute Error.

    Args:
        pred: Predicted depth (B, 1, H, W)
        target: Ground truth depth (B, 1, H, W)
        mask: Valid depth mask (B, 1, H, W), optional

    Returns:
        MAE value
    """
    if mask is not None:
        valid = mask > 0
        if valid.sum() > 0:
            return torch.mean(torch.abs(pred[valid] - target[valid]))
        else:
            return torch.tensor(0.0)

    return torch.mean(torch.abs(pred - target))


def compute_abs_rel(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute Absolute Relative Error.

    Args:
        pred: Predicted depth (B, 1, H, W)
        target: Ground truth depth (B, 1, H, W)
        mask: Valid depth mask (B, 1, H, W), optional

    Returns:
        Absolute relative error
    """
    if mask is not None:
        valid = mask > 0
        if valid.sum() > 0:
            return torch.mean(torch.abs(pred[valid] - target[valid]) / target[valid])
        else:
            return torch.tensor(0.0)

    return torch.mean(torch.abs(pred - target) / target)


def compute_sq_rel(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute Squared Relative Error.

    Args:
        pred: Predicted depth (B, 1, H, W)
        target: Ground truth depth (B, 1, H, W)
        mask: Valid depth mask (B, 1, H, W), optional

    Returns:
        Squared relative error
    """
    if mask is not None:
        valid = mask > 0
        if valid.sum() > 0:
            return torch.mean(((pred[valid] - target[valid]) ** 2) / target[valid])
        else:
            return torch.tensor(0.0)

    return torch.mean(((pred - target) ** 2) / target)


def compute_all_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, float]:
    """
    Compute all depth estimation metrics.

    Args:
        pred: Predicted depth (B, 1, H, W)
        target: Ground truth depth (B, 1, H, W)
        mask: Valid depth mask (B, 1, H, W), optional

    Returns:
        Dictionary containing all metrics
    """

    return {
        "rmse": compute_rmse(pred, target, mask).item(),
        "mae": compute_mae(pred, target, mask).item(),
        "abs_rel": compute_abs_rel(pred, target, mask).item(),
        "sq_rel": compute_sq_rel(pred, target, mask).item(),
    }
