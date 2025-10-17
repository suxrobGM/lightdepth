# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Evaluation script for LightDepth model

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

import torch
from rich import print
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightdepth.data import create_dataloaders
from lightdepth.models import LightDepthNet
from lightdepth.utils import Config, compute_all_metrics, save_depth_map


def evaluate(
    model: LightDepthNet,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    """Evaluate model on test set with optional mixed precision.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for test data
        device: Device to run evaluation on
        use_amp: Whether to use automatic mixed precision (faster on CUDA)

    Returns:
        Dictionary containing average metrics across the test set
    """
    model.eval()

    # Initialize metric accumulators
    metrics_sum = {
        "rmse": 0.0,
        "mae": 0.0,
        "abs_rel": 0.0,
        "sq_rel": 0.0,
    }

    print("Evaluating...")
    with torch.no_grad():
        for images, depths, masks in tqdm(dataloader, desc="Evaluation"):
            # Use non_blocking transfers for better performance
            images, depths, masks = (
                images.to(device, non_blocking=True),
                depths.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
            )

            # Forward with optional mixed precision
            if use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    pred_depths = model(images)
                    batch_metrics = compute_all_metrics(pred_depths, depths, masks)
            else:
                pred_depths = model(images)
                batch_metrics = compute_all_metrics(pred_depths, depths, masks)

            # Accumulate metrics
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]

    # Compute averages
    num_batches = len(dataloader)
    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}

    return avg_metrics


def main() -> None:
    """Main evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate LightDepth model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML (optional)"
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    print("=" * 70)
    print("LightDepth Evaluation")
    print("=" * 70)
    print(f"Device: {device}")

    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mixed Precision (AMP): Enabled")
    else:
        print("Mixed Precision (AMP): Disabled (CPU mode)")

    print("=" * 70)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = LightDepthNet(pretrained=False).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Store checkpoint info
    checkpoint_info = {}

    # Load model state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        # Print checkpoint info if available
        if "epoch" in checkpoint:
            checkpoint_info["epoch"] = checkpoint["epoch"]
            print(f"Checkpoint epoch: {checkpoint["epoch"]}")
        if "val_rmse" in checkpoint:
            checkpoint_info["val_rmse"] = checkpoint["val_rmse"]
            print(f"Checkpoint validation RMSE: {checkpoint["val_rmse"]:.4f}")
    else:
        model.load_state_dict(checkpoint)

    total, trainable = model.count_parameters()
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # Load test data
    print("\nLoading test data...")
    _, test_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        img_size=(config.img_height, config.img_width),
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Total batches: {len(test_loader)}\n")

    # Evaluate with AMP if using CUDA
    metrics = evaluate(model, test_loader, device, use_amp=use_cuda)

    # Display results
    print(f"\n{"="*70}")
    print("Evaluation Results")
    print(f"{"="*70}")
    print("\nError Metrics (lower is better):")
    print(f"RMSE:       {metrics["rmse"]:.4f}")
    print(f"MAE:        {metrics["mae"]:.4f}")
    print(f"Abs Rel:    {metrics["abs_rel"]:.4f}")
    print(f"Sq Rel:     {metrics["sq_rel"]:.4f}")
    print(f"{"="*70}\n")


if __name__ == "__main__":
    main()
