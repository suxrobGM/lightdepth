# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Evaluation script for LightDepth model

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print

from lightdepth.models import LightDepthNet
from lightdepth.data import create_dataloaders
from lightdepth.utils import Config, compute_rmse


def evaluate(model: LightDepthNet, dataloader: DataLoader, device: torch.device):
    """Evaluate model on test set."""
    model.eval()
    total_rmse = 0

    print("Evaluating...")
    with torch.no_grad():
        for images, depths, masks in tqdm(dataloader):
            images, depths, masks = images.to(device), depths.to(device), masks.to(device)

            # Forward
            pred_depths = model(images)
            rmse = compute_rmse(pred_depths, depths, masks)
            total_rmse += rmse.item()

    avg_rmse = total_rmse / len(dataloader)
    return avg_rmse


def main():
    """Main evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LightDepth model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (optional)")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = LightDepthNet(pretrained=False).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    total, trainable = model.count_parameters()
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # Load test data
    print("Loading test data...")
    _, test_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        img_size=(config.img_height, config.img_width),
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate
    test_rmse = evaluate(model, test_loader, device)

    print(f"\n{"="*50}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"{"="*50}\n")


if __name__ == "__main__":
    main()
