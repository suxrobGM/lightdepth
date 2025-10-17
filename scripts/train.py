# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Training script for LightDepth model

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print

from lightdepth.models import LightDepthNet
from lightdepth.data import create_dataloaders
from lightdepth.utils import Config, DepthLoss, compute_rmse


def train_epoch(
    model: LightDepthNet, 
    dataloader: DataLoader, 
    criterion: DepthLoss, 
    optimizer: optim.Optimizer, 
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, depths, masks in pbar:
        images, depths, masks = images.to(device), depths.to(device), masks.to(device)

        # Forward
        pred_depths = model(images)
        loss: torch.Tensor = criterion(pred_depths, depths, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def validate(
    model: LightDepthNet, 
    dataloader: DataLoader, 
    criterion: DepthLoss, 
    device: torch.device
) -> tuple[float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0
    total_rmse = 0

    pbar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for images, depths, masks in pbar:
            images, depths, masks = images.to(device), depths.to(device), masks.to(device)

            # Forward
            pred_depths = model(images)
            loss: torch.Tensor = criterion(pred_depths, depths, masks)
            rmse = compute_rmse(pred_depths, depths, masks)

            total_loss += loss.item()
            total_rmse += rmse.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "rmse": f"{rmse.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    avg_rmse = total_rmse / len(dataloader)
    return avg_loss, avg_rmse


def main() -> None:
    """Main training loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Train LightDepth model")
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

    # Model
    print("Creating model...")
    model = LightDepthNet(pretrained=True).to(device)
    total, trainable = model.count_parameters()
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # Data
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        img_size=(config.img_height, config.img_width),
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Loss and optimizer
    criterion = DepthLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    print(f"\nTraining for {config.num_epochs} epochs...\n")
    best_rmse = float("inf")

    for epoch in range(1, config.num_epochs + 1):
        print(f"Epoch {epoch}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_rmse = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": val_rmse,
            }, "checkpoints/best_model.pth")
            print(f"Saved best model (RMSE: {val_rmse:.4f})")

        print()

    print(f"Training complete! Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
