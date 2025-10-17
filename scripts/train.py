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
from rich import print
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightdepth.data import create_dataloaders
from lightdepth.models import LightDepthNet
from lightdepth.utils import Config, DepthLoss, compute_rmse


def train_epoch(
    model: LightDepthNet,
    dataloader: DataLoader,
    criterion: DepthLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> float:
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, depths, masks in pbar:
        images, depths, masks = (
            images.to(device, non_blocking=True),
            depths.to(device, non_blocking=True),
            masks.to(device, non_blocking=True),
        )

        # Forward with mixed precision
        if scaler is not None:
            with autocast(device_type="cuda", dtype=torch.float16):
                pred_depths = model(images)
                loss: torch.Tensor = criterion(pred_depths, depths, masks)

            # Backward with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward/backward
            pred_depths = model(images)
            loss: torch.Tensor = criterion(pred_depths, depths, masks)

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
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float]:
    """Validate model with optional mixed precision."""
    model.eval()
    total_loss = 0
    total_rmse = 0

    pbar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for images, depths, masks in pbar:
            images, depths, masks = (
                images.to(device, non_blocking=True),
                depths.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
            )

            # Forward with optional mixed precision
            if use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    pred_depths = model(images)
                    loss: torch.Tensor = criterion(pred_depths, depths, masks)
                    rmse = compute_rmse(pred_depths, depths, masks)
            else:
                pred_depths = model(images)
                loss: torch.Tensor = criterion(pred_depths, depths, masks)
                rmse = compute_rmse(pred_depths, depths, masks)

            total_loss += loss.item()
            total_rmse += rmse.item()
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "rmse": f"{rmse.item():.4f}"}
            )

    avg_loss = total_loss / len(dataloader)
    avg_rmse = total_rmse / len(dataloader)
    return avg_loss, avg_rmse


def main() -> None:
    """Main training loop with mixed precision, resume capability, and detailed logging."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Train LightDepth model")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML (optional)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load config from YAML or use defaults
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override resume path if provided via command line
    if args.resume:
        config.resume_from = args.resume

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    print("=" * 70)
    print("LightDepth Training Configuration")
    print("=" * 70)
    print(f"Device: {device}")

    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Mixed Precision (AMP): Enabled")
    else:
        print("Mixed Precision (AMP): Disabled (CPU mode)")

    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Num Workers: {config.num_workers}")
    print(f"Image Size: {config.img_height}x{config.img_width}")
    print(f"Data Root: {config.data_root}")
    print("=" * 70)

    # Model
    print("\nCreating model...")
    model = LightDepthNet(pretrained=True).to(device)
    total, trainable = model.count_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Data
    print("\nLoading datasets...")
    train_loader, val_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        img_size=(config.img_height, config.img_width),
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Total training samples: {len(train_loader) * config.batch_size}")
    print(f"Total validation samples: {len(val_loader) * config.batch_size}")

    # Loss and optimizer
    criterion = DepthLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler (cosine annealing with warmup)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    # Mixed precision setup
    scaler = GradScaler() if use_cuda else None
    if scaler:
        print("Using automatic mixed precision (AMP) for faster training")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_rmse = float("inf")

    if config.resume_from:
        print(f"\nResuming from checkpoint: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_rmse = checkpoint.get("val_rmse", float("inf"))

        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state if available and using AMP
        if scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best RMSE so far: {best_rmse:.4f}")
        print(f"Starting from epoch {start_epoch}")

    # Training loop
    print("\n" + "=" * 70)
    print(f"Starting Training (Epochs {start_epoch}-{config.num_epochs})")
    print("=" * 70 + "\n")

    total_training_time = 0

    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()

        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*70}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Train
        train_start = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        train_time = time.time() - train_start

        print(f"[TRAIN] Loss: {train_loss:.4f} | Time: {train_time:.1f}s")

        # Validate
        val_start = time.time()
        val_loss, val_rmse = validate(
            model, val_loader, criterion, device, use_amp=use_cuda
        )
        val_time = time.time() - val_start

        print(
            f"[VAL] Loss: {val_loss:.4f} | RMSE: {val_rmse:.4f} | Time: {val_time:.1f}s"
        )

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_rmse < best_rmse:
            improvement = best_rmse - val_rmse
            best_rmse = val_rmse
            Path("checkpoints").mkdir(exist_ok=True)

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_rmse": val_rmse,
                "val_loss": val_loss,
                "train_loss": train_loss,
            }

            if scaler:
                checkpoint_data["scaler_state_dict"] = scaler.state_dict()

            torch.save(checkpoint_data, "checkpoints/best_model.pth")
            print(
                f"New best model, RMSE improved by {improvement:.4f} (now {val_rmse:.4f})"
            )

        # Save latest checkpoint every 5 epochs
        if epoch % 5 == 0:
            Path("checkpoints").mkdir(exist_ok=True)
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_rmse": val_rmse,
                "val_loss": val_loss,
                "train_loss": train_loss,
            }

            if scaler:
                checkpoint_data["scaler_state_dict"] = scaler.state_dict()

            torch.save(checkpoint_data, f"checkpoints/checkpoint_epoch_{epoch}.pth")
            print(f"Checkpoint saved: checkpoint_epoch_{epoch}.pth")

        epoch_time = time.time() - epoch_start
        total_training_time += epoch_time

        print(
            f"\nEpoch Time: {epoch_time:.1f}s | Total Training Time: {total_training_time/60:.1f}min"
        )
        print(f"Best RMSE: {best_rmse:.4f}")

    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best Validation RMSE: {best_rmse:.4f}")
    print(
        f"Total Training Time: {total_training_time/60:.1f} minutes ({total_training_time/3600:.2f} hours)"
    )
    print(f"Best model saved at: checkpoints/best_model.pth")
    print("=" * 70)


if __name__ == "__main__":
    main()
