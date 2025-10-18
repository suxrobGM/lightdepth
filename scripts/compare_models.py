# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-17
# Description: Compare LightDepth model with Depth Anything V2 on NYU Depth v2 test set
# Parameters:
#   --lightdepth-checkpoint: Path to LightDepth model checkpoint
#   --config: Optional path to config YAML file. If not provided, defaults are used.
#   --dav2-model: Name of Depth Anything V2 model variant. Default: "depth-anything/Depth-Anything-V2-Small-hf"
#   --output: Path to save comparison results JSON. Default: "output/comparison_results.json"
#   --visualize: Number of samples to visualize (0 to disable). Default: 0

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
from datetime import datetime

import torch
import torchvision.transforms.functional as TF
from rich import print
from rich.console import Console
from rich.table import Table
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from lightdepth.data import create_dataloaders
from lightdepth.models import LightDepth
from lightdepth.utils import Config, compute_all_metrics, save_depth_map


def evaluate_lightdepth(
    model: LightDepth,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    """Evaluate LightDepth model."""
    model.eval()

    metrics_sum = {
        "rmse": 0.0,
        "mae": 0.0,
        "abs_rel": 0.0,
        "sq_rel": 0.0,
    }

    print("[cyan]Evaluating LightDepth...[/cyan]")
    with torch.no_grad():
        for images, depths, masks in tqdm(dataloader, desc="LightDepth"):
            images, depths, masks = (
                images.to(device, non_blocking=True),
                depths.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
            )

            if use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    pred_depths = model(images)
                    batch_metrics = compute_all_metrics(pred_depths, depths, masks)
            else:
                pred_depths = model(images)
                batch_metrics = compute_all_metrics(pred_depths, depths, masks)

            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]

    num_batches = len(dataloader)
    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}

    return avg_metrics


def evaluate_depth_anything_v2(
    model,
    image_processor,
    dataloader: DataLoader,
    device: torch.device,
    img_size: tuple[int, int],
) -> dict[str, float]:
    """Evaluate Depth Anything V2 model."""
    model.eval()

    metrics_sum = {
        "rmse": 0.0,
        "mae": 0.0,
        "abs_rel": 0.0,
        "sq_rel": 0.0,
    }

    print("[cyan]Evaluating Depth Anything V2...[/cyan]")
    with torch.no_grad():
        for images, depths, masks in tqdm(dataloader, desc="Depth Anything V2"):
            batch_size = images.shape[0]
            depths = depths.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Convert batch of tensors to list of PIL images for Depth Anything V2
            # Need to denormalize first since images are normalized with ImageNet stats
            pil_images = []
            for i in range(batch_size):
                img = images[i].cpu()
                denorm_img = denormalize_image(img)
                pil_img = TF.to_pil_image(denorm_img)
                pil_images.append(pil_img)

            # Process images through Depth Anything V2
            inputs = image_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # Post-process to get depth maps at original resolution
            target_sizes = [(img_size[0], img_size[1])] * batch_size
            post_processed = image_processor.post_process_depth_estimation(
                outputs, target_sizes=target_sizes
            )

            # Stack predicted depths into a batch tensor
            pred_depths = torch.stack(
                [p["predicted_depth"] for p in post_processed]
            ).unsqueeze(1)
            pred_depths = pred_depths.to(device)

            # Compute metrics
            batch_metrics = compute_all_metrics(pred_depths, depths, masks)

            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]

    num_batches = len(dataloader)
    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}

    return avg_metrics


def create_comparison_table(
    lightdepth_metrics: dict[str, float],
    dav2_metrics: dict[str, float],
    lightdepth_params: int,
    dav2_params: int,
) -> Table:
    """Create a rich table comparing both models."""
    table = Table(title="Model Comparison", show_header=True, header_style="bold cyan")

    table.add_column("Metric", style="white", width=20)
    table.add_column("LightDepth", justify="right", style="yellow")
    table.add_column("Depth Anything V2", justify="right", style="green")
    table.add_column("Winner", justify="center", style="bold")

    # Model parameters
    table.add_row(
        "Parameters",
        f"{lightdepth_params:,}",
        f"{dav2_params:,}",
        "LightDepth" if lightdepth_params < dav2_params else "DAv2",
    )

    table.add_section()

    # Error metrics (lower is better)
    error_metrics = ["rmse", "mae", "abs_rel", "sq_rel"]
    for metric in error_metrics:
        ld_val = lightdepth_metrics[metric]
        dav2_val = dav2_metrics[metric]
        winner = "LightDepth" if ld_val < dav2_val else "DAv2"
        improvement = abs((ld_val - dav2_val) / dav2_val * 100)

        table.add_row(
            metric.upper(),
            f"{ld_val:.4f}",
            f"{dav2_val:.4f}",
            f"{winner} ({improvement:.1f}%)",
        )

    table.add_section()
    return table


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize image tensor from ImageNet normalization.

    Args:
        tensor: Normalized image tensor (C, H, W)

    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)


def visualize_comparison(
    lightdepth_model: LightDepth,
    dav2_model,
    dav2_processor,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int,
    output_dir: Path,
    use_amp: bool = False,
    img_size: tuple[int, int] = (480, 640),
):
    """Generate side-by-side visualizations comparing both models."""

    output_dir.mkdir(exist_ok=True, parents=True)
    print(
        f"\n[cyan]Generating comparison visualizations for {num_samples} samples...[/cyan]"
    )

    lightdepth_model.eval()
    dav2_model.eval()
    vis_count = 0

    with torch.no_grad():
        for images, depths, masks in dataloader:
            batch_size = images.shape[0]

            for i in range(batch_size):
                if vis_count >= num_samples:
                    break

                # Get single image
                image = images[i : i + 1].to(device)
                depth_gt = depths[i : i + 1].to(device)

                # LightDepth prediction
                if use_amp:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        pred_lightdepth = lightdepth_model(image)
                else:
                    pred_lightdepth = lightdepth_model(image)

                # Denormalize image for visualization and Depth Anything V2
                denorm_img = denormalize_image(images[i].cpu())
                pil_img = TF.to_pil_image(denorm_img)

                # Depth Anything V2 prediction
                dav2_inputs = dav2_processor(images=pil_img, return_tensors="pt")
                dav2_inputs = {k: v.to(device) for k, v in dav2_inputs.items()}
                dav2_outputs = dav2_model(**dav2_inputs)
                dav2_processed = dav2_processor.post_process_depth_estimation(
                    dav2_outputs, target_sizes=[(img_size[0], img_size[1])]
                )
                pred_dav2 = (
                    dav2_processed[0]["predicted_depth"].unsqueeze(0).unsqueeze(0)
                )

                # Save input image
                img_path = output_dir / f"sample_{vis_count:04d}_input.png"
                pil_img.save(img_path)

                # Save ground truth
                gt_path = output_dir / f"sample_{vis_count:04d}_gt.png"
                save_depth_map(depth_gt.cpu(), str(gt_path), colormap="plasma")

                # Save LightDepth prediction
                ld_path = output_dir / f"sample_{vis_count:04d}_lightdepth.png"
                save_depth_map(pred_lightdepth.cpu(), str(ld_path), colormap="plasma")

                # Save Depth Anything V2 prediction
                dav2_path = output_dir / f"sample_{vis_count:04d}_dav2.png"
                save_depth_map(pred_dav2.cpu(), str(dav2_path), colormap="plasma")

                vis_count += 1

            if vis_count >= num_samples:
                break

    print(f"[green]Visualizations saved to: {output_dir.absolute()}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LightDepth with Depth Anything V2"
    )
    parser.add_argument(
        "--lightdepth-checkpoint",
        type=str,
        required=True,
        help="Path to LightDepth checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML (optional)"
    )
    parser.add_argument(
        "--dav2-model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Depth Anything V2 model name (default: Small variant)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/comparison_results.json",
        help="Path to save comparison results",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="Number of samples to visualize (0 to disable)",
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

    console = Console()
    console.print("=" * 70, style="cyan")
    console.print(
        "Model Comparison: LightDepth vs Depth Anything V2", style="bold cyan"
    )
    console.print("=" * 70, style="cyan")
    console.print(f"Device: {device}")

    if use_cuda:
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        console.print("Mixed Precision (AMP): Enabled")
    else:
        console.print("Mixed Precision (AMP): Disabled (CPU mode)")

    console.print("=" * 70, style="cyan")

    # Load LightDepth model
    print("\n[yellow]Loading LightDepth model...[/yellow]")
    lightdepth_model = LightDepth(pretrained=False).to(device)
    checkpoint = torch.load(args.lightdepth_checkpoint, map_location=device)

    checkpoint_info = {}

    if "model_state_dict" in checkpoint:
        lightdepth_model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            checkpoint_info["epoch"] = checkpoint["epoch"]
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if "val_rmse" in checkpoint:
            checkpoint_info["val_rmse"] = checkpoint["val_rmse"]
            print(f"Checkpoint validation RMSE: {checkpoint['val_rmse']:.4f}")
    else:
        lightdepth_model.load_state_dict(checkpoint)

    total_ld, trainable_ld = lightdepth_model.count_parameters()
    print(f"Parameters: {total_ld:,} total, {trainable_ld:,} trainable")

    # Load Depth Anything V2 model
    print(f"\n[yellow]Loading Depth Anything V2 ({args.dav2_model})...[/yellow]")
    dav2_processor = AutoImageProcessor.from_pretrained(args.dav2_model)
    dav2_model = AutoModelForDepthEstimation.from_pretrained(args.dav2_model).to(device)

    # Count DAv2 parameters
    total_dav2 = sum(p.numel() for p in dav2_model.parameters())
    trainable_dav2 = sum(p.numel() for p in dav2_model.parameters() if p.requires_grad)
    print(f"Parameters: {total_dav2:,} total, {trainable_dav2:,} trainable")

    # Load test data
    print("\n[yellow]Loading test data...[/yellow]")
    _, test_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        img_size=(config.img_height, config.img_width),
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Total batches: {len(test_loader)}\n")

    # Evaluate both models
    lightdepth_metrics = evaluate_lightdepth(
        lightdepth_model, test_loader, device, use_amp=use_cuda
    )

    dav2_metrics = evaluate_depth_anything_v2(
        dav2_model,
        dav2_processor,
        test_loader,
        device,
        (config.img_height, config.img_width),
    )

    # Display comparison table
    table = create_comparison_table(
        lightdepth_metrics, dav2_metrics, total_ld, total_dav2
    )
    console.print(table)

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "lightdepth": {
            "checkpoint": str(Path(args.lightdepth_checkpoint).absolute()),
            "checkpoint_info": checkpoint_info,
            "total_parameters": total_ld,
            "trainable_parameters": trainable_ld,
            "metrics": lightdepth_metrics,
        },
        "depth_anything_v2": {
            "model_name": args.dav2_model,
            "total_parameters": total_dav2,
            "trainable_parameters": trainable_dav2,
            "metrics": dav2_metrics,
        },
        "dataset": {
            "test_samples": len(test_loader.dataset),
            "batch_size": config.batch_size,
            "image_size": [config.img_height, config.img_width],
        },
        "device": str(device),
        "mixed_precision": use_cuda,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[green]Results saved to: {output_path}[/green]")

    # Generate visualizations if requested
    if args.visualize > 0:
        vis_dir = Path("output")
        visualize_comparison(
            lightdepth_model,
            dav2_model,
            dav2_processor,
            test_loader,
            device,
            args.visualize,
            vis_dir,
            use_amp=use_cuda,
            img_size=(config.img_height, config.img_width),
        )


if __name__ == "__main__":
    main()
