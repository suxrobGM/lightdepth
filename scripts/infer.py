# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Inference script for LightDepth model on single images
# Parameters:
#   --checkpoint: Path to model checkpoint
#   --input: Path to input image
#   --output: Path to save output depth image Default: output/depth.png
#   --colormap: Colormap to apply (plasma, viridis, magma, inferno, gray) for depth visualization. Default: plasma

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

import numpy as np
import torch
from PIL import Image
from rich import print

from lightdepth.models import LightDepth
from lightdepth.utils import save_depth_map


def load_and_preprocess_image(
    image_path: str, img_size: tuple[int, int] = (480, 640)
) -> torch.Tensor:
    """Load and preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize (width, height for PIL)
    image = image.resize((img_size[1], img_size[0]))

    # Convert to numpy and transpose to CHW
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np.transpose(2, 0, 1)

    # Normalize using ImageNet statistics
    # These are the mean RGB values across millions of ImageNet images
    # [Red: 0.485, Green: 0.456, Blue: 0.406] - ImageNet channel means
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)

    # [Red: 0.229, Green: 0.224, Blue: 0.225] - ImageNet channel standard deviations
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image_np = (image_np - mean) / std

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_np).float().unsqueeze(0)
    return image_tensor


def main() -> None:
    """Main inference."""

    parser = argparse.ArgumentParser(description="Run inference with LightDepth")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--output", type=str, default="output/depth.png", help="Output depth image path"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="plasma",
        help="Colormap to apply (plasma, viridis, magma, inferno, gray). Use 'gray' for grayscale output.",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = LightDepth(pretrained=False).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully")

    # Load and preprocess image
    print(f"Processing image: {args.input}")
    image_tensor = load_and_preprocess_image(args.input)
    image_tensor = image_tensor.to(device)

    # Predict depth
    with torch.no_grad():
        depth_pred = model(image_tensor)

    # Convert to numpy
    depth_map = depth_pred.squeeze().cpu().numpy()

    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_depth_map(depth_map, str(output_path), colormap=args.colormap)
    print(f"Depth map saved to: {output_path} (colormap: {args.colormap})")

    # Also save raw numpy array
    numpy_path = output_path.with_suffix(".npy")
    np.save(numpy_path, depth_map)
    print(f"Raw depth saved to: {numpy_path}")

    print(f"\nDepth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")


if __name__ == "__main__":
    main()
