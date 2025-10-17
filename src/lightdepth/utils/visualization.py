# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Visualization utilities for depth maps

import numpy as np
from PIL import Image

# Colormaps for depth visualization
# Each colormap is a list of RGB tuples normalized to [0, 1]
COLORMAPS = {
    "gray": None,  # Special case for grayscale
    "plasma": [
        (0.050383, 0.029803, 0.527975),
        (0.186213, 0.018803, 0.587228),
        (0.287076, 0.010855, 0.627295),
        (0.381047, 0.001814, 0.653068),
        (0.471457, 0.005678, 0.659897),
        (0.558148, 0.051545, 0.641509),
        (0.640603, 0.124403, 0.596422),
        (0.716387, 0.214982, 0.524578),
        (0.784195, 0.319397, 0.430538),
        (0.843229, 0.433717, 0.326009),
        (0.893616, 0.554642, 0.224595),
        (0.935015, 0.679874, 0.142720),
        (0.967671, 0.807245, 0.091796),
        (0.992440, 0.934251, 0.131326),
    ],
    "viridis": [
        (0.267004, 0.004874, 0.329415),
        (0.282623, 0.140926, 0.457517),
        (0.253935, 0.265254, 0.529983),
        (0.206756, 0.371758, 0.553117),
        (0.163625, 0.471133, 0.558148),
        (0.127568, 0.566949, 0.550556),
        (0.134692, 0.658636, 0.517649),
        (0.266941, 0.748751, 0.440573),
        (0.477504, 0.821444, 0.318195),
        (0.741388, 0.873449, 0.149561),
        (0.993248, 0.906157, 0.143936),
    ],
    "magma": [
        (0.001462, 0.000466, 0.013866),
        (0.078815, 0.054184, 0.211667),
        (0.232077, 0.059889, 0.437695),
        (0.390384, 0.100379, 0.501864),
        (0.550287, 0.161158, 0.505719),
        (0.716387, 0.214982, 0.475855),
        (0.868793, 0.287728, 0.409303),
        (0.967671, 0.439703, 0.359828),
        (0.994738, 0.624470, 0.427397),
        (0.995891, 0.812325, 0.572470),
        (0.987053, 0.991438, 0.749504),
    ],
    "inferno": [
        (0.001462, 0.000466, 0.013866),
        (0.087411, 0.044556, 0.224813),
        (0.258234, 0.038571, 0.406485),
        (0.416331, 0.090203, 0.432943),
        (0.579304, 0.148039, 0.404411),
        (0.735683, 0.215906, 0.330245),
        (0.865006, 0.316822, 0.226055),
        (0.955552, 0.451577, 0.109545),
        (0.987622, 0.621785, 0.068213),
        (0.963394, 0.812325, 0.181692),
        (0.988362, 0.998364, 0.644924),
    ],
}


def apply_colormap(values: np.ndarray, colormap_name: str = "plasma") -> np.ndarray:
    """Apply a colormap to normalized values [0, 1].

    Args:
        values: Normalized array with values in [0, 1]
        colormap_name: Name of colormap to apply

    Returns:
        RGB image as uint8 array with shape (H, W, 3)
    """
    if colormap_name == "gray":
        # Grayscale
        gray = (values * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    if colormap_name not in COLORMAPS:
        print(f"[Warning] Unknown colormap '{colormap_name}', using 'plasma'")
        colormap_name = "plasma"

    # Get colormap
    cmap = COLORMAPS[colormap_name]
    if cmap is None:
        # Fallback to gray
        gray = (values * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    # Interpolate colors
    n_colors = len(cmap)
    indices = values * (n_colors - 1)
    lower_idx = np.floor(indices).astype(np.int32)
    upper_idx = np.ceil(indices).astype(np.int32)
    fraction = indices - lower_idx

    # Clip indices to valid range
    lower_idx = np.clip(lower_idx, 0, n_colors - 1)
    upper_idx = np.clip(upper_idx, 0, n_colors - 1)

    # Get colors
    lower_colors = np.array([cmap[i] for i in lower_idx.flat]).reshape(
        (*values.shape, 3)
    )
    upper_colors = np.array([cmap[i] for i in upper_idx.flat]).reshape(
        (*values.shape, 3)
    )

    # Interpolate
    fraction = fraction[..., np.newaxis]
    interpolated = lower_colors * (1 - fraction) + upper_colors * fraction

    # Convert to uint8
    rgb = (interpolated * 255).astype(np.uint8)
    return rgb


def save_depth_map(
    depth_map: np.ndarray, output_path: str, colormap: str = "plasma"
) -> None:
    """Save depth map as colored image using built-in colormaps.

    Args:
        depth_map: Depth map array
        output_path: Path to save the output
        colormap: Colormap name (plasma, viridis, magma, inferno, gray)
    """
    # Normalize depth to 0-1
    depth_norm = (depth_map - depth_map.min()) / (
        depth_map.max() - depth_map.min() + 1e-6
    )

    # Apply colormap
    colored_rgb = apply_colormap(depth_norm, colormap)

    # Save as RGB image
    Image.fromarray(colored_rgb, mode="RGB").save(output_path)
