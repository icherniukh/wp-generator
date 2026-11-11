from typing import Optional, Sequence, Tuple
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator

from engine.colors import oklab_to_rgb, rgb_to_oklab
from engine.gradients.spatial import (
    generate_noise_plane,
    normalize_scalar_field,
    random_anchor_points,
)


def colors_to_oklab(colors: Sequence[Sequence[float]]) -> np.ndarray:
    """Converts a sequence of sRGB colors (0-255) into an Oklab ndarray."""
    return rgb_to_oklab(np.asarray(colors, dtype=np.float32))


def interpolate_oklab(
    c1: Sequence[float], c2: Sequence[float], alpha: float
) -> np.ndarray:
    """Interpolates two sRGB colors through perceptually uniform Oklab space."""
    lab1 = rgb_to_oklab(c1)
    lab2 = rgb_to_oklab(c2)
    mixed = lab1 * (1.0 - alpha) + lab2 * alpha
    return np.clip(np.rint(oklab_to_rgb(mixed) * 255.0), 0, 255).astype(np.uint8)


def oklab_image_to_pil(
    values: np.ndarray, size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """Converts an (H, W, 3) Oklab array into a PIL RGB Image, optionally upscaling to size."""
    srgb = np.clip(oklab_to_rgb(values), 0.0, 1.0)
    image = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))
    if size is not None and image.size != size:
        return image.resize(size, Image.LANCZOS)
    return image


def validate_palette_colors(
    colors: Sequence[Sequence[float]], width: int, height: int
) -> Optional[Image.Image]:
    """Validates non-empty palette and returns a solid image if only one color is given."""
    if len(colors) == 0:
        raise ValueError("At least one color must be provided.")
    if len(colors) == 1:
        c = tuple(int(v) for v in colors[0])
        return Image.new("RGB", (width, height), c)  # type: ignore[arg-type]
    return None


def interpolate_color_curve(
    t_values: np.ndarray, colors_oklab: np.ndarray
) -> np.ndarray:
    """Maps a normalized scalar field in [0, 1] through a PCHIP spline in Oklab space."""
    num_colors = len(colors_oklab)
    nodes = np.linspace(0.0, 1.0, num_colors)
    interpolator = PchipInterpolator(nodes, colors_oklab, axis=0)

    original_shape = t_values.shape
    flat_t = np.clip(t_values.reshape(-1), 0.0, 1.0)
    flat_colors = interpolator(flat_t)
    return flat_colors.reshape((*original_shape, 3)).astype(np.float32)


def evaluate_decoupled_palette(
    colors_oklab: np.ndarray,
    u_lightness: np.ndarray,
    u_chroma: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Evaluates decoupled Oklab Lightness and Chroma/Hue scalar fields along color curves."""
    num_colors = len(colors_oklab)
    nodes = np.linspace(0.0, 1.0, num_colors)
    interp_L = PchipInterpolator(nodes, colors_oklab[:, 0])
    interp_ab = PchipInterpolator(nodes, colors_oklab[:, 1:3], axis=0)

    flat_L = interp_L(np.clip(u_lightness.reshape(-1), 0.0, 1.0))
    flat_ab = interp_ab(np.clip(u_chroma.reshape(-1), 0.0, 1.0))
    return np.column_stack([flat_L, flat_ab]).reshape((height, width, 3)).astype(np.float32)


__all__ = [
    "colors_to_oklab",
    "evaluate_decoupled_palette",
    "generate_noise_plane",
    "interpolate_color_curve",
    "interpolate_oklab",
    "normalize_scalar_field",
    "oklab_image_to_pil",
    "random_anchor_points",
    "validate_palette_colors",
]
