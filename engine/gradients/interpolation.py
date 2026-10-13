import random
from typing import List, Sequence, Tuple
import numpy as np
from PIL import Image
import scipy.ndimage
from scipy.interpolate import PchipInterpolator

from engine.colors import oklab_to_rgb, rgb_to_oklab

try:
    import noise
except ImportError:
    noise = None


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


def oklab_image_to_pil(values: np.ndarray) -> Image.Image:
    """Converts an (H, W, 3) Oklab array into a PIL RGB Image."""
    srgb = np.clip(oklab_to_rgb(values), 0.0, 1.0)
    return Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))


def random_anchor_points(count: int, margin: float = 0.08) -> List[Tuple[float, float]]:
    """Generates normalized anchor coordinates placed within canvas margins."""
    return [
        (random.uniform(margin, 1.0 - margin), random.uniform(margin, 1.0 - margin))
        for _ in range(count)
    ]


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


def generate_noise_plane(
    h: int,
    w: int,
    scale: float = 1.0,
    octaves: int = 2,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Generates a continuous 2D scalar noise plane via Simplex noise or sinusoidal harmonic fallback."""
    if noise is None:
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32) * scale + offset[1]
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32) * scale + offset[0]
        xv, yv = np.meshgrid(x, y)
        return np.sin(xv * np.pi) * np.cos(yv * np.pi)

    sample_h = max(8, h // 4)
    sample_w = max(8, w // 4)
    y = np.linspace(-1.0, 1.0, sample_h, dtype=np.float32) * scale + offset[1]
    x = np.linspace(-1.0, 1.0, sample_w, dtype=np.float32) * scale + offset[0]
    grid = np.zeros((sample_h, sample_w), dtype=np.float32)
    for iy in range(sample_h):
        for ix in range(sample_w):
            grid[iy, ix] = noise.snoise2(x[ix], y[iy], octaves=octaves)
    zoom_y = h / sample_h
    zoom_x = w / sample_w
    return scipy.ndimage.zoom(grid, (zoom_y, zoom_x), order=3)[:h, :w]
