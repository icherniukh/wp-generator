import random
from typing import Any, List, Tuple
import numpy as np
import scipy.ndimage

from engine.gradients.noise import snoise2


def normalize_scalar_field(field: np.ndarray) -> np.ndarray:
    """Normalizes a scalar field to [0.0, 1.0] via min-max scaling."""
    f_min, f_max = field.min(), field.max()
    if f_max > f_min:
        return np.clip((field - f_min) / (f_max - f_min), 0.0, 1.0)
    return np.zeros_like(field, dtype=np.float32)


def random_anchor_points(count: int, margin: float = 0.08, rng: Any = random) -> List[Tuple[float, float]]:
    """Generates normalized anchor coordinates placed within canvas margins."""
    return [
        (rng.uniform(margin, 1.0 - margin), rng.uniform(margin, 1.0 - margin))
        for _ in range(count)
    ]


def generate_noise_plane(
    h: int,
    w: int,
    scale: float = 1.0,
    octaves: int = 2,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Generates a continuous 2D scalar noise plane by upsampling a coarse Simplex noise grid."""
    sample_h = max(8, h // 4)
    sample_w = max(8, w // 4)
    y = np.linspace(-1.0, 1.0, sample_h, dtype=np.float32) * scale + offset[1]
    x = np.linspace(-1.0, 1.0, sample_w, dtype=np.float32) * scale + offset[0]
    xv, yv = np.meshgrid(x, y)
    grid = snoise2(xv, yv, octaves=octaves)
    zoom_y = h / sample_h
    zoom_x = w / sample_w
    return scipy.ndimage.zoom(grid, (zoom_y, zoom_x), order=3)[:h, :w]


def warp_coordinates_perlin(
    width: int,
    height: int,
    complexity: int,
    warp_range: Tuple[float, float] = (0.2, 0.6),
    octaves: int = 3,
    rng: Any = random,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warps a normalized [0, 1]^2 coordinate grid using coarse-to-fine Perlin displacement."""
    from engine.stages.frame import full_resolution_grid
    from engine.stages.warp import perlin_displacement

    coords = perlin_displacement(
        full_resolution_grid(width=width, height=height),
        strength_range=warp_range,
        complexity=complexity,
        octaves=octaves,
        rng=rng,
    )
    return coords.x, coords.y


def trig_fbm_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    octaves: int = 3,
    lacunarity: float = 1.8,
    gain: float = 0.45,
    rng: Any = random,
) -> np.ndarray:
    """Evaluates multi-octave rotated trigonometric fBm over 2D coordinate arrays."""
    val = np.zeros_like(x_grid)
    freq = 0.65
    amp = 1.0
    for _ in range(octaves):
        theta = rng.uniform(0, np.pi)
        rx = x_grid * np.cos(theta) - y_grid * np.sin(theta)
        ry = x_grid * np.sin(theta) + y_grid * np.cos(theta)
        phase = rng.uniform(0, 2 * np.pi)
        val += amp * (np.sin(rx * freq + phase) * np.cos(ry * freq + phase * 0.7))
        freq *= lacunarity
        amp *= gain
    return val

