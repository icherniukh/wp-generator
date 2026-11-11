import random
from typing import List, Tuple
import numpy as np
import scipy.ndimage

try:
    import noise
except ImportError:
    noise = None


def normalize_scalar_field(field: np.ndarray) -> np.ndarray:
    """Normalizes a scalar field to [0.0, 1.0] via min-max scaling."""
    f_min, f_max = field.min(), field.max()
    if f_max > f_min:
        return np.clip((field - f_min) / (f_max - f_min), 0.0, 1.0)
    return np.zeros_like(field, dtype=np.float32)


def random_anchor_points(count: int, margin: float = 0.08) -> List[Tuple[float, float]]:
    """Generates normalized anchor coordinates placed within canvas margins."""
    return [
        (random.uniform(margin, 1.0 - margin), random.uniform(margin, 1.0 - margin))
        for _ in range(count)
    ]


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


def warp_coordinates_perlin(
    width: int,
    height: int,
    complexity: int,
    warp_range: Tuple[float, float] = (0.2, 0.6),
    octaves: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warps a normalized [0, 1]^2 coordinate grid using coarse-to-fine Perlin displacement."""
    if noise is None:
        raise ImportError("The 'noise' package is required.")

    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    small_h = max(2, int(height * 0.1))
    small_w = max(2, int(width * 0.1))

    ox = random.uniform(0, 1000)
    oy = random.uniform(0, 1000)
    warp_strength = random.uniform(*warp_range)
    scale = random.uniform(0.002, 0.006) * complexity

    dx = np.zeros((small_h, small_w), dtype=np.float32)
    dy = np.zeros((small_h, small_w), dtype=np.float32)
    for sy in range(small_h):
        for sx in range(small_w):
            nx, ny = sx * scale, sy * scale
            dx[sy, sx] = noise.pnoise2(nx + ox, ny + oy, octaves=octaves)
            dy[sy, sx] = noise.pnoise2(nx + ox + 500, ny + oy + 500, octaves=octaves)

    dx = scipy.ndimage.zoom(dx, (height / small_h, width / small_w), order=1)
    dy = scipy.ndimage.zoom(dy, (height / small_h, width / small_w), order=1)
    if dx.shape != xv.shape:
        pad_h = max(0, xv.shape[0] - dx.shape[0])
        pad_w = max(0, xv.shape[1] - dx.shape[1])
        dx = np.pad(dx, ((0, pad_h), (0, pad_w)))[: xv.shape[0], : xv.shape[1]]
        dy = np.pad(dy, ((0, pad_h), (0, pad_w)))[: yv.shape[0], : yv.shape[1]]

    return xv + dx * warp_strength, yv + dy * warp_strength


def trig_fbm_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    octaves: int = 3,
    lacunarity: float = 1.8,
    gain: float = 0.45,
) -> np.ndarray:
    """Evaluates multi-octave rotated trigonometric fBm over 2D coordinate arrays."""
    val = np.zeros_like(x_grid)
    freq = 0.65
    amp = 1.0
    for _ in range(octaves):
        theta = random.uniform(0, np.pi)
        rx = x_grid * np.cos(theta) - y_grid * np.sin(theta)
        ry = x_grid * np.sin(theta) + y_grid * np.cos(theta)
        phase = random.uniform(0, 2 * np.pi)
        val += amp * (np.sin(rx * freq + phase) * np.cos(ry * freq + phase * 0.7))
        freq *= lacunarity
        amp *= gain
    return val

