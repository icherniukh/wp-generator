import random
from typing import List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image
import scipy.ndimage

from engine.gradients.interpolation import (
    colors_to_oklab,
    interpolate_color_curve,
    oklab_image_to_pil,
)
from engine.gradients.spatial import (
    generate_noise_plane,
    normalize_scalar_field,
    random_anchor_points,
)

try:
    import noise
except ImportError:
    noise = None


class FluidMeshGradient:
    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        fluid_mode: str = "random",
    ) -> Image.Image:
        """Generates a fluid mesh gradient using bicubic interpolation of an Oklab control grid."""
        aspect_ratio = self.width / self.height
        min_complexity = max(4, complexity)
        rows = max(2, int(np.sqrt(min_complexity / aspect_ratio))) * 2
        cols = max(2, int(rows * aspect_ratio)) * 2

        colors_arr = colors_to_oklab(colors)
        angle = random.uniform(0, 2 * np.pi)

        y_indices = np.linspace(-0.5, 0.5, rows, dtype=np.float32)
        x_indices = np.linspace(-0.5, 0.5, cols, dtype=np.float32)
        xv, yv = np.meshgrid(x_indices, y_indices)

        t_base = xv * np.cos(angle) + yv * np.sin(angle)
        fold = np.sin(xv * np.pi) * np.cos(yv * np.pi) * 0.25
        t_values = normalize_scalar_field(t_base + fold)

        grid_oklab = interpolate_color_curve(t_values, colors_arr)
        res = scipy.ndimage.zoom(
            grid_oklab, (self.height / rows, self.width / cols, 1), order=3, mode="nearest"
        )
        return oklab_image_to_pil(res)


class SimplexMeshGradient:
    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        fluid_mode: str = "random",
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generates continuous flowing waves evaluated over a continuous scalar field."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        colors_arr = colors_to_oklab(colors)

        scale_factor = 0.25
        h = max(16, int(self.height * scale_factor))
        w = max(16, int(self.width * scale_factor))

        x = np.linspace(-0.5, 0.5, w, dtype=np.float32)
        y = np.linspace(-0.5, 0.5, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        angle = random.uniform(0, 2 * np.pi)
        t_base = xv * np.cos(angle) + yv * np.sin(angle)

        if noise is not None:
            ox = random.uniform(0, 1000)
            oy = random.uniform(0, 1000)
            noise_scale = 0.5 * (1.0 + complexity * 0.2)
            t_noise = generate_noise_plane(h, w, scale=noise_scale, octaves=2, offset=(ox, oy))
            t_values = t_base * 0.6 + t_noise * 1.5
        else:
            fold = np.sin(xv * np.pi) * np.cos(yv * np.pi) * 0.5
            t_values = t_base + fold

        grid_oklab = interpolate_color_curve(normalize_scalar_field(t_values), colors_arr)
        return oklab_image_to_pil(grid_oklab, (self.width, self.height))


class MeshGradient:
    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        points: Optional[List[Tuple[float, float]]] = None,
        p: float = 2.0,
        softness: float = 0.12,
    ) -> Image.Image:
        """Generates a gradient using Inverse Distance Weighting (IDW) with softened anchors."""
        num_colors = len(colors)
        if points is None:
            points = random_anchor_points(num_colors)

        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        distances = np.zeros((num_colors, self.height, self.width), dtype=np.float32)
        for i, (px, py) in enumerate(points):
            distances[i] = np.sqrt((xv - px) ** 2 + (yv - py) ** 2)

        distances = np.maximum(distances, softness)
        weights = 1.0 / (distances**p)
        weights /= np.sum(weights, axis=0)

        colors_arr = np.asarray(colors, dtype=np.float32)
        res = np.einsum("ihw,ic->hwc", weights, colors_arr)
        return Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))
