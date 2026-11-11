import random
from typing import Optional, Sequence
import numpy as np
from PIL import Image

from engine.gradients.interpolation import (
    colors_to_oklab,
    interpolate_color_curve,
    oklab_image_to_pil,
)
from engine.gradients.spatial import warp_coordinates_perlin
from engine.parameter_ranges import PARAMETER_RANGES


class SplineWaveGradient:
    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        warp_strength: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Directional sweep using Monotonic Cubic Spline (PCHIP) in Oklab space warped by curl noise."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        colors_oklab = colors_to_oklab(colors)
        warp_range = (
            (float(warp_strength), float(warp_strength))
            if warp_strength is not None
            else PARAMETER_RANGES.get_bounds("spline_warp")
        )
        xv_warped, yv_warped = warp_coordinates_perlin(
            self.width, self.height, complexity, warp_range=warp_range, octaves=2
        )

        angle = random.uniform(0, 2 * np.pi)
        t_values = xv_warped * np.cos(angle) + yv_warped * np.sin(angle)

        t_min, t_max = np.percentile(t_values, [1, 99])
        if t_max > t_min:
            t_values = (t_values - t_min) / (t_max - t_min)

        t_values = np.clip(t_values, 0.0, 1.0)
        t_values = t_values * t_values * (3.0 - 2.0 * t_values)

        return oklab_image_to_pil(interpolate_color_curve(t_values, colors_oklab))
