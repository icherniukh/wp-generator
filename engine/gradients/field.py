import random
from typing import Optional, Sequence
import numpy as np
from PIL import Image
import scipy.ndimage

from engine.gradients.interpolation import (
    colors_to_oklab,
    interpolate_color_curve,
    oklab_image_to_pil,
    validate_palette_colors,
)
from engine.gradients.spatial import (
    generate_noise_plane,
    normalize_scalar_field,
    warp_coordinates_perlin,
)
from engine.parameter_ranges import PARAMETER_RANGES


class DomainWarpGradient:
    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        warp_strength: Optional[float] = None,
        transverse_warp: Optional[float] = None,
        angle: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generates a gradient using nested 2D vector domain warping in Oklab space."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        colors_oklab = colors_to_oklab(colors)

        scale_factor = 0.25
        h = max(16, int(self.height * scale_factor))
        w = max(16, int(self.width * scale_factor))

        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        if angle is None:
            angle = random.uniform(0, 2 * np.pi)

        eff_warp = (
            float(warp_strength)
            if warp_strength is not None
            else float(PARAMETER_RANGES.sample("warp_strength"))
        )
        eff_transverse = (
            float(transverse_warp)
            if transverse_warp is not None
            else float(PARAMETER_RANGES.sample("transverse_warp"))
        )

        u = xv * np.cos(angle) + yv * np.sin(angle)
        v = -xv * np.sin(angle) + yv * np.cos(angle)

        base_freq = 0.5 + (complexity * 0.08)
        ox1, oy1 = random.uniform(0, 500), random.uniform(0, 500)
        ox2, oy2 = random.uniform(500, 1000), random.uniform(500, 1000)
        qx = generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox1, oy1))
        qy = generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox2, oy2))

        u_warped = u + eff_warp * qx
        v_warped = v + eff_transverse * qy

        ox3, oy3 = random.uniform(1000, 1500), random.uniform(1000, 1500)
        rx = generate_noise_plane(h, w, scale=base_freq * 1.3, octaves=2, offset=(ox3, oy3))

        if eff_transverse > 0.0:
            grid_y, grid_x = np.meshgrid(
                np.arange(h, dtype=np.float32),
                np.arange(w, dtype=np.float32),
                indexing="ij",
            )
            warped_coords = [
                grid_y + (eff_transverse * 0.45 * qy) * h,
                grid_x + (eff_transverse * 0.45 * qx) * w,
            ]
            rx_warped = scipy.ndimage.map_coordinates(
                rx, warped_coords, order=3, mode="reflect"
            )
            transverse_bend = eff_transverse * 0.35 * np.sin(v_warped * 1.5)
        else:
            rx_warped = rx
            transverse_bend = 0.0

        t_values = normalize_scalar_field(
            u_warped + transverse_bend + (eff_warp * 0.35) * rx_warped
        )
        grid_oklab = interpolate_color_curve(t_values, colors_oklab)
        return oklab_image_to_pil(grid_oklab, (self.width, self.height))


class RadialBasisGradient:
    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        warp_strength: float = 1.0,
        ambient_weight: float = 0.15,
        dispersion: float = 0.35,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generates a 2D color field using radial basis emitters and coordinate domain warping."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        solid = validate_palette_colors(colors, self.width, self.height)
        if solid is not None:
            return solid

        num_colors = len(colors)
        colors_oklab = colors_to_oklab(colors)

        scale_factor = 0.25
        h = max(16, int(self.height * scale_factor))
        w = max(16, int(self.width * scale_factor))

        y = np.linspace(0.0, 1.0, h, dtype=np.float32)
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        effective_warp = max(0.0, float(warp_strength))
        if effective_warp > 0.0:
            base_freq = 0.6 + (max(1, complexity) * 0.08)
            ox1, oy1 = random.uniform(0, 500), random.uniform(0, 500)
            ox2, oy2 = random.uniform(500, 1000), random.uniform(500, 1000)
            qx = generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox1, oy1))
            qy = generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox2, oy2))
            warp_amp = effective_warp * 0.35
            xv = xv + warp_amp * qx
            yv = yv + warp_amp * qy

        weights = np.zeros((num_colors, h, w), dtype=np.float32)
        weights[0] = max(0.01, float(ambient_weight))

        num_emitters = max(num_colors - 1, max(1, complexity))
        for i in range(num_emitters):
            c_idx = 1 + (i % (num_colors - 1))
            angle_pos = i * (2.0 * np.pi / num_emitters) + random.uniform(-0.35, 0.35)
            dist_center = random.uniform(0.18, 0.44)
            cx = float(np.clip(0.5 + dist_center * np.cos(angle_pos), 0.1, 0.9))
            cy = float(np.clip(0.5 + dist_center * np.sin(angle_pos), 0.1, 0.9))
            sigma = max(0.05, float(dispersion) * random.uniform(0.75, 1.25))
            aspect = random.uniform(0.7, 1.4)
            flow_angle = random.uniform(0.0, np.pi)

            dx = xv - cx
            dy = yv - cy
            dx_rot = dx * np.cos(flow_angle) + dy * np.sin(flow_angle)
            dy_rot = -dx * np.sin(flow_angle) + dy * np.cos(flow_angle)
            dist = np.sqrt(dx_rot**2 + (dy_rot * aspect) ** 2)

            w_norm = np.clip(1.0 - (dist / (sigma * 1.8)), 0.0, 1.0)
            w_emit = (w_norm**2) * (3.0 - 2.0 * w_norm)
            weights[c_idx] += w_emit

        weight_sum = np.sum(weights, axis=0, keepdims=True)
        weights /= np.maximum(weight_sum, 1e-6)

        grid_oklab = np.einsum("nhw,nc->hwc", weights, colors_oklab)
        return oklab_image_to_pil(grid_oklab, (self.width, self.height))


RBFFieldGradient = RadialBasisGradient


class FlowFieldGradient:
    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        fluid_mode: str = "random",
        warp_strength: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generates an organic fluid gradient by warping coordinates along curl-noise flow vectors."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        num_colors = len(colors)
        colors_oklab = colors_to_oklab(colors)

        warp_range = (
            (float(warp_strength), float(warp_strength))
            if warp_strength is not None
            else PARAMETER_RANGES.get_bounds("flow_warp")
        )
        xv_warped, yv_warped = warp_coordinates_perlin(
            self.width, self.height, complexity, warp_range=warp_range, octaves=3
        )

        points = []
        for i in range(num_colors):
            angle = i * (2 * np.pi / num_colors) + random.uniform(0, np.pi)
            dist = random.uniform(0.5, 1.2)
            px = 0.5 + np.cos(angle) * dist
            py = 0.5 + np.sin(angle) * dist
            points.append((px, py))

        distances = np.zeros((num_colors, self.height, self.width), dtype=np.float32)
        for i, (px, py) in enumerate(points):
            distances[i] = np.sqrt((xv_warped - px) ** 2 + (yv_warped - py) ** 2)

        distances[0] *= 0.6
        distances = np.maximum(distances, 0.15)

        weights = 1.0 / (distances**2.0)
        weights /= np.sum(weights, axis=0)

        res = np.einsum("ihw,ic->hwc", weights, colors_oklab)
        return oklab_image_to_pil(res)


class TopographicGradient:
    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        warp_strength: float = 0.85,
        angle: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generates continuous elevation hypsometric contours mapped through Oklab space."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        solid = validate_palette_colors(colors, self.width, self.height)
        if solid is not None:
            return solid

        colors_oklab = colors_to_oklab(colors)

        scale_factor = 0.25
        h = max(16, int(self.height * scale_factor))
        w = max(16, int(self.width * scale_factor))

        aspect = w / h
        y = np.linspace(-1.5, 1.5, h, dtype=np.float32)
        x = np.linspace(-1.5 * aspect, 1.5 * aspect, w, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        if angle is None:
            angle = random.uniform(0, 2 * np.pi)

        u = xv * np.cos(angle) + yv * np.sin(angle)
        v = -xv * np.sin(angle) + yv * np.cos(angle)

        base_freq = 0.7 + (complexity * 0.1)
        h1 = np.sin(u * base_freq * 1.4) * np.cos(v * base_freq * 1.1)
        h2 = np.sin((u + v) * base_freq * 2.1 + 1.1) * 0.5
        h3 = np.cos(np.sqrt(u**2 + v**2) * base_freq * 2.8) * 0.25
        raw_elevation = h1 + h2 + h3

        effective_warp = max(0.0, float(warp_strength))
        if effective_warp > 0.0:
            ox = random.uniform(0, 500)
            oy = random.uniform(500, 1000)
            warp_plane = generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox, oy))
            warped = raw_elevation + (effective_warp * 0.6) * warp_plane
        else:
            warped = raw_elevation

        grid_oklab = interpolate_color_curve(normalize_scalar_field(warped), colors_oklab)
        return oklab_image_to_pil(grid_oklab, (self.width, self.height))

