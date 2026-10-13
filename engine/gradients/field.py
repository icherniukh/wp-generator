import random
from typing import Optional, Sequence
import numpy as np
from PIL import Image
import scipy.ndimage

from engine.colors import oklab_to_rgb
from engine.gradients.interpolation import (
    colors_to_oklab,
    generate_noise_plane,
    interpolate_color_curve,
)

try:
    import noise
except ImportError:
    noise = None


class DomainWarpGradient:
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
        """Generates a gradient using nested vector domain warping in Oklab space."""
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

        u = xv * np.cos(angle) + yv * np.sin(angle)
        v = -xv * np.sin(angle) + yv * np.cos(angle)

        base_freq = 0.5 + (complexity * 0.08)
        ox1, oy1 = random.uniform(0, 500), random.uniform(0, 500)
        ox2, oy2 = random.uniform(500, 1000), random.uniform(500, 1000)
        qx = generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox1, oy1))
        qy = generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox2, oy2))

        u_warped = u + warp_strength * qx
        ox3, oy3 = random.uniform(1000, 1500), random.uniform(1000, 1500)
        rx = generate_noise_plane(h, w, scale=base_freq * 1.3, octaves=2, offset=(ox3, oy3))

        t = u_warped + (warp_strength * 0.4) * rx
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
        t_values = np.clip(t, 0.0, 1.0)

        grid_oklab = interpolate_color_curve(t_values, colors_oklab)
        srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
        img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))
        return img_mod.resize((self.width, self.height), Image.LANCZOS)


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

        num_colors = len(colors)
        if num_colors == 0:
            raise ValueError("At least one color must be provided.")
        if num_colors == 1:
            c = tuple(int(v) for v in colors[0])
            return Image.new("RGB", (self.width, self.height), c)

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
        srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
        img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))
        return img_mod.resize((self.width, self.height), Image.LANCZOS)


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
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generates an organic fluid gradient by warping coordinates along curl-noise flow vectors."""
        if noise is None:
            raise ImportError("The 'noise' package is required for FlowFieldGradient.")

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        num_colors = len(colors)
        colors_oklab = colors_to_oklab(colors)
        color_order = list(range(num_colors))

        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        res_scale = 0.1
        small_h = max(2, int(self.height * res_scale))
        small_w = max(2, int(self.width * res_scale))

        ox = random.uniform(0, 1000)
        oy = random.uniform(0, 1000)

        warp_strength = random.uniform(0.2, 0.6)
        scale = random.uniform(0.002, 0.006) * complexity

        dx = np.zeros((small_h, small_w), dtype=np.float32)
        dy = np.zeros((small_h, small_w), dtype=np.float32)

        for sy in range(small_h):
            for sx in range(small_w):
                nx, ny = sx * scale, sy * scale
                dx[sy, sx] = noise.pnoise2(nx + ox, ny + oy, octaves=3)
                dy[sy, sx] = noise.pnoise2(nx + ox + 500, ny + oy + 500, octaves=3)

        dx = scipy.ndimage.zoom(dx, (self.height / small_h, self.width / small_w), order=1)
        dy = scipy.ndimage.zoom(dy, (self.height / small_h, self.width / small_w), order=1)

        if dx.shape != xv.shape:
            dx = np.pad(dx, ((0, max(0, xv.shape[0] - dx.shape[0])), (0, max(0, xv.shape[1] - dx.shape[1]))))[: xv.shape[0], : xv.shape[1]]
            dy = np.pad(dy, ((0, max(0, yv.shape[0] - dy.shape[0])), (0, max(0, yv.shape[1] - dy.shape[1]))))[: yv.shape[0], : yv.shape[1]]

        xv_warped = xv + dx * warp_strength
        yv_warped = yv + dy * warp_strength

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

        p = 2.0
        weights = 1.0 / (distances**p)
        weights /= np.sum(weights, axis=0)

        res = np.einsum("ihw,ic->hwc", weights, colors_oklab[color_order])
        srgb = np.clip(oklab_to_rgb(res), 0.0, 1.0)
        return Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))


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

        num_colors = len(colors)
        if num_colors == 0:
            raise ValueError("At least one color must be provided.")
        if num_colors == 1:
            c = tuple(int(v) for v in colors[0])
            return Image.new("RGB", (self.width, self.height), c)

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

        e_min, e_max = warped.min(), warped.max()
        t_values = (warped - e_min) / (e_max - e_min) if e_max > e_min else np.zeros_like(warped)
        t_values = np.clip(t_values, 0.0, 1.0)

        grid_oklab = interpolate_color_curve(t_values, colors_oklab)
        srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
        img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))
        return img_mod.resize((self.width, self.height), Image.LANCZOS)
