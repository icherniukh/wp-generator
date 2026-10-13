import random
from typing import Optional, Sequence, Tuple
import numpy as np
from PIL import Image
import scipy.ndimage

from engine.colors import oklab_to_rgb
from engine.gradients.interpolation import (
    colors_to_oklab,
    evaluate_decoupled_palette,
)


class FractalFieldGradient:
    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        fractal_type: str = "fluid",
        zoom: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
        angle: Optional[float] = None,
        phase_offset: float = 0.0,
        frequency: float = 1.0,
        trap_weight: float = 0.35,
        channel_offset: float = 0.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generates a continuous complex fractal field with dynamic radius scaling and focal archetypes."""
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

        if zoom is not None:
            eff_zoom = float(zoom)
        else:
            if fractal_type == "mandelbrot":
                eff_zoom = random.uniform(8.0, 35.0)
            else:
                radius_tier = random.choices(["macro", "medium", "detail"], weights=[0.45, 0.40, 0.15])[0]
                if radius_tier == "macro":
                    eff_zoom = random.uniform(0.40, 0.85)
                elif radius_tier == "medium":
                    eff_zoom = random.uniform(0.85, 1.75)
                else:
                    eff_zoom = random.uniform(1.75, 3.20)

        eff_angle = float(angle) if angle is not None else random.uniform(0.0, 2.0 * np.pi)

        if center is not None:
            cx, cy = float(center[0]), float(center[1])
        else:
            if fractal_type == "mandelbrot":
                mandel_points = [
                    (-0.7436438870371587, 0.1318259042053119),
                    (-0.1011, 0.9563),
                    (-0.75, 0.1),
                    (-0.748, 0.065),
                    (-1.25, 0.0),
                    (0.275, 0.0),
                ]
                cx, cy = random.choice(mandel_points)
            else:
                focal_archetype = random.choices(["offscreen", "corner", "centered"], weights=[0.45, 0.35, 0.20])[0]
                if focal_archetype == "offscreen":
                    theta_off = random.uniform(0.0, 2.0 * np.pi)
                    dist_off = random.uniform(1.4, 2.8) / eff_zoom
                    cx = dist_off * aspect * np.cos(theta_off)
                    cy = dist_off * np.sin(theta_off)
                elif focal_archetype == "corner":
                    sign_x = random.choice([-1.0, 1.0])
                    sign_y = random.choice([-1.0, 1.0])
                    cx = sign_x * random.uniform(0.40, 0.95) * aspect / eff_zoom
                    cy = sign_y * random.uniform(0.35, 0.90) / eff_zoom
                else:
                    cx = random.uniform(-0.35, 0.35) * aspect / eff_zoom
                    cy = random.uniform(-0.35, 0.35) / eff_zoom

        y = np.linspace(-1.3 / eff_zoom, 1.3 / eff_zoom, h, dtype=np.float32)
        x = np.linspace(-1.3 * aspect / eff_zoom, 1.3 * aspect / eff_zoom, w, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        dx = xv - cx
        dy = yv - cy
        rf = np.sqrt(dx**2 + dy**2)

        warp_amp = 0.18
        qx = np.sin(yv * 2.6 + 1.2) * np.cos(xv * 2.0)
        qy = np.cos(xv * 2.6 + 0.8) * np.sin(yv * 2.0)
        xv_warped = xv + warp_amp * qx
        yv_warped = yv + warp_amp * qy

        rot_x = (xv_warped - cx) * np.cos(eff_angle) - (yv_warped - cy) * np.sin(eff_angle)
        rot_y = (xv_warped - cx) * np.sin(eff_angle) + (yv_warped - cy) * np.cos(eff_angle)
        z_grid = rot_x + 1j * rot_y

        safe_complexity = max(1, int(complexity))
        eff_freq = max(0.1, float(frequency))
        eff_phase = float(phase_offset) % 1.0
        eff_trap_weight = np.clip(float(trap_weight), 0.0, 1.0)
        eff_chan_offset = float(channel_offset)

        if fractal_type == "fluid":
            def fbm_field(x_grid, y_grid, octaves=3, lacunarity=1.8, gain=0.45):
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

            q_x = fbm_field(rot_x, rot_y, octaves=2)
            q_y = fbm_field(rot_x + 3.8, rot_y + 1.5, octaves=2)
            r_x = fbm_field(rot_x + 1.8 * q_x + 1.4, rot_y + 1.8 * q_y + 2.8, octaves=2)
            r_y = fbm_field(rot_x + 1.8 * q_x + 4.5, rot_y + 1.8 * q_y + 1.6, octaves=2)

            radial_freq = random.uniform(0.35, 0.80)
            radial_carrier = 0.5 + 0.5 * np.cos(rf * radial_freq)

            sweep_angle = eff_angle + random.uniform(-0.5, 0.5)
            ambient_sweep = 0.5 * (np.cos(sweep_angle) * (xv / aspect) + np.sin(sweep_angle) * yv)

            fluid_folds = fbm_field(rot_x + 2.2 * r_x, rot_y + 2.2 * r_y, octaves=max(2, min(5, safe_complexity)))
            field = 0.30 * radial_carrier + 0.25 * ambient_sweep + 0.45 * fluid_folds

            f_min, f_max = field.min(), field.max()
            u_base = (field - f_min) / max(1e-5, (f_max - f_min))

            u = 0.5 - 0.5 * np.cos(np.pi * (u_base * eff_freq + eff_phase))
            u_L = 0.5 - 0.5 * np.cos(np.pi * (u_base * eff_freq + eff_phase + eff_chan_offset))

            grid_oklab = evaluate_decoupled_palette(colors_oklab, u_L, u, h, w)
            srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
            img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))
            return img_mod.resize((self.width, self.height), Image.LANCZOS)

        max_iter = max(16, 16 + safe_complexity * 3)
        escape_radius = 16.0
        ln_r = np.log(escape_radius)

        smooth_pot = np.zeros(z_grid.shape, dtype=np.float32)
        trap = np.full(z_grid.shape, 1e6, dtype=np.float32)
        mask = np.ones(z_grid.shape, dtype=bool)

        if fractal_type == "mandelbrot":
            z = np.zeros_like(z_grid)
            c = z_grid
        else:
            theta = random.uniform(0.0, 2.0 * np.pi)
            if random.random() < 0.3:
                c_bound = -1.0 + 0.25 * np.exp(1j * theta)
            else:
                c_bound = 0.5 * np.exp(1j * theta) - 0.25 * np.exp(2j * theta)
            eps = random.uniform(0.015, 0.04) * np.exp(1j * random.uniform(0.0, 2.0 * np.pi))
            c = c_bound + eps
            z = z_grid.copy()

        for i in range(max_iter):
            z[mask] = z[mask] ** 2 + (c[mask] if isinstance(c, np.ndarray) else c)
            abs_z = np.abs(z)

            curr_trap = np.minimum(np.abs(z.real), np.abs(z.imag)) + 0.15 * abs_z
            trap[mask] = np.minimum(trap[mask], curr_trap[mask])

            escaped = abs_z > escape_radius
            newly_escaped = escaped & mask
            if np.any(newly_escaped):
                smooth_pot[newly_escaped] = (
                    i + 1.0 - (np.log(np.log(abs_z[newly_escaped])) - np.log(ln_r)) / np.log(2.0)
                )
            mask = mask & (~escaped)
            if not np.any(mask):
                break

        if np.any(mask):
            final_abs = np.clip(np.abs(z[mask]), 0.0, escape_radius)
            smooth_pot[mask] = max_iter - np.log1p(escape_radius - final_abs)

        smooth_pot = scipy.ndimage.gaussian_filter(smooth_pot, sigma=1.8)
        trap = scipy.ndimage.gaussian_filter(trap, sigma=1.5)

        p_min, p_max = smooth_pot.min(), smooth_pot.max()
        pot_norm = (smooth_pot - p_min) / max(1e-5, (p_max - p_min))

        pot_smooth = 1.0 / (1.0 + np.exp(-4.0 * (pot_norm - 0.5)))
        pot_smooth = (pot_smooth - pot_smooth.min()) / max(1e-5, (pot_smooth.max() - pot_smooth.min()))

        trap_p90 = np.percentile(trap, 90)
        trap_norm = np.clip(trap / max(1e-5, trap_p90), 0.0, 1.0)
        trap_smooth = 1.0 - np.exp(-2.2 * trap_norm)

        ambient_carrier = 0.5 + 0.5 * (np.sin(rot_x * 0.5) * np.cos(rot_y * 0.5))

        u_field = (
            (1.0 - eff_trap_weight * 0.5) * (0.45 * ambient_carrier + 0.55 * pot_smooth)
            + eff_trap_weight * 0.5 * trap_smooth
        )

        u = 0.5 - 0.5 * np.cos(np.pi * (u_field * eff_freq + eff_phase))
        u_L = 0.5 - 0.5 * np.cos(np.pi * (u_field * eff_freq + eff_phase + eff_chan_offset))

        grid_oklab = evaluate_decoupled_palette(colors_oklab, u_L, u, h, w)
        srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
        img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))
        return img_mod.resize((self.width, self.height), Image.LANCZOS)
