import random
from typing import Optional, Sequence, Tuple
import numpy as np
from PIL import Image
import scipy.ndimage

from engine.gradients.interpolation import (
    colors_to_oklab,
    evaluate_decoupled_palette,
    oklab_image_to_pil,
    validate_palette_colors,
)
from engine.gradients.spatial import normalize_scalar_field, trig_fbm_field

_MANDELBROT_FOCAL_POINTS = [
    (-0.7436438870371587, 0.1318259042053119),
    (-0.1011, 0.9563),
    (-0.75, 0.1),
    (-0.748, 0.065),
    (-1.25, 0.0),
    (0.275, 0.0),
]


class FractalFieldGradient:
    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    @staticmethod
    def _resolve_camera(
        fractal_type: str,
        zoom: Optional[float],
        center: Optional[Tuple[float, float]],
        angle: Optional[float],
        aspect: float,
    ) -> Tuple[float, float, float, float]:
        if zoom is not None:
            eff_zoom = float(zoom)
        elif fractal_type == "mandelbrot":
            eff_zoom = random.uniform(8.0, 35.0)
        else:
            radius_tier = random.choices(
                ["macro", "medium", "detail"], weights=[0.45, 0.40, 0.15]
            )[0]
            zoom_ranges = {"macro": (0.40, 0.85), "medium": (0.85, 1.75), "detail": (1.75, 3.20)}
            eff_zoom = random.uniform(*zoom_ranges[radius_tier])

        eff_angle = float(angle) if angle is not None else random.uniform(0.0, 2.0 * np.pi)

        if center is not None:
            cx, cy = float(center[0]), float(center[1])
        elif fractal_type == "mandelbrot":
            cx, cy = random.choice(_MANDELBROT_FOCAL_POINTS)
        else:
            focal_archetype = random.choices(
                ["offscreen", "corner", "centered"], weights=[0.45, 0.35, 0.20]
            )[0]
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

        return eff_zoom, eff_angle, cx, cy

    @staticmethod
    def _compute_fluid_field(
        xv: np.ndarray,
        yv: np.ndarray,
        rot_x: np.ndarray,
        rot_y: np.ndarray,
        rf: np.ndarray,
        eff_angle: float,
        aspect: float,
        safe_complexity: int,
    ) -> np.ndarray:
        q_x = trig_fbm_field(rot_x, rot_y, octaves=2)
        q_y = trig_fbm_field(rot_x + 3.8, rot_y + 1.5, octaves=2)
        r_x = trig_fbm_field(rot_x + 1.8 * q_x + 1.4, rot_y + 1.8 * q_y + 2.8, octaves=2)
        r_y = trig_fbm_field(rot_x + 1.8 * q_x + 4.5, rot_y + 1.8 * q_y + 1.6, octaves=2)

        radial_freq = random.uniform(0.35, 0.80)
        radial_carrier = 0.5 + 0.5 * np.cos(rf * radial_freq)

        sweep_angle = eff_angle + random.uniform(-0.5, 0.5)
        ambient_sweep = 0.5 * (np.cos(sweep_angle) * (xv / aspect) + np.sin(sweep_angle) * yv)

        fluid_folds = trig_fbm_field(
            rot_x + 2.2 * r_x, rot_y + 2.2 * r_y, octaves=max(2, min(5, safe_complexity))
        )
        field = 0.30 * radial_carrier + 0.25 * ambient_sweep + 0.45 * fluid_folds
        return normalize_scalar_field(field)

    @staticmethod
    def _compute_escape_time_field(
        rot_x: np.ndarray,
        rot_y: np.ndarray,
        fractal_type: str,
        safe_complexity: int,
        eff_trap_weight: float,
    ) -> np.ndarray:
        z_grid = rot_x + 1j * rot_y
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

        pot_norm = normalize_scalar_field(smooth_pot)
        pot_smooth = normalize_scalar_field(1.0 / (1.0 + np.exp(-4.0 * (pot_norm - 0.5))))

        trap_p90 = np.percentile(trap, 90)
        trap_norm = np.clip(trap / max(1e-5, trap_p90), 0.0, 1.0)
        trap_smooth = 1.0 - np.exp(-2.2 * trap_norm)

        ambient_carrier = 0.5 + 0.5 * (np.sin(rot_x * 0.5) * np.cos(rot_y * 0.5))

        return (
            (1.0 - eff_trap_weight * 0.5) * (0.45 * ambient_carrier + 0.55 * pot_smooth)
            + eff_trap_weight * 0.5 * trap_smooth
        )

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

        solid = validate_palette_colors(colors, self.width, self.height)
        if solid is not None:
            return solid

        colors_oklab = colors_to_oklab(colors)

        scale_factor = 0.25
        h = max(16, int(self.height * scale_factor))
        w = max(16, int(self.width * scale_factor))
        aspect = w / h

        eff_zoom, eff_angle, cx, cy = self._resolve_camera(
            fractal_type, zoom, center, angle, aspect
        )

        y = np.linspace(-1.3 / eff_zoom, 1.3 / eff_zoom, h, dtype=np.float32)
        x = np.linspace(-1.3 * aspect / eff_zoom, 1.3 * aspect / eff_zoom, w, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        rf = np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2)

        warp_amp = 0.18
        xv_warped = xv + warp_amp * (np.sin(yv * 2.6 + 1.2) * np.cos(xv * 2.0))
        yv_warped = yv + warp_amp * (np.cos(xv * 2.6 + 0.8) * np.sin(yv * 2.0))

        rot_x = (xv_warped - cx) * np.cos(eff_angle) - (yv_warped - cy) * np.sin(eff_angle)
        rot_y = (xv_warped - cx) * np.sin(eff_angle) + (yv_warped - cy) * np.cos(eff_angle)

        safe_complexity = max(1, int(complexity))
        eff_freq = max(0.1, float(frequency))
        eff_phase = float(phase_offset) % 1.0
        eff_trap_weight = np.clip(float(trap_weight), 0.0, 1.0)
        eff_chan_offset = float(channel_offset)

        if fractal_type == "fluid":
            u_field = self._compute_fluid_field(
                xv, yv, rot_x, rot_y, rf, eff_angle, aspect, safe_complexity
            )
        else:
            u_field = self._compute_escape_time_field(
                rot_x, rot_y, fractal_type, safe_complexity, eff_trap_weight
            )

        u = 0.5 - 0.5 * np.cos(np.pi * (u_field * eff_freq + eff_phase))
        u_L = 0.5 - 0.5 * np.cos(np.pi * (u_field * eff_freq + eff_phase + eff_chan_offset))

        grid_oklab = evaluate_decoupled_palette(colors_oklab, u_L, u, h, w)
        return oklab_image_to_pil(grid_oklab, (self.width, self.height))


class FluidFractalGradient(FractalFieldGradient):
    """Dedicated FBM domain-warped fluid fractal engine."""

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        zoom: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
        angle: Optional[float] = None,
        phase_offset: float = 0.0,
        frequency: float = 1.0,
        channel_offset: float = 0.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        return super().generate(
            colors=colors,
            complexity=complexity,
            fractal_type="fluid",
            zoom=zoom,
            center=center,
            angle=angle,
            phase_offset=phase_offset,
            frequency=frequency,
            channel_offset=channel_offset,
            seed=seed,
        )


class EscapeTimeFractalGradient(FractalFieldGradient):
    """Dedicated complex-plane escape-time (Julia / Mandelbrot) fractal engine."""

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        fractal_type: str = "julia",
        zoom: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
        angle: Optional[float] = None,
        phase_offset: float = 0.0,
        frequency: float = 1.0,
        trap_weight: float = 0.35,
        channel_offset: float = 0.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        eff_type = fractal_type if fractal_type in ("julia", "mandelbrot") else "julia"
        return super().generate(
            colors=colors,
            complexity=complexity,
            fractal_type=eff_type,
            zoom=zoom,
            center=center,
            angle=angle,
            phase_offset=phase_offset,
            frequency=frequency,
            trap_weight=trap_weight,
            channel_offset=channel_offset,
            seed=seed,
        )


FractalGradient = FractalFieldGradient
