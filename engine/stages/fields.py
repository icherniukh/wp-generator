"""Field stages: turn sample positions into a scalar field or per-color weights."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage

from engine.gradients.spatial import generate_noise_plane, normalize_scalar_field, trig_fbm_field
from engine.stages.types import Coords, Palette, ScalarField, WeightField


def folded_noise(
    coords: Coords,
    *,
    warp: float,
    transverse: float,
    frequency: float,
    rng: Any,
) -> ScalarField:
    """A linear ramp along x, folded by a second noise plane.

    With transverse > 0 the noise plane is itself warped by the previous warp's
    displacement and the ramp bends along y.
    """
    if coords.displacement is None:
        raise ValueError("folded_noise needs a warp stage before it")
    h, w = coords.shape
    qx, qy = coords.displacement
    ox3, oy3 = rng.uniform(1000, 1500), rng.uniform(1000, 1500)
    rx = generate_noise_plane(h, w, scale=frequency * 1.3, octaves=2, offset=(ox3, oy3))

    if transverse > 0.0:
        grid_y, grid_x = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        warped_coords = [
            grid_y + (transverse * 0.45 * qy) * h,
            grid_x + (transverse * 0.45 * qx) * w,
        ]
        rx_warped = scipy.ndimage.map_coordinates(rx, warped_coords, order=3, mode="reflect")
        transverse_bend = transverse * 0.35 * np.sin(coords.y * 1.5)
    else:
        rx_warped = rx
        transverse_bend = 0.0

    return ScalarField(normalize_scalar_field(coords.x + transverse_bend + (warp * 0.35) * rx_warped))


def radial_emitters(
    coords: Coords,
    palette: Palette,
    *,
    complexity: int,
    ambient_weight: float,
    dispersion: float,
    rng: Any,
) -> WeightField:
    """Soft elliptical spots of colors 1..n around the center over color 0.

    Each spot falls off with a smoothstep to zero at 1.8 x its radius.
    """
    num_colors = palette.size
    h, w = coords.shape
    xv, yv = coords.x, coords.y

    weights = np.zeros((num_colors, h, w), dtype=np.float32)
    weights[0] = max(0.01, float(ambient_weight))

    num_emitters = max(num_colors - 1, max(1, complexity))
    for i in range(num_emitters):
        c_idx = 1 + (i % (num_colors - 1))
        angle_pos = i * (2.0 * np.pi / num_emitters) + rng.uniform(-0.35, 0.35)
        dist_center = rng.uniform(0.18, 0.44)
        cx = float(np.clip(0.5 + dist_center * np.cos(angle_pos), 0.1, 0.9))
        cy = float(np.clip(0.5 + dist_center * np.sin(angle_pos), 0.1, 0.9))
        sigma = max(0.05, float(dispersion) * rng.uniform(0.75, 1.25))
        aspect = rng.uniform(0.7, 1.4)
        flow_angle = rng.uniform(0.0, np.pi)

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
    return WeightField(weights, background=0)


def noisy_ramp(coords: Coords, *, complexity: int, rng: Any) -> ScalarField:
    """A linear ramp along x plus a stronger simplex-noise plane."""
    h, w = coords.shape
    ox = rng.uniform(0, 1000)
    oy = rng.uniform(0, 1000)
    noise_scale = 0.5 * (1.0 + complexity * 0.2)
    t_noise = generate_noise_plane(h, w, scale=noise_scale, octaves=2, offset=(ox, oy))
    return ScalarField(normalize_scalar_field(coords.x * 0.6 + t_noise * 1.5))


def folded_ramp(coords: Coords) -> ScalarField:
    """A linear ramp along x plus one fixed fold over the unrotated grid."""
    fold = np.sin(coords.base_x * np.pi) * np.cos(coords.base_y * np.pi) * 0.25
    return ScalarField(normalize_scalar_field(coords.x + fold))


def smooth_ramp(coords: Coords) -> ScalarField:
    """A linear ramp along x, stretched between its 1st and 99th percentile and smoothstepped."""
    t_values = coords.x
    t_min, t_max = np.percentile(t_values, [1, 99])
    if t_max > t_min:
        t_values = (t_values - t_min) / (t_max - t_min)
    t_values = np.clip(t_values, 0.0, 1.0)
    return ScalarField(t_values * t_values * (3.0 - 2.0 * t_values))


def elevation(coords: Coords, *, complexity: int, noise: float, rng: Any) -> ScalarField:
    """Three interfering sine waves as terrain height, plus simplex noise added to the height."""
    h, w = coords.shape
    u, v = coords.x, coords.y
    base_freq = 0.7 + (complexity * 0.1)
    h1 = np.sin(u * base_freq * 1.4) * np.cos(v * base_freq * 1.1)
    h2 = np.sin((u + v) * base_freq * 2.1 + 1.1) * 0.5
    h3 = np.cos(np.sqrt(u**2 + v**2) * base_freq * 2.8) * 0.25
    raw_elevation = h1 + h2 + h3

    effective = max(0.0, float(noise))
    if effective > 0.0:
        ox = rng.uniform(0, 500)
        oy = rng.uniform(500, 1000)
        noise_plane = generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox, oy))
        raw_elevation = raw_elevation + (effective * 0.6) * noise_plane
    return ScalarField(normalize_scalar_field(raw_elevation))


def orbiting_emitters(coords: Coords, palette: Palette, *, rng: Any) -> WeightField:
    """One emitter per color on a ring around the center, partly off-screen;
    inverse-square weights, with color 0 pulled closer."""
    num_colors = palette.size
    h, w = coords.shape
    points = []
    for i in range(num_colors):
        angle = i * (2 * np.pi / num_colors) + rng.uniform(0, np.pi)
        dist = rng.uniform(0.5, 1.2)
        points.append((0.5 + np.cos(angle) * dist, 0.5 + np.sin(angle) * dist))

    distances = np.zeros((num_colors, h, w), dtype=np.float32)
    for i, (px, py) in enumerate(points):
        distances[i] = np.sqrt((coords.x - px) ** 2 + (coords.y - py) ** 2)
    distances[0] *= 0.6
    distances = np.maximum(distances, 0.15)
    weights = 1.0 / (distances**2.0)
    weights /= np.sum(weights, axis=0)
    return WeightField(weights)


def anchor_points(
    coords: Coords,
    palette: Palette,
    *,
    power: float,
    softness: float = 0.12,
    points: Optional[Sequence[Tuple[float, float]]] = None,
    margin: float = 0.08,
    rng: Any,
) -> WeightField:
    """One anchor per color, random unless given; inverse-distance weights (IDW)."""
    num_colors = palette.size
    h, w = coords.shape
    if points is None:
        points = [
            (rng.uniform(margin, 1.0 - margin), rng.uniform(margin, 1.0 - margin))
            for _ in range(num_colors)
        ]
    distances = np.zeros((num_colors, h, w), dtype=np.float32)
    for i, (px, py) in enumerate(points):
        distances[i] = np.sqrt((coords.x - px) ** 2 + (coords.y - py) ** 2)
    distances = np.maximum(distances, softness)
    weights = 1.0 / (distances**power)
    weights /= np.sum(weights, axis=0)
    return WeightField(weights)


def fluid_fractal_field(
    xv: np.ndarray,
    yv: np.ndarray,
    rot_x: np.ndarray,
    rot_y: np.ndarray,
    rf: np.ndarray,
    angle: float,
    aspect: float,
    complexity: int,
    rng: Any,
) -> np.ndarray:
    """Two levels of trigonometric-fBm domain warp, mixed with a radial carrier and a sweep."""
    q_x = trig_fbm_field(rot_x, rot_y, octaves=2, rng=rng)
    q_y = trig_fbm_field(rot_x + 3.8, rot_y + 1.5, octaves=2, rng=rng)
    r_x = trig_fbm_field(rot_x + 1.8 * q_x + 1.4, rot_y + 1.8 * q_y + 2.8, octaves=2, rng=rng)
    r_y = trig_fbm_field(rot_x + 1.8 * q_x + 4.5, rot_y + 1.8 * q_y + 1.6, octaves=2, rng=rng)

    radial_freq = rng.uniform(0.35, 0.80)
    radial_carrier = 0.5 + 0.5 * np.cos(rf * radial_freq)

    sweep_angle = angle + rng.uniform(-0.5, 0.5)
    ambient_sweep = 0.5 * (np.cos(sweep_angle) * (xv / aspect) + np.sin(sweep_angle) * yv)

    fluid_folds = trig_fbm_field(
        rot_x + 2.2 * r_x, rot_y + 2.2 * r_y, octaves=max(2, min(5, complexity)), rng=rng
    )
    field = 0.30 * radial_carrier + 0.25 * ambient_sweep + 0.45 * fluid_folds
    return normalize_scalar_field(field)


def escape_time_field(
    rot_x: np.ndarray,
    rot_y: np.ndarray,
    kind: str,
    complexity: int,
    trap_weight: float,
    rng: Any,
) -> np.ndarray:
    """Smoothed escape-time potential of a Mandelbrot or Julia set, blended with an orbit trap.

    Julia sets pick c near the main cardioid or the period-2 bulb boundary.
    """
    z_grid = rot_x + 1j * rot_y
    max_iter = max(16, 16 + complexity * 3)
    escape_radius = 16.0
    ln_r = np.log(escape_radius)

    smooth_pot = np.zeros(z_grid.shape, dtype=np.float32)
    trap = np.full(z_grid.shape, 1e6, dtype=np.float32)
    mask = np.ones(z_grid.shape, dtype=bool)

    if kind == "mandelbrot":
        z = np.zeros_like(z_grid)
        c = z_grid
    else:
        theta = rng.uniform(0.0, 2.0 * np.pi)
        if rng.random() < 0.3:
            c_bound = -1.0 + 0.25 * np.exp(1j * theta)
        else:
            c_bound = 0.5 * np.exp(1j * theta) - 0.25 * np.exp(2j * theta)
        eps = rng.uniform(0.015, 0.04) * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
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
        (1.0 - trap_weight * 0.5) * (0.45 * ambient_carrier + 0.55 * pot_smooth)
        + trap_weight * 0.5 * trap_smooth
    )


def fractal_field(coords: Coords, *, kind: str, complexity: int, trap_weight: float, rng: Any) -> ScalarField:
    """kind "fluid": nested fBm domain warp over a radial carrier and a sweep.
    Otherwise an escape-time fractal (mandelbrot, or a Julia set) with an orbit trap.

    Needs fractal_camera, and the camera-frame positions in x, y.
    """
    cam = coords.camera
    if cam is None:
        raise ValueError("fractal_field needs fractal_camera before it")
    safe_complexity = max(1, int(complexity))
    if kind == "fluid":
        cx, cy = cam.center
        rf = np.sqrt((coords.base_x - cx) ** 2 + (coords.base_y - cy) ** 2)
        t = fluid_fractal_field(
            coords.base_x, coords.base_y, coords.x, coords.y, rf, cam.angle, cam.aspect, safe_complexity, rng
        )
    else:
        t = escape_time_field(coords.x, coords.y, kind, safe_complexity, np.clip(float(trap_weight), 0.0, 1.0), rng)
    return ScalarField(t)
