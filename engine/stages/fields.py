"""Field stages: turn sample positions into a scalar field or per-color weights."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.ndimage

from engine.gradients.spatial import generate_noise_plane, normalize_scalar_field
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
    coords: Coords, palette: Palette, *, power: float, softness: float = 0.12, margin: float = 0.08, rng: Any
) -> WeightField:
    """One random anchor per color; inverse-distance weights (IDW)."""
    num_colors = palette.size
    h, w = coords.shape
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


def fractal_field(coords: Coords, *, kind: str, complexity: int, trap_weight: float) -> ScalarField:
    """kind "fluid": nested fBm domain warp over a radial carrier and a sweep.
    Otherwise an escape-time fractal (mandelbrot, or a Julia set) with an orbit trap.

    Needs fractal_camera, and the camera-frame positions in x, y.
    """
    from engine.gradients.fractal import FractalFieldGradient

    cam = coords.camera
    if cam is None:
        raise ValueError("fractal_field needs fractal_camera before it")
    safe_complexity = max(1, int(complexity))
    if kind == "fluid":
        cx, cy = cam.center
        rf = np.sqrt((coords.base_x - cx) ** 2 + (coords.base_y - cy) ** 2)
        t = FractalFieldGradient._compute_fluid_field(
            coords.base_x, coords.base_y, coords.x, coords.y, rf, cam.angle, cam.aspect, safe_complexity
        )
    else:
        t = FractalFieldGradient._compute_escape_time_field(
            coords.x, coords.y, kind, safe_complexity, np.clip(float(trap_weight), 0.0, 1.0)
        )
    return ScalarField(t)
