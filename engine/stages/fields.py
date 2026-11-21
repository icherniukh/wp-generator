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
