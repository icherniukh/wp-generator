"""Warp stages: move sample positions before a field is evaluated."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import scipy.ndimage

from engine.gradients.noise import pnoise2
from engine.gradients.spatial import generate_noise_plane
from engine.stages.types import Coords


def noise_displacement(
    coords: Coords,
    *,
    amount_x: float,
    amount_y: float,
    frequency: float,
    octaves: int = 2,
    skip_when_zero: bool = False,
    rng: Any,
) -> Coords:
    """Shifts x and y by two independent simplex-noise planes."""
    if skip_when_zero and amount_x <= 0.0 and amount_y <= 0.0:
        return coords
    h, w = coords.shape
    ox1, oy1 = rng.uniform(0, 500), rng.uniform(0, 500)
    ox2, oy2 = rng.uniform(500, 1000), rng.uniform(500, 1000)
    qx = generate_noise_plane(h, w, scale=frequency, octaves=octaves, offset=(ox1, oy1))
    qy = generate_noise_plane(h, w, scale=frequency, octaves=octaves, offset=(ox2, oy2))
    return coords.moved(coords.x + amount_x * qx, coords.y + amount_y * qy, (qx, qy))


def perlin_displacement(
    coords: Coords,
    *,
    strength_range: Tuple[float, float],
    complexity: int,
    octaves: int,
    rng: Any,
) -> Coords:
    """Shifts positions by coarse Perlin noise, bilinearly enlarged.

    The strength is drawn from strength_range and the noise frequency scales
    with complexity, in grid samples, so the pattern changes with resolution.
    """
    height, width = coords.shape
    small_h = max(2, int(height * 0.1))
    small_w = max(2, int(width * 0.1))

    ox = rng.uniform(0, 1000)
    oy = rng.uniform(0, 1000)
    strength = rng.uniform(*strength_range)
    scale = rng.uniform(0.002, 0.006) * complexity

    # float64 sample positions, as the legacy per-point loop passed Python floats.
    nx = np.arange(small_w, dtype=np.float64) * scale + ox
    ny = np.arange(small_h, dtype=np.float64) * scale + oy
    nxv, nyv = np.meshgrid(nx, ny)
    dx = pnoise2(nxv, nyv, octaves=octaves)
    dy = pnoise2(nxv + 500, nyv + 500, octaves=octaves)

    dx = scipy.ndimage.zoom(dx, (height / small_h, width / small_w), order=1)
    dy = scipy.ndimage.zoom(dy, (height / small_h, width / small_w), order=1)
    if dx.shape != coords.shape:
        pad_h = max(0, height - dx.shape[0])
        pad_w = max(0, width - dx.shape[1])
        dx = np.pad(dx, ((0, pad_h), (0, pad_w)))[:height, :width]
        dy = np.pad(dy, ((0, pad_h), (0, pad_w)))[:height, :width]

    return coords.moved(coords.x + dx * strength, coords.y + dy * strength, (dx, dy))


def sine_warp(coords: Coords, *, amplitude: float = 0.18) -> Coords:
    """A fixed, smooth sinusoidal bend of both axes."""
    xv, yv = coords.x, coords.y
    xw = xv + amplitude * (np.sin(yv * 2.6 + 1.2) * np.cos(xv * 2.0))
    yw = yv + amplitude * (np.cos(xv * 2.6 + 0.8) * np.sin(yv * 2.0))
    return coords.moved(xw, yw, coords.displacement)
