"""Warp stages: move sample positions before a field is evaluated."""

from __future__ import annotations

from typing import Any

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
