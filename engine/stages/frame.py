"""Frame stages: the working grid and its orientation."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from engine.stages.types import Coords


def sample_grid(
    *,
    width: int,
    height: int,
    scale: float = 0.25,
    extent: Tuple[float, float] = (-1.0, 1.0),
    min_size: int = 16,
) -> Coords:
    """A grid at a fraction of the output size, spanning extent on both axes.

    Both axes span the same extent whatever the aspect ratio, so patterns stretch
    on wide screens; that matches the current engines.
    """
    h = max(min_size, int(height * scale))
    w = max(min_size, int(width * scale))
    y = np.linspace(extent[0], extent[1], h, dtype=np.float32)
    x = np.linspace(extent[0], extent[1], w, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    return Coords(x=xv, y=yv, base_x=xv, base_y=yv, output_size=(width, height))


def rotate(coords: Coords, *, angle: Optional[float] = None, rng: Any) -> Coords:
    """Rotates the grid by angle (radians), drawn at random when not given."""
    if angle is None:
        angle = rng.uniform(0, 2 * np.pi)
    u = coords.x * np.cos(angle) + coords.y * np.sin(angle)
    v = -coords.x * np.sin(angle) + coords.y * np.cos(angle)
    return Coords(x=u, y=v, base_x=u, base_y=v, output_size=coords.output_size)
