"""Frame stages: the working grid, its orientation and camera."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from engine.stages.types import Camera, Coords


def sample_grid(
    *,
    width: int,
    height: int,
    scale: float = 0.25,
    extent: Tuple[float, float] = (-1.0, 1.0),
    aspect_correct: bool = False,
    min_size: int = 16,
) -> Coords:
    """A grid at a fraction of the output size spanning extent.

    Without aspect_correct both axes span the same extent whatever the aspect
    ratio, so patterns stretch on wide screens (what most engines do today).
    With it, x spans extent times the grid's aspect ratio.
    """
    h = max(min_size, int(height * scale))
    w = max(min_size, int(width * scale))
    y = np.linspace(extent[0], extent[1], h, dtype=np.float32)
    if aspect_correct:
        aspect = w / h
        x = np.linspace(extent[0] * aspect, extent[1] * aspect, w, dtype=np.float32)
    else:
        x = np.linspace(extent[0], extent[1], w, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    return Coords(x=xv, y=yv, base_x=xv, base_y=yv, output_size=(width, height))


def full_resolution_grid(*, width: int, height: int) -> Coords:
    """One sample per output pixel over [0, 1] on both axes."""
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    return Coords(x=xv, y=yv, base_x=xv, base_y=yv, output_size=(width, height))


def control_lattice(*, width: int, height: int, complexity: int) -> Coords:
    """A coarse lattice of a few control points per axis, matched to the aspect ratio."""
    aspect_ratio = width / height
    min_complexity = max(4, complexity)
    rows = max(2, int(np.sqrt(min_complexity / aspect_ratio))) * 2
    cols = max(2, int(rows * aspect_ratio)) * 2
    y = np.linspace(-0.5, 0.5, rows, dtype=np.float32)
    x = np.linspace(-0.5, 0.5, cols, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    return Coords(x=xv, y=yv, base_x=xv, base_y=yv, output_size=(width, height))


def rotate(coords: Coords, *, angle: Optional[float] = None, rng: Any) -> Coords:
    """Rotates positions by angle (radians), drawn at random when not given."""
    if angle is None:
        angle = rng.uniform(0, 2 * np.pi)
    u = coords.x * np.cos(angle) + coords.y * np.sin(angle)
    v = -coords.x * np.sin(angle) + coords.y * np.cos(angle)
    return coords.moved(u, v, coords.displacement)


def fractal_camera(
    *,
    width: int,
    height: int,
    kind: str,
    zoom: Optional[float],
    center: Optional[Tuple[float, float]],
    angle: Optional[float],
) -> Coords:
    """Picks zoom, angle and center (random unless given) and samples the view.

    The view spans 1.3 / zoom vertically around the origin, aspect-corrected.
    """
    from engine.gradients.fractal import FractalFieldGradient

    h = max(16, int(height * 0.25))
    w = max(16, int(width * 0.25))
    aspect = w / h
    eff_zoom, eff_angle, cx, cy = FractalFieldGradient._resolve_camera(kind, zoom, center, angle, aspect)
    y = np.linspace(-1.3 / eff_zoom, 1.3 / eff_zoom, h, dtype=np.float32)
    x = np.linspace(-1.3 * aspect / eff_zoom, 1.3 * aspect / eff_zoom, w, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    return Coords(
        x=xv,
        y=yv,
        base_x=xv,
        base_y=yv,
        output_size=(width, height),
        camera=Camera(zoom=eff_zoom, angle=eff_angle, center=(cx, cy), aspect=aspect),
    )


def rotate_about_camera(coords: Coords) -> Coords:
    """Moves positions into the camera frame: centered on it and rotated by its angle."""
    cam = coords.camera
    if cam is None:
        raise ValueError("rotate_about_camera needs fractal_camera before it")
    cx, cy = cam.center
    a = cam.angle
    rot_x = (coords.x - cx) * np.cos(a) - (coords.y - cy) * np.sin(a)
    rot_y = (coords.x - cx) * np.sin(a) + (coords.y - cy) * np.cos(a)
    return coords.moved(rot_x, rot_y, coords.displacement)
