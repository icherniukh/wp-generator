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


_MANDELBROT_FOCAL_POINTS = [
    (-0.7436438870371587, 0.1318259042053119),
    (-0.1011, 0.9563),
    (-0.75, 0.1),
    (-0.748, 0.065),
    (-1.25, 0.0),
    (0.275, 0.0),
]


def resolve_fractal_camera(
    kind: str,
    zoom: Optional[float],
    center: Optional[Tuple[float, float]],
    angle: Optional[float],
    aspect: float,
    rng: Any,
) -> Tuple[float, float, float, float]:
    """Zoom, angle and center for a fractal view; each drawn at random unless given.

    Mandelbrot views zoom into one of a few known detailed points. Other kinds
    pick a zoom tier (macro, medium, detail) and place the center off-screen,
    in a corner or near the middle.
    """
    if zoom is not None:
        eff_zoom = float(zoom)
    elif kind == "mandelbrot":
        eff_zoom = rng.uniform(8.0, 35.0)
    else:
        radius_tier = rng.choices(["macro", "medium", "detail"], weights=[0.45, 0.40, 0.15])[0]
        zoom_ranges = {"macro": (0.40, 0.85), "medium": (0.85, 1.75), "detail": (1.75, 3.20)}
        eff_zoom = rng.uniform(*zoom_ranges[radius_tier])

    eff_angle = float(angle) if angle is not None else rng.uniform(0.0, 2.0 * np.pi)

    if center is not None:
        cx, cy = float(center[0]), float(center[1])
    elif kind == "mandelbrot":
        cx, cy = rng.choice(_MANDELBROT_FOCAL_POINTS)
    else:
        focal_archetype = rng.choices(["offscreen", "corner", "centered"], weights=[0.45, 0.35, 0.20])[0]
        if focal_archetype == "offscreen":
            theta_off = rng.uniform(0.0, 2.0 * np.pi)
            dist_off = rng.uniform(1.4, 2.8) / eff_zoom
            cx = dist_off * aspect * np.cos(theta_off)
            cy = dist_off * np.sin(theta_off)
        elif focal_archetype == "corner":
            sign_x = rng.choice([-1.0, 1.0])
            sign_y = rng.choice([-1.0, 1.0])
            cx = sign_x * rng.uniform(0.40, 0.95) * aspect / eff_zoom
            cy = sign_y * rng.uniform(0.35, 0.90) / eff_zoom
        else:
            cx = rng.uniform(-0.35, 0.35) * aspect / eff_zoom
            cy = rng.uniform(-0.35, 0.35) / eff_zoom

    return eff_zoom, eff_angle, cx, cy


def fractal_camera(
    *,
    width: int,
    height: int,
    kind: str,
    zoom: Optional[float],
    center: Optional[Tuple[float, float]],
    angle: Optional[float],
    rng: Any,
) -> Coords:
    """Picks zoom, angle and center (random unless given) and samples the view.

    The view spans 1.3 / zoom vertically around the origin, aspect-corrected.
    """
    h = max(16, int(height * 0.25))
    w = max(16, int(width * 0.25))
    aspect = w / h
    eff_zoom, eff_angle, cx, cy = resolve_fractal_camera(kind, zoom, center, angle, aspect, rng)
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
