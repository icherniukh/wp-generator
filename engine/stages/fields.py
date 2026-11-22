"""Field stages: turn sample positions into a scalar field or per-color weights."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage

from engine.gradients.spatial import generate_noise_plane, normalize_scalar_field, trig_fbm_field
from engine.stages.types import Coords, Field, Palette, ScalarField, WeightField


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
    # Distances in units of the image height, so anchors pull in circles on a
    # wide screen; softness rounds the falloff off near each anchor instead of
    # clamping it, which left a flat disc ringed by the next color.
    width, height = coords.output_size
    aspect = width / max(height, 1)
    weights = np.zeros((num_colors, h, w), dtype=np.float32)
    for i, (px, py) in enumerate(points):
        squared = ((coords.x - px) * aspect) ** 2 + (coords.y - py) ** 2
        weights[i] = (squared + softness**2) ** (-power / 2.0)
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


def _polar(coords: Coords) -> Tuple[np.ndarray, np.ndarray]:
    """Distance from the image center in half short sides, and angle."""
    from engine.stages.warp import Frame

    u, v = Frame(coords).from_xy(coords.x, coords.y)
    return np.sqrt(u**2 + v**2), np.arctan2(v, u)


def _mirror(t: np.ndarray) -> np.ndarray:
    """Folds a repeating 0..1 value into 0..1..0, so palette ends meet without a seam."""
    return 1.0 - np.abs(2.0 * (t % 1.0) - 1.0)


def conic(coords: Coords, *, turns: float = 1.0, rng: Any) -> ScalarField:
    """Sweeps the palette around the center, there and back turns times."""
    _, theta = _polar(coords)
    offset = rng.uniform(0.0, 1.0)
    return ScalarField(_mirror(theta / (2 * np.pi) * turns + offset).astype(np.float32))


def spiral(coords: Coords, *, turns: float = 1.0, tightness: float = 1.5, rng: Any) -> ScalarField:
    """A conic sweep that advances with distance from the center."""
    r, theta = _polar(coords)
    offset = rng.uniform(0.0, 1.0)
    return ScalarField(_mirror(theta / (2 * np.pi) * turns + r * tightness + offset).astype(np.float32))


def radial(coords: Coords) -> ScalarField:
    """Distance from the center, 0 there and 1 at the farthest corner."""
    r, _ = _polar(coords)
    return ScalarField(normalize_scalar_field(r))


def voronoi(
    coords: Coords, *, cells: int = 12, mode: str = "distance", jitter: float = 1.0, rng: Any
) -> ScalarField:
    """Cells around random seed points.

    mode "distance": distance to the nearest point (soft bubbles, F1).
    mode "edges": second-nearest minus nearest distance (cracks between cells, F2-F1).
    mode "f2": distance to the second-nearest point (F2, rounded facets).
    mode "cells": one random value per cell (flat mosaic).
    """
    from scipy.spatial import cKDTree

    # Seed points on the image's own frame, so cells stay in view after rotation or zoom.
    x0, x1 = float(coords.x.min()), float(coords.x.max())
    y0, y1 = float(coords.y.min()), float(coords.y.max())
    count = max(2, int(cells))
    # Jittered grid points keep cells similar in size; jitter=0 gives a regular grid.
    aspect = (x1 - x0) / max(y1 - y0, 1e-9)
    nx = max(1, int(round(np.sqrt(count * aspect))))
    ny = max(1, int(round(count / nx)))
    points = []
    for j in range(ny):
        for i in range(nx):
            px = x0 + (i + 0.5 + jitter * rng.uniform(-0.5, 0.5)) * (x1 - x0) / nx
            py = y0 + (j + 0.5 + jitter * rng.uniform(-0.5, 0.5)) * (y1 - y0) / ny
            points.append((px, py))
    values = [rng.uniform(0.0, 1.0) for _ in points]
    tree = cKDTree(np.asarray(points))
    query = np.column_stack([coords.x.ravel(), coords.y.ravel()])
    dist, idx = tree.query(query, k=2)
    if mode == "edges":
        t = (dist[:, 1] - dist[:, 0]).reshape(coords.shape)
    elif mode == "cells":
        t = np.asarray(values, dtype=np.float32)[idx[:, 0]].reshape(coords.shape)
        return ScalarField(t)
    elif mode == "distance":
        t = dist[:, 0].reshape(coords.shape)
    elif mode == "f2":
        t = dist[:, 1].reshape(coords.shape)
    else:
        raise ValueError(f"voronoi mode must be distance, edges, f2 or cells, not {mode!r}")
    return ScalarField(normalize_scalar_field(t.astype(np.float32)))


def _octave_noise(
    coords: Coords, *, scale: float, octaves: int, fold: Any, gain: float, lacunarity: float, rng: Any
) -> ScalarField:
    """Sums octaves of simplex noise, each passed through fold, on a half-size grid."""
    from engine.gradients.noise import snoise2

    h, w = coords.shape
    sample_h, sample_w = max(8, h // 2), max(8, w // 2)
    # mode="nearest" matters: the default "constant" returns 0 for a last sample that
    # float rounding puts a hair past the edge, which put the noise at position 0 there
    # and smeared a band along the bottom (or right) edge in about one grid size in eight.
    sx = scipy.ndimage.zoom(coords.x.astype(np.float64), (sample_h / h, sample_w / w), order=1, mode="nearest")
    sy = scipy.ndimage.zoom(coords.y.astype(np.float64), (sample_h / h, sample_w / w), order=1, mode="nearest")
    ox, oy = rng.uniform(0, 1000), rng.uniform(0, 1000)
    total = np.zeros_like(sx)
    amp, freq, norm = 1.0, float(scale), 0.0
    for _ in range(max(1, int(octaves))):
        n = snoise2(sx * freq + ox, sy * freq + oy, octaves=1)
        total += amp * fold(n)
        norm += amp
        amp *= gain
        freq *= lacunarity
    grid = scipy.ndimage.zoom(total / norm, (h / total.shape[0], w / total.shape[1]), order=3, mode="nearest")[:h, :w]
    return ScalarField(normalize_scalar_field(grid.astype(np.float32)))


def ridged_noise(
    coords: Coords,
    *,
    scale: float = 1.5,
    octaves: int = 4,
    sharpness: float = 2.0,
    gain: float = 0.5,
    lacunarity: float = 2.0,
    rng: Any,
) -> ScalarField:
    """Ridged multifractal: each octave contributes (1 - |noise|)^sharpness, so
    zero crossings of the noise become sharp crests, like mountain ridges."""
    return _octave_noise(
        coords, scale=scale, octaves=octaves, fold=lambda n: (1.0 - np.abs(n)) ** sharpness,
        gain=gain, lacunarity=lacunarity, rng=rng,
    )


def billow_noise(
    coords: Coords, *, scale: float = 1.5, octaves: int = 4, gain: float = 0.5, lacunarity: float = 2.0, rng: Any
) -> ScalarField:
    """Billow noise: each octave contributes |noise|, so zero crossings become
    creases between rounded puffs (the inverse look of ridged)."""
    return _octave_noise(
        coords, scale=scale, octaves=octaves, fold=np.abs, gain=gain, lacunarity=lacunarity, rng=rng
    )


def fbm_noise(
    coords: Coords, *, scale: float = 1.2, octaves: int = 5, gain: float = 0.5, lacunarity: float = 2.0, rng: Any
) -> ScalarField:
    """Plain fractal Brownian motion: octaves of simplex noise summed as they are."""
    return _octave_noise(
        coords, scale=scale, octaves=octaves, fold=lambda n: n, gain=gain, lacunarity=lacunarity, rng=rng
    )


def _frame_extent(coords: Coords) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Frame positions (half short sides, centered) and the half width and half
    height of the image in them, following any rotation or zoom (from the corners)."""
    from engine.stages.warp import Frame

    u, v = Frame(coords).from_xy(coords.x, coords.y)
    corners = (0, -1)
    umax = max(float(np.max(np.abs(u[np.ix_(corners, corners)]))), 1e-9)
    vmax = max(float(np.max(np.abs(v[np.ix_(corners, corners)]))), 1e-9)
    return u, v, umax, vmax


def _image_unit_square(coords: Coords) -> Tuple[np.ndarray, np.ndarray]:
    """Positions as 0..1 across the image's width and height."""
    u, v, umax, vmax = _frame_extent(coords)
    return (u / umax + 1.0) / 2.0, (v / vmax + 1.0) / 2.0


def patch_mesh(
    coords: Coords,
    palette: Palette,
    *,
    columns: int = 4,
    rows: int = 3,
    jitter: float = 0.3,
    rng: Any,
) -> WeightField:
    """A grid of colored control points blended by a smooth bicubic spline.

    columns x rows vertices each take one palette color (every color appears at
    least once when the grid has room). The colors pass through their vertices
    and blend across each patch with continuous slope, so neighbouring colors
    meet without the halos of point-distance weights. jitter moves the inner
    vertices by up to that fraction of a cell; edge vertices slide only along
    their edge, so the mesh still fills the image.
    """
    columns, rows = max(2, int(columns)), max(2, int(rows))
    n = palette.size
    count = columns * rows
    slots = [i % n for i in range(count)]
    # Shuffle with rng.uniform keys so the draw count does not depend on n.
    keys = [rng.uniform(0.0, 1.0) for _ in range(count)]
    order = [slot for _, slot in sorted(zip(keys, slots))]
    vertex_colors = np.asarray(order).reshape(rows, columns)

    offsets = np.zeros((2, rows, columns), dtype=np.float64)
    for j in range(rows):
        for i in range(columns):
            du, dv = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
            offsets[0, j, i] = 0.0 if i in (0, columns - 1) else du
            offsets[1, j, i] = 0.0 if j in (0, rows - 1) else dv

    s, t = _image_unit_square(coords)
    gx, gy = s * (columns - 1), t * (rows - 1)
    sample = [gy.ravel(), gx.ravel()]
    # Moving a vertex by d pulls its color along; sampling at p - d(p) is the
    # first-order inverse of that move and holds for jitter below half a cell.
    shift_x = scipy.ndimage.map_coordinates(offsets[0], sample, order=3, mode="nearest")
    shift_y = scipy.ndimage.map_coordinates(offsets[1], sample, order=3, mode="nearest")
    sample = [np.clip(gy.ravel() - shift_y, 0, rows - 1), np.clip(gx.ravel() - shift_x, 0, columns - 1)]

    h, w = coords.shape
    weights = np.zeros((n, h, w), dtype=np.float32)
    for color in range(n):
        one_hot = (vertex_colors == color).astype(np.float64)
        if not one_hot.any():
            continue
        weights[color] = scipy.ndimage.map_coordinates(one_hot, sample, order=3, mode="nearest").reshape(h, w)
    # The interpolating spline overshoots slightly next to sharp color changes.
    weights = np.clip(weights, 0.0, None)
    weights /= np.maximum(weights.sum(axis=0), 1e-6)
    return WeightField(weights)


def composite_over(alphas: Sequence[np.ndarray], colors: Sequence[int], base: int, num_colors: int) -> np.ndarray:
    """Per-color weights of layers stacked with the "over" operator on a base color.

    Layer k (bottom first) has coverage alphas[k] and palette index colors[k].
    Over-compositing fixed colors is linear in the colors, so the result is a
    weight per color: layer k keeps a_k times the transparency of everything
    above it, and the base keeps what no layer covers. The weights sum to 1
    without normalizing, and an opaque layer hides what is under it.
    """
    shape = alphas[0].shape if len(alphas) else ()
    weights = np.zeros((num_colors, *shape), dtype=np.float32)
    remaining = np.ones(shape, dtype=np.float32)
    for alpha, color in zip(reversed(alphas), reversed(colors)):
        a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        weights[color] += a * remaining
        remaining *= 1.0 - a
    weights[base] += remaining
    return weights


def _blurred_ellipse(d: np.ndarray, blur: float) -> np.ndarray:
    """Coverage of a unit ellipse blurred by a Gaussian of sigma blur (in radii),
    at elliptic distance d: an erfc edge, scaled so the center reaches what a
    blurred disc reaches there, 1 - exp(-1 / (2 blur^2)). A blur wider than the
    shape lowers its peak, as a large layer blur does to a small ellipse."""
    from scipy.special import erfc

    s = max(float(blur), 1e-3)
    peak = 1.0 - np.exp(-1.0 / (2.0 * s * s))
    edge = erfc((d - 1.0) / (np.sqrt(2.0) * s)) / erfc(-1.0 / (np.sqrt(2.0) * s))
    return peak * np.clip(edge, 0.0, 1.0)


def blob_layers(
    coords: Coords,
    palette: Palette,
    *,
    blobs: int = 4,
    blur: float = 0.5,
    spread: float = 1.1,
    rng: Any,
) -> WeightField:
    """Soft elliptical blobs stacked over the darkest color, each covering what
    is under it rather than averaging with it.

    The first blob is large (1.0 to 1.5 half short sides) and sits at the
    bottom; the others are smaller (0.5 to 0.95) and drawn over it. Centers may
    lie up to spread times the half width or height from the middle, so a blob
    can sit partly off the image. Every color except the base appears when
    there are enough blobs, and the blob of the lightest color is moved toward
    an edge. blur is the Gaussian sigma in blob radii.
    """
    n = palette.size
    count = max(1, int(blobs))
    lightness = np.asarray(palette.oklab)[:, 0]
    base = int(np.argmin(lightness))
    others = [i for i in range(n) if i != base] or [base]
    lightest = int(np.argmax(lightness))

    # A fixed number of draws per blob, independent of the palette size.
    keys = [rng.uniform(0.0, 1.0) for _ in range(count)]
    layout = [
        (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(0.0, 1.0),
         rng.uniform(0.55, 1.0), rng.uniform(0.0, np.pi), rng.uniform(0.7, 1.0))
        for _ in range(count)
    ]
    slots = [others[i % len(others)] for i in range(count)]
    colors = [slot for _, slot in sorted(zip(keys, slots))]

    u, v, umax, vmax = _frame_extent(coords)
    alphas = []
    for k, ((cu, cv, size, ratio, angle, opacity), color) in enumerate(zip(layout, colors)):
        if color == lightest and n > 1:
            # Off-center, near an edge: at least 0.8 of the way out on one axis.
            reach = max(abs(cu), abs(cv))
            cu, cv = (cu / reach * 0.85, cv / reach * 0.85) if reach > 1e-6 else (0.85, 0.0)
        radius = 1.0 + 0.5 * size if k == 0 else 0.5 + 0.45 * size
        du = u - cu * spread * umax
        dv = v - cv * spread * vmax
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        along = (du * cos_a + dv * sin_a) / radius
        across = (-du * sin_a + dv * cos_a) / (radius * ratio)
        alphas.append(opacity * _blurred_ellipse(np.sqrt(along**2 + across**2), blur))
    return WeightField(composite_over(alphas, colors, base, n), background=base)


def _stroke_path(
    start: Tuple[float, float], angle: float, length: float, bend: float, wobble: float, points: int = 24
) -> np.ndarray:
    """A smooth curve of length (about) from start along angle, as (points, 2).

    A cubic Bezier whose two inner control points are pushed sideways, bend and
    wobble in lengths, in opposite directions for an S shape."""
    direction = np.array([np.cos(angle), np.sin(angle)])
    side = np.array([-direction[1], direction[0]])
    p0 = np.asarray(start, dtype=np.float64) - direction * length / 2.0
    p3 = p0 + direction * length
    p1 = p0 + direction * length / 3.0 + side * bend * length
    p2 = p0 + direction * 2.0 * length / 3.0 + side * wobble * length
    t = np.linspace(0.0, 1.0, points)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def _capsule_distance(
    u: np.ndarray, v: np.ndarray, path: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """Distance to the polyline in units of the local radius (radii per path
    point, interpolated along it): 1 on the outline of a tapering capsule chain."""
    best = np.full(u.shape, np.inf, dtype=np.float32)
    for (ax, ay), (bx, by), ra, rb in zip(path[:-1], path[1:], radii[:-1], radii[1:]):
        abx, aby = bx - ax, by - ay
        along = np.clip(((u - ax) * abx + (v - ay) * aby) / max(abx * abx + aby * aby, 1e-12), 0.0, 1.0)
        dist = np.hypot(u - (ax + along * abx), v - (ay + along * aby))
        best = np.minimum(best, dist / (ra + along * (rb - ra)))
    return best


# The range of stroke directions, in radians (about 125 degrees).
STROKE_SPREAD = 2.2


def strokes(
    coords: Coords,
    palette: Palette,
    *,
    strokes: int = 5,
    width: float = 0.9,
    blur: float = 0.5,
    rng: Any,
) -> WeightField:
    """A few thick, heavily blurred strokes stacked over the darkest color, like
    smeared paint: a liquid-gradient base.

    Each stroke is a smooth curve running across the image, drawn as a capsule
    chain of radius width / 2 (in half short sides) that thins toward its ends,
    with a Gaussian edge of sigma blur (in radii). The strokes' directions are
    spread evenly over about 125 degrees around a random flow direction (each
    jittered inside its own slice, the slices assigned to strokes at random), and
    each bends and wobbles a good deal, so no two lie parallel and the picture
    never reads as stripes. Each stroke has its own color from the palette (every
    color but the darkest appears when there are enough strokes), and each covers
    what is under it, as with blob_layers.
    """
    n = palette.size
    count = max(1, int(strokes))
    lightness = np.asarray(palette.oklab)[:, 0]
    base = int(np.argmin(lightness))
    others = [i for i in range(n) if i != base] or [base]

    # A fixed number of draws per stroke, independent of the palette size.
    flow = rng.uniform(0.0, np.pi)
    keys = [rng.uniform(0.0, 1.0) for _ in range(count)]
    layout = [
        (rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9), rng.uniform(0.15, 0.85), rng.uniform(2.0, 3.8),
         rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(0.75, 1.25), rng.uniform(0.75, 1.0),
         rng.uniform(0.0, 1.0))
        for _ in range(count)
    ]
    slots = [others[i % len(others)] for i in range(count)]
    colors = [slot for _, slot in sorted(zip(keys, slots))]
    # Each stroke gets its own slice of the direction range, so the strokes fan out.
    slice_of = {k: rank for rank, k in enumerate(sorted(range(count), key=lambda k: layout[k][8]))}

    u, v, umax, vmax = _frame_extent(coords)
    taper = 0.55 + 0.45 * np.sin(np.linspace(0.0, 1.0, 24) * np.pi)
    alphas = []
    for k, (cu, cv, jitter, length, bend, wobble, thickness, opacity, _) in enumerate(layout):
        turn = ((slice_of[k] + jitter) / count - 0.5) * STROKE_SPREAD
        path = _stroke_path((cu * umax, cv * vmax), flow + turn, length, bend, wobble)
        radii = max(float(width), 1e-3) / 2.0 * thickness * taper
        alphas.append(opacity * _blurred_ellipse(_capsule_distance(u, v, path, radii), blur))
    return WeightField(composite_over(alphas, colors, base, n), background=base)


SHAPE_KINDS = ("circle", "half-circle", "quarter-circle", "rectangle", "bar", "rotated-rectangle", "ring")
# Edge softness at hardness 0, in half short sides; it falls with the square of 1 - hardness.
_SHAPE_SOFTNESS = 0.8
# The edge ramp of a hard shape spans this many pixels (a smoothstep over it is about 0.8 px
# between 10% and 90%): one pixel of anti-aliasing and a little margin.
_SHAPE_EDGE_PIXELS = 1.3


def shape_coverage(distance: np.ndarray, hardness: float = 0.97) -> np.ndarray:
    """Coverage in [0, 1] of the inside of a signed distance (negative inside).

    The distance is turned into pixels by its own gradient across the grid, as a
    shader does with fwidth, so a hard edge is about one pixel wide on any grid
    and stays so after a warp stretches or squeezes the image. Softness grows
    with the square of 1 - hardness (see _SHAPE_SOFTNESS), so the default 0.97
    is still sharp and a low hardness gives blob-like edges. Smoothstep keeps
    coverage inside [0, 1] with no overshoot."""
    grad_y, grad_x = np.gradient(distance)
    pixel = np.maximum(np.hypot(grad_x, grad_y), 1e-6)
    soft = _SHAPE_SOFTNESS * (1.0 - min(max(float(hardness), 0.0), 1.0)) ** 2
    width = np.hypot(_SHAPE_EDGE_PIXELS, soft / pixel)
    t = np.clip(0.5 - distance / (pixel * width), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def shape_distance(kind: str, p: np.ndarray, q: np.ndarray, size: float, ratio: float) -> np.ndarray:
    """Signed distance (negative inside) to a shape centered on the origin of the
    local coordinates p, q, in the same units as p and q.

    size is the radius, or the half width of a box; ratio is the half height of a
    box in sizes, or the thickness of a ring in sizes. Half and quarter circles
    are centered on their bounding box, flat side down and corner at the
    bottom left, before rotation."""
    if kind == "circle":
        return np.hypot(p, q) - size
    if kind == "half-circle":
        q = q + size / 2.0
        return np.maximum(np.hypot(p, q) - size, -q)
    if kind == "quarter-circle":
        p, q = p + size / 2.0, q + size / 2.0
        return np.maximum(np.hypot(p, q) - size, np.maximum(-p, -q))
    if kind in ("rectangle", "bar", "rotated-rectangle"):
        reach_u = np.abs(p) - size * (1.8 if kind == "bar" else 1.0)
        reach_v = np.abs(q) - size * ratio
        return np.hypot(np.maximum(reach_u, 0.0), np.maximum(reach_v, 0.0)) + np.minimum(
            np.maximum(reach_u, reach_v), 0.0
        )
    if kind == "ring":
        thickness = size * ratio
        return np.abs(np.hypot(p, q) - (size - thickness / 2.0)) - thickness / 2.0
    raise ValueError(f"shape kind must be one of {', '.join(SHAPE_KINDS)}, not {kind!r}")


def _shape_kinds(kinds: Any) -> Tuple[str, ...]:
    chosen = (kinds,) if isinstance(kinds, str) else tuple(kinds)
    unknown = [k for k in chosen if k not in SHAPE_KINDS]
    if unknown or not chosen:
        raise ValueError(f"shape kinds must come from {', '.join(SHAPE_KINDS)}; got {list(chosen)!r}")
    return chosen


def shapes(
    coords: Coords,
    palette: Palette,
    *,
    count: int = 6,
    hardness: float = 0.97,
    kinds: Sequence[str] = SHAPE_KINDS,
    rng: Any,
) -> WeightField:
    """Hard-edged geometric shapes stacked over the darkest color, a poster or
    Bauhaus composition: each shape is one palette color and covers what is
    under it, as with blob_layers.

    The image is cut into a loose grid of cells about 1.25 half short sides
    across (at least 2 by 2). Shapes take distinct cells in turn, wander up to
    a third of a cell from the center, and are big: radius 0.4 to 0.8 of a
    cell, the bottom one 1.3 times that, so edges overlap. About 4 in 10 are
    pushed against the nearest frame edge so the frame cuts them. kinds are
    the allowed SHAPE_KINDS (half and quarter circles, boxes and bars sit at
    right angles; rotated rectangles lean 15 to 75 degrees).

    Coverage comes from analytic signed distances (shape_distance) through
    shape_coverage: at hardness 1 an edge is one pixel of anti-aliasing, and
    it softens toward blob-like edges as hardness drops. The edge is only as
    sharp as the grid, so use a full-resolution grid (see catalog.compose).
    """
    allowed = _shape_kinds(kinds)
    n = palette.size
    total = max(1, int(count))
    lightness = np.asarray(palette.oklab)[:, 0]
    base = int(np.argmin(lightness))
    others = [i for i in range(n) if i != base] or [base]

    u, v, umax, vmax = _frame_extent(coords)
    cell = 1.25
    cols, rows = max(2, round(2.0 * umax / cell)), max(2, round(2.0 * vmax / cell))
    cell_u, cell_v = 2.0 * umax / cols, 2.0 * vmax / rows

    # A fixed number of draws per shape, independent of the palette size and kinds.
    keys = [rng.uniform(0.0, 1.0) for _ in range(total)]
    layout = [tuple(rng.uniform(0.0, 1.0) for _ in range(10)) for _ in range(total)]
    slots = [others[i % len(others)] for i in range(total)]
    colors = [slot for _, slot in sorted(zip(keys, slots))]

    free = list(range(cols * rows))
    alphas = []
    for k, (pick, jit_u, jit_v, bleed, push, size, ratio, quarter, lean, kind_draw) in enumerate(layout):
        if not free:
            free = list(range(cols * rows))
        index = free.pop(min(int(pick * len(free)), len(free) - 1))
        cu = -umax + (index % cols + 0.5 + (jit_u - 0.5) * 0.66) * cell_u
        cv = -vmax + (index // cols + 0.5 + (jit_v - 0.5) * 0.66) * cell_v
        radius = cell * (0.4 + 0.4 * size) * (1.3 if k == 0 else 1.0)
        if bleed < 0.4:
            # Against the nearer frame edge on the axis the shape sits closest to, overhanging it.
            overhang = radius * (0.1 + 0.6 * push)
            if abs(cu) / umax >= abs(cv) / vmax:
                cu = np.sign(cu or 1.0) * (umax - radius + overhang)
            else:
                cv = np.sign(cv or 1.0) * (vmax - radius + overhang)
        kind = allowed[min(int(kind_draw * len(allowed)), len(allowed) - 1)]
        angle = int(quarter * 4.0) * (np.pi / 2.0)
        if kind == "rotated-rectangle":
            angle += np.pi / 12.0 + lean * (np.pi / 3.0)
        if kind == "bar":
            ratio = 0.07 + 0.13 * ratio
        elif kind == "ring":
            ratio = 0.18 + 0.22 * ratio
        else:
            ratio = 0.5 + 0.5 * ratio
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        du, dv = u - cu, v - cv
        distance = shape_distance(kind, du * cos_a + dv * sin_a, -du * sin_a + dv * cos_a, radius, ratio)
        alphas.append(shape_coverage(distance, hardness))
    return WeightField(composite_over(alphas, colors, base, n), background=base)


# Julia constants whose sets are connected or dendritic and fill a frame with fine filaments.
GLOW_JULIA_CONSTANTS = (
    (-0.8, 0.156), (-0.7269, 0.1889), (0.285, 0.01), (-0.4, 0.6), (-0.835, -0.2321),
    (-0.70176, -0.3842), (0.355, 0.355), (-0.1, 0.651),
)
GLOW_KINDS = ("julia", "mandelbrot")
_GLOW_BAILOUT = 1.0e3
_GLOW_COMPACT_EVERY = 4
_GLOW_SHARPNESS = 0.8  # < 1 gives the glow a sharp bright core and a long soft tail


def _distance_estimate(
    z0: np.ndarray, c: np.ndarray, *, mandelbrot: bool, iterations: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Escape-time orbit with derivative, for every point at once.

    Returns the distance estimate 0.5 |z| ln|z| / |dz| to the set boundary (in the
    complex plane), the smooth iteration count, and a mask of points that escaped
    within the budget. Escaped points leave the working arrays every few steps; the
    estimate and the smooth count do not change with extra squarings, so the delay
    only costs time. float64 throughout."""
    shape = z0.shape
    z = z0.ravel().astype(np.complex128)
    cc = np.broadcast_to(c, shape).ravel().astype(np.complex128)
    dz = np.full(z.shape, 1.0 if not mandelbrot else 0.0, dtype=np.complex128)
    index = np.arange(z.size)
    distance = np.zeros(z.size)
    smooth = np.zeros(z.size)
    escaped = np.zeros(z.size, dtype=bool)
    bailout2 = _GLOW_BAILOUT**2
    with np.errstate(all="ignore"):
        for step in range(1, iterations + 1):
            dz = 2.0 * z * dz + 1.0 if mandelbrot else 2.0 * z * dz
            z = z * z + cc
            if step % _GLOW_COMPACT_EVERY and step != iterations:
                continue
            r2 = z.real * z.real + z.imag * z.imag
            out = r2 > bailout2
            if out.any():
                hit = index[out]
                zo, dzo = z[out], dz[out]
                mag = np.abs(zo)
                distance[hit] = 0.5 * mag * np.log(mag) / np.maximum(np.abs(dzo), 1e-300)
                smooth[hit] = step + 1.0 - np.log2(np.log(mag))
                escaped[hit] = True
                keep = ~out
                z, dz, cc, index = z[keep], dz[keep], cc[keep], index[keep]
                if z.size == 0:
                    break
    return distance.reshape(shape), smooth.reshape(shape), escaped.reshape(shape)


def fractal_glow(
    coords: Coords,
    *,
    kind: str = "julia",
    zoom: float = 1.0,
    center: Optional[Sequence[float]] = None,
    c: Optional[Sequence[float]] = None,
    fade: float = 0.005,
    iterations: int = 200,
    interior: float = 0.0,
    rng: Any,
) -> ScalarField:
    """A neon fractal: 1.0 on the boundary of a Julia or Mandelbrot set, falling
    to 0 over a soft band around it.

    The value is exp(-(d / width)^0.8), d the distance estimate to the boundary and
    width fade times the short side of the image, so the filaments are crisp at any
    resolution and the fade scales with the image. Pair it with a palette that
    starts dark and the field maps to a dark ground with bright filaments (bloom
    then lights the band). It is computed on its own view of the plane, not on the
    fractal camera, one sample per output pixel (see catalog.compose).

    kind: "julia" (constant c; a curated one is drawn unless c is given) or
    "mandelbrot". zoom magnifies the view (1 shows a Julia set filling the frame
    with some of it cut by the edge); center is the view's middle in the complex
    plane (drawn near the origin for Julia, or near the cardioid for Mandelbrot,
    when unset). The view also turns by a random angle. Inside the set the glow
    continues symmetrically, with the distance measured in pixels. interior in 0..1
    multiplies the glow outside the set by the smooth iteration count's own bands,
    for stripes in the fade band; 0 leaves the plain glow. Always draws the same
    number of random values, so setting any parameter does not shift the others.
    """
    if kind not in GLOW_KINDS:
        raise ValueError(f"fractal-glow kind must be one of {', '.join(GLOW_KINDS)}, not {kind!r}")
    mandelbrot = kind == "mandelbrot"
    draws = [rng.uniform(0.0, 1.0) for _ in range(6)]
    constant = GLOW_JULIA_CONSTANTS[min(int(draws[0] * len(GLOW_JULIA_CONSTANTS)), len(GLOW_JULIA_CONSTANTS) - 1)]
    c_value = complex(*c) if c is not None else complex(*constant)
    angle = draws[1] * 2.0 * np.pi
    zoom = max(float(zoom), 1e-3)
    if center is not None:
        mid = complex(*center)
    elif mandelbrot:
        mid = complex(-0.55 + 0.3 * (draws[2] - 0.5), 0.5 * (draws[3] - 0.5))
    else:
        mid = complex(0.5 * (draws[2] - 0.5), 0.5 * (draws[3] - 0.5))
    # Complex units per half short side: a Julia set (radius about 1.5) overhangs a 16:9 frame a little.
    span = (0.95 if mandelbrot else 0.75) * (1.0 + 0.2 * (draws[4] - 0.5)) / zoom

    u, v, _, _ = _frame_extent(coords)
    h, w = coords.shape
    spin = np.exp(1j * angle)
    p = mid + spin * span * (u.astype(np.float64) + 1j * v.astype(np.float64))
    distance, smooth, escaped = _distance_estimate(
        np.zeros_like(p) if mandelbrot else p, p if mandelbrot else c_value, mandelbrot=mandelbrot,
        iterations=max(1, int(iterations)),
    )

    width = max(float(fade), 1e-4) * 2.0 * span  # the fade, in complex units
    pixel = 2.0 * span / max(min(w, h), 1)
    # The profile has no cusp at the boundary (a third of a pixel of softening, rescaled
    # so the peak stays 1): a sample that flips between escaped and not, or a distance
    # that shifts by rounding, then moves the value by a small amount, not a jump.
    soft = 0.35 * pixel

    def profile(d: np.ndarray) -> np.ndarray:
        reach = np.sqrt(d * d + soft * soft) / width
        return np.exp(-(reach**_GLOW_SHARPNESS - (soft / width) ** _GLOW_SHARPNESS))

    glow = np.where(escaped, profile(distance), 0.0)
    if not escaped.all() and escaped.any():
        # Inside the set (or too slow to escape): the distance to the nearest escaped sample.
        inner = scipy.ndimage.distance_transform_edt(~escaped) * pixel
        glow = np.where(escaped, glow, profile(np.maximum(inner - pixel, 0.0)))
    mix = float(np.clip(interior, 0.0, 1.0))
    if mix > 0.0:
        bands = 0.5 + 0.5 * np.cos(smooth * (np.pi / 3.0))
        glow = np.where(escaped, glow * ((1.0 - mix) + mix * bands), glow)
    return ScalarField(np.clip(glow, 0.0, 1.0).astype(np.float32))


BLEND_MODES = ("add", "multiply", "screen", "difference", "overlay", "min", "max")


def blend_fields(
    field: ScalarField, other: ScalarField, *, mode: str = "screen", amount: float = 1.0, normalize: bool = True
) -> ScalarField:
    """Combines two scalar fields as image layers: field is the bottom layer and
    other the top. amount mixes from field alone (0) to the full blend (1);
    normalize stretches the result back to 0..1, which keeps a difference of two
    similar fields from mapping to a narrow slice of the palette."""
    a = np.asarray(field.t, dtype=np.float32)
    b = np.asarray(other.t, dtype=np.float32)
    if mode == "add":
        out = np.clip(a + b, 0.0, 1.0)
    elif mode == "multiply":
        out = a * b
    elif mode == "screen":
        out = 1.0 - (1.0 - a) * (1.0 - b)
    elif mode == "difference":
        out = np.abs(a - b)
    elif mode == "overlay":
        out = np.where(a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    elif mode == "min":
        out = np.minimum(a, b)
    elif mode == "max":
        out = np.maximum(a, b)
    else:
        raise ValueError(f"blend mode must be one of {', '.join(BLEND_MODES)}, not {mode!r}")
    out = a + float(amount) * (out - a)
    return ScalarField(normalize_scalar_field(out) if normalize else np.clip(out, 0.0, 1.0).astype(np.float32))


def _resize_plane(plane: np.ndarray, size: Tuple[int, int], method: str = "bicubic") -> np.ndarray:
    """Resizes one float plane to size (width, height): PIL bicubic, which aligns
    pixel centers as the export's Lanczos does, or "spline", the cubic spline
    that bicubic_upscale uses, which aligns the corner samples (the two disagree
    by tens of pixels on a lattice as coarse as fluid-wave's)."""
    from PIL import Image

    if method == "spline":
        width, height = size
        rows, cols = plane.shape
        return scipy.ndimage.zoom(plane, (height / rows, width / cols), order=3, mode="nearest").astype(np.float32)
    image = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32)


def field_grain(field: Field, coords: Coords, *, amount: float = 0.1, resize: str = "bicubic", rng: Any) -> Field:
    """Adds noise to the field itself, before it is mapped to colors, so the
    grain takes the palette's own colors: neighboring hues speckle into each
    other instead of grey noise lying over them (finish-stage film grain).

    Resolution. Engines evaluate the field on a grid a quarter of the output
    size (sample_grid), map it there and enlarge the colors in the export
    stage, which would blur noise added at that size into blotches four pixels
    wide. So a field coarser than coords.output_size is first enlarged to it
    (the mapping then runs at output resolution and the export has nothing left
    to resize), and the noise is drawn per output pixel. resize picks the
    enlargement so the gradient matches the one the plan makes without grain:
    "bicubic" where the export enlarges the colors, "spline" for plans that
    enlarge them with bicubic_upscale (fluid-wave). A field
    that is already at output size, as in spline-wave, flow-field and idw, is
    used as is. The returned field is at output size; the plan's own field,
    which contours and shading read, is left untouched.

    Scalar fields. Zero-mean Gaussian noise of standard deviation amount times
    the field's robust range (its 1st to 99th percentile span, at least 0.25 so
    a nearly flat field still grains, at most 1 since mappings read 0..1).
    Every mapping reads the field as a position on the palette in 0..1 and
    clips it to that (none renormalizes it), so amount is a fraction of the
    ramp, for escape-fractal as for simplex. Bands and cosine cycles also
    multiply the field by their cycle count or frequency; the plan divides
    amount by that before calling (catalog._with_field_grain). The result is folded back into
    0..1 by reflection rather than clipped, so pixels sitting on the first or
    last color still speckle toward their neighbor instead of half of them
    staying flat.

    Weight fields (blobs, patch-mesh, strokes, rbf, idw, flow-field). Noise
    moves weight between layers: each layer's weight gets its own Gaussian
    noise of standard deviation amount, with the mean over layers removed so
    the total is unchanged, then weights are clipped at 0 and renormalized to
    sum to 1. A pixel of one color thus picks up specks of the others, in
    proportion to amount, with no order between colors (a weight field has
    none). Chosen from renders over a single shared noise plane pushing every
    pixel along one fixed random direction in layer space: that clips at zero
    for most pixels, so it came out as sparse same-hue specks (confetti) on a
    clean background, where independent per-layer noise reads as even grain.
    Specks of the other colors lighten an area of one dark color slightly.

    The noise comes from one draw of rng, which seeds a numpy generator, so it
    is reproducible per seed and unaffected by the field's size.
    """
    width, height = coords.output_size
    generator = np.random.default_rng(rng.getrandbits(63))
    amount = float(amount)

    if isinstance(field, ScalarField):
        t = np.asarray(field.t, dtype=np.float32)
        if t.shape != (height, width):
            t = np.clip(_resize_plane(t, (width, height), resize), 0.0, 1.0)
        low, high = np.percentile(t, (1.0, 99.0))
        span = float(np.clip(high - low, 0.25, 1.0))
        noisy = t + generator.standard_normal(t.shape, dtype=np.float32) * np.float32(amount * span)
        # Reflect at 0 and 1: a triangle wave of period 2.
        folded = np.abs((noisy + 1.0) % 2.0 - 1.0)
        return ScalarField(folded.astype(np.float32))

    if isinstance(field, WeightField):
        weights = np.asarray(field.weights, dtype=np.float32)
        count, rows, cols = weights.shape
        if (rows, cols) != (height, width):
            weights = np.stack([_resize_plane(plane, (width, height), resize) for plane in weights])
        noise = generator.standard_normal(weights.shape, dtype=np.float32)
        noise -= noise.mean(axis=0, keepdims=True)
        weights = np.clip(weights + noise * np.float32(amount), 0.0, None)
        weights /= np.maximum(weights.sum(axis=0), 1e-6)
        return WeightField(weights, background=field.background)

    raise TypeError(f"cannot add grain to {type(field).__name__}")
