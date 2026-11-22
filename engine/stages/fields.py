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
    sx = scipy.ndimage.zoom(coords.x.astype(np.float64), (sample_h / h, sample_w / w), order=1)
    sy = scipy.ndimage.zoom(coords.y.astype(np.float64), (sample_h / h, sample_w / w), order=1)
    ox, oy = rng.uniform(0, 1000), rng.uniform(0, 1000)
    total = np.zeros_like(sx)
    amp, freq, norm = 1.0, float(scale), 0.0
    for _ in range(max(1, int(octaves))):
        n = snoise2(sx * freq + ox, sy * freq + oy, octaves=1)
        total += amp * fold(n)
        norm += amp
        amp *= gain
        freq *= lacunarity
    grid = scipy.ndimage.zoom(total / norm, (h / total.shape[0], w / total.shape[1]), order=3)[:h, :w]
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
    with a Gaussian edge of sigma blur (in radii). Strokes share a flow
    direction, within about 40 degrees, each with its own color from the palette
    (every color but the darkest appears when there are enough strokes), and
    each covers what is under it, as with blob_layers.
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
        (rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9), rng.uniform(-0.6, 0.6), rng.uniform(3.0, 4.2),
         rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0.75, 1.25), rng.uniform(0.75, 1.0))
        for _ in range(count)
    ]
    slots = [others[i % len(others)] for i in range(count)]
    colors = [slot for _, slot in sorted(zip(keys, slots))]

    u, v, umax, vmax = _frame_extent(coords)
    taper = 0.55 + 0.45 * np.sin(np.linspace(0.0, 1.0, 24) * np.pi)
    alphas = []
    for (cu, cv, turn, length, bend, wobble, thickness, opacity) in layout:
        path = _stroke_path((cu * umax, cv * vmax), flow + turn, length, bend, wobble)
        radii = max(float(width), 1e-3) / 2.0 * thickness * taper
        alphas.append(opacity * _blurred_ellipse(_capsule_distance(u, v, path, radii), blur))
    return WeightField(composite_over(alphas, colors, base, n), background=base)


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
