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


class Frame:
    """Image-proportional positions: centered, in half short sides, square pixels.

    Built from the corners of the current x, y, so it holds on a plain grid, an
    unequally scaled one (simplex spans -0.5..0.5 on both axes) and a rotated
    or zoomed fractal camera alike. to_xy maps frame positions back.
    """

    def __init__(self, coords: Coords) -> None:
        x, y = coords.x, coords.y
        h, w = coords.shape
        self.center = (
            float(np.mean((x[0, 0], x[0, -1], x[-1, 0], x[-1, -1]))),
            float(np.mean((y[0, 0], y[0, -1], y[-1, 0], y[-1, -1]))),
        )
        half = max(min(w, h) / 2.0, 1e-9)
        # Basis: one frame unit along the image's width and height, in x, y units.
        across = np.array([x[0, -1] - x[0, 0], y[0, -1] - y[0, 0]], dtype=np.float64) / max(w - 1, 1) * half
        down = np.array([x[-1, 0] - x[0, 0], y[-1, 0] - y[0, 0]], dtype=np.float64) / max(h - 1, 1) * half
        self.basis = np.column_stack([across, down])
        if abs(np.linalg.det(self.basis)) < 1e-18:
            self.basis = np.eye(2)
        self.inverse = np.linalg.inv(self.basis)

    def from_xy(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx, dy = x - self.center[0], y - self.center[1]
        return self.inverse[0, 0] * dx + self.inverse[0, 1] * dy, self.inverse[1, 0] * dx + self.inverse[1, 1] * dy

    def to_xy(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        b = self.basis
        return self.center[0] + b[0, 0] * u + b[0, 1] * v, self.center[1] + b[1, 0] * u + b[1, 1] * v


def _apply_in_frame(coords: Coords, move: Any) -> Coords:
    frame = Frame(coords)
    u, v = frame.from_xy(coords.x, coords.y)
    nu, nv = move(u, v)
    x, y = frame.to_xy(nu, nv)
    return coords.moved(x.astype(coords.x.dtype), y.astype(coords.y.dtype), coords.displacement)


def twist(coords: Coords, *, amount: float = 2.0, radius: float = 1.0) -> Coords:
    """Swirls positions around the grid center: rotation of amount radians at
    the center, fading to none at radius (in half short sides)."""

    def move(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = np.sqrt(u**2 + v**2) / max(radius, 1e-6)
        theta = amount * np.clip(1.0 - r, 0.0, 1.0) ** 2
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        return u * cos_t - v * sin_t, u * sin_t + v * cos_t

    return _apply_in_frame(coords, move)


def bulge(coords: Coords, *, amount: float = 0.5, radius: float = 1.0) -> Coords:
    """Magnifies (amount > 0) or pinches (amount < 0) around the grid center,
    like a lens, fading out at radius (in half short sides)."""

    def move(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = np.sqrt(u**2 + v**2) / max(radius, 1e-6)
        scale = 1.0 - amount * np.clip(1.0 - r, 0.0, 1.0) ** 2
        return u * scale, v * scale

    return _apply_in_frame(coords, move)


def wave(
    coords: Coords, *, amplitude: float = 0.08, waves: float = 3.0, angle: float = 0.0
) -> Coords:
    """Shifts positions sideways by a sine running along angle (radians).

    amplitude is in half short sides; waves counts crests across the short side.
    """
    ux, uy = np.cos(angle), np.sin(angle)

    def move(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shift = amplitude * np.sin((u * ux + v * uy) * np.pi * waves / 2.0)
        return u - uy * shift, v + ux * shift

    return _apply_in_frame(coords, move)


def sphere(coords: Coords, *, amount: float = 1.0, radius: float = 1.0) -> Coords:
    """Wraps the disc of radius (in half short sides) around a sphere seen head on.

    A point at relative distance r inside the disc samples the surface at
    asin(0.96 r) / asin(0.96), so the center swells and the rim compresses; amount
    blends from no change (0) to the full sphere (1). Outside the disc nothing moves.
    The 0.96 keeps the rim's compression finite (about 2.7x instead of
    unbounded), which would otherwise alias into speckle.
    """
    rim = 0.96

    def move(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = np.sqrt(u**2 + v**2) / max(radius, 1e-6)
        inside = r < 1.0
        target = np.where(inside, np.arcsin(rim * np.clip(r, 0.0, 1.0)) / np.arcsin(rim), r)
        scale = np.where(r > 1e-9, (r + amount * (target - r)) / np.maximum(r, 1e-9), 1.0)
        return u * scale, v * scale

    return _apply_in_frame(coords, move)


def lens(coords: Coords, *, amount: float = 0.3) -> Coords:
    """Radial lens distortion over the whole image, r -> r * (1 + amount * r^2),
    with r relative to the corner distance. amount > 0 pulls the corners inwards
    (barrel); amount < 0 pushes them out (pincushion), and folds back below -1/3."""

    def move(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r2 = u**2 + v**2
        corner2 = max(float(np.max(r2)), 1e-12)
        scale = 1.0 + amount * r2 / corner2
        return u * scale, v * scale

    return _apply_in_frame(coords, move)


# Largest speed of the curl of two-octave simplex noise, in noise units (the
# 99.9th percentile of the gradient magnitude is about 7.1, the maximum 8.9),
# so the normalized velocity stays within about 1.
_CURL_PEAK = 7.5


def liquify(coords: Coords, *, amount: float = 0.4, scale: float = 0.8, steps: int = 6, rng: Any) -> Coords:
    """Smears positions along a divergence-free flow, like stirring liquid.

    The velocity is the curl of a smooth simplex-noise potential,
    v = (dpsi/dy, -dpsi/dx), normalized so its speed stays within about 1 at any
    scale. Positions are advected in steps equal moves of -(amount / steps) * v,
    with v looked up where they are, so nearby positions move together and the
    pattern is dragged into swirls rather than torn. amount is the farthest a
    position can travel, in half short sides; scale is the potential's frequency
    per half short side.

    A large amount or scale stretches the pattern into thin filaments; far
    enough (amount * scale above about 0.35 at 6 steps) the explicit steps
    overshoot and the finest filaments fold over. Raise steps to push that out.

    The potential is one noise plane over a square lattice that covers the
    positions plus a margin, and v is its finite-difference gradient, read back
    bilinearly: far cheaper than evaluating noise at every position and step.
    """
    ox, oy = rng.uniform(0.0, 500.0), rng.uniform(0.0, 500.0)
    steps = max(1, int(steps))
    if amount == 0.0:
        return coords
    scale = max(float(scale), 1e-3)
    h, w = coords.shape

    def move(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        margin = abs(amount) * 1.3 + 0.05
        u0, u1 = float(u.min()) - margin, float(u.max()) + margin
        v0, v1 = float(v.min()) - margin, float(v.max()) + margin
        span = max(u1 - u0, v1 - v0)
        density = float(np.clip(min(h, w) / 2.0, 32.0, 96.0))  # lattice samples per half short side
        n = int(np.clip(np.ceil(span * density) + 1, 16, 2048))
        cell = span / (n - 1)
        origin_u, origin_v = (u0 + u1) / 2.0 - span / 2.0, (v0 + v1) / 2.0 - span / 2.0
        # The plane spans 2 * plane_scale noise units over the lattice, scale per half short side.
        psi = generate_noise_plane(n, n, scale=scale * span / 2.0, octaves=2, offset=(ox, oy)).astype(np.float64)
        per_cell = 2.0 * (scale * span / 2.0) / (n - 1)  # noise units per lattice cell
        d_row, d_col = np.gradient(psi)
        norm = 1.0 / (per_cell * _CURL_PEAK)
        flow_u, flow_v = d_row * norm, -d_col * norm

        dt = float(amount) / steps
        for _ in range(steps):
            at = [(v - origin_v).ravel() / cell, (u - origin_u).ravel() / cell]
            u = u - dt * scipy.ndimage.map_coordinates(flow_u, at, order=1, mode="nearest").reshape(u.shape)
            v = v - dt * scipy.ndimage.map_coordinates(flow_v, at, order=1, mode="nearest").reshape(v.shape)
        return u, v

    return _apply_in_frame(coords, move)
