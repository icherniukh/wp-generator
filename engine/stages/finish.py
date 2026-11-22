"""Finish stages: image effects applied to the exported 8-bit image, in a fixed order.

Sizes in pixels (glass blur, contour width and smoothing) are given for an
image whose short side is REFERENCE_SHORT_SIDE and scale with the image, so a
small preview looks like the full render reduced.

Contours can follow the image's luminance (default) or the engine's scalar
field, and sit under the glass and aberration (frosted, default) or over them
(sharp).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage
from PIL import Image

from engine.colors import rgb_to_oklab
from engine.filters import WallpaperFilters
from engine.stages.runner import Stage, StageParams
from engine.stages.types import Field, ScalarField


# The default device (MacBook Pro 14", 3024 x 1964), where the pixel sizes below were tuned.
REFERENCE_SHORT_SIDE = 1964


def size_scale(p: StageParams) -> float:
    """The image's short side relative to REFERENCE_SHORT_SIDE."""
    return min(int(p["width"]), int(p["height"])) / REFERENCE_SHORT_SIDE


def contour_lines(
    image: Image.Image,
    *,
    intensity: float,
    levels: int,
    line_width: float,
    mode: str,
    smoothing_radius: float = 2.5,
) -> Image.Image:
    """Isolines of the image's smoothed luminance, drawn over it."""
    return WallpaperFilters.apply_contour_lines(
        image,
        levels=levels,
        line_width=line_width,
        intensity=intensity,
        mode=mode,
        smoothing_radius=smoothing_radius,
    )


def field_elevation(field: ScalarField, size: Tuple[int, int]) -> np.ndarray:
    """The field's values enlarged bicubically to size (width, height), in [0, 1]."""
    t = Image.fromarray(np.asarray(field.t, dtype=np.float32), mode="F")
    if t.size != size:
        t = t.resize(size, Image.Resampling.BICUBIC)
    return np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)


def field_contour_lines(
    image: Image.Image,
    field: ScalarField,
    *,
    intensity: float,
    levels: int,
    line_width: float,
    mode: str,
    smoothing_radius: float = 2.5,
) -> Image.Image:
    """Isolines of the engine's scalar field, so they follow the shape the
    engine computed rather than the colors the palette gave it."""
    return WallpaperFilters.apply_contour_lines(
        image,
        levels=levels,
        line_width=line_width,
        intensity=intensity,
        mode=mode,
        smoothing_radius=smoothing_radius,
        elevation=field_elevation(field, image.size),
    )


def frosted_glass(image: Image.Image, *, blur_radius: int, noise_amount: float) -> Image.Image:
    """Gaussian blur at half size, then a screen-blended speckle."""
    return WallpaperFilters.apply_frosted_glass(image, blur_radius=blur_radius, noise_amount=noise_amount)


def chromatic_aberration(image: Image.Image, *, strength: float) -> Image.Image:
    """Shifts red outward and blue inward, growing toward the edges."""
    return WallpaperFilters.apply_chromatic_aberration(image, strength=strength)


def film_grain(
    image: Image.Image,
    *,
    amount: float,
    grain_size: float = 1.0,
    grain_color: str = "mono",
    grain_blend: str = "add",
) -> Image.Image:
    """Fine plus coarse Gaussian noise, weaker in deep shadows and highlights.

    grain_size enlarges the grain (1.0 is the default grain), grain_color "color" draws
    the noise per channel, and grain_blend applies it by add, overlay, soft-light or multiply.
    """
    return WallpaperFilters.apply_film_grain(
        image, amount=amount, grain_size=grain_size, grain_color=grain_color, grain_blend=grain_blend
    )


def vignette(image: Image.Image, *, strength: float) -> Image.Image:
    """Darkens toward the corners."""
    return WallpaperFilters.apply_vignette(image, strength=strength)


# The steepest surface tilt shade() allows, as rise over run (2.5 is about 68 degrees).
MAX_SLOPE = 2.5


def shade(
    final: Image.Image,
    field: Optional[Field],
    *,
    strength: float,
    depth: float,
    light_angle: float,
    light_elevation: float,
    specular: float,
    shininess: float,
    rim: float,
    refraction: float,
    smoothing: float = 0.0,
) -> Image.Image:
    """Lights the image as a relief, for a glossy 3D glass or embossed look.

    The height is the engine's scalar field, or the image's Oklab lightness
    when the field is a weight field (or absent); it is smoothed with a
    Gaussian of sigma `smoothing` pixels. Normals come from the height gradient
    times `depth`, measured per short side of the image, so the relief does not
    depend on the resolution. Lambert diffuse is normalized so a flat surface is
    unchanged, the Blinn-Phong highlight is zero on a flat surface, and a
    Schlick-Fresnel rim brightens steep slopes. light_angle (degrees,
    counterclockwise from the right) and light_elevation (degrees above the
    image plane) set the light. With refraction > 0 the image is sampled that
    many pixels along the normal first, like a lens. Deterministic; no RNG.
    """
    rgb = np.asarray(final.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]

    if isinstance(field, ScalarField):
        height = field_elevation(field, final.size)
    else:
        height = rgb_to_oklab(rgb)[..., 0].astype(np.float32)
    if smoothing > 0.0:
        height = scipy.ndimage.gaussian_filter(height, smoothing, mode="nearest")

    gy, gx = np.gradient(height)
    slope = depth * min(w, h)
    nx, ny = -gx * slope, -gy * slope
    # Soft-limit the tilt: an edge in the height is a near-vertical wall whose rim
    # and highlight would otherwise saturate and break into 8-bit noise.
    steepness = np.hypot(nx, ny)
    limit = np.tanh(steepness / MAX_SLOPE) * MAX_SLOPE / np.maximum(steepness, 1e-6)
    nx, ny = nx * limit, ny * limit
    inverse_length = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    nx, ny, nz = nx * inverse_length, ny * inverse_length, inverse_length

    base = rgb
    if refraction > 0.0:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        coordinates = np.array([y + ny * refraction, x + nx * refraction])
        base = np.stack(
            [scipy.ndimage.map_coordinates(rgb[..., c], coordinates, order=1, mode="nearest") for c in range(3)],
            axis=-1,
        )

    azimuth, elevation = np.radians(light_angle), np.radians(light_elevation)
    lx = np.cos(azimuth) * np.cos(elevation)
    ly = -np.sin(azimuth) * np.cos(elevation)  # image y grows downward
    lz = np.sin(elevation)

    diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0) / max(lz, 1e-3)

    # Half vector between the light and the viewer (looking along z). The
    # highlight is the Blinn-Phong lobe minus its flat-surface value, scaled so
    # the peak is 1: a flat field has no highlight.
    hx, hy, hz = lx, ly, lz + 1.0
    half_length = np.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hy, hz = hx / half_length, hy / half_length, hz / half_length
    facing = nx * hx + ny * hy + nz * hz
    flat = hz**shininess
    highlight = np.clip((np.clip(facing, 0.0, 1.0) ** shininess - flat) / max(1.0 - flat, 1e-6), 0.0, 1.0)
    fresnel = (1.0 - nz) ** 5

    shine = (specular * highlight + rim * fresnel)[..., None] * 255.0
    out = base * (1.0 - strength + strength * diffuse[..., None]) + strength * shine
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def _contours(p: StageParams) -> Dict:
    scale = size_scale(p)
    return {
        "intensity": p["contours"],
        "levels": p["contour_levels"],
        "line_width": p["contour_width"] * scale,
        "mode": p["contour_mode"],
        "smoothing_radius": 2.5 * scale,
    }


def _glass(p: StageParams) -> Dict:
    # frosted_glass blurs at half size with radius blur_radius // 2, so keep at least 2.
    return {"blur_radius": max(2, int(round(60 * size_scale(p)))), "noise_amount": 0.15}


def _grain(p: StageParams) -> Dict:
    args: Dict = {"amount": p["noise"]}
    size = p.get("grain_size", 1.0)
    # The size is given for the reference image and scales with it; the default stays 1 pixel.
    args["grain_size"] = 1.0 if size == 1.0 else max(1.0, size * size_scale(p))
    args["grain_color"] = p.get("grain_color", "mono")
    args["grain_blend"] = p.get("grain_blend", "add")
    return args


CONTOURS = Stage("contours", "finish", contour_lines, ("final",), "final", _contours)
FIELD_CONTOURS = Stage("field contours", "finish", field_contour_lines, ("final", "field"), "final", _contours)
GLASS = Stage("glass", "finish", frosted_glass, ("final",), "final", _glass)
ABERRATION = Stage("chromatic aberration", "finish", chromatic_aberration, ("final",), "final",
                   lambda p: {"strength": p["chromatic_aberration"]})
GRAIN = Stage("film grain", "finish", film_grain, ("final",), "final", _grain)
VIGNETTE = Stage("vignette", "finish", vignette, ("final",), "final", lambda p: {"strength": p["vignette"]})


def _shade(p: StageParams) -> Dict:
    # Pixel sizes are for REFERENCE_SHORT_SIDE and scale with the image; depth is
    # per short side, so it needs no scaling. The smoothing is wide because the
    # height of a weight-field image is its lightness, whose 8-bit noise and
    # dark-level steps would otherwise show as grain in the highlights.
    scale = size_scale(p)
    return {
        "strength": p["shading"],
        "depth": 3.5,
        "light_angle": 135.0,
        "light_elevation": 50.0,
        "specular": 0.45,
        "shininess": 10.0,
        "rim": 1.0,
        "refraction": 12.0 * scale,
        "smoothing": 30.0 * scale,
    }


SHADE = Stage("shading", "finish", shade, ("final", "field"), "final", _shade)


def finish_stages(p: StageParams) -> Tuple[Stage, ...]:
    """The enabled effects, in their fixed order.

    Contours go first (frosted: the glass softens them) or after the glass and
    aberration (sharp). contour_source "field" draws them from the field.
    """
    with p.reading_for("finish plan"):
        stages: List[Stage] = []
        contours = None
        if p["contours"] > 0.0:
            contours = FIELD_CONTOURS if p.get("contour_source", "image") == "field" else CONTOURS
        sharp = p.get("contour_placement", "frosted") == "sharp"
        if contours is not None and not sharp:
            stages.append(contours)
        if p["glass"]:
            stages.append(GLASS)
        if p.get("shading", 0.0) > 0.0:
            stages.append(SHADE)
        if p["chromatic_aberration"] > 0.0:
            stages.append(ABERRATION)
        if contours is not None and sharp:
            stages.append(contours)
        if p["noise"] > 0.0:
            stages.append(GRAIN)
        if p["vignette"] > 0.0:
            stages.append(VIGNETTE)
    return tuple(stages)
