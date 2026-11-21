"""Finish stages: image effects applied to the exported 8-bit image, in a fixed order.

Contours can follow the image's luminance (default) or the engine's scalar
field, and sit under the glass and aberration (frosted, default) or over them
(sharp).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from engine.filters import WallpaperFilters
from engine.stages.runner import Stage, StageParams
from engine.stages.types import ScalarField


def contour_lines(image: Image.Image, *, intensity: float, levels: int, line_width: float, mode: str) -> Image.Image:
    """Isolines of the image's smoothed luminance, drawn over it."""
    return WallpaperFilters.apply_contour_lines(
        image, levels=levels, line_width=line_width, intensity=intensity, mode=mode
    )


def field_elevation(field: ScalarField, size: Tuple[int, int]) -> np.ndarray:
    """The field's values enlarged bicubically to size (width, height), in [0, 1]."""
    t = Image.fromarray(np.asarray(field.t, dtype=np.float32), mode="F")
    if t.size != size:
        t = t.resize(size, Image.Resampling.BICUBIC)
    return np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)


def field_contour_lines(
    image: Image.Image, field: ScalarField, *, intensity: float, levels: int, line_width: float, mode: str
) -> Image.Image:
    """Isolines of the engine's scalar field, so they follow the shape the
    engine computed rather than the colors the palette gave it."""
    return WallpaperFilters.apply_contour_lines(
        image,
        levels=levels,
        line_width=line_width,
        intensity=intensity,
        mode=mode,
        elevation=field_elevation(field, image.size),
    )


def frosted_glass(image: Image.Image, *, blur_radius: int, noise_amount: float) -> Image.Image:
    """Gaussian blur at half size, then a screen-blended speckle."""
    return WallpaperFilters.apply_frosted_glass(image, blur_radius=blur_radius, noise_amount=noise_amount)


def chromatic_aberration(image: Image.Image, *, strength: float) -> Image.Image:
    """Shifts red outward and blue inward, growing toward the edges."""
    return WallpaperFilters.apply_chromatic_aberration(image, strength=strength)


def film_grain(image: Image.Image, *, amount: float) -> Image.Image:
    """Fine plus coarse Gaussian noise, weaker in deep shadows and highlights."""
    return WallpaperFilters.apply_film_grain(image, amount=amount)


def vignette(image: Image.Image, *, strength: float) -> Image.Image:
    """Darkens toward the corners."""
    return WallpaperFilters.apply_vignette(image, strength=strength)


def _contours(p: StageParams) -> Dict:
    return {
        "intensity": p["contours"],
        "levels": p["contour_levels"],
        "line_width": p["contour_width"],
        "mode": p["contour_mode"],
    }


CONTOURS = Stage("contours", "finish", contour_lines, ("final",), "final", _contours)
FIELD_CONTOURS = Stage("field contours", "finish", field_contour_lines, ("final", "field"), "final", _contours)
GLASS = Stage("glass", "finish", frosted_glass, ("final",), "final",
              lambda p: {"blur_radius": 60, "noise_amount": 0.15})
ABERRATION = Stage("chromatic aberration", "finish", chromatic_aberration, ("final",), "final",
                   lambda p: {"strength": p["chromatic_aberration"]})
GRAIN = Stage("film grain", "finish", film_grain, ("final",), "final", lambda p: {"amount": p["noise"]})
VIGNETTE = Stage("vignette", "finish", vignette, ("final",), "final", lambda p: {"strength": p["vignette"]})


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
        if p["chromatic_aberration"] > 0.0:
            stages.append(ABERRATION)
        if contours is not None and sharp:
            stages.append(contours)
        if p["noise"] > 0.0:
            stages.append(GRAIN)
        if p["vignette"] > 0.0:
            stages.append(VIGNETTE)
    return tuple(stages)
