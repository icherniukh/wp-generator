"""Finish stages: image effects applied to the exported 8-bit image, in a fixed order."""

from __future__ import annotations

from typing import Dict, List, Tuple

from PIL import Image

from engine.filters import WallpaperFilters
from engine.stages.runner import Stage, StageParams


def contour_lines(image: Image.Image, *, intensity: float, levels: int, line_width: float, mode: str) -> Image.Image:
    """Isolines of the image's smoothed luminance, drawn over it."""
    return WallpaperFilters.apply_contour_lines(
        image, levels=levels, line_width=line_width, intensity=intensity, mode=mode
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
GLASS = Stage("glass", "finish", frosted_glass, ("final",), "final",
              lambda p: {"blur_radius": 60, "noise_amount": 0.15})
ABERRATION = Stage("chromatic aberration", "finish", chromatic_aberration, ("final",), "final",
                   lambda p: {"strength": p["chromatic_aberration"]})
GRAIN = Stage("film grain", "finish", film_grain, ("final",), "final", lambda p: {"amount": p["noise"]})
VIGNETTE = Stage("vignette", "finish", vignette, ("final",), "final", lambda p: {"strength": p["vignette"]})


def finish_stages(p: StageParams) -> Tuple[Stage, ...]:
    """The enabled effects, in their fixed order."""
    with p.reading_for("finish plan"):
        stages: List[Stage] = []
        if p["contours"] > 0.0:
            stages.append(CONTOURS)
        if p["glass"]:
            stages.append(GLASS)
        if p["chromatic_aberration"] > 0.0:
            stages.append(ABERRATION)
        if p["noise"] > 0.0:
            stages.append(GRAIN)
        if p["vignette"] > 0.0:
            stages.append(VIGNETTE)
    return tuple(stages)
