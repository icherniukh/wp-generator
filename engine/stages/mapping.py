"""Mapping stages: color a field with the palette."""

from __future__ import annotations

import numpy as np
import scipy.ndimage

from engine.gradients.interpolation import evaluate_decoupled_palette, interpolate_color_curve
from engine.stages.types import Field, OklabImage, Palette, ScalarField, SrgbImage, WeightField


def map_colors(field: Field, palette: Palette) -> OklabImage:
    """A scalar field runs through the palette in order on a PCHIP curve in Oklab;
    a weight field blends the colors by weight in Oklab."""
    if isinstance(field, ScalarField):
        return OklabImage(interpolate_color_curve(field.t, palette.oklab))
    if isinstance(field, WeightField):
        return OklabImage(np.einsum("nhw,nc->hwc", field.weights, palette.oklab))
    raise TypeError(f"cannot map {type(field).__name__}")


def blend_srgb(field: WeightField, palette: Palette) -> SrgbImage:
    """Blends the colors by weight in sRGB (darker, muddier mixes than Oklab)."""
    return SrgbImage(np.einsum("ihw,ic->hwc", field.weights, palette.srgb))


def cosine_cycles(
    field: ScalarField,
    palette: Palette,
    *,
    frequency: float,
    phase: float,
    channel_offset: float,
) -> OklabImage:
    """Runs the field through the palette frequency times, back and forth on a cosine.

    Lightness follows the same curve shifted by channel_offset, so light and hue
    bands can separate.
    """
    eff_freq = max(0.1, float(frequency))
    eff_phase = float(phase) % 1.0
    eff_chan = float(channel_offset)
    u = 0.5 - 0.5 * np.cos(np.pi * (field.t * eff_freq + eff_phase))
    u_L = 0.5 - 0.5 * np.cos(np.pi * (field.t * eff_freq + eff_phase + eff_chan))
    h, w = field.t.shape
    return OklabImage(evaluate_decoupled_palette(palette.oklab, u_L, u, h, w))


def bicubic_upscale(image: OklabImage, *, width: int, height: int) -> OklabImage:
    """Enlarges the mapped image to the output size with cubic splines (colors, not the field)."""
    rows, cols, _ = image.values.shape
    return OklabImage(
        scipy.ndimage.zoom(image.values, (height / rows, width / cols, 1), order=3, mode="nearest")
    )


def bands(field: ScalarField, palette: Palette, *, cycles: float = 3.0, repeat: str = "mirror", phase: float = 0.0) -> OklabImage:
    """Runs the palette cycles times across the field.

    repeat "mirror" goes there and back (smooth bands); "repeat" jumps back to
    the first color each cycle (hard edges).
    """
    t = field.t * float(cycles) + float(phase)
    if repeat == "mirror":
        t = 1.0 - np.abs(2.0 * (t % 1.0) - 1.0)
    elif repeat == "repeat":
        t = t % 1.0
    else:
        raise ValueError(f"repeat must be mirror or repeat, not {repeat!r}")
    return OklabImage(interpolate_color_curve(t, palette.oklab))
