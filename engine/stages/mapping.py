"""Mapping stage: color a field with the palette."""

from __future__ import annotations

import numpy as np

from engine.gradients.interpolation import interpolate_color_curve
from engine.stages.types import Field, OklabImage, Palette, ScalarField, WeightField


def map_colors(field: Field, palette: Palette) -> OklabImage:
    """A scalar field runs through the palette in order on a PCHIP curve in Oklab;
    a weight field blends the colors by weight in Oklab."""
    if isinstance(field, ScalarField):
        return OklabImage(interpolate_color_curve(field.t, palette.oklab))
    if isinstance(field, WeightField):
        return OklabImage(np.einsum("nhw,nc->hwc", field.weights, palette.oklab))
    raise TypeError(f"cannot map {type(field).__name__}")
