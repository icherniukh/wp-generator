"""Palette stage: the colors a render maps through."""

from __future__ import annotations

from typing import Sequence

from engine.gradients.interpolation import colors_to_oklab
from engine.stages.types import Palette


def make_palette(*, colors: Sequence[Sequence[float]]) -> Palette:
    rgb = tuple(tuple(int(v) for v in color[:3]) for color in colors)
    return Palette(rgb=rgb, oklab=colors_to_oklab(colors))
