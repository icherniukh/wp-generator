"""Data passed between render stages.

Each stage reads some of these from the render state and writes one back, so a
stage's inputs and output are visible from its declaration (see runner.Stage).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Palette:
    """Colors in render order, as sRGB 0-255 and as Oklab."""

    rgb: Tuple[Tuple[int, int, int], ...]
    oklab: np.ndarray  # (n, 3)

    @property
    def size(self) -> int:
        return len(self.rgb)


@dataclass(frozen=True)
class Coords:
    """Sample positions on the working grid.

    x, y are the current positions (after any warp); base_x, base_y are the
    positions before warping, which some fields combine with the warped ones.
    displacement holds the last warp's unscaled noise planes, for fields that
    reuse them.
    """

    x: np.ndarray  # (h, w)
    y: np.ndarray  # (h, w)
    base_x: np.ndarray
    base_y: np.ndarray
    output_size: Tuple[int, int]  # final (width, height); the grid is usually smaller
    displacement: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.x.shape

    def moved(
        self,
        x: np.ndarray,
        y: np.ndarray,
        displacement: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "Coords":
        return replace(self, x=x, y=y, displacement=displacement)


@dataclass(frozen=True)
class ScalarField:
    """One value per sample in [0, 1], mapped through the palette in order."""

    t: np.ndarray  # (h, w)


@dataclass(frozen=True)
class WeightField:
    """One weight per color per sample, summing to 1; colors blend with no order.

    background is the color index that fills where no emitter reaches, if any.
    """

    weights: np.ndarray  # (n, h, w)
    background: Optional[int] = None


Field = Union[ScalarField, WeightField]


@dataclass(frozen=True)
class OklabImage:
    """A mapped image on the working grid, still in floating-point Oklab."""

    values: np.ndarray  # (h, w, 3)
