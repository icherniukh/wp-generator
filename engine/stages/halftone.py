"""Halftone and ordered-dither screens for the finish stage of the same name.

The five modes turn the finished image into something printed:

dots    an amplitude-modulated (AM) screen on a rotated square grid. Each cell
        gets a round dot whose area is set by its luminance: paper and ink in
        that share average to the cell's luminance, which is 1 - luminance / paper
        luminance for a black ink. Dark cells are mostly ink, light cells mostly paper.
cmyk    four such screens at the classic angles (C 15, M 75, Y 0, K 45 degrees),
        composited subtractively (multiplied) on the paper.
lines   a line screen; the line's width follows the same tone.
dither  an 8 x 8 Bayer ordered dither to `levels` per channel, in blocks of
        `cell` pixels.
pixel   a mosaic of `cell` pixel blocks, quantized to `levels` per channel.

Paper and ink (dots, lines, cmyk). The page is the image's own lightest tone,
lifted toward white until its luma is at least PAPER_LUMA, so it reads as stock
and not as a grey wash, and a dark image still prints on light paper. The ink of
a dot is the color of the cell it sits in, darkened and made more saturated in
Oklab: a spot-colour (risograph) look in which the dots take the image's hues,
where black ink on paper would give grey newspaper. The area rule keeps the
cell's tone, so the picture still reads as the original from a distance.

The dot edge is anti-aliased analytically, from the distance to the dot boundary
across one pixel, never by resampling a larger image, so a screen at any angle
has no moire from the renderer. The tone of a cell is the average of the pixels
the cell covers, not a point sample.

Deterministic; no RNG.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage
from PIL import Image

from engine.colors import oklab_to_rgb, rgb_to_oklab

MODES: Tuple[str, ...] = ("dots", "cmyk", "lines", "dither", "pixel")

# The classic process-colour screen angles, in degrees.
CMYK_ANGLES = {"C": 15.0, "M": 75.0, "Y": 0.0, "K": 45.0}
# Ink colors of the four plates (a cyan, magenta, yellow and key black of offset printing).
CMYK_INKS = {
    "C": (0.0, 174.0, 239.0),
    "M": (236.0, 0.0, 140.0),
    "Y": (255.0, 242.0, 0.0),
    "K": (30.0, 26.0, 28.0),
}

# A screen needs a few pixels per cell to be drawn at all.
MIN_SCREEN_CELL = 2.0
# The luma the paper is lifted to at least (0 to 1, of the gamma-encoded color).
PAPER_LUMA = 0.9
# How the ink differs from the cell color, in Oklab: the chroma is boosted, and the
# lightness is scaled by INK_NEUTRAL for a grey cell (a near-black ink) up to
# INK_COLORED for a cell of INK_FULL_CHROMA or more (a deep spot colour).
INK_NEUTRAL = 0.25
INK_COLORED = 0.65
INK_FULL_CHROMA = 0.12
INK_CHROMA = 1.5

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 weights on the gamma-encoded channels, scaled to 0 to 1."""
    return (np.asarray(rgb, dtype=np.float32) @ _LUMA) / 255.0


def find_paper(rgb: np.ndarray) -> np.ndarray:
    """The image's lightest tone (the mean of its lightest 1 percent), lifted toward white.

    Returns float RGB on the 0-255 scale."""
    flat = rgb.reshape(-1, 3).astype(np.float32)
    lightness = flat @ _LUMA
    cut = np.percentile(lightness, 99.0)
    tone = flat[lightness >= cut].mean(axis=0)
    y = float(luma(tone))
    if y < PAPER_LUMA:
        t = (PAPER_LUMA - y) / max(1.0 - y, 1e-6)
        tone = tone + (255.0 - tone) * t
    return np.clip(tone, 0.0, 255.0)


def _bayer(n: int) -> np.ndarray:
    m = np.zeros((1, 1), dtype=np.int64)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return m


BAYER_8 = _bayer(8)


@lru_cache(maxsize=1)
def _area_to_radius_table() -> Tuple[np.ndarray, np.ndarray]:
    """Dot radius (in cells) for each covered area, once the dots touch and overlap.

    A dot of radius r <= 0.5 covers pi r^2 of its cell. Beyond that it spills into
    the neighbours and the union of the circles covers less than a lone circle
    would, so the radius for an area is read from the distance to the nearest dot
    centre over a unit cell: the area covered by radius r is the share of the cell
    within r of some centre.
    """
    n = 256
    offsets = (np.arange(n, dtype=np.float64) + 0.5) / n
    x, y = np.meshgrid(offsets, offsets)
    nearest = np.full(x.shape, np.inf)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            nearest = np.minimum(nearest, np.hypot(x - (du + 0.5), y - (dv + 0.5)))
    areas = np.linspace(np.pi / 4.0, 1.0, 257)
    return areas, np.quantile(nearest.ravel(), areas)


def radius_for_area(area: np.ndarray) -> np.ndarray:
    """Dot radius, in cells, whose dots cover `area` (0 to 1) of the plane."""
    area = np.clip(area, 0.0, 1.0)
    areas, radii = _area_to_radius_table()
    return np.where(area <= np.pi / 4.0, np.sqrt(area / np.pi), np.interp(area, areas, radii))


class _Grid:
    """A square cell grid rotated by `angle` degrees about the image centre."""

    def __init__(self, height: int, width: int, cell: float, angle: float) -> None:
        self.cell = float(cell)
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        x = x + 0.5 - width / 2.0
        y = y + 0.5 - height / 2.0
        t = np.radians(angle)
        c, s = np.cos(t), np.sin(t)
        self.u = (x * c + y * s) / np.float32(cell)
        self.v = (-x * s + y * c) / np.float32(cell)
        iu = np.floor(self.u).astype(np.int64)
        iv = np.floor(self.v).astype(np.int64)
        # One cell of margin so every neighbour of every pixel's cell is in the grid.
        # u and v keep their own origin: a grid index plus (offset_u, offset_v) is a cell's number in u, v.
        self.offset_u, self.offset_v = int(iu.min()) - 1, int(iv.min()) - 1
        self.iu = iu - self.offset_u
        self.iv = iv - self.offset_v
        self.shape = (int(self.iu.max()) + 2, int(self.iv.max()) + 2)
        self.ids = self.iu * self.shape[1] + self.iv

    def means(self, values: np.ndarray) -> np.ndarray:
        """The average of values (H, W, k) over each cell's pixels, as a grid (nu, nv, k).

        Cells no pixel falls in stay zero."""
        count = np.bincount(self.ids.ravel(), minlength=self.shape[0] * self.shape[1]).astype(np.float32)
        out = np.zeros((self.shape[0] * self.shape[1], values.shape[-1]), dtype=np.float32)
        for k in range(values.shape[-1]):
            out[:, k] = np.bincount(self.ids.ravel(), weights=values[..., k].ravel(), minlength=out.shape[0])
        out /= np.maximum(count, 1.0)[:, None]
        return out.reshape(self.shape + (values.shape[-1],))

    def dots(self, area: np.ndarray, colors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Anti-aliased coverage (H, W) of round dots whose cells' areas are `area` (nu, nv).

        With colors (nu, nv, 3) also the color of the dot that covers each pixel."""
        radius = radius_for_area(area).astype(np.float32) * np.float32(self.cell)  # pixels
        # Dots no wider than their cell (less the pixel of anti-aliasing) cannot reach a neighbour.
        spill = float(radius.max()) + 1.0 > self.cell / 2.0
        steps = [(du, dv) for du in (-1, 0, 1) for dv in (-1, 0, 1)] if spill else [(0, 0)]
        coverage = np.zeros(self.u.shape, dtype=np.float32)
        color = None if colors is None else np.zeros(self.u.shape + (3,), dtype=np.float32)
        for du, dv in steps:
            iu, iv = self.iu + du, self.iv + dv
            r = radius[iu, iv]
            d = np.hypot(self.u - (iu + self.offset_u + 0.5), self.v - (iv + self.offset_v + 0.5))
            d *= np.float32(self.cell)
            # A dot narrower than a pixel fades out with its area rather than keeping a half-covered pixel.
            c = np.clip(r - d + 0.5, 0.0, 1.0) * np.clip(r * 2.0, 0.0, 1.0) ** 2
            if color is not None:
                better = c > coverage
                color[better] = colors[iu, iv][better]
            coverage = np.maximum(coverage, c)
        return coverage, color


def _ink_colors(mean_rgb: np.ndarray) -> np.ndarray:
    """The ink for cells of these colors (0-255): the same hue, darker and more saturated."""
    lab = rgb_to_oklab(np.clip(mean_rgb, 0.0, 255.0))
    chroma = np.clip(np.hypot(lab[..., 1], lab[..., 2]) / INK_FULL_CHROMA, 0.0, 1.0)
    lightness = lab[..., 0] * (INK_NEUTRAL + (INK_COLORED - INK_NEUTRAL) * chroma)
    lab = np.concatenate([lightness[..., None], lab[..., 1:] * INK_CHROMA], axis=-1)
    return np.clip(oklab_to_rgb(lab), 0.0, 1.0) * 255.0


def _paper_array(rgb: np.ndarray, paper: Optional[Sequence[float]]) -> np.ndarray:
    if paper is None:
        return find_paper(rgb)
    value = np.asarray(paper, dtype=np.float32)
    if value.shape != (3,):
        raise ValueError(f"paper must be an RGB triple, not {paper!r}")
    return value


def _area_of_tone(tone: np.ndarray, paper: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """The share of a cell inked for a cell of this luma (0 to 1).

    Paper and ink in the right shares average to the cell's luma:
    (paper - luma) / (paper - ink luma), which is 1 - luma / paper luma for a black ink."""
    paper_luma = float(luma(paper))
    return np.clip((paper_luma - tone) / np.maximum(paper_luma - luma(ink), 0.05), 0.0, 1.0)


def dot_coverage(
    rgb: np.ndarray, *, cell: float, angle: float, paper: Optional[Sequence[float]] = None
) -> np.ndarray:
    """The ink coverage (H, W, 0 to 1) of the dots mode for an image (H, W, 3, 0-255)."""
    return _dots(rgb, cell, angle, _paper_array(rgb, paper))[1]


def _dots(rgb: np.ndarray, cell: float, angle: float, paper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    grid = _Grid(rgb.shape[0], rgb.shape[1], cell, angle)
    mean = grid.means(rgb)
    ink = _ink_colors(mean)
    area = _area_of_tone(luma(mean), paper, ink)
    coverage, color = grid.dots(area, ink)
    out = paper[None, None, :] * (1.0 - coverage[..., None]) + color * coverage[..., None]
    return out, coverage


def _plates(rgb: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """C, M, Y, K ink amounts (H, W, 4, 0 to 1) for the image printed on this paper."""
    r = np.clip(rgb / np.maximum(paper, 1.0), 0.0, 1.0)
    k = 1.0 - r.max(axis=-1)
    denominator = np.maximum(1.0 - k, 1e-4)[..., None]
    cmy = (1.0 - r - k[..., None]) / denominator
    return np.concatenate([np.clip(cmy, 0.0, 1.0), k[..., None]], axis=-1)


def _cmyk(rgb: np.ndarray, cell: float, paper: np.ndarray) -> np.ndarray:
    plates = _plates(rgb, paper)
    out = np.broadcast_to(paper / 255.0, rgb.shape).astype(np.float32).copy()
    for index, name in enumerate("CMYK"):
        grid = _Grid(rgb.shape[0], rgb.shape[1], cell, CMYK_ANGLES[name])
        area = grid.means(plates[..., index:index + 1])[..., 0]
        coverage, _ = grid.dots(area)
        ink = np.asarray(CMYK_INKS[name], dtype=np.float32) / 255.0
        out *= 1.0 - coverage[..., None] * (1.0 - ink)
    return out * 255.0


def _lines(rgb: np.ndarray, cell: float, angle: float, paper: np.ndarray) -> np.ndarray:
    smooth = np.stack(
        [scipy.ndimage.gaussian_filter(rgb[..., c], cell * 0.5, mode="nearest") for c in range(3)], axis=-1
    )
    ink = _ink_colors(smooth)
    width = _area_of_tone(luma(smooth), paper, ink) * cell  # pixels
    t = np.radians(angle)
    h, w = rgb.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    across = (-(x + 0.5 - w / 2.0) * np.sin(t) + (y + 0.5 - h / 2.0) * np.cos(t)) / cell
    distance = np.abs(across - np.floor(across) - 0.5) * cell
    coverage = np.clip(width / 2.0 - distance + 0.5, 0.0, 1.0) * np.clip(width, 0.0, 1.0)
    return paper[None, None, :] * (1.0 - coverage[..., None]) + ink * coverage[..., None]


def _blocks(rgb: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-block means of rgb in blocks of size pixels, and each pixel's block row and column."""
    h, w = rgb.shape[:2]
    rows, cols = np.arange(0, h, size), np.arange(0, w, size)
    sums = np.add.reduceat(np.add.reduceat(rgb, rows, axis=0), cols, axis=1)
    count = np.outer(np.diff(np.append(rows, h)), np.diff(np.append(cols, w)))[..., None]
    return sums / count, np.arange(h) // size, np.arange(w) // size


def _quantized(rgb: np.ndarray, cell: float, levels: int, ordered: bool) -> np.ndarray:
    size = max(1, int(round(cell)))
    mean, block_row, block_col = _blocks(rgb, size)
    value = mean / 255.0 * (levels - 1)
    if ordered:
        rows, cols = np.arange(mean.shape[0]) % 8, np.arange(mean.shape[1]) % 8
        threshold = (BAYER_8[np.ix_(rows, cols)] + 0.5) / 64.0
        steps = np.floor(value + threshold[..., None])
    else:
        steps = np.floor(value + 0.5)
    steps = np.clip(steps, 0, levels - 1)
    return (steps / (levels - 1) * 255.0)[block_row][:, block_col]


def halftone(
    final: Image.Image,
    *,
    mode: str,
    cell: float,
    angle: float,
    mix: float,
    levels: int,
    paper: Optional[Sequence[float]] = None,
) -> Image.Image:
    """Prints the image: a halftone screen, a line screen, or an ordered dither.

    `cell` is the screen's cell (dot spacing, line pitch, dither or pixel block) in
    pixels; the stage scales it with the image. `angle` (degrees) turns the dot
    and line screens (cmyk uses the classic angles; dither and pixel ignore it).
    `levels` (2 or more) is the number of levels per channel of dither and pixel.
    `mix` (0 to 1) blends the result with the original; 0 returns it unchanged.
    `paper` (RGB, 0-255) overrides the paper, the image's lightest tone lifted
    toward white (see the module docstring for the dot ink).
    """
    if mode not in MODES:
        raise ValueError(f"halftone mode must be one of {', '.join(MODES)}, not {mode!r}")
    if not 0.0 <= mix <= 1.0:
        raise ValueError(f"mix must be from 0 to 1, not {mix!r}")
    if not cell > 0.0:
        raise ValueError(f"cell must be positive, not {cell!r}")
    if mode in ("dither", "pixel") and (int(levels) != levels or levels < 2):
        raise ValueError(f"levels must be an integer of at least 2, not {levels!r}")
    if mix == 0.0:
        return final
    original = np.asarray(final.convert("RGB"), dtype=np.float32)
    levels = int(levels)

    if mode in ("dither", "pixel"):
        effect = _quantized(original, cell, levels, ordered=mode == "dither")
    else:
        screen_cell = max(float(cell), MIN_SCREEN_CELL)
        page = _paper_array(original, paper)
        if mode == "dots":
            effect = _dots(original, screen_cell, angle, page)[0]
        elif mode == "cmyk":
            effect = _cmyk(original, screen_cell, page)
        else:
            effect = _lines(original, screen_cell, angle, page)
    out = original + (effect - original) * mix if mix < 1.0 else effect
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
