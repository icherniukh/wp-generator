"""Export stage: working image to an 8-bit sRGB PIL image.

The image stays in floating point through the enlargement to the output size,
and is quantized once at the end with triangular dither, so smooth gradients
do not band. The dither pattern comes from a fixed generator, so it never
consumes draws from the render's random state.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from PIL import Image

from engine.colors import oklab_to_rgb
from engine.stages.types import Coords, OklabImage, SrgbImage

_DITHER_SEED = 0x5EED


def _resize_float(values: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Lanczos-resizes an (h, w, c) float array per channel to size (width, height)."""
    channels = [
        np.asarray(Image.fromarray(np.ascontiguousarray(values[..., c], dtype=np.float32), mode="F").resize(
            size, Image.Resampling.LANCZOS
        ))
        for c in range(values.shape[2])
    ]
    return np.stack(channels, axis=-1)


def quantize(values: np.ndarray, *, dither: bool = True) -> np.ndarray:
    """0-255 floats to uint8; with dither, adds triangular noise of one level first."""
    if dither:
        rng = np.random.default_rng(_DITHER_SEED)
        h, w = values.shape[:2]
        noise = rng.random((h, w, 1), dtype=np.float32) - rng.random((h, w, 1), dtype=np.float32)
        values = values + noise
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def to_srgb_image(
    image: Union[OklabImage, SrgbImage], coords: Coords, *, resize: bool = True, dither: bool = True
) -> Image.Image:
    """Converts to sRGB, enlarges to the output size (with resize), then quantizes to 8 bits."""
    if isinstance(image, SrgbImage):
        values = np.asarray(image.values, dtype=np.float32)
    else:
        values = np.clip(oklab_to_rgb(image.values), 0.0, 1.0).astype(np.float32) * 255.0
    if resize and (values.shape[1], values.shape[0]) != tuple(coords.output_size):
        values = _resize_float(values, tuple(coords.output_size))
    return Image.fromarray(quantize(values, dither=dither))
