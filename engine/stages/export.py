"""Export stage: working image to an 8-bit sRGB PIL image."""

from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image

from engine.gradients.interpolation import oklab_image_to_pil
from engine.stages.types import Coords, OklabImage, SrgbImage


def to_srgb_image(image: Union[OklabImage, SrgbImage], coords: Coords, *, resize: bool = True) -> Image.Image:
    """Quantizes to 8-bit sRGB; with resize, then enlarges to the output size with Lanczos."""
    if isinstance(image, SrgbImage):
        return Image.fromarray(np.clip(image.values, 0, 255).astype(np.uint8))
    return oklab_image_to_pil(image.values, coords.output_size if resize else None)
