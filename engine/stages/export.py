"""Export stage: Oklab working image to a full-size sRGB PIL image."""

from __future__ import annotations

from PIL import Image

from engine.gradients.interpolation import oklab_image_to_pil
from engine.stages.types import Coords, OklabImage


def to_srgb_image(image: OklabImage, coords: Coords) -> Image.Image:
    """Quantizes to 8-bit sRGB, then resizes to the output size with Lanczos."""
    return oklab_image_to_pil(image.values, coords.output_size)
