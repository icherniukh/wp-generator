"""A wallpaper's embedded metadata must reproduce it exactly."""

import numpy as np
import pytest

from engine.metadata import extract_image_metadata
from engine.pipeline import RenderConfiguration, WallpaperPipeline

PRESETS = ["afterglow", "balanced", "cupertino", "cyberpunk", "fractal", "ink", "nocturne", "nothing", "pastel", "sleek"]


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("seed", [3, 41])
def test_metadata_replays_the_same_image(preset, seed):
    pipeline = WallpaperPipeline()
    _, original = pipeline.render_png_bytes(RenderConfiguration(preset=preset, width=160, height=100, seed=seed))
    png, _ = pipeline.render_png_bytes(RenderConfiguration(preset=preset, width=160, height=100, seed=seed))
    metadata = extract_image_metadata(png)
    assert metadata is not None
    replayed = pipeline.render(metadata.to_render_configuration())
    diff = np.abs(np.asarray(original.image, np.int16) - np.asarray(replayed.image, np.int16))
    assert diff.max() == 0, f"{preset} seed {seed}: max diff {diff.max()}"
