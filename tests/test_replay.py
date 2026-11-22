"""A wallpaper's embedded metadata must reproduce it exactly."""

import numpy as np
import pytest

from engine.metadata import extract_image_metadata
from engine.pipeline import RenderConfiguration, WallpaperPipeline

PRESETS = ["afterglow", "aura", "balanced", "cupertino", "cyberpunk", "fractal", "holo", "ink", "mesh", "nocturne", "nothing", "pastel", "sleek"]


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


COMPOSED = [
    dict(engine="simplex", field_stage="voronoi", warps=["twist"], mapping="bands",
         stage_params={"voronoi": {"mode": "edges"}, "twist": {"amount": 3.0}}),
    dict(engine="domain-warp", field_stage="spiral"),
    dict(engine="fractal", warps=["none", "bulge"]),
    dict(engine="rbf", field_stage="ridged", mapping="cycles"),
    dict(engine="patch-mesh", warps=["sphere"], stage_params={"patch-mesh": {"jitter": 0.4}}),
    dict(engine="simplex", field_stage="patch-mesh"),
    dict(engine="blobs", warps=["noise"], stage_params={"blob-layers": {"blur": 0.8}}),
    dict(engine="simplex", field_stage="conic", blend="radial:difference", mapping="bands",
         stage_params={"blend": {"amount": 0.8}}),
    dict(engine="simplex", field_stage="ridged", shading=0.6, glass=True),
    dict(engine="blobs", shading=0.6),
    dict(engine="simplex", width=640, height=400, grain_size=3.0, grain_color="color", grain_blend="soft-light"),
    dict(engine="blobs", width=640, height=400, grain_size=6.0, grain_blend="overlay"),
    dict(engine="simplex", grain_color="color", grain_blend="multiply"),
]


@pytest.mark.parametrize("composition", COMPOSED, ids=lambda c: f"{c['engine']}+{c.get('field_stage') or c.get('warps')}")
def test_composed_render_replays_from_metadata(composition):
    pipeline = WallpaperPipeline()
    config = RenderConfiguration(**{"width": 160, "height": 100, "seed": 8, "noise": 0.2, **composition})
    png, original = pipeline.render_png_bytes(config)
    metadata = extract_image_metadata(png)
    replayed = pipeline.render(metadata.to_render_configuration())
    diff = np.abs(np.asarray(original.image, np.int16) - np.asarray(replayed.image, np.int16))
    assert diff.max() == 0


def test_composition_on_custom_engine_fails_loudly():
    from engine.gradients.registry import GRADIENT_REGISTRY, register_gradient_engine

    class Flat:
        def __init__(self, width, height):
            self.size = (width, height)

        def generate(self, colors):
            from PIL import Image
            return Image.new("RGB", self.size, tuple(colors[0]))

    register_gradient_engine("flat-test", Flat)
    try:
        with pytest.raises(ValueError, match="custom class"):
            WallpaperPipeline().render(RenderConfiguration(engine="flat-test", field_stage="conic", width=20, height=20, seed=1))
    finally:
        GRADIENT_REGISTRY.pop("flat-test", None)
