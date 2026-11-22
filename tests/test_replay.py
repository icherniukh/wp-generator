"""A wallpaper's embedded metadata must reproduce it exactly."""

import numpy as np
import pytest

from engine.metadata import extract_image_metadata
from engine.pipeline import RenderConfiguration, WallpaperPipeline

PRESETS = ["afterglow", "aura", "balanced", "cupertino", "cyberpunk", "fractal", "gel", "holo", "ink", "liquid", "mesh", "nocturne", "nothing", "pastel", "poster", "sleek"]


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
    dict(engine="simplex", warps=["liquify"], stage_params={"liquify": {"amount": 0.6, "steps": 8}}),
    dict(engine="simplex", field_stage="strokes", stage_params={"strokes": {"strokes": 5, "blur": 0.7}}),
    dict(engine="simplex", field_stage="conic", blend="radial:difference", mapping="bands",
         stage_params={"blend": {"amount": 0.8}}),
    dict(engine="simplex", field_stage="ridged", shading=0.6, glass=True),
    dict(engine="blobs", shading=0.6),
    dict(engine="simplex", width=240, height=160, halftone="dots", stage_params={"halftone": {"cell": 60.0}}),
    dict(engine="blobs", width=240, height=160, halftone="cmyk", noise=0.1, stage_params={"halftone": {"cell": 60.0, "mix": 0.8}}),
    dict(engine="simplex", width=240, height=160, halftone="dither", stage_params={"halftone": {"cell": 20.0, "levels": 3}}),
    dict(engine="simplex", width=640, height=400, grain_size=3.0, grain_color="color", grain_blend="soft-light"),
    dict(engine="blobs", width=640, height=400, grain_size=6.0, grain_blend="overlay"),
    dict(engine="simplex", grain_color="color", grain_blend="multiply"),
    dict(engine="simplex", field_stage="conic", blend="radial:difference", field_grain=0.15),
    dict(engine="blobs", field_grain=0.2, shading=0.5),
    dict(engine="fluid-wave", field_grain=0.1, contours=0.5, contour_source="field"),
    dict(engine="shapes", warps=["liquify"]),
    dict(engine="simplex", field_stage="shapes",
         stage_params={"shapes": {"kinds": ["circle", "ring"], "hardness": 0.8}}),
    dict(engine="blobs", shading=0.6, stage_params={"shading": {"light_angle": 45, "refraction": 0, "specular": 0.8, "tint": 0.5}}),
    dict(engine="shapes", bloom=0.6),
    dict(engine="simplex", field_stage="fractal-glow", bloom=0.6,
         colors=[(11, 15, 42), (255, 61, 129), (124, 77, 255), (0, 229, 255), (255, 224, 102)],
         stage_params={"fractal-glow": {"kind": "julia", "c": [-0.4, 0.6], "fade": 0.02, "interior": 0.5}}),
    dict(engine="blobs", width=640, height=400, bloom=0.5, shading=0.4,
         stage_params={"bloom": {"rays": 0.6, "ray_origin": [0.3, -0.1], "radii": [8, 30], "ray_length": 500}}),
    dict(engine="simplex", contours=0.5, noise=0.3, width=640, height=400,
         stage_params={"contours": {"line_width": 3.0}, "film-grain": {"grain_size": 4.0}, "twist": {"amount": 1.0}}, warps=["twist"]),
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
