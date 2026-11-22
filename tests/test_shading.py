"""Height-field shading: lighting from a scalar field or a weight field's lightness."""

import random

import numpy as np
from PIL import Image

from engine.metadata import extract_image_metadata
from engine.pipeline import RenderConfiguration, WallpaperPipeline
from engine.stages.finish import SHADE, finish_stages, shade
from engine.stages.runner import StageParams, run_stages
from engine.stages.types import ScalarField, WeightField

SIZE = (64, 40)
PARAMS = dict(
    strength=0.7, depth=3.0, light_angle=135.0, light_elevation=50.0,
    specular=0.5, shininess=10.0, rim=1.0, refraction=0.0, smoothing=0.0,
)
DIFFUSE_ONLY = {**PARAMS, "specular": 0.0, "rim": 0.0}
INNER = (slice(8, 32), slice(8, 56))  # clear of the gradient's edge rows and columns


def solid(color=(90, 120, 200), size=SIZE):
    return Image.new("RGB", size, color)


def ramp_right(size=SIZE):
    """A field rising from 0 to 1 across the width, so its slope faces left."""
    w, h = size
    return ScalarField(np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1)))


def ramp_down(size=SIZE):
    """A field falling from 1 to 0 down the height, so its slope faces down."""
    w, h = size
    return ScalarField(np.tile(np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None], (1, w)))


def test_flat_field_has_no_specular_and_only_the_uniform_diffuse_change():
    image = solid()
    flat = ScalarField(np.full((SIZE[1], SIZE[0]), 0.5, dtype=np.float32))
    out = np.asarray(shade(image, flat, **PARAMS), dtype=np.int16)
    # The same change everywhere, and the highlight and rim contribute nothing to it.
    assert (out == out[0, 0]).all()
    assert np.array_equal(out, np.asarray(shade(image, flat, **DIFFUSE_ONLY), dtype=np.int16))
    # The diffuse term is normalized so a surface facing the viewer keeps its color.
    assert np.abs(out - np.asarray(image, dtype=np.int16)).max() <= 1


def test_light_direction_flips_which_side_brightens():
    image = solid((100, 100, 100))
    from_left = np.asarray(shade(image, ramp_right(), **{**DIFFUSE_ONLY, "light_angle": 180.0}), dtype=np.float32)
    from_right = np.asarray(shade(image, ramp_right(), **{**DIFFUSE_ONLY, "light_angle": 0.0}), dtype=np.float32)
    assert from_left[INNER].mean() > 100.0 > from_right[INNER].mean()

    from_above = np.asarray(shade(image, ramp_down(), **{**DIFFUSE_ONLY, "light_angle": 90.0}), dtype=np.float32)
    from_below = np.asarray(shade(image, ramp_down(), **{**DIFFUSE_ONLY, "light_angle": 270.0}), dtype=np.float32)
    assert from_below[INNER].mean() > 100.0 > from_above[INNER].mean()


def test_weight_field_uses_the_images_lightness_as_height():
    w, h = SIZE
    lightness = np.tile(np.linspace(30, 220, w).astype(np.uint8), (h, 1))
    image = Image.fromarray(np.stack([lightness] * 3, axis=-1))
    field = WeightField(np.full((2, h, w), 0.5, dtype=np.float32))
    params = {**DIFFUSE_ONLY, "light_angle": 180.0}
    shaded = np.asarray(shade(image, field, **params), dtype=np.float32)
    plain = np.asarray(image, dtype=np.float32)
    assert shaded.shape == plain.shape
    # Brighter to the right: the slope faces left, so a light from the left brightens it.
    assert shaded[INNER].mean() > plain[INNER].mean()
    # With the light on the other side it darkens.
    away = np.asarray(shade(image, field, **{**params, "light_angle": 0.0}), dtype=np.float32)
    assert away[INNER].mean() < plain[INNER].mean()
    # A missing field is lit the same way.
    assert np.array_equal(np.asarray(shade(image, None, **params)), np.asarray(shade(image, field, **params)))


def test_refraction_moves_pixels_along_the_normal():
    w, h = SIZE
    columns = np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))
    image = Image.fromarray(np.stack([columns] * 3, axis=-1))
    params = {**PARAMS, "strength": 0.0}
    straight = np.asarray(shade(image, ramp_right(), **params), dtype=np.int16)
    bent = np.asarray(shade(image, ramp_right(), **{**params, "refraction": 4.0}), dtype=np.int16)
    assert np.array_equal(straight, np.asarray(image, dtype=np.int16))
    assert bent.shape == straight.shape
    assert np.abs(bent - straight).max() > 0


def test_shade_is_deterministic_and_uses_no_rng():
    assert SHADE.uses_rng is False
    state = random.getstate()
    first = np.asarray(shade(solid(), ramp_right(), **PARAMS))
    second = np.asarray(shade(solid(), ramp_right(), **PARAMS))
    assert np.array_equal(first, second)
    assert random.getstate() == state


def test_output_is_clipped_not_wrapped():
    out = np.asarray(shade(solid((250, 250, 250)), ramp_right(), **{**PARAMS, "strength": 1.0, "specular": 1.0, "rim": 1.0}))
    assert out.dtype == np.uint8 and out.max() == 255 and out[INNER].min() > 100


def finish_params(**overrides):
    values = dict(
        contours=0.0, contour_levels=8, contour_width=2.0, contour_mode="lines", contour_source="image",
        contour_placement="frosted", glass=False, chromatic_aberration=0.0, noise=0.0, vignette=0.0,
        shading=0.0, width=64, height=40,
    )
    values.update(overrides)
    return StageParams(values)


def names(**overrides):
    return [stage.name for stage in finish_stages(finish_params(**overrides))]


def test_finish_stages_leave_shading_out_when_off():
    assert names() == []
    assert names(glass=True, noise=0.1, vignette=0.2) == ["glass", "film grain", "vignette"]
    assert "shading" not in names(contours=0.5, glass=True, chromatic_aberration=0.01, noise=0.1, vignette=0.2)


def test_finish_stages_place_shading_after_glass_and_before_aberration():
    everything = dict(contours=0.5, glass=True, chromatic_aberration=0.01, noise=0.1, vignette=0.2, shading=0.5)
    assert names(**everything) == ["contours", "glass", "shading", "chromatic aberration", "film grain", "vignette"]
    # Sharp contours go over the shading and the aberration.
    assert names(**everything, contour_placement="sharp") == [
        "glass", "shading", "chromatic aberration", "contours", "film grain", "vignette",
    ]
    assert names(shading=0.5, contours=0.5, contour_source="field") == ["field contours", "shading"]


def test_shading_stage_arguments_scale_with_the_image():
    small = SHADE.bind(finish_params(shading=0.6, width=480, height=270))
    reference = SHADE.bind(finish_params(shading=0.6, width=3024, height=1964))
    assert small["strength"] == 0.6
    assert small["depth"] == reference["depth"]
    for key in ("smoothing", "refraction"):
        assert np.isclose(small[key] / reference[key], 270 / 1964)

    params = finish_params(shading=0.6, width=480, height=270)
    state = run_stages(finish_stages(params), params, rng=random, state={"final": solid(size=(480, 270)), "field": None})
    assert state["final"].size == (480, 270)


def render(**kwargs):
    config = RenderConfiguration(
        width=96, height=60, seed=5, colors=[(18, 24, 64), (220, 80, 120), (250, 190, 90)], harmonize="none",
        vignette=0.0, contours=0.0, glass=False, chromatic_aberration=0.0, **kwargs,
    )
    return WallpaperPipeline().render(config)


def test_pipeline_off_is_unchanged_and_on_changes_the_image():
    off = render(engine="simplex", noise=0.0)
    zero = render(engine="simplex", noise=0.0, shading=0.0)
    on = render(engine="simplex", noise=0.0, shading=0.6)
    assert (off.shading, on.shading) == (0.0, 0.6)
    assert np.array_equal(np.asarray(off.image), np.asarray(zero.image))
    assert not np.array_equal(np.asarray(off.image), np.asarray(on.image))


def test_shading_does_not_shift_the_grain_that_follows_it():
    # Film grain draws from the shared RNG after the shading; shading must leave it alone,
    # so the grain pattern (the difference from the noise-free render) is identical.
    plain = np.asarray(render(engine="blobs", noise=0.0, shading=0.5).image, dtype=np.int16)
    grainy = np.asarray(render(engine="blobs", noise=0.3, shading=0.5).image, dtype=np.int16)
    reference = np.asarray(render(engine="blobs", noise=0.3).image, dtype=np.int16)
    unshaded = np.asarray(render(engine="blobs", noise=0.0).image, dtype=np.int16)
    assert np.abs((grainy - plain)).mean() > 0
    assert np.corrcoef((grainy - plain).ravel(), (reference - unshaded).ravel())[0, 1] > 0.95


def test_metadata_round_trips_shading():
    pipeline = WallpaperPipeline()
    png, result = pipeline.render_png_bytes(RenderConfiguration(width=96, height=60, seed=5, engine="blobs", shading=0.45))
    metadata = extract_image_metadata(png)
    assert metadata.parameters["shading"] == 0.45
    restored = metadata.to_render_configuration()
    assert restored.shading == 0.45
    replayed = pipeline.render(restored)
    assert np.array_equal(np.asarray(result.image), np.asarray(replayed.image))


def test_metadata_without_shading_replays_as_off():
    png, _ = WallpaperPipeline().render_png_bytes(RenderConfiguration(width=64, height=40, seed=2, engine="simplex"))
    metadata = extract_image_metadata(png)
    assert not metadata.parameters.get("shading")
