"""Bloom and light rays: a soft glow around bright areas, in linear light."""

import numpy as np
import pytest
from PIL import Image

from engine.metadata import extract_image_metadata
from engine.pipeline import RenderConfiguration, WallpaperPipeline
from engine.stages.catalog import PlanCompositionError
from engine.stages.finish import BLOOM, REFERENCE_SHORT_SIDE, bloom, finish_stages
from engine.stages.runner import StageParams, run_stages

SIZE = (160, 100)
SMALL = dict(radii=(2.0, 6.0, 20.0))


def solid(color, size=SIZE):
    return Image.new("RGB", size, color)


def dot(size=SIZE, at=(80, 50), color=(255, 255, 255), background=(0, 0, 0), half=1):
    pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    pixels[:] = background
    x, y = at
    pixels[y - half : y + half, x - half : x + half] = color
    return Image.fromarray(pixels)


def array(image):
    return np.asarray(image, dtype=np.int16)


def test_black_image_stays_black():
    out = bloom(solid((0, 0, 0)), strength=1.0, rays=1.0, **SMALL)
    assert not array(out).any()


def test_image_below_the_threshold_is_unchanged():
    # sRGB 100 is about 0.13 linear, under threshold - knee = 0.2: nothing passes the bright pass.
    image = solid((100, 90, 120))
    out = bloom(image, strength=1.0, threshold=0.5, knee=0.3, rays=1.0, **SMALL)
    assert np.array_equal(array(out), array(image))
    # A bit above the threshold changes only a little at full strength: a flat field's glow is
    # its own excess, screened on.
    near = solid((150, 150, 150))  # about 0.3 linear, inside the knee
    assert np.abs(array(bloom(near, strength=0.3, **SMALL)) - array(near)).max() <= 6


def test_a_bright_dot_spreads_and_falls_off_with_distance():
    image = dot(size=(240, 160), at=(120, 80), half=4)
    out = array(bloom(image, strength=1.0, **{"radii": (3.0, 10.0, 30.0)})).max(axis=-1)
    # The dot covers x 116..123 and y 76..83. Away from it the glow only falls.
    row = out[79, 124:]
    assert row[0] > row[6] > row[15] > row[30] > 0
    assert (np.diff(row) <= 0).all()
    column = out[84:, 119]
    assert (np.diff(column) <= 0).all() and column[0] > column[15] > 0
    # Left and right of the dot match.
    assert np.abs(out[79, 124:184] - out[79, 115:55:-1]).max() <= 2
    # Only a small share of the image brightens much.
    assert (out > 8).mean() < 0.35


def test_the_glow_is_tinted_like_its_source_and_does_not_clip_to_white():
    image = dot(size=(240, 160), at=(120, 80), color=(255, 120, 40), half=4)
    out = array(bloom(image, strength=1.0, threshold=0.2, knee=0.2, radii=(3.0, 10.0, 30.0)))
    halo = out[79, 130]  # a few pixels from the dot's edge
    assert halo[0] > halo[1] > halo[2] >= 0
    bright = solid((255, 230, 200))
    glowing = array(bloom(bright, strength=1.0, threshold=0.2, knee=0.2, **SMALL))
    assert glowing.min() >= array(bright).min() and glowing.max() <= 255
    # Screen never darkens, and a mid-gray image under a strong glow keeps its detail.
    ramp = np.tile(np.linspace(120, 250, 160).astype(np.uint8)[None, :, None], (100, 1, 3))
    lit = array(bloom(Image.fromarray(ramp), strength=1.0, threshold=0.2, knee=0.2, **SMALL))
    assert (lit >= array(Image.fromarray(ramp))).all()
    assert len(np.unique(lit[50, :, 0])) > 40


def test_wider_radii_spread_the_glow_further():
    image = dot(size=(240, 160), at=(120, 80), half=4)
    narrow = array(bloom(image, strength=1.0, radii=(2.0,))).max(axis=-1)
    wide = array(bloom(image, strength=1.0, radii=(2.0, 20.0))).max(axis=-1)
    assert wide[79, 124 + 16] > narrow[79, 124 + 16] + 5


def test_rays_elongate_away_from_the_origin():
    image = dot(size=(240, 160), at=(120, 60), half=3)
    options = dict(strength=1.0, rays=1.0, ray_length=80.0, **{"radii": (2.0, 6.0, 20.0)})
    # The origin above: the streak runs down from the dot, not up.
    out = array(bloom(image, ray_origin=(0.5, -1.0), **options)).max(axis=-1)
    below, above = out[60 + 5 + 30, 120], out[60 - 5 - 30, 120]
    assert below > above + 10
    # ... and it is long: still bright well beyond the plain bloom's reach.
    plain = array(bloom(image, **{**options, "rays": 0.0})).max(axis=-1)
    assert out[60 + 5 + 30, 120] > plain[60 + 5 + 30, 120] + 10
    # The origin on the left sends the streak to the right.
    sideways = array(bloom(image, ray_origin=(-1.0, 0.375), **options)).max(axis=-1)
    assert sideways[60, 120 + 5 + 30] > sideways[60, 120 - 5 - 30] + 10
    # Rays never darken, and the streak is a stripe, not a fill.
    assert (out >= plain - 1).all()
    assert out[60 + 5 + 30, 120] > out[60 + 5 + 30, 120 + 40] + 10


def test_rays_off_is_the_plain_bloom():
    image = dot(size=(240, 160), half=2)
    assert np.array_equal(
        array(bloom(image, strength=0.8, rays=0.0, ray_length=50.0, **SMALL)),
        array(bloom(image, strength=0.8, **SMALL)),
    )


def test_strength_zero_gives_identical_pixels():
    image = dot(size=(240, 160), half=3)
    assert np.array_equal(array(bloom(image, strength=0.0, rays=1.0, **SMALL)), array(image))


def test_validation():
    with pytest.raises(ValueError):
        bloom(solid((9, 9, 9)), strength=-0.1)
    with pytest.raises(ValueError):
        bloom(solid((9, 9, 9)), strength=0.5, radii=())
    with pytest.raises(ValueError):
        bloom(solid((9, 9, 9)), strength=0.5, radii=(0.0, 4.0))
    with pytest.raises(ValueError):
        bloom(solid((9, 9, 9)), strength=0.5, threshold=-1.0)


def test_is_deterministic():
    image = dot(size=(240, 160), half=3)
    a = bloom(image, strength=0.7, rays=0.5, **SMALL)
    b = bloom(image, strength=0.7, rays=0.5, **SMALL)
    assert np.array_equal(array(a), array(b))


def finish_params(**overrides):
    values = dict(
        contours=0.0, contour_levels=8, contour_width=2.0, contour_mode="lines", contour_source="image",
        contour_placement="frosted", glass=False, chromatic_aberration=0.0, noise=0.0, vignette=0.0,
        shading=0.0, bloom=0.0, width=SIZE[0], height=SIZE[1],
    )
    values.update(overrides)
    return StageParams(values)


def names(**overrides):
    return [stage.name for stage in finish_stages(finish_params(**overrides))]


def arguments(**overrides):
    p = finish_params(bloom=0.5, **overrides)
    (stage,) = [s for s in finish_stages(p) if s.name == "bloom"]
    return stage.bind(p)


def test_stage_runs_only_when_bloom_is_set():
    assert names() == []
    assert names(bloom=0.4) == ["bloom"]
    # Overrides never enable it.
    assert names(stage_params={"bloom": {"rays": 0.5}}) == []


def test_finish_order_places_bloom_after_shading_and_before_aberration():
    everything = dict(
        contours=0.5, glass=True, shading=0.5, bloom=0.5, chromatic_aberration=0.01, noise=0.1, vignette=0.2
    )
    assert names(**everything) == [
        "contours", "glass", "shading", "bloom", "chromatic aberration", "film grain", "vignette",
    ]
    assert names(**everything, contour_placement="sharp") == [
        "glass", "shading", "bloom", "chromatic aberration", "contours", "film grain", "vignette",
    ]
    assert names(bloom=0.5, glass=True, vignette=0.2) == ["glass", "bloom", "vignette"]


def test_defaults_bind_the_option_and_leave_rays_off():
    args = arguments()
    assert args["strength"] == 0.5 and args["rays"] == 0.0
    assert args["ray_origin"] == (0.5, -0.25)
    assert set(args) == {"strength", "threshold", "knee", "radii", "rays", "ray_origin", "ray_length", "ray_samples"}


def test_pixel_arguments_scale_with_the_image():
    small = arguments(width=480, height=300)
    large = arguments(width=1920, height=1200)
    ratio = 1200 / 300
    assert np.allclose(np.array(large["radii"]) / np.array(small["radii"]), ratio)
    assert large["ray_length"] / small["ray_length"] == pytest.approx(ratio)
    assert small["radii"][0] == pytest.approx(6.0 * 300 / REFERENCE_SHORT_SIDE)
    # Relative arguments do not scale.
    assert small["ray_origin"] == large["ray_origin"] and small["threshold"] == large["threshold"]


def test_overrides_are_given_at_reference_scale_and_scale_with_the_image():
    overrides = {"bloom": {"radii": [10, 40], "ray_length": 500.0, "rays": 0.7, "threshold": 0.3, "ray_origin": [0.2, -0.1]}}
    for width, height in ((480, 300), (1920, 1200)):
        args = arguments(width=width, height=height, stage_params=overrides)
        scale = height / REFERENCE_SHORT_SIDE
        assert args["radii"] == pytest.approx((10 * scale, 40 * scale))
        assert args["ray_length"] == pytest.approx(500.0 * scale)
        assert args["rays"] == 0.7 and args["threshold"] == 0.3
        assert tuple(args["ray_origin"]) == (0.2, -0.1)


def test_an_unknown_argument_raises():
    with pytest.raises(PlanCompositionError, match="no argument raze"):
        finish_stages(finish_params(bloom=0.5, stage_params={"bloom": {"raze": 1}}))


def test_runs_as_a_stage_and_changes_a_bright_image():
    image = dot(size=SIZE, half=4)
    p = finish_params(bloom=0.8, stage_params={"bloom": {"rays": 0.5}})
    state = run_stages(finish_stages(p), p, rng=None, state={"final": image})
    assert not np.array_equal(array(state["final"]), array(image))


def render(**kwargs):
    config = RenderConfiguration(
        width=200, height=120, seed=5, colors=[(18, 24, 64), (250, 220, 150), (255, 250, 235)],
        harmonize="none", vignette=0.0, contours=0.0, glass=False, chromatic_aberration=0.0, noise=0.0,
        engine="shapes", **kwargs,
    )
    return WallpaperPipeline().render(config)


def test_pipeline_off_is_identical_and_on_changes_pixels():
    base = array(render(bloom=0.0).image)
    assert np.array_equal(base, array(render().image))
    # An override cannot turn it on.
    assert np.array_equal(base, array(render(stage_params={"bloom": {"rays": 0.9}}).image))
    glowing = render(bloom=0.7)
    assert glowing.bloom == 0.7
    assert (array(glowing.image) >= base).all() and (array(glowing.image) != base).any()


def test_bloom_and_rays_replay_from_metadata():
    pipeline = WallpaperPipeline()
    config = RenderConfiguration(
        width=200, height=120, seed=9, engine="shapes", bloom=0.6, noise=0.1,
        stage_params={"bloom": {"rays": 0.6, "ray_length": 400}},
    )
    png, original = pipeline.render_png_bytes(config)
    metadata = extract_image_metadata(png)
    assert metadata.parameters["bloom"] == 0.6
    replayed = pipeline.render(metadata.to_render_configuration())
    assert np.array_equal(array(original.image), array(replayed.image))
