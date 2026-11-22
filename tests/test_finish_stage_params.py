"""stage_params overrides for the finish stages (contours, glass, shading, grain, ...)."""

import random

import numpy as np
import pytest
from PIL import Image

from engine.metadata import extract_image_metadata
from engine.pipeline import RenderConfiguration, WallpaperPipeline
from engine.stages import Composition, render_staged
from engine.stages.catalog import PlanCompositionError, compose
from engine.stages.finish import (
    FINISH_KEYS,
    GLASS,
    REFERENCE_SHORT_SIDE,
    check_finish_stage_params,
    finish_stages,
    shade,
    split_stage_params,
)
from engine.stages.plans import STAGED_PLANS
from engine.stages.runner import StageParams, run_stages
from engine.stages.types import ScalarField

SIZE = (96, 60)
PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90)]
SHADE_ARGS = dict(
    strength=0.7, depth=3.0, light_angle=135.0, light_elevation=50.0,
    specular=0.5, shininess=10.0, rim=1.0, refraction=4.0, smoothing=1.0,
)


def finish_params(**overrides):
    values = dict(
        contours=0.0, contour_levels=8, contour_width=2.0, contour_mode="lines", contour_source="image",
        contour_placement="frosted", glass=False, chromatic_aberration=0.0, noise=0.0, vignette=0.0,
        shading=0.0, width=SIZE[0], height=SIZE[1],
    )
    values.update(overrides)
    return StageParams(values)


def arguments_of(params, name):
    (stage,) = [s for s in finish_stages(params) if s.name == name]
    return stage.bind(params)


def render(**kwargs):
    config = RenderConfiguration(
        width=SIZE[0], height=SIZE[1], seed=5, colors=PALETTE, harmonize="none",
        vignette=0.0, contours=0.0, glass=False, chromatic_aberration=0.0, noise=0.0, **kwargs,
    )
    return WallpaperPipeline().render(config)


def pixels(result):
    return np.asarray(result.image, dtype=np.int16)


def hill():
    w, h = SIZE
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return ScalarField(np.exp(-(((x - w / 2) / 20) ** 2 + ((y - h / 2) / 14) ** 2)).astype(np.float32))


# --- keys and binding -------------------------------------------------------------------------


def test_every_finish_stage_has_a_key_named_like_compose_names_stages():
    assert sorted(FINISH_KEYS) == [
        "chromatic-aberration", "contours", "film-grain", "glass", "halftone", "shading", "vignette",
    ]
    for key, stages in FINISH_KEYS.items():
        for stage in stages:
            assert key == stage.name.replace(" ", "-") or (key == "contours" and "contours" in stage.name)


def test_an_override_replaces_the_bound_arguments():
    plain = finish_params(shading=0.6)
    overridden = finish_params(shading=0.6, stage_params={"shading": {"light_angle": 45, "specular": 0.8}})
    before, after = arguments_of(plain, "shading"), arguments_of(overridden, "shading")
    assert (before["light_angle"], before["specular"]) == (135.0, 0.45)
    assert (after["light_angle"], after["specular"]) == (45, 0.8)
    # Everything not named keeps its bound value.
    assert {k: v for k, v in after.items() if k not in ("light_angle", "specular")} == {
        k: v for k, v in before.items() if k not in ("light_angle", "specular")
    }


def test_overrides_reach_every_finish_stage():
    p = finish_params(
        contours=0.5, glass=True, chromatic_aberration=0.01, noise=0.1, vignette=0.2, shading=0.5,
        stage_params={
            "contours": {"levels": 5}, "glass": {"noise_amount": 0.3}, "shading": {"rim": 0.0},
            "chromatic-aberration": {"strength": 0.05}, "film-grain": {"amount": 0.4}, "vignette": {"strength": 0.9},
        },
    )
    assert arguments_of(p, "contours")["levels"] == 5
    assert arguments_of(p, "glass")["noise_amount"] == 0.3
    assert arguments_of(p, "shading")["rim"] == 0.0
    assert arguments_of(p, "chromatic aberration")["strength"] == 0.05
    assert arguments_of(p, "film grain")["amount"] == 0.4
    assert arguments_of(p, "vignette")["strength"] == 0.9


def test_the_contours_key_reaches_the_field_contours_too():
    p = finish_params(contours=0.5, contour_source="field", stage_params={"contours": {"levels": 3}})
    assert arguments_of(p, "field contours")["levels"] == 3


def test_an_override_changes_the_output():
    image = Image.new("RGB", SIZE, (90, 120, 200))
    rendered = []
    for stage_params in (None, {"shading": {"light_angle": 300, "refraction": 0, "specular": 0.8}}):
        p = finish_params(shading=0.6, stage_params=stage_params)
        state = run_stages(finish_stages(p), p, rng=random, state={"final": image, "field": hill()})
        rendered.append(np.asarray(state["final"]))
    assert not np.array_equal(rendered[0], rendered[1])


# --- validation -------------------------------------------------------------------------------


def test_an_unknown_argument_raises_a_clear_value_error():
    with pytest.raises(ValueError, match=r"stage 'shading' has no argument speculer; arguments: .*specular"):
        finish_stages(finish_params(shading=0.6, stage_params={"shading": {"speculer": 1.0}}))
    with pytest.raises(PlanCompositionError):
        check_finish_stage_params({"film-grain": {"amount": 0.1, "colour": "color"}})
    with pytest.raises(ValueError, match="must map argument names"):
        check_finish_stage_params({"vignette": 0.5})
    # The stage's input names and rng are not arguments.
    with pytest.raises(ValueError):
        check_finish_stage_params({"shading": {"final": None}})


def test_an_unknown_argument_is_refused_even_when_the_stage_is_off():
    with pytest.raises(ValueError, match="no argument typo"):
        finish_stages(finish_params(stage_params={"glass": {"typo": 1}}))


def test_the_pipeline_refuses_a_bad_finish_argument_before_rendering():
    with pytest.raises(ValueError, match="no argument typo"):
        render(engine="simplex", shading=0.5, stage_params={"shading": {"typo": 1}})


# --- a stage that is off stays off -------------------------------------------------------------


def test_an_override_never_enables_a_stage():
    p = finish_params(stage_params={
        "shading": {"strength": 1.0}, "glass": {"blur_radius": 9}, "film-grain": {"amount": 0.5},
        "vignette": {"strength": 0.8}, "contours": {"intensity": 1.0}, "chromatic-aberration": {"strength": 0.1},
    })
    assert finish_stages(p) == ()

    off = render(engine="blobs")
    overridden = render(engine="blobs", stage_params={"shading": {"strength": 1.0}, "film-grain": {"amount": 0.5}})
    assert np.array_equal(pixels(off), pixels(overridden))


def test_a_stage_stays_in_its_place_in_the_order():
    p = finish_params(glass=True, shading=0.5, vignette=0.2, stage_params={"shading": {"depth": 1.0}})
    assert [s.name for s in finish_stages(p)] == ["glass", "shading", "vignette"]


# --- engine keys and finish keys do not trip each other ---------------------------------------


def test_compose_ignores_finish_keys_and_still_checks_engine_keys():
    stages = STAGED_PLANS["simplex"].stages
    composed = compose(stages, field="voronoi", stage_params={"voronoi": {"mode": "edges"}, "shading": {"rim": 0.0}})
    assert [s.name for s in composed] == [s.name for s in compose(stages, field="voronoi")]
    with pytest.raises(PlanCompositionError, match="no stage named twistt"):
        compose(stages, stage_params={"twistt": {"amount": 1}, "shading": {"rim": 0.0}})


def test_finish_stages_ignore_engine_keys():
    p = finish_params(shading=0.6, stage_params={"voronoi": {"mode": "edges"}, "twist": {"amount": 3.0}})
    assert [s.name for s in finish_stages(p)] == ["shading"]
    assert arguments_of(p, "shading")["light_angle"] == 135.0
    assert split_stage_params({"voronoi": {}, "shading": {}, "glass": {}}) == ({"voronoi": {}}, {"shading": {}, "glass": {}})


def test_engine_stage_params_work_unchanged_next_to_finish_ones():
    engine_only = {"voronoi": {"mode": "edges"}}
    both = {**engine_only, "shading": {"light_angle": 300}, "film-grain": {"amount": 0.2}}
    a = render(engine="simplex", field_stage="voronoi", stage_params=engine_only)
    b = render(engine="simplex", field_stage="voronoi", stage_params=both)
    # With nothing enabled in the finish, the finish keys change nothing.
    assert np.array_equal(pixels(a), pixels(b))
    c = render(engine="simplex", field_stage="voronoi", stage_params={"voronoi": {"mode": "f2"}})
    assert not np.array_equal(pixels(a), pixels(c))

    composition = Composition(field="voronoi", stage_params=engine_only)
    direct = render_staged("simplex", dict(
        width=SIZE[0], height=SIZE[1], colors=PALETTE, complexity=4, seed=5), composition=composition)
    assert direct.size == SIZE


def test_finish_only_stage_params_do_not_count_as_a_composition():
    # A custom engine has no stage plan and refuses engine composition; finish keys are no such thing.
    from engine.gradients.registry import GRADIENT_REGISTRY, register_gradient_engine

    class Flat:
        def __init__(self, width, height):
            self.size = (width, height)

        def generate(self, colors):
            return Image.new("RGB", self.size, tuple(colors[0]))

    register_gradient_engine("flat-finish-test", Flat)
    try:
        config = dict(engine="flat-finish-test", width=64, height=40, seed=1, colors=PALETTE, harmonize="none",
                      noise=0.0, vignette=0.0, contours=0.0, glass=False, chromatic_aberration=0.0)
        out = WallpaperPipeline().render(RenderConfiguration(**config, shading=0.5, stage_params={"shading": {"rim": 0.0}}))
        assert out.image.size == (64, 40)
        with pytest.raises(ValueError, match="custom class"):
            WallpaperPipeline().render(RenderConfiguration(**config, stage_params={"twist": {"amount": 1}}))
    finally:
        GRADIENT_REGISTRY.pop("flat-finish-test", None)


# --- pixel sizes are given at reference scale -------------------------------------------------


def test_pixel_size_overrides_are_given_at_reference_scale_and_scaled_like_the_defaults():
    base = dict(contours=0.5, glass=True, shading=0.5, noise=0.1)
    overrides = {
        "contours": {"line_width": 3.0, "smoothing_radius": 8.0},
        "glass": {"blur_radius": 100},
        "shading": {"refraction": 20.0, "smoothing": 50.0},
        "film-grain": {"grain_size": 4.0},
    }
    reference = finish_params(**base, width=3024, height=REFERENCE_SHORT_SIDE, stage_params=overrides)
    half = finish_params(**base, width=1512, height=REFERENCE_SHORT_SIDE // 2, stage_params=overrides)

    at_reference = {
        "contours": arguments_of(reference, "contours"), "glass": arguments_of(reference, "glass"),
        "shading": arguments_of(reference, "shading"), "grain": arguments_of(reference, "film grain"),
    }
    assert at_reference["contours"]["line_width"] == 3.0
    assert at_reference["contours"]["smoothing_radius"] == 8.0
    assert at_reference["glass"]["blur_radius"] == 100
    assert at_reference["shading"]["refraction"] == 20.0
    assert at_reference["shading"]["smoothing"] == 50.0
    assert at_reference["grain"]["grain_size"] == 4.0

    assert arguments_of(half, "contours")["line_width"] == 1.5
    assert arguments_of(half, "contours")["smoothing_radius"] == 4.0
    assert arguments_of(half, "glass")["blur_radius"] == 50
    assert arguments_of(half, "shading")["refraction"] == 10.0
    assert arguments_of(half, "shading")["smoothing"] == 25.0
    assert arguments_of(half, "film grain")["grain_size"] == 2.0

    tiny = finish_params(glass=True, width=40, height=20, stage_params={"glass": {"blur_radius": 100}})
    assert arguments_of(tiny, "glass")["blur_radius"] == GLASS.bind(finish_params(glass=True, width=40, height=20))["blur_radius"]
    # A size left alone keeps today's default, and non-pixel arguments are never scaled.
    default = arguments_of(finish_params(shading=0.5, width=1512, height=REFERENCE_SHORT_SIDE // 2), "shading")
    other = arguments_of(finish_params(
        shading=0.5, width=1512, height=REFERENCE_SHORT_SIDE // 2, stage_params={"shading": {"depth": 2.0, "tint": 0.5}}), "shading")
    assert other["refraction"] == default["refraction"] and other["smoothing"] == default["smoothing"]
    assert (other["depth"], other["tint"]) == (2.0, 0.5)


def test_an_override_of_the_default_value_is_a_no_op():
    p = finish_params(shading=0.5, noise=0.1, glass=True, width=480, height=270)
    q = finish_params(
        shading=0.5, noise=0.1, glass=True, width=480, height=270,
        stage_params={"shading": {"refraction": 12.0, "smoothing": 30.0}, "glass": {"blur_radius": 60}},
    )
    for name in ("shading", "glass"):
        assert arguments_of(p, name) == arguments_of(q, name)


# --- shade(tint=) -----------------------------------------------------------------------------


def test_tint_zero_is_pixel_identical_to_the_old_behaviour():
    w, h = SIZE
    ramp = np.tile(np.linspace(40, 200, w, dtype=np.uint8), (h, 1))
    images = [Image.new("RGB", SIZE, (90, 120, 200)), Image.fromarray(np.stack([ramp, ramp[::-1], ramp], axis=-1))]
    for field, image in ((hill(), images[0]), (None, images[1])):
        old = np.asarray(shade(image, field, **SHADE_ARGS))
        assert np.array_equal(old, np.asarray(shade(image, field, **SHADE_ARGS, tint=0.0)))
        assert not np.array_equal(old, np.asarray(shade(image, field, **SHADE_ARGS, tint=0.8)))


def test_tint_pulls_the_highlight_toward_the_local_base_color():
    base = np.array([30.0, 60.0, 100.0])
    image = Image.new("RGB", SIZE, tuple(int(c) for c in base))
    args = {**SHADE_ARGS, "strength": 1.0, "refraction": 0.0, "specular": 0.3, "rim": 0.0}
    lit = lambda tint: np.asarray(shade(image, hill(), **args, tint=tint), dtype=np.float32)
    none = np.asarray(shade(image, hill(), **{**args, "specular": 0.0}), dtype=np.float32)
    white, half, full = (lit(t) - none for t in (0.0, 0.5, 1.0))
    peak = np.unravel_index(white[..., 0].argmax(), white.shape[:2])
    amount = white[peak][0]
    assert amount > 40
    # tint 0 adds the same white in every channel; tint 1 adds the amount times the base color.
    assert np.allclose(white[peak], amount, atol=1.5)
    assert np.allclose(full[peak], amount / 255.0 * base, atol=1.5)
    assert np.allclose(half[peak], (white[peak] + full[peak]) / 2, atol=1.5)
    with pytest.raises(ValueError, match="tint"):
        shade(image, None, **SHADE_ARGS, tint=1.5)


# --- end to end and replay --------------------------------------------------------------------


def test_stage_params_change_a_pipeline_render_and_replay_from_metadata():
    off = render(engine="blobs", shading=0.6)
    on = render(engine="blobs", shading=0.6, stage_params={"shading": {"light_angle": 300, "tint": 0.6}})
    assert not np.array_equal(pixels(off), pixels(on))
    assert on.stage_params == {"shading": {"light_angle": 300, "tint": 0.6}}

    pipeline = WallpaperPipeline()
    config = RenderConfiguration(
        engine="blobs", width=SIZE[0], height=SIZE[1], seed=5, noise=0.2, shading=0.6, contours=0.4,
        stage_params={"shading": {"light_angle": 300, "tint": 0.6}, "film-grain": {"grain_size": 3.0}},
    )
    png, original = pipeline.render_png_bytes(config)
    replayed = pipeline.render(extract_image_metadata(png).to_render_configuration())
    assert np.array_equal(np.asarray(original.image), np.asarray(replayed.image))
    # The traced finish arguments show the overrides.
    from engine.stages.runner import Trace

    trace = Trace()
    pipeline.render(config, trace=trace)
    shading = next(r for r in trace.records if r.name == "shading")
    assert (shading.arguments["light_angle"], shading.arguments["tint"]) == (300, 0.6)
    assert "stage_params" not in trace.unread_params()
