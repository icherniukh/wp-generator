"""The staged path must reproduce the legacy engines pixel for pixel."""

import random

import numpy as np
import pytest

from engine.gradients.registry import render_gradient
from engine.stages import STAGED_PLANS, ScalarField, Trace, WeightField

PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90), (60, 170, 200)]

CASES = [
    dict(complexity=4, warp_strength=0.6, transverse_warp=0.4, fractal_type="julia", trap_weight=0.6),
    dict(complexity=2, warp_strength=0.0, transverse_warp=0.0, fractal_type="mandelbrot", frequency=2.3),
    dict(complexity=6, warp_strength=1.3, transverse_warp=0.0, ambient_weight=0.4, dispersion=0.2,
         fractal_type="orbit_trap", phase_offset=0.3, channel_offset=0.15),
    dict(complexity=3, warp_strength=None, transverse_warp=None, fractal_type="fluid",
         fractal_zoom=1.4, fractal_center=(0.2, -0.1), fractal_angle=0.7),
]
SIZES = [(96, 60), (200, 50), (40, 90)]


def _render(engine, staged, size, seed, **kwargs):
    random.seed(999)  # state before the call must not matter when a seed is given
    image = render_gradient(
        engine, width=size[0], height=size[1], colors=PALETTE, seed=seed, staged=staged, **kwargs
    )
    return np.asarray(image)


@pytest.mark.parametrize("engine", sorted(STAGED_PLANS))
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("size", SIZES)
def test_staged_matches_legacy(engine, case, size):
    for seed in (1, 77, 2024):
        legacy = _render(engine, False, size, seed, **case)
        staged = _render(engine, True, size, seed, **case)
        assert legacy.shape == staged.shape
        assert np.array_equal(legacy, staged), f"{engine} {case} {size} seed {seed}"


def test_unseeded_render_draws_from_global_state_like_legacy():
    random.seed(5)
    np.random.seed(5)
    legacy = np.asarray(render_gradient("domain-warp", width=64, height=40, colors=PALETTE, staged=False))
    random.seed(5)
    np.random.seed(5)
    staged = np.asarray(render_gradient("domain-warp", width=64, height=40, colors=PALETTE, staged=True))
    assert np.array_equal(legacy, staged)


def test_single_color_rbf_is_solid():
    image = render_gradient("rbf", width=20, height=10, colors=[(10, 20, 30)], seed=1, staged=True)
    assert image.getcolors() == [(200, (10, 20, 30))]


def test_trace_records_stages_and_unread_params():
    trace = Trace()
    render_gradient("rbf", width=64, height=40, colors=PALETTE, seed=3, trace=trace)
    names = [r.name for r in trace.records]
    assert names == ["palette", "grid", "noise warp", "radial emitters", "map", "export"]
    field = next(r for r in trace.records if r.phase == "field").output
    assert isinstance(field, WeightField)
    assert field.weights.shape[0] == len(PALETTE)
    np.testing.assert_allclose(field.weights.sum(axis=0), 1.0, rtol=1e-5)
    unread = set(trace.unread_params())
    # rbf has no transverse warp, fractal or fluid-mode controls.
    assert {"transverse_warp", "fluid_mode", "fractal_type", "frequency"} <= unread
    assert "dispersion" not in unread and "warp_strength" not in unread


def test_trace_scalar_field_is_normalized():
    trace = Trace()
    render_gradient("domain-warp", width=64, height=40, colors=PALETTE, seed=3,
                    warp_strength=0.5, transverse_warp=0.3, trace=trace)
    field = next(r for r in trace.records if r.phase == "field").output
    assert isinstance(field, ScalarField)
    assert field.t.min() == 0.0 and field.t.max() == 1.0
    assert "ambient_weight" in trace.unread_params()


def test_pipeline_trace_does_not_change_the_image_and_draws_a_filmstrip():
    from engine.pipeline import RenderConfiguration, WallpaperPipeline
    from engine.stages.filmstrip import render_filmstrip

    pipeline = WallpaperPipeline()
    for config in (
        RenderConfiguration(engine="domain-warp", colors=PALETTE, width=160, height=100, seed=4,
                            contours=0.5, glass=True, noise=0.05, vignette=0.2),
        RenderConfiguration(preset="sleek", width=160, height=100, seed=4),
    ):
        trace = Trace()
        traced = pipeline.render(config, trace=trace)
        plain = pipeline.render(config)
        assert np.array_equal(np.asarray(traced.image), np.asarray(plain.image))
        assert trace.records[0].name == "palette"
        sheet = render_filmstrip(trace, title="test")
        assert sheet.size[0] > 0 and sheet.size[1] > 0


def test_every_registry_engine_has_a_stage_plan():
    from engine.gradients.registry import GRADIENT_REGISTRY

    assert set(GRADIENT_REGISTRY) <= set(STAGED_PLANS)


@pytest.mark.parametrize("engine", sorted(STAGED_PLANS))
def test_trace_covers_every_engine(engine):
    trace = Trace()
    render_gradient(engine, width=80, height=50, colors=PALETTE, seed=9, complexity=3, trace=trace)
    phases = [r.phase for r in trace.records]
    assert phases[0] == "palette" and phases[-1] == "export"
    assert "field" in phases


def test_engine_registered_over_a_builtin_name_runs_its_own_class():
    from PIL import Image

    from engine.gradients.registry import GRADIENT_REGISTRY, register_gradient_engine

    class Solid:
        def __init__(self, width, height):
            self.size = (width, height)

        def generate(self, colors, **_):
            return Image.new("RGB", self.size, (1, 2, 3))

    original = GRADIENT_REGISTRY["simplex"]
    register_gradient_engine("simplex", Solid)
    try:
        image = render_gradient("simplex", width=8, height=4, colors=PALETTE, seed=1)
        assert image.getcolors() == [(32, (1, 2, 3))]
    finally:
        register_gradient_engine("simplex", original)
