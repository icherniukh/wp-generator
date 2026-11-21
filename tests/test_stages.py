"""The staged path must reproduce the legacy engines pixel for pixel."""

import random

import numpy as np
import pytest

from engine.gradients.registry import render_gradient
from engine.stages import STAGED_PLANS, ScalarField, Trace, WeightField

PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90), (60, 170, 200)]

CASES = [
    dict(complexity=4, warp_strength=0.6, transverse_warp=0.4),
    dict(complexity=2, warp_strength=0.0, transverse_warp=0.0),
    dict(complexity=6, warp_strength=1.3, transverse_warp=0.0, ambient_weight=0.4, dispersion=0.2),
    dict(complexity=3, warp_strength=None, transverse_warp=None),
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
    legacy = np.asarray(render_gradient("domain-warp", width=64, height=40, colors=PALETTE))
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
        RenderConfiguration(preset="sleek", width=160, height=100, seed=4),  # engine not staged yet
    ):
        trace = Trace()
        traced = pipeline.render(config, trace=trace)
        plain = pipeline.render(config)
        assert np.array_equal(np.asarray(traced.image), np.asarray(plain.image))
        assert trace.records[0].name == "palette"
        sheet = render_filmstrip(trace, title="test")
        assert sheet.size[0] > 0 and sheet.size[1] > 0
