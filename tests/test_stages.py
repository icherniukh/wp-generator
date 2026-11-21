"""Stage plans: traces, parameter tracking and registry dispatch.

Pixel output is pinned by the golden renders (tests/test_golden.py).
"""


import numpy as np
import pytest

from engine.gradients.registry import render_gradient
from engine.stages import STAGED_PLANS, ScalarField, Trace, WeightField

PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90), (60, 170, 200)]

def test_single_color_rbf_is_solid():
    image = render_gradient("rbf", width=20, height=10, colors=[(10, 20, 30)], seed=1)
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
