"""Fractal shapes: self-similar shape compositions with soft halos."""

import random
import time

import numpy as np
import pytest

import engine.stages.fields as fields
from engine.stages import STAGED_PLANS, Composition, render_staged
from engine.stages.catalog import FIELDS, compose
from engine.stages.fields import SHAPE_KINDS, fractal_shapes
from engine.stages.frame import sample_grid
from engine.stages.palette import make_palette

PALETTE = [(11, 15, 42), (255, 61, 129), (124, 77, 255), (0, 229, 255), (255, 224, 102)]


def _field(seed=4, width=160, height=90, colors=PALETTE, **kwargs):
    coords = sample_grid(width=width, height=height, scale=1.0, extent=(0.0, 1.0))
    return fractal_shapes(coords, make_palette(colors=colors), rng=random.Random(seed), **kwargs)


def _params(width, height, **extra):
    return dict(width=width, height=height, colors=PALETTE, complexity=5, seed=7, **extra)


@pytest.mark.parametrize("fade", [0.0, 0.35, 1.0])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_weights_sum_to_one_and_are_finite(seed, fade):
    weights = _field(seed=seed, fade=fade, count=3).weights
    assert np.isfinite(weights).all()
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-5)
    assert weights.min() >= -1e-6 and weights.max() <= 1.0 + 1e-6


def test_deterministic_per_seed():
    assert np.array_equal(_field(seed=9).weights, _field(seed=9).weights)
    assert not np.array_equal(_field(seed=9).weights, _field(seed=10).weights)
    first = render_staged("simplex", _params(160, 90), composition=Composition(field="fractal-shapes"))
    again = render_staged("simplex", _params(160, 90), composition=Composition(field="fractal-shapes"))
    assert np.array_equal(np.asarray(first), np.asarray(again))


def test_draws_are_a_multiple_of_the_per_shape_count_and_ignore_the_palette():
    class Counting(random.Random):
        calls = 0

        def uniform(self, a, b):
            Counting.calls += 1
            return super().uniform(a, b)

    def draws(colors):
        Counting.calls = 0
        coords = sample_grid(width=64, height=36, scale=1.0, extent=(0.0, 1.0))
        fractal_shapes(coords, make_palette(colors=colors), rng=Counting(3), kinds=SHAPE_KINDS)
        return Counting.calls

    assert draws(PALETTE) % 14 == 0
    assert draws(PALETTE[:3]) == draws(PALETTE)


def test_the_darkest_color_stays_the_background_and_the_others_appear():
    field = _field(count=3)
    assert field.background == 0
    for color in range(1, len(PALETTE)):
        assert field.weights[color].max() > 0.99, color
    assert (field.weights[0] > 0.99).mean() > 0.05  # bare background remains


def test_depth_adds_smaller_copies_in_the_next_colors():
    flat = _field(count=2, depth=1, fade=0.0, hardness=1.0).weights
    deep = _field(count=2, depth=4, fade=0.0, hardness=1.0).weights
    assert not np.array_equal(flat, deep)
    assert len(np.unique(deep.argmax(axis=0))) > len(np.unique(flat.argmax(axis=0)))


def test_fade_adds_a_partial_halo_outside_a_crisp_body():
    hard = _field(seed=2, count=1, depth=1, fade=0.0, hardness=1.0).weights
    soft = _field(seed=2, count=1, depth=1, fade=0.6, hardness=1.0).weights
    bare = hard[0] > 0.999
    halo = bare & (soft[0] < 0.999)
    assert halo.sum() > 0
    assert 0.3 < soft[0][halo].min() and soft[0][halo].max() < 0.999  # partial, not opaque
    assert (soft[0][hard[0] < 0.001] < 0.01).all()  # the body is as solid as without the halo


def test_the_cap_limits_the_shape_count(monkeypatch):
    full = _field(count=4, depth=5, fade=0.0).weights
    monkeypatch.setattr(fields, "FRACTAL_SHAPE_CAP", 4)
    capped = _field(count=4, depth=5, fade=0.0).weights
    assert not np.array_equal(capped, full)
    assert np.allclose(capped.sum(axis=0), 1.0, atol=1e-5)


def test_a_tiny_frame_works():
    small = _field(width=24, height=16, count=2, depth=2)
    assert np.allclose(small.weights.sum(axis=0), 1.0, atol=1e-5)


def test_a_2560x1440_field_is_quick():
    started = time.perf_counter()
    weights = _field(width=2560, height=1440, count=3).weights
    assert time.perf_counter() - started < 6.0  # about 0.5 s measured; slack for loaded CI
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-4)


def test_composed_field_gets_a_full_resolution_grid_and_stage_params_reach_it():
    for engine in ("simplex", "idw"):
        state = {}
        render_staged(engine, _params(160, 90), keep_state=state, composition=Composition(
            field="fractal-shapes", stage_params={"fractal-shapes": {"depth": 1, "fade": 0.8}}))
        assert state["coords"].shape == (90, 160), engine
        assert np.allclose(state["field"].weights.sum(axis=0), 1.0, atol=1e-5), engine
    stages = compose(STAGED_PLANS["simplex"].stages, field="fractal-shapes",
                     stage_params={"fractal-shapes": {"count": 2, "depth": 2, "ratio": 0.5, "fade": 0.2,
                                                      "hardness": 0.8, "kinds": ["circle", "ring"]}})
    arguments = next(s for s in stages if s.phase == "field").bind({"complexity": 5})
    assert (arguments["count"], arguments["depth"], arguments["ratio"], arguments["fade"],
            arguments["hardness"], arguments["kinds"]) == (2, 2, 0.5, 0.2, 0.8, ["circle", "ring"])
    assert FIELDS["fractal-shapes"].full_resolution


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="shape kinds"):
        _field(kinds=["hexagon"])


@pytest.mark.parametrize("kinds", [["circle"], ["rectangle"], ["half-circle", "quarter-circle"]])
def test_restricted_kinds_render(kinds):
    weights = _field(kinds=kinds, count=2).weights
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-5)
