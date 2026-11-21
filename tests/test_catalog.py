"""Every engine plan combined with every catalogue stage either renders or refuses clearly."""

import itertools

import numpy as np
import pytest

from engine.stages import STAGED_PLANS, Composition, render_staged
from engine.stages.catalog import FIELDS, MAPPINGS, WARPS, PlanCompositionError, compose, parse_stage_params

PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90), (60, 170, 200)]
ENGINES = sorted(set(STAGED_PLANS) - {"radial-basis"})


def _params(**extra):
    return dict(
        width=48, height=30, colors=PALETTE, complexity=4, fluid_mode="random", warp_strength=None,
        transverse_warp=None, ambient_weight=0.15, dispersion=0.35, fractal_type="fluid", fractal_zoom=None,
        fractal_center=None, fractal_angle=None, phase_offset=0.0, frequency=1.0, trap_weight=0.35,
        channel_offset=0.0, seed=5, **extra,
    )


def _render(engine, composition):
    try:
        image = render_staged(engine, _params(), composition=composition)
    except PlanCompositionError:
        return None
    assert image.size == (48, 30), f"{engine} {composition}: size {image.size}"
    return np.asarray(image.convert("RGB"))


COMBOS = [
    (engine, Composition(field=field, mapping=mapping))
    for engine, field, mapping in itertools.product(ENGINES, [None, *FIELDS], [None, *MAPPINGS])
    if field or mapping
] + [(engine, Composition(warps=(warp,))) for engine, warp in itertools.product(ENGINES, WARPS)]


@pytest.mark.parametrize("engine,composition", COMBOS, ids=lambda v: str(v) if isinstance(v, str) else repr(v)[11:])
def test_composition_renders_or_refuses(engine, composition):
    first = _render(engine, composition)
    second = _render(engine, composition)
    if first is None:
        assert second is None
        return
    assert np.array_equal(first, second), "same seed, different pixels"
    assert first.std() > 0.5 or composition.mapping == "srgb-blend", "flat image"


def test_every_new_field_renders_on_simplex():
    for name in ("conic", "spiral", "radial", "voronoi", "ridged"):
        assert _render("simplex", Composition(field=name)) is not None, name


def test_warp_none_drops_engine_warps():
    stages = compose(STAGED_PLANS["domain-warp"].stages, warps=["none", "twist"], field="radial")
    assert [s.name for s in stages if s.phase == "warp"] == ["twist"]


def test_folded_noise_needs_a_noise_warp():
    with pytest.raises(PlanCompositionError, match="displacement"):
        compose(STAGED_PLANS["simplex"].stages, field="folded-noise")
    compose(STAGED_PLANS["simplex"].stages, field="folded-noise", warps=["noise"])


def test_mapping_must_fit_the_field():
    with pytest.raises(PlanCompositionError, match="needs a scalar field"):
        compose(STAGED_PLANS["rbf"].stages, mapping_name="bands")


def test_stage_params_override_and_unknown_stage_fails():
    plain = _render("simplex", Composition(field="voronoi"))
    edges = _render("simplex", Composition(field="voronoi", stage_params={"voronoi": {"mode": "edges"}}))
    assert not np.array_equal(plain, edges)
    with pytest.raises(PlanCompositionError, match="no stage named"):
        compose(STAGED_PLANS["simplex"].stages, stage_params={"twist": {"amount": 1}})


def test_parse_stage_params():
    assert parse_stage_params(["twist.amount=3", "voronoi.mode=edges", "noisy-ramp.complexity=2"]) == {
        "twist": {"amount": 3}, "voronoi": {"mode": "edges"}, "noisy-ramp": {"complexity": 2}
    }
    with pytest.raises(PlanCompositionError):
        parse_stage_params(["twist=3"])


def test_no_composition_is_the_plain_plan():
    assert np.array_equal(_render("simplex", None), _render("simplex", Composition()))
