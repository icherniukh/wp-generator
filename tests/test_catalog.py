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
    for name in ("conic", "spiral", "radial", "voronoi", "ridged", "billow", "fbm"):
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


def _grid(width=41, height=21):
    from engine.stages.frame import sample_grid

    return sample_grid(width=width, height=height, extent=(-1.0, 1.0), aspect_correct=True)


def test_sphere_keeps_center_and_rim_and_swells_inside():
    from engine.stages.warp import Frame, sphere

    coords = _grid()
    moved = sphere(coords, amount=1.0, radius=1.0)
    frame = Frame(coords)
    u0, v0 = frame.from_xy(coords.x, coords.y)
    u1, v1 = frame.from_xy(moved.x, moved.y)
    r0, r1 = np.hypot(u0, v0), np.hypot(u1, v1)
    outside = r0 >= 1.0
    assert np.allclose(r1[outside], r0[outside], atol=1e-5)
    inside = (r0 > 0.05) & (r0 < 0.95)
    # Inside the disc each pixel samples closer to the center, asin(r) / (pi/2) < r,
    # so the middle is magnified.
    assert np.all(r1[inside] < r0[inside])
    assert np.allclose(sphere(coords, amount=0.0).x, coords.x, atol=1e-6)


def test_lens_moves_corners_by_amount():
    from engine.stages.warp import Frame, lens

    coords = _grid()
    frame = Frame(coords)
    u0, v0 = frame.from_xy(coords.x, coords.y)
    for amount in (0.3, -0.2):
        moved = lens(coords, amount=amount)
        u1, v1 = frame.from_xy(moved.x, moved.y)
        ratio = np.hypot(u1, v1)[0, 0] / np.hypot(u0, v0)[0, 0]
        assert ratio == pytest.approx(1.0 + amount, rel=1e-4)


def test_voronoi_f2_differs_from_f1():
    f1 = _render("simplex", Composition(field="voronoi"))
    f2 = _render("simplex", Composition(field="voronoi", stage_params={"voronoi": {"mode": "f2"}}))
    assert not np.array_equal(f1, f2)


def test_noise_folds_differ():
    images = {name: _render("simplex", Composition(field=name)) for name in ("ridged", "billow", "fbm")}
    assert not np.array_equal(images["ridged"], images["billow"])
    assert not np.array_equal(images["billow"], images["fbm"])


@pytest.mark.parametrize("engine", ENGINES)
def test_blend_renders_or_refuses(engine):
    composition = Composition(blend="conic:difference")
    first = _render(engine, composition)
    if first is None:
        return
    assert np.array_equal(first, _render(engine, composition))
    assert not np.array_equal(first, _render(engine, Composition()))


def test_blend_modes():
    from engine.stages.fields import BLEND_MODES, blend_fields
    from engine.stages.types import ScalarField

    a = ScalarField(np.array([[0.2, 0.8]], np.float32))
    b = ScalarField(np.array([[0.5, 0.5]], np.float32))
    expect = {
        "add": [0.7, 1.0], "multiply": [0.1, 0.4], "screen": [0.6, 0.9], "difference": [0.3, 0.3],
        "overlay": [0.2, 0.8], "min": [0.2, 0.5], "max": [0.5, 0.8],
    }
    assert set(expect) == set(BLEND_MODES)
    for mode, values in expect.items():
        out = blend_fields(a, b, mode=mode, normalize=False).t
        assert np.allclose(out[0], values), mode
    assert np.allclose(blend_fields(a, b, mode="screen", amount=0.0, normalize=False).t, a.t)


def test_blend_refuses_weights_and_unknown_names():
    with pytest.raises(PlanCompositionError, match="two scalar fields"):
        compose(STAGED_PLANS["rbf"].stages, blend="conic")
    with pytest.raises(PlanCompositionError, match="two scalar fields"):
        compose(STAGED_PLANS["simplex"].stages, blend="blob-layers")
    with pytest.raises(PlanCompositionError, match="blend mode"):
        compose(STAGED_PLANS["simplex"].stages, blend="conic:dodge")
    stages = compose(STAGED_PLANS["simplex"].stages, blend="radial:max", stage_params={"blend": {"amount": 0.5}})
    names = [s.name for s in stages]
    assert names[names.index("noisy ramp") + 1:][:2] == ["blend radial", "blend"]
