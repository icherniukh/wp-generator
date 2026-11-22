"""Hard-edged shape layers: crisp analytic edges at full resolution, no ringing."""

import random

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from engine.colors import rgb_to_oklab
from engine.pipeline import RenderConfiguration, WallpaperPipeline
from engine.stages import STAGED_PLANS, Composition, render_staged
from engine.stages.catalog import PlanCompositionError, compose
from engine.stages.fields import SHAPE_KINDS, shape_coverage, shape_distance, shapes
from engine.stages.frame import sample_grid
from engine.stages.palette import make_palette

PALETTE = [(30, 60, 150), (220, 90, 70), (240, 200, 90), (70, 190, 160)]
NO_FINISH = dict(noise=0.0, vignette=0.0, contours=0.0, glass=False, chromatic_aberration=0.0)


def _field(seed=4, width=96, height=54, **kwargs):
    coords = sample_grid(width=width, height=height, scale=1.0, extent=(0.0, 1.0))
    return shapes(coords, make_palette(colors=PALETTE), rng=random.Random(seed), **kwargs)


def _params(width, height, **extra):
    return dict(width=width, height=height, colors=PALETTE, complexity=5, seed=7, **extra)


@pytest.mark.parametrize("hardness", [0.0, 0.5, 0.97, 1.0])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_weights_sum_to_one_and_are_finite(seed, hardness):
    weights = _field(seed=seed, hardness=hardness).weights
    assert np.isfinite(weights).all()
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-5)
    assert weights.min() >= 0.0 and weights.max() <= 1.0


def test_deterministic_per_seed():
    assert np.array_equal(_field(seed=9).weights, _field(seed=9).weights)
    assert not np.array_equal(_field(seed=9).weights, _field(seed=10).weights)
    first = render_staged("shapes", _params(160, 90))
    assert np.array_equal(np.asarray(first), np.asarray(render_staged("shapes", _params(160, 90))))


def test_every_color_but_the_base_appears_and_the_darkest_is_the_base():
    field = _field(count=6)
    assert field.background == int(np.argmin(make_palette(colors=PALETTE).oklab[:, 0]))
    for color in range(len(PALETTE)):
        assert field.weights[color].max() > 0.99, color


@pytest.mark.parametrize("kind", SHAPE_KINDS)
def test_each_kind_draws_one_flat_colored_shape(kind):
    field = _field(count=1, kinds=[kind], hardness=1.0)
    weights, base = field.weights, field.background
    assert (weights.max(axis=0) > 0.99).mean() > 0.9  # almost every pixel is one flat color
    covered = (weights[base] < 0.5).mean()
    assert 0.0 < covered < 1.0


def test_edge_is_at_most_two_pixels_wide_at_full_hd():
    """Across a known circle: pixels that are neither inside nor outside, per pixel of edge length."""
    width, height = 1920, 1080
    coords = sample_grid(width=width, height=height, scale=1.0, extent=(0.0, 1.0))
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    radius_px = 300.0
    # Distance in half short sides, as the shapes use.
    half = height / 2.0
    distance = (np.hypot(x - 900.3, y - 520.7) - radius_px) / half
    assert coords.shape == (height, width)
    coverage = shape_coverage(distance.astype(np.float32), hardness=1.0)
    partial = (coverage > 0.02) & (coverage < 0.98)
    assert partial.sum() / (2 * np.pi * radius_px) <= 2.0
    # Along the middle row the transition is at most 2 pixels, and it is anti-aliased (not a step).
    row = coverage[521, :]
    assert (0.02 < row).sum() - (row > 0.98).sum() <= 2
    assert partial.any()
    # The edge sits where the circle does: area within half a pixel's worth of pi r^2.
    assert abs(coverage.sum() - np.pi * radius_px**2) < 0.002 * np.pi * radius_px**2


def test_edge_softens_as_hardness_drops():
    x, y = np.meshgrid(np.arange(400, dtype=np.float32), np.arange(400, dtype=np.float32))
    distance = ((np.hypot(x - 200, y - 200) - 120) / 200.0).astype(np.float32)
    widths = [((c > 0.02) & (c < 0.98)).sum() for c in (shape_coverage(distance, h) for h in (1.0, 0.97, 0.9, 0.7, 0.4))]
    assert widths == sorted(widths) and widths[-1] > 20 * widths[0]
    # The default is still crisp: under 3 pixels of transition around a 120 pixel radius.
    assert widths[1] / (2 * np.pi * 120) < 3.0


def _hull_excess(image, palette):
    """How far each pixel lies outside the convex hull of the palette, in Oklab."""
    hull = ConvexHull(rgb_to_oklab(np.asarray(palette, dtype=np.float64)))
    lab = rgb_to_oklab(np.asarray(image, dtype=np.float64).reshape(-1, 3))
    return (lab @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)


def _render_edges(width, height, composition=None):
    config = RenderConfiguration(engine="shapes", colors=PALETTE, harmonize="none", width=width, height=height,
                                 seed=5, **NO_FINISH)
    if composition is None:
        return np.asarray(WallpaperPipeline().render(config).image.convert("RGB"))
    return np.asarray(render_staged("shapes", _params(width, height), composition=composition).convert("RGB"))


def test_no_ringing_or_colors_outside_the_palette_hull():
    image = _render_edges(1920, 1080)
    assert image.shape == (1080, 1920, 3)
    # 8-bit rounding and one level of dither move a color by under 0.005 in Oklab
    # (0.0023 measured); a coarse grid enlarged with Lanczos overshoots to 0.016.
    assert _hull_excess(image, PALETTE).max() < 0.006


def test_a_coarse_grid_enlarged_would_ring():
    """The check above sees ringing: sampling at a quarter size and enlarging overshoots."""
    coarse = _render_edges(1920, 1080, Composition(stage_params={"grid": {"scale": 0.25}}))
    assert _hull_excess(coarse, PALETTE).max() > 0.01


def test_composed_shapes_field_gets_a_full_resolution_grid():
    for engine in ("simplex", "topographic", "idw", "patch-mesh"):
        state = {}
        render_staged(engine, _params(160, 90), composition=Composition(field="shapes"), keep_state=state)
        assert state["coords"].shape == (90, 160), engine
        assert np.allclose(state["field"].weights.sum(axis=0), 1.0, atol=1e-5), engine


def test_grid_scale_can_still_be_overridden():
    state = {}
    render_staged("simplex", _params(160, 90), keep_state=state,
                  composition=Composition(field="shapes", stage_params={"grid": {"scale": 0.5}}))
    assert state["coords"].shape == (45, 80)


def test_engines_that_cannot_sample_at_full_size_refuse():
    for engine in ("fluid-wave", "fractal"):
        with pytest.raises(PlanCompositionError, match="full-resolution"):
            render_staged(engine, _params(48, 30, fractal_type="fluid", fractal_zoom=None, fractal_center=None,
                                          fractal_angle=None), composition=Composition(field="shapes"))


def test_kinds_hardness_and_count_are_stage_arguments():
    stages = compose(STAGED_PLANS["shapes"].stages,
                     stage_params={"shapes": {"kinds": ["circle", "ring"], "count": 2, "hardness": 0.5}})
    field = next(s for s in stages if s.phase == "field")
    arguments = field.bind({"complexity": 5})
    assert (arguments["kinds"], arguments["count"], arguments["hardness"]) == (["circle", "ring"], 2, 0.5)
    image = render_staged("shapes", _params(96, 54), composition=Composition(
        stage_params={"shapes": {"kinds": ["circle"], "count": 2}}))
    assert image.size == (96, 54)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="shape kinds"):
        _field(kinds=["hexagon"])
    with pytest.raises(ValueError, match="shape kinds"):
        _field(kinds=[])
    with pytest.raises(ValueError, match="shape kind"):
        shape_distance("hexagon", np.zeros(2), np.zeros(2), 1.0, 1.0)


def test_count_follows_complexity():
    from engine.stages.plans import shape_layers

    counts = [shape_layers({"complexity": c})["count"] for c in (1, 4, 10, 30)]
    assert counts == sorted(counts) and counts[0] >= 3 and counts[-1] <= 9
