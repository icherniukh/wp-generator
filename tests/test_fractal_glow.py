"""fractal-glow: a neon escape-time fractal field at full resolution."""

import random

import numpy as np
import pytest

from engine.stages import Composition, render_staged
from engine.stages.catalog import FIELDS
from engine.stages.fields import GLOW_JULIA_CONSTANTS, fractal_glow
from engine.stages.frame import sample_grid

PALETTE = [(11, 15, 42), (255, 61, 129), (124, 77, 255), (0, 229, 255), (255, 224, 102)]


def _grid(width=160, height=90):
    return sample_grid(width=width, height=height, scale=1.0, extent=(-0.5, 0.5))


def _field(seed=3, **kwargs):
    return fractal_glow(_grid(), rng=random.Random(seed), **kwargs).t


@pytest.mark.parametrize("kind", ["julia", "mandelbrot"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_field_shape_range_and_content(kind, seed):
    t = _field(seed=seed, kind=kind)
    assert t.shape == (90, 160)
    assert np.isfinite(t).all()
    assert t.min() >= 0.0 and t.max() <= 1.0
    assert t.max() > 0.99  # filaments reach the top of the ramp
    assert 0.02 < float((t < 0.1).mean()) < 0.98  # and there is a dark ground


def test_deterministic_per_seed_and_seeds_differ():
    assert np.array_equal(_field(seed=5), _field(seed=5))
    assert not np.array_equal(_field(seed=5), _field(seed=6))


def test_parameters_do_not_shift_the_random_draws():
    """Giving c or center leaves the angle, and the rest, as the seed set them."""
    base = random.Random(4)
    fractal_glow(_grid(), rng=base)
    other = random.Random(4)
    fractal_glow(_grid(), rng=other, c=[-0.4, 0.6], center=[0.0, 0.0], kind="mandelbrot")
    assert base.random() == other.random()


def test_wider_fade_lights_more_of_the_frame():
    narrow, wide = _field(fade=0.004), _field(fade=0.05)
    assert wide.mean() > narrow.mean()
    assert np.all(wide >= narrow - 1e-6)


def test_every_curated_constant_draws_a_fractal():
    for c in GLOW_JULIA_CONSTANTS:
        t = _field(c=list(c))
        assert t.max() > 0.99 and t.min() < 0.1, c


def test_interior_blend_only_dims_the_band():
    plain, banded = _field(interior=0.0), _field(interior=1.0)
    assert np.all(banded <= plain + 1e-6)
    assert not np.array_equal(plain, banded)


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        _field(kind="newton")


def test_wide_fade_is_steady_when_the_boundary_flips():
    """A sample that escapes or not within the budget, as across platforms, moves the
    field by a small amount at a wide fade (the golden case relies on this)."""
    c = GLOW_JULIA_CONSTANTS[0]
    a = _field(fade=0.2, c=list(c))
    b = _field(fade=0.2, c=[c[0] + 1e-9, c[1]])  # orbits diverge after a few dozen steps
    assert float(np.abs(a - b).max()) < 0.03


def test_catalogue_entry_has_full_resolution_grid_and_a_dark_ground():
    assert FIELDS["fractal-glow"].full_resolution
    for engine in ("simplex", "topographic"):
        state = {}
        image = render_staged(
            engine,
            dict(width=160, height=90, colors=PALETTE, complexity=5, seed=7),
            composition=Composition(field="fractal-glow"),
            keep_state=state,
        )
        assert state["coords"].shape == (90, 160), engine
        rgb = np.asarray(image.convert("RGB"), dtype=float)
        luminance = rgb @ [0.2126, 0.7152, 0.0722]
        assert np.median(luminance) < 60, engine  # dark ground
        assert luminance.max() > 200, engine  # bright filaments
