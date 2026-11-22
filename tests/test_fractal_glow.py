"""fractal-glow: a neon escape-time fractal field at full resolution."""

import random

import numpy as np
import pytest
import scipy.ndimage

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
    assert t.max() > 0.9  # the line nearly reaches the top of the ramp (sub-pixel averaging keeps the peak up)
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
        assert t.max() > 0.9 and t.min() < 0.1, c


def test_interior_blend_only_dims_the_band():
    plain, banded = _field(interior=0.0), _field(interior=1.0)
    assert np.all(banded <= plain + 1e-6)
    assert not np.array_equal(plain, banded)


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        _field(kind="newton")


def test_wide_fade_is_steady_when_the_boundary_flips():
    """A sample that escapes or not within the budget, as across platforms, moves the
    field by a small amount at a wide fade without the line (the golden case relies on
    this): shifting c by 1e-9 and cutting the budget by 20 moves the field by under 0.03."""
    c = GLOW_JULIA_CONSTANTS[0]
    kwargs = dict(fade=0.2, core=0.0, glow=0.6, ambient=0.3, samples=4)
    a = _field(c=list(c), iterations=300, **kwargs)
    b = _field(c=[c[0] + 1e-9, c[1]], iterations=280, **kwargs)  # orbits diverge after a few dozen steps
    assert float(np.abs(a - b).max()) < 0.03


def test_line_is_thin_and_the_halo_is_dimmer():
    """The line is about core pixels wide at peak 1; away from it the value is the halo,
    capped near glow, so a dense boundary is lines over a dim fill and not a white blob."""
    line = _field(kind="mandelbrot", samples=1, core=1.0, glow=0.4, fade=0.03)
    halo = _field(kind="mandelbrot", samples=1, core=0.0, glow=0.4, fade=0.03)
    assert halo.max() <= 0.4 + 1e-5
    assert line.max() > 0.9
    assert float((line > 0.8).mean()) < 0.1  # a thin line, not a filled region
    assert np.all(line >= halo - 1e-6)
    wide = _field(kind="mandelbrot", samples=1, core=6.0, glow=0.4, fade=0.03)
    assert float((wide > 0.5).mean()) > float((line > 0.5).mean())


def test_dense_boundary_does_not_clip_to_a_clump():
    """Where filaments are denser than a pixel, the lines are dimmed by how much line
    is around them: no large patch sits at the top of the ramp."""
    t = _field(seed=2, kind="mandelbrot", zoom=40.0, center=[-0.745, 0.113], glow=0.45)
    blob = scipy.ndimage.uniform_filter((t > 0.97).astype(float), 9)
    assert blob.max() < 0.5


def test_supersampling_antialiases_the_line():
    """More sub-pixel samples change the line pixels by partial coverage (values between
    the ground and the peak), and leave the smooth ground alone."""
    one = _field(kind="mandelbrot", samples=1)
    two = _field(kind="mandelbrot", samples=2)
    four = _field(kind="mandelbrot", samples=4)
    assert not np.array_equal(one, two)
    assert float(np.abs(two - four).mean()) < float(np.abs(one - four).mean())
    far = one < 0.01
    assert np.allclose(one[far], two[far], atol=2e-3)


def test_ambient_lifts_the_ground_near_the_set_only():
    plain, lifted = _field(ambient=0.0), _field(ambient=0.4)
    assert np.all(lifted >= plain - 1e-6)
    assert lifted.mean() > plain.mean() + 0.05
    assert lifted.min() < 0.4  # still darker far from the set than near it
    assert plain.min() < 0.01


def test_iterations_auto_and_explicit():
    from engine.stages.fields import _glow_iterations

    assert _glow_iterations("auto", False, 1.0) == 300
    assert _glow_iterations("auto", False, 50.0) == 300
    assert _glow_iterations("auto", True, 1.0) == 500
    assert _glow_iterations("auto", True, 0.2) == 500
    assert _glow_iterations("auto", True, 8.0) == 1100
    assert _glow_iterations("auto", True, 2.0**40) == 2000
    assert _glow_iterations(123, True, 1.0) == 123
    assert _glow_iterations("77", False, 1.0) == 77
    with pytest.raises(ValueError):
        _glow_iterations("deep", True, 1.0)
    # a deeper explicit budget finds more of the boundary than a shallow one
    shallow = _field(kind="mandelbrot", iterations=12, samples=1)
    deep = _field(kind="mandelbrot", iterations="auto", samples=1)
    assert not np.array_equal(shallow, deep)


def test_periodic_orbits_do_not_change_the_classification():
    """Orbits that settle are dropped early; the escape mask equals a plain run's."""
    from engine.stages import fields

    p = np.linspace(-0.76, -0.70, 60)[None, :] + 1j * np.linspace(0.05, 0.12, 40)[:, None]
    d, _, escaped = fields._distance_estimate(np.zeros_like(p), p, mandelbrot=True, iterations=400)
    z = np.zeros_like(p)
    with np.errstate(all="ignore"):
        for _ in range(400):
            z = z * z + p
            z = np.where(np.abs(z) > 1e6, 1e6, z)  # keep escaped points from overflowing
    reference = np.abs(z) > 1e3
    assert np.mean(escaped == reference) > 0.995


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


def test_chunking_and_threads_do_not_change_the_field(monkeypatch):
    from engine.stages import fields

    whole = _field(kind="mandelbrot", seed=2)
    monkeypatch.setattr(fields, "_GLOW_CHUNK", 997)
    monkeypatch.setattr(fields, "_GLOW_THREADS", 4)
    assert np.array_equal(whole, _field(kind="mandelbrot", seed=2))
    monkeypatch.setattr(fields, "_GLOW_THREADS", 1)
    assert np.array_equal(whole, _field(kind="mandelbrot", seed=2))
