"""The liquify warp moves positions smoothly along a bounded flow; strokes layer to weights."""

import random

import numpy as np

from engine.stages.fields import strokes
from engine.stages.frame import sample_grid
from engine.stages.palette import make_palette
from engine.stages.warp import Frame, liquify

PALETTE = [(220, 80, 120), (18, 24, 64), (250, 190, 90), (60, 170, 200)]


def _grid():
    return sample_grid(width=160, height=90, scale=1.0, extent=(-1.0, 1.0))


def _liquify(seed=3, **kwargs):
    coords = _grid()
    return coords, liquify(coords, rng=random.Random(seed), **kwargs)


def test_deterministic_per_seed_and_seeds_differ():
    _, a = _liquify(seed=3)
    _, b = _liquify(seed=3)
    _, c = _liquify(seed=4)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.x, c.x)


def test_zero_amount_is_the_identity():
    coords, moved = _liquify(amount=0.0)
    assert np.array_equal(moved.x, coords.x) and np.array_equal(moved.y, coords.y)


def test_displacement_is_bounded_by_about_amount():
    coords = _grid()
    frame = Frame(coords)
    u0, v0 = frame.from_xy(coords.x, coords.y)
    for amount in (0.2, 0.5, 1.0):
        for seed in range(4):
            moved = liquify(coords, amount=amount, rng=random.Random(seed))
            u1, v1 = frame.from_xy(moved.x, moved.y)
            farthest = float(np.max(np.hypot(u1 - u0, v1 - v0)))
            assert 0.0 < farthest <= 1.25 * amount, (amount, seed, farthest)


def _jacobian(coords, moved):
    frame = Frame(coords)
    u0, v0 = frame.from_xy(coords.x, coords.y)
    u1, v1 = frame.from_xy(moved.x, moved.y)
    du_dy, du_dx = np.gradient(u1)
    dv_dy, dv_dx = np.gradient(v1)
    cell_dy, cell_dx = np.gradient(v0)[0].mean(), np.gradient(u0)[1].mean()
    return (du_dx * dv_dy - du_dy * dv_dx) / (cell_dx * cell_dy)


def _fine_grid():
    return sample_grid(width=480, height=270, scale=1.0, extent=(-1.0, 1.0))


def test_flow_does_not_fold_and_is_nearly_area_preserving():
    coords = _fine_grid()
    for seed in range(4):
        moved = liquify(coords, rng=random.Random(seed))
        jacobian = _jacobian(coords, moved)
        # Positive everywhere: no position overtakes its neighbour (no fold).
        assert jacobian.min() > 0.1, (seed, jacobian.min())
        # Curl flow preserves area; the explicit steps only drift slightly.
        assert abs(float(np.median(jacobian)) - 1.0) < 0.05, seed


def test_strong_flow_stays_finite_and_in_range():
    coords = _fine_grid()
    frame = Frame(coords)
    u0, v0 = frame.from_xy(coords.x, coords.y)
    moved = liquify(coords, amount=1.0, scale=1.5, steps=2, rng=random.Random(1))
    u1, v1 = frame.from_xy(moved.x, moved.y)
    assert np.isfinite(u1).all() and np.isfinite(v1).all()
    assert np.hypot(u1 - u0, v1 - v0).max() <= 1.25


def test_flow_is_smooth():
    coords = _fine_grid()
    for seed in range(4):
        moved = liquify(coords, rng=random.Random(seed))
        u, v = Frame(coords).from_xy(moved.x, moved.y)
        assert np.abs(np.diff(u, n=2, axis=1)).max() < 0.015
        assert np.abs(np.diff(v, n=2, axis=0)).max() < 0.015


def _strokes(seed=4, **kwargs):
    return strokes(_grid(), make_palette(colors=PALETTE), rng=random.Random(seed), **kwargs)


def test_strokes_weights_sum_to_one_and_are_finite():
    for count in (1, 3, 6):
        field = _strokes(strokes=count)
        assert field.weights.shape[0] == 4
        assert np.isfinite(field.weights).all()
        assert (field.weights >= 0.0).all()
        assert np.allclose(field.weights.sum(axis=0), 1.0, atol=1e-5)
        assert field.background == 1


def test_strokes_deterministic_and_every_color_appears():
    assert np.array_equal(_strokes().weights, _strokes().weights)
    assert not np.array_equal(_strokes().weights, _strokes(seed=5).weights)
    field = _strokes(strokes=6)
    for color in (0, 2, 3):
        assert field.weights[color].max() > 0.3, color


def _orientation_dominance(field, bins=18):
    """Share of the gradient magnitude within one histogram bin of the dominant orientation."""
    lightness = np.asarray(make_palette(colors=PALETTE).oklab)[:, 0]
    height = np.tensordot(lightness, field.weights, axes=1)
    grad_y, grad_x = np.gradient(height)
    theta = np.mod(np.arctan2(grad_y, grad_x), np.pi)
    hist, _ = np.histogram(theta, bins=bins, range=(0.0, np.pi), weights=np.hypot(grad_x, grad_y))
    hist = hist / hist.sum()
    peak = int(hist.argmax())
    return float(sum(hist[(peak + d) % bins] for d in (-1, 0, 1)))


def test_strokes_fan_out_instead_of_running_in_parallel_stripes():
    # Seed 21 used to come out as near-parallel stripes: every stroke shared one flow direction.
    shares = [_orientation_dominance(_strokes(seed=seed)) for seed in range(24)]
    assert _orientation_dominance(_strokes(seed=21)) < 0.45
    assert max(shares) < 0.6
    assert float(np.mean(shares)) < 0.45
