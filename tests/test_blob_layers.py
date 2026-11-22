"""Blob layers cover what is under them instead of averaging with it."""

import random

import numpy as np

from engine.stages.fields import blob_layers, composite_over
from engine.stages.frame import sample_grid
from engine.stages.palette import make_palette

PALETTE = [(220, 80, 120), (18, 24, 64), (250, 190, 90), (60, 170, 200)]


def test_opaque_top_layer_gives_its_own_color():
    ones = np.ones((3, 3), np.float32)
    weights = composite_over([ones * 0.7, ones], [2, 3], base=1, num_colors=4)
    assert np.array_equal(weights[3], ones)
    assert not weights[[0, 1, 2]].any()


def test_weights_sum_to_one_without_normalizing():
    rng = np.random.default_rng(0)
    alphas = [rng.uniform(0, 1, (5, 7)).astype(np.float32) for _ in range(4)]
    weights = composite_over(alphas, [0, 2, 2, 3], base=1, num_colors=4)
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-6)
    # Bottom layer under a half-transparent top: a1 * (1 - a2).
    two = composite_over([np.full((1,), 0.8), np.full((1,), 0.5)], [0, 2], base=1, num_colors=3)
    assert np.allclose(two[:, 0], [0.4, 0.1, 0.5])


def _blobs(**kwargs):
    coords = sample_grid(width=64, height=36, scale=1.0, extent=(0.0, 1.0))
    return blob_layers(coords, make_palette(colors=PALETTE), rng=random.Random(4), **kwargs)


def test_darkest_color_is_the_base_and_every_color_appears():
    field = _blobs(blobs=4)
    assert field.background == 1
    assert np.allclose(field.weights.sum(axis=0), 1.0, atol=1e-5)
    for color in (0, 2, 3):
        assert field.weights[color].max() > 0.3, color


def test_wider_blur_lowers_the_peak():
    sharp, soft = _blobs(blobs=1, blur=0.3), _blobs(blobs=1, blur=1.5)
    assert sharp.weights[1].min() < soft.weights[1].min()
    assert np.array_equal(_blobs().weights, _blobs().weights)
