"""The patch mesh passes through its vertex colors and blends smoothly between them."""

import random

import numpy as np

from engine.stages.fields import patch_mesh
from engine.stages.frame import sample_grid
from engine.stages.palette import make_palette

PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90), (60, 170, 200)]


def _mesh(width=61, height=41, **kwargs):
    coords = sample_grid(width=width, height=height, scale=1.0, extent=(0.0, 1.0))
    rng = random.Random(3)
    return patch_mesh(coords, make_palette(colors=PALETTE), rng=rng, **kwargs).weights


def test_weights_are_a_partition_and_every_color_appears():
    weights = _mesh(columns=4, rows=3)
    assert weights.shape[0] == len(PALETTE)
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-5)
    assert weights.min() >= 0.0
    assert all(weights[i].max() > 0.9 for i in range(len(PALETTE)))


def test_without_jitter_vertices_hold_one_color():
    weights = _mesh(width=61, height=41, columns=4, rows=3, jitter=0.0)
    # Vertices of a 4x3 mesh on a 61x41 grid fall on pixels 0, 20, 40, 60 and 0, 20, 40.
    for y in (0, 20, 40):
        for x in (0, 20, 40, 60):
            assert weights[:, y, x].max() > 0.999


def test_same_rng_same_mesh_and_jitter_moves_it():
    assert np.array_equal(_mesh(), _mesh())
    assert not np.allclose(_mesh(jitter=0.0), _mesh(jitter=0.4))
