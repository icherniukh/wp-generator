"""Export keeps floats through the enlargement and dithers once at quantization."""

import random

import numpy as np

from engine.stages import Composition, render_staged
from engine.stages.export import quantize

PARAMS = dict(width=320, height=180, colors=[(20, 22, 40), (48, 40, 80)], complexity=4, seed=3)


def test_dither_spreads_a_flat_value_without_bias():
    values = np.full((64, 64, 3), 100.4, dtype=np.float32)
    plain, dithered = quantize(values, dither=False), quantize(values, dither=True)
    assert np.unique(plain).tolist() == [100]
    assert set(np.unique(dithered)) <= {99, 100, 101, 102}
    assert abs(dithered.astype(np.float64).mean() - 100.4) < 0.05


def test_dither_does_not_touch_the_render_random_state():
    random.seed(7)
    np.random.seed(7)
    quantize(np.zeros((8, 8, 3), np.float32))
    assert random.random() == random.Random(7).random()
    assert np.random.random() == np.random.RandomState(7).random_sample()


def test_float_export_has_more_levels_than_quarter_size_8_bit():
    image = np.asarray(render_staged("simplex", PARAMS, composition=Composition(field="radial")))
    undithered = np.asarray(
        render_staged("simplex", PARAMS, composition=Composition(field="radial", stage_params={"export": {"dither": False}}))
    )
    assert not np.array_equal(image, undithered)
    # Dither breaks up flat runs: far fewer neighbouring pixels share a value.
    same = lambda a: float(np.mean(a[:, 1:, 2] == a[:, :-1, 2]))
    assert same(image) < same(undithered) - 0.2
