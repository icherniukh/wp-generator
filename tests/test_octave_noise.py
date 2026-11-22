"""Octave-noise fields (ridged, billow, fbm) have no band along the grid's edges.

The half-size coordinate grid used to be built with scipy's default zoom mode,
which turned the last sample into 0 when float rounding put it a hair past the
edge. That pulled the noise to position 0 for the last half-size row (or
column), and the cubic enlargement smeared it over the last four or so rows.
"""

import random

import numpy as np
import pytest

import engine.gradients.noise as noise
from engine.stages import fields
from engine.stages.frame import rotate, sample_grid

OCTAVE_FIELDS = {
    "ridged": lambda c, rng: fields.ridged_noise(c, scale=0.5, sharpness=4, rng=rng),
    "billow": lambda c, rng: fields.billow_noise(c, scale=0.5, rng=rng),
    "fbm": lambda c, rng: fields.fbm_noise(c, scale=0.5, rng=rng),
}


@pytest.mark.parametrize("height", range(16, 130))
def test_half_size_samples_span_the_grid_to_its_last_row_and_column(height, monkeypatch):
    seen = []
    real = noise.snoise2

    def spy(x, y, **kwargs):
        seen.append((x, y))
        return real(x, y, **kwargs)

    monkeypatch.setattr(noise, "snoise2", spy)
    for width in (height, int(height * 16 / 9)):
        seen.clear()
        coords = sample_grid(width=width, height=height, scale=1.0, extent=(-1.0, 1.0))
        fields.fbm_noise(coords, octaves=1, scale=1.0, rng=random.Random(0))
        x, y = seen[0]
        # One octave at frequency 1 shifts the samples by a constant, so their extent
        # must still be the grid's full span of 2.0 along each axis.
        assert float(x.max() - x.min()) == pytest.approx(2.0, abs=1e-5), (width, height)
        assert float(y.max() - y.min()) == pytest.approx(2.0, abs=1e-5), (width, height)


@pytest.mark.parametrize("name", OCTAVE_FIELDS)
@pytest.mark.parametrize("rotated", [False, True])
def test_edge_rows_match_the_interior(name, rotated):
    # 640x360 at the default grid scale gives the 160x90 grid where the band showed.
    rows, cols = [], []
    for seed in range(1, 25):
        rng = random.Random(seed)
        coords = sample_grid(width=640, height=360, scale=0.25, extent=(-1.0, 1.0))
        if rotated:
            coords = rotate(coords, rng=rng)
        t = OCTAVE_FIELDS[name](coords, rng).t.astype(np.float64)
        rows.append(t.mean(axis=1))
        cols.append(t.mean(axis=0))
    rows, cols = np.mean(rows, axis=0), np.mean(cols, axis=0)

    # A band is a steep run in the last few rows: the last row sits far from the one 5
    # rows in, relative to how much the seed-averaged profile moves between neighbors.
    for profile in (rows, cols):
        interior = np.abs(np.diff(profile[8:-8])).mean()
        assert abs(profile[-1] - profile[-6]) < 4 * interior + 0.02, (name, profile[-8:])
        assert abs(profile[0] - profile[5]) < 4 * interior + 0.02, (name, profile[:8])
