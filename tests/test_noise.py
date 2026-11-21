"""Parity of engine.gradients.noise with the `noise` C extension it replaces.

tests/fixtures/noise_reference.json holds values computed by noise 1.2.2
(snoise2/pnoise2) for float32-exact inputs. The C code works in 32-bit float
and its arm64 build uses fused multiply-add, so a few ulp of difference are
expected; the tolerance below is absolute.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from engine.gradients.noise import pnoise2, snoise2

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "noise_reference.json").read_text())
X = np.array(FIXTURE["x"], dtype=np.float32)
Y = np.array(FIXTURE["y"], dtype=np.float32)
FUNCS = {"snoise2": snoise2, "pnoise2": pnoise2}
ATOL = 1e-6


def _case_id(case):
    kw = ",".join(f"{k}={v}" for k, v in case["kwargs"].items())
    return f"{case['func']}({kw})"


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=_case_id)
def test_matches_reference(case):
    got = FUNCS[case["func"]](X, Y, **case["kwargs"])
    expected = np.array(case["values"], dtype=np.float64)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got.astype(np.float64), expected, rtol=0, atol=ATOL)


def test_broadcasts_and_keeps_shape():
    xv, yv = np.meshgrid(X[:6], Y[:4])
    grid = pnoise2(xv, yv, octaves=3)
    assert grid.shape == (4, 6)
    for r in range(4):
        for c in range(6):
            assert grid[r, c] == pnoise2(xv[r, c], yv[r, c], octaves=3)
    assert snoise2(X[:5], np.float32(0.25)).shape == (5,)


@pytest.mark.parametrize("func", [snoise2, pnoise2])
def test_rejects_non_positive_octaves(func):
    with pytest.raises(ValueError):
        func(0.5, 0.5, octaves=0)


def test_rejects_out_of_table_base():
    with pytest.raises(ValueError):
        pnoise2(0.5, 0.5, base=2)
