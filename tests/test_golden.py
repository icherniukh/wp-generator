"""Pins the current renders; see tests/golden.py for how to regenerate them."""

import numpy as np
import pytest

from tests.golden import golden_cases, load_golden, render_case

# libm and numpy builds differ by a few ULPs across platforms; those can flip a
# rounding step or an escape-time iteration, so allow rare, small differences.
MAX_DIFF = 3
MAX_CHANGED_FRACTION = 0.002

GOLDEN = load_golden()


@pytest.mark.parametrize("case", golden_cases(), ids=lambda c: c.name)
def test_render_matches_golden(case):
    assert case.name in GOLDEN, f"no golden for {case.name}; run: uv run python -m tests.golden --update"
    image = render_case(case)
    ref = GOLDEN[case.name]
    assert image.shape == ref.shape
    diff = np.abs(image.astype(np.int16) - ref.astype(np.int16)).max(axis=2)
    changed = float(np.mean(diff > MAX_DIFF))
    assert changed <= MAX_CHANGED_FRACTION, (
        f"{case.name}: {changed:.2%} of pixels differ by more than {MAX_DIFF} "
        f"(max {diff.max()}); if intended, run: uv run python -m tests.golden --update"
    )
