import unittest

import numpy as np

from engine.gradient import _interpolate_oklab, _random_anchor_points


class TestPerceptualGradientInterpolation(unittest.TestCase):
    def test_midpoint_is_interpolated_in_oklab(self):
        midpoint = _interpolate_oklab(
            np.array([0, 0, 0], dtype=np.float32),
            np.array([255, 255, 255], dtype=np.float32),
            0.5,
        )

        np.testing.assert_allclose(midpoint, [99, 99, 99], atol=2)

    def test_random_anchors_stay_away_from_image_boundaries(self):
        points = _random_anchor_points(100, margin=0.1)

        self.assertTrue(all(0.1 <= x <= 0.9 and 0.1 <= y <= 0.9 for x, y in points))


if __name__ == "__main__":
    unittest.main()
