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


class TestDomainWarpGradient(unittest.TestCase):
    def test_domain_warp_generates_correct_dimensions_and_mode(self):
        from engine.gradient import DomainWarpGradient
        engine = DomainWarpGradient(width=160, height=100)
        colors = [(20, 20, 40), (200, 80, 120), (50, 150, 220)]
        img = engine.generate(colors, complexity=4, warp_strength=0.8, seed=42)

        self.assertEqual(img.size, (160, 100))
        self.assertEqual(img.mode, "RGB")

    def test_domain_warp_is_deterministic_with_seed(self):
        from engine.gradient import DomainWarpGradient
        engine = DomainWarpGradient(width=120, height=80)
        colors = [(10, 20, 30), (240, 100, 50)]

        img1 = engine.generate(colors, complexity=5, warp_strength=1.0, seed=777)
        img2 = engine.generate(colors, complexity=5, warp_strength=1.0, seed=777)

        np.testing.assert_array_equal(np.asarray(img1), np.asarray(img2))


if __name__ == "__main__":
    unittest.main()
