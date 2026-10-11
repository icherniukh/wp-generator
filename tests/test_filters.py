import unittest

import numpy as np
from PIL import Image

from engine.filters import WallpaperFilters


class TestWallpaperDither(unittest.TestCase):
    def test_dither_is_subtle_and_monochromatic(self):
        np.random.seed(7)
        image = Image.new("RGB", (64, 64), (128, 128, 128))

        result = np.asarray(WallpaperFilters.apply_dither(image, amount=0.2))
        delta = result.astype(np.int16) - 128

        self.assertLessEqual(np.abs(delta).max(), 15)
        np.testing.assert_array_equal(delta[..., 0], delta[..., 1])
        np.testing.assert_array_equal(delta[..., 1], delta[..., 2])

    def test_vignette_keeps_center_brighter_than_edges(self):
        image = Image.new("RGB", (101, 101), (200, 200, 200))

        result = np.asarray(WallpaperFilters.apply_vignette(image, strength=0.1))

        self.assertGreater(result[50, 50, 0], result[0, 0, 0])
        self.assertGreaterEqual(result[50, 50, 0], 198)


if __name__ == "__main__":
    unittest.main()
