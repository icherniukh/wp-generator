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

    def test_chromatic_aberration_separates_channels_radially(self):
        # Create an image with a sharp edge circle in the corner
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:30, :30] = 255
        image = Image.fromarray(arr)

        aberrated = WallpaperFilters.apply_chromatic_aberration(image, strength=0.03)
        res_arr = np.asarray(aberrated)

        # Zero strength returns identical image
        identity = WallpaperFilters.apply_chromatic_aberration(image, strength=0.0)
        np.testing.assert_array_equal(np.asarray(identity), arr)

        # Center pixel (50, 50) has zero displacement
        self.assertEqual(res_arr[50, 50, 0], arr[50, 50, 0])

        # Near edge where displacement is maximum, red and blue diverge
        self.assertFalse(np.array_equal(res_arr[..., 0], res_arr[..., 2]))

    def test_film_grain_increases_pixel_variance(self):
        image = Image.new("RGB", (80, 80), (128, 128, 128))
        grained = WallpaperFilters.apply_film_grain(image, amount=0.08)
        res_arr = np.asarray(grained).astype(np.float32)

        # Variance is substantially higher than clean image
        self.assertGreater(np.var(res_arr), 10.0)

        # Amount 0 is identity
        no_grain = WallpaperFilters.apply_film_grain(image, amount=0.0)
        np.testing.assert_array_equal(np.asarray(no_grain), np.asarray(image))


if __name__ == "__main__":
    unittest.main()
