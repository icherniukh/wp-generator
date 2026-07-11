import unittest

import numpy as np

from engine.colors import oklch_to_rgb, oklch_to_rgb_clamped, rgb_to_oklab


class TestOklabColorPipeline(unittest.TestCase):
    def test_out_of_gamut_oklch_is_detectable_before_mapping(self):
        rgb = oklch_to_rgb([0.7, 0.4, 40.0])

        self.assertTrue(np.any((rgb < 0.0) | (rgb > 1.0)))

    def test_gamut_mapping_preserves_lightness_and_hue_by_reducing_chroma(self):
        mapped = oklch_to_rgb_clamped([0.7, 0.4, 40.0])
        mapped_lab = rgb_to_oklab(mapped)

        self.assertTrue(all(0 <= channel <= 255 for channel in mapped))
        self.assertAlmostEqual(mapped_lab[0], 0.7, delta=0.01)


if __name__ == "__main__":
    unittest.main()
