import unittest

import numpy as np

from engine.colors import (
    ColorTheoryHarmonizer, StyleHarmonizer, clamp_oklch, contrast_ratio,
    evaluate_palette_contrast, oklch_to_rgb, oklch_to_rgb_clamped,
    rgb_to_oklab, rgb_to_oklch,
)


class TestOklabColorPipeline(unittest.TestCase):
    def test_out_of_gamut_oklch_is_detectable_before_mapping(self):
        rgb = oklch_to_rgb([0.7, 0.4, 40.0])

        self.assertTrue(np.any((rgb < 0.0) | (rgb > 1.0)))

    def test_gamut_mapping_preserves_lightness_and_hue_by_reducing_chroma(self):
        mapped = oklch_to_rgb_clamped([0.7, 0.4, 40.0])
        mapped_lab = rgb_to_oklab(mapped)

        self.assertTrue(all(0 <= channel <= 255 for channel in mapped))
        self.assertAlmostEqual(mapped_lab[0], 0.7, delta=0.01)

    def test_dark_integer_rgb_is_not_treated_as_normalized_rgb(self):
        integer = rgb_to_oklch((1, 0, 0))
        normalized = rgb_to_oklch((1.0 / 255.0, 0.0, 0.0))
        np.testing.assert_allclose(integer, normalized, atol=1e-6)

    def test_clamp_preserves_hue_and_respects_requested_bounds(self):
        result = clamp_oklch([0.95, 0.3, 275.0],
                             lightness_bounds=(0.2, 0.8), chroma_bounds=(0.0, 0.15))
        np.testing.assert_allclose(result, [0.8, 0.15, 275.0])

    def test_gamut_mapping_clamps_out_of_range_lightness(self):
        self.assertEqual(oklch_to_rgb_clamped([-0.5, 0.2, 120.0]), (0, 0, 0))
        self.assertEqual(oklch_to_rgb_clamped([1.5, 0.2, 120.0]), (255, 255, 255))

    def test_invalid_clamping_bounds_fail(self):
        with self.assertRaises(ValueError):
            clamp_oklch([0.5, 0.1, 90.0], lightness_bounds=(0.8, 0.2))
        with self.assertRaises(ValueError):
            clamp_oklch([0.5, 0.1, 90.0], chroma_bounds=(-0.1, 0.2))


class TestPerceptualHarmonization(unittest.TestCase):
    def test_default_harmonizer_preserves_each_hue_and_input_order(self):
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
        result = ColorTheoryHarmonizer.harmonize(colors)
        self.assertEqual(len(result), len(colors))
        for original, adjusted in zip(colors, result):
            before, after = rgb_to_oklch(original), rgb_to_oklch(adjusted)
            hue_difference = abs((after[2] - before[2] + 180.0) % 360.0 - 180.0)
            self.assertLess(hue_difference, 2.0)
            self.assertTrue(0.119 <= after[0] <= 0.901)
            self.assertLessEqual(after[1], 0.251)

    def test_neutral_palette_stays_neutral(self):
        colors = [(0, 0, 0), (128, 128, 128), (255, 255, 255)]
        for harmonizer in (ColorTheoryHarmonizer.harmonize, StyleHarmonizer.adjust_for_nothing):
            for adjusted in harmonizer(colors):
                self.assertEqual(adjusted[0], adjusted[1])
                self.assertEqual(adjusted[1], adjusted[2])

    def test_nothing_adjustment_preserves_saturated_hues(self):
        for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)):
            adjusted = StyleHarmonizer.adjust_for_nothing([color])[0]
            before, after = rgb_to_oklch(color), rgb_to_oklch(adjusted)
            self.assertLess(abs((after[2] - before[2] + 180) % 360 - 180), 2.0)

    def test_explicit_classical_modes_remain_available(self):
        colors = [(220, 50, 60), (60, 80, 200)]
        for mode in ColorTheoryHarmonizer.MODES:
            result = ColorTheoryHarmonizer.harmonize(colors, mode=mode)
            self.assertEqual(len(result), len(colors))
            self.assertTrue(all(0 <= channel <= 255 for color in result for channel in color))


class TestPaletteContrast(unittest.TestCase):
    def test_black_white_contrast_is_21_and_order_independent(self):
        self.assertAlmostEqual(contrast_ratio((0, 0, 0), (255, 255, 255)), 21.0)
        self.assertAlmostEqual(contrast_ratio((255, 255, 255), (0, 0, 0)), 21.0)
        self.assertEqual(contrast_ratio((100, 100, 100), (100, 100, 100)), 1.0)

    def test_luminance_weights_are_applied_to_the_correct_channels(self):
        self.assertAlmostEqual(contrast_ratio((0, 0, 0), (255, 0, 0)), 5.252)
        self.assertAlmostEqual(contrast_ratio((0, 0, 0), (0, 255, 0)), 15.304)
        self.assertAlmostEqual(contrast_ratio((0, 0, 0), (0, 0, 255)), 2.444)

    def test_each_accent_is_checked_against_the_anchor(self):
        contrast = evaluate_palette_contrast([(0, 0, 0), (255, 255, 255), (20, 20, 20)])
        self.assertEqual(len(contrast.ratios), 2)
        self.assertAlmostEqual(contrast.ratios[0], 21.0)
        self.assertFalse(contrast.passes)
        self.assertTrue(evaluate_palette_contrast([(0, 0, 0), (255, 255, 255)]).passes)

    def test_display_rounding_does_not_change_threshold_result(self):
        contrast = evaluate_palette_contrast([(255, 255, 255), (119, 119, 119)])
        self.assertEqual(round(contrast.ratios[0], 1), 4.5)
        self.assertLess(contrast.ratios[0], 4.5)
        self.assertFalse(contrast.passes)

    def test_empty_or_single_color_palette_does_not_pass(self):
        self.assertFalse(evaluate_palette_contrast([]).passes)
        self.assertFalse(evaluate_palette_contrast([(0, 0, 0)]).passes)


if __name__ == "__main__":
    unittest.main()
