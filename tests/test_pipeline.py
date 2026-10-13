import unittest
import io
import numpy as np
from PIL import Image

from engine.colors import contrast_ratio, rgb_to_oklch

from engine.pipeline import (
    DEVICE_RESOLUTIONS,
    PipelineConfigurationError,
    PresetNotFoundError,
    RenderConfiguration,
    WallpaperPipeline,
    parse_hex_color,
)


class TestWallpaperPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = WallpaperPipeline()

    def test_parse_hex_color(self):
        self.assertEqual(parse_hex_color("#ff5500"), (255, 85, 0))
        self.assertEqual(parse_hex_color("00387e"), (0, 56, 126))
        with self.assertRaises(PipelineConfigurationError):
            parse_hex_color("invalid")

    def test_list_presets_contains_registered_presets(self):
        presets = self.pipeline.list_presets()
        ids = [p["id"] for p in presets]
        self.assertIn("nocturne", ids)
        self.assertIn("afterglow", ids)
        self.assertIn("sleek", ids)

    def test_render_produces_matching_dimensions(self):
        cfg = RenderConfiguration(preset="nocturne", width=200, height=150, seed=123)
        result = self.pipeline.render(cfg)
        self.assertEqual(result.image.size, (200, 150))
        self.assertEqual(result.width, 200)
        self.assertEqual(result.height, 150)
        self.assertEqual(result.seed, 123)
        self.assertGreaterEqual(len(result.colors), 2)

    def test_render_png_bytes_outputs_decodable_png(self):
        cfg = RenderConfiguration(preset="nocturne", width=160, height=100, seed=456)
        png_bytes, result = self.pipeline.render_png_bytes(cfg)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

        decoded = Image.open(io.BytesIO(png_bytes))
        self.assertEqual(decoded.size, (160, 100))
        self.assertEqual(decoded.format, "PNG")

    def test_deterministic_output_with_same_seed(self):
        cfg1 = RenderConfiguration(preset="nocturne", width=100, height=100, seed=777)
        cfg2 = RenderConfiguration(preset="nocturne", width=100, height=100, seed=777)

        res1 = self.pipeline.render(cfg1)
        res2 = self.pipeline.render(cfg2)

        arr1 = np.array(res1.image)
        arr2 = np.array(res2.image)
        np.testing.assert_array_equal(arr1, arr2)
        self.assertEqual(res1.color_hexes, res2.color_hexes)

    def test_custom_colors_override_preset(self):
        custom = [(255, 0, 0), (0, 0, 255)]
        cfg = RenderConfiguration(
            colors=custom, width=120, height=80, seed=10, engine="simplex"
        )
        res = self.pipeline.render(cfg)
        self.assertEqual(res.colors, custom)
        self.assertEqual(res.color_hexes, ["#ff0000", "#0000ff"])

    def test_invalid_dimensions_raise_error(self):
        with self.assertRaises(PipelineConfigurationError):
            self.pipeline.render(RenderConfiguration(width=0, height=100))

    def test_unknown_preset_raises_error(self):
        with self.assertRaises(PresetNotFoundError):
            self.pipeline.render(RenderConfiguration(preset="non_existent_preset_xyz"))

    def test_capture_base_image_stores_unfiltered_canvas(self):
        cfg = RenderConfiguration(
            preset="nocturne",
            width=100,
            height=80,
            seed=42,
            noise=0.1,
            dots=0.5,
            capture_base_image=True,
        )
        res = self.pipeline.render(cfg)
        self.assertIsNotNone(res.base_image)
        self.assertEqual(res.base_image.size, (100, 80))
        # Ensure filtered and base image differ due to applied noise and dots
        self.assertFalse(np.array_equal(np.array(res.image), np.array(res.base_image)))

    def test_chromatic_aberration_configured_in_pipeline(self):
        cfg = RenderConfiguration(
            preset="nocturne",
            width=120,
            height=80,
            seed=99,
            chromatic_aberration=0.03,
        )
        res = self.pipeline.render(cfg)
        self.assertEqual(res.chromatic_aberration, 0.03)
        self.assertEqual(res.image.size, (120, 80))

    def test_domain_warp_engine_in_pipeline(self):
        cfg = RenderConfiguration(
            engine="domain-warp",
            width=140,
            height=90,
            warp_strength=1.2,
            colors=[(10, 10, 20), (220, 100, 150)],
            seed=42,
        )
        res = self.pipeline.render(cfg)
        self.assertEqual(res.engine, "domain-warp")
        self.assertEqual(res.warp_strength, 1.2)
        self.assertEqual(res.image.size, (140, 90))

    def test_rbf_engine_in_pipeline(self):
        cfg = RenderConfiguration(
            engine="rbf",
            width=140,
            height=90,
            warp_strength=1.1,
            ambient_weight=0.2,
            dispersion=0.4,
            colors=[(10, 10, 20), (220, 100, 150), (40, 180, 220)],
            seed=42,
        )
        res = self.pipeline.render(cfg)
        self.assertEqual(res.engine, "rbf")
        self.assertEqual(res.warp_strength, 1.1)
        self.assertEqual(res.ambient_weight, 0.2)
        self.assertEqual(res.dispersion, 0.4)
        self.assertEqual(res.image.size, (140, 90))

    def test_perceptual_harmonization_preserves_input_palette_order_and_hues(self):
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
        result = self.pipeline.render(RenderConfiguration(
            colors=colors, harmonize="perceptual", width=80, height=60, seed=42,
        ))
        for original, adjusted in zip(colors, result.colors):
            before, after = rgb_to_oklch(original), rgb_to_oklch(adjusted)
            self.assertLess(abs((after[2] - before[2] + 180) % 360 - 180), 2.0)

    def test_contrast_is_evaluated_after_harmonization(self):
        result = self.pipeline.render(RenderConfiguration(
            colors=[(0, 0, 0), (255, 255, 255)], harmonize="nothing",
            width=80, height=60, seed=42,
        ))
        self.assertAlmostEqual(result.palette_contrast.ratios[0],
                               contrast_ratio(*result.colors))
        self.assertNotEqual(result.palette_contrast.ratios[0], 21.0)

    def test_failing_contrast_does_not_recolor_explicit_palette(self):
        colors = [(255, 255, 255), (119, 119, 119)]
        result = self.pipeline.render(RenderConfiguration(
            colors=colors, harmonize="none", width=80, height=60, seed=42,
        ))
        self.assertEqual(result.colors, colors)
        self.assertFalse(result.palette_contrast.passes)

    def test_invalid_harmonizer_is_rejected(self):
        with self.assertRaises(PipelineConfigurationError):
            self.pipeline.render(RenderConfiguration(harmonize="unknown", width=80, height=60))

    def test_preset_can_define_perceptual_aesthetic_bounds(self):
        result = self.pipeline.render(RenderConfiguration(
            colors=[(255, 0, 0), (0, 255, 0)], width=80, height=60, seed=42,
            preset_data={
                "overrides": {"harmonize": "perceptual"},
                "harmonization": {
                    "lightness_bounds": [0.4, 0.5], "chroma_bounds": [0.0, 0.08],
                },
            },
        ))
        for color in result.colors:
            lightness, chroma, _ = rgb_to_oklch(color)
            self.assertTrue(0.399 <= lightness <= 0.501)
            self.assertLessEqual(chroma, 0.081)

    def test_invalid_preset_bounds_are_configuration_errors(self):
        with self.assertRaises(PipelineConfigurationError):
            self.pipeline.render(RenderConfiguration(
                width=80, height=60,
                preset_data={
                    "overrides": {"harmonize": "perceptual"},
                    "harmonization": {"lightness_bounds": [0.8, 0.2]},
                },
            ))


if __name__ == "__main__":
    unittest.main()
