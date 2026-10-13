import unittest
import random
from engine.presets import PresetEngine

class TestPresetEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PresetEngine(preset_dir="presets")
        
        self.mock_preset = {
            "preset_id": "test_preset",
            "color_count": {"type": "uniform_int", "min": 3, "max": 3},
            "palette_structure": [
                {
                    "role": "grounding_anchor",
                    "hue": {"type": "uniform", "min": 0.0, "max": 1.0},
                    "lightness": {"type": "gaussian", "mu": 0.2, "sigma": 0.05, "clip": [0.0, 0.4]},
                    "chroma": {"type": "beta", "alpha": 2, "beta": 5, "scale": 0.3}
                },
                {
                    "role": "accent_core",
                    "hue": {"type": "relative", "target": "grounding_anchor", "mu_offset": 90, "sigma": 10},
                    "lightness": {"type": "gaussian", "mu": 0.7, "sigma": 0.1, "clip": [0.5, 0.9]},
                    "chroma": {"type": "beta", "alpha": 5, "beta": 2, "scale": 0.5, "offset": 0.5}
                }
            ],
            "overrides": {
                "noise": {"type": "gaussian", "mu": 0.05, "sigma": 0.01, "clip": [0.0, 0.1]},
                "fluid_mode": "spatial"
            }
        }

    def test_sample_distribution_uniform(self):
        conf = {"type": "uniform", "min": 10, "max": 20}
        samples = [self.engine._sample_distribution(conf) for _ in range(50)]
        for val in samples:
            self.assertGreaterEqual(val, 10)
            self.assertLessEqual(val, 20)

    def test_sample_distribution_clip(self):
        conf = {"type": "gaussian", "mu": 0.5, "sigma": 1.0, "clip": [0.3, 0.7]}
        samples = [self.engine._sample_distribution(conf) for _ in range(100)]
        for val in samples:
            self.assertGreaterEqual(val, 0.3)
            self.assertLessEqual(val, 0.7)

    def test_generate_palette_count(self):
        colors = self.engine.generate_palette(self.mock_preset)
        self.assertEqual(len(colors), 3)
        # Check type
        for c in colors:
            self.assertIsInstance(c, tuple)
            self.assertEqual(len(c), 3)
            for channel in c:
                self.assertIsInstance(channel, int)
                self.assertTrue(0 <= channel <= 255)
                
    def test_get_engine_overrides(self):
        overrides = self.engine.get_engine_overrides(self.mock_preset)
        self.assertEqual(overrides["fluid_mode"], "spatial")
        self.assertIn("noise", overrides)

    def test_live_preset_loading(self):
        # Ensure cupertino.json is valid
        data = self.engine.load_preset("cupertino")
        self.assertEqual(data["preset_id"], "cupertino")
        colors = self.engine.generate_palette(data)
        self.assertGreaterEqual(len(colors), 2)
        self.assertLessEqual(len(colors), 5)

    def test_afterglow_preset_has_reference_derived_defaults(self):
        data = self.engine.load_preset("afterglow")

        self.assertEqual(data["color_space"], "oklch")
        self.assertGreaterEqual(len(data["palette_families"]), 4)
        self.assertEqual(data["overrides"]["engine"], "idw")
        self.assertEqual(data["overrides"]["harmonize"], "none")
        self.assertGreaterEqual(data["overrides"]["noise"], 0.015)
        self.assertGreater(data["overrides"]["vignette"], 0.0)
        self.assertEqual(len(data["palette_families"]), 10)
        self.assertTrue(all(len(family) == 3 for family in data["palette_families"]))

    def test_curated_palette_family_is_preserved(self):
        preset = {
            "palette_families": [
                ["#652ec1", "#52d0bf", "#cd6519"],
            ]
        }

        colors = self.engine.generate_palette(preset)

        self.assertEqual(colors, [(101, 46, 193), (82, 208, 191), (205, 101, 25)])

    def test_curated_palette_families_can_be_cycled(self):
        preset = {"palette_families": [["#000000"], ["#ffffff"]]}

        self.assertEqual(self.engine.generate_palette(preset, family_index=0), [(0, 0, 0)])
        self.assertEqual(self.engine.generate_palette(preset, family_index=1), [(255, 255, 255)])
        self.assertEqual(self.engine.generate_palette(preset, family_index=2), [(0, 0, 0)])

    def test_nocturne_preset_has_tuned_defaults(self):
        data = self.engine.load_preset("nocturne")
        self.assertEqual(data["color_space"], "oklch")
        self.assertEqual(data["overrides"]["engine"], "fluid-wave")
        self.assertEqual(data["overrides"]["dots"], 0.0)
        self.assertEqual(data["overrides"]["glass"], False)
        colors = self.engine.generate_palette(data, count=3)
        self.assertEqual(len(colors), 3)

    def test_fractal_preset_has_curated_harmonic_families(self):
        from engine.colors import evaluate_palette_contrast
        data = self.engine.load_preset("fractal")
        self.assertEqual(data["color_space"], "oklch")
        self.assertIn("palette_families", data)
        self.assertEqual(len(data["palette_families"]), 12)
        for family in data["palette_families"]:
            self.assertEqual(len(family), 4)
            rgb = [tuple(int(c.lstrip("#")[j:j+2], 16) for j in (0, 2, 4)) for c in family]
            contrast = evaluate_palette_contrast(rgb)
            self.assertTrue(contrast.passes, f"Family {family} failed contrast with {contrast.ratios}")

    def test_procedural_harmonic_progression_with_relative_accents(self):
        colors = self.engine.generate_palette(self.mock_preset, count=4)
        self.assertEqual(len(colors), 4)
        self.assertTrue(all(isinstance(c, tuple) and len(c) == 3 for c in colors))


if __name__ == '__main__':
    unittest.main()
