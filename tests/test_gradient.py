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


class TestRadialBasisGradient(unittest.TestCase):
    def test_rbf_generates_correct_dimensions_and_mode(self):
        from engine.gradient import RadialBasisGradient
        engine = RadialBasisGradient(width=160, height=100)
        colors = [(15, 20, 35), (230, 80, 140), (50, 180, 220), (240, 190, 60)]
        img = engine.generate(
            colors,
            complexity=4,
            warp_strength=1.0,
            ambient_weight=0.15,
            dispersion=0.35,
            seed=42,
        )

        self.assertEqual(img.size, (160, 100))
        self.assertEqual(img.mode, "RGB")

    def test_rbf_is_deterministic_with_seed(self):
        from engine.gradient import RadialBasisGradient
        engine = RadialBasisGradient(width=120, height=80)
        colors = [(10, 15, 25), (220, 90, 130), (40, 160, 210)]

        img1 = engine.generate(colors, complexity=5, warp_strength=1.2, ambient_weight=0.2, dispersion=0.4, seed=555)
        img2 = engine.generate(colors, complexity=5, warp_strength=1.2, ambient_weight=0.2, dispersion=0.4, seed=555)

        np.testing.assert_array_equal(np.asarray(img1), np.asarray(img2))

    def test_rbf_handles_single_color(self):
        from engine.gradient import RadialBasisGradient
        engine = RadialBasisGradient(width=40, height=30)
        img = engine.generate([(255, 0, 128)])
        self.assertEqual(img.size, (40, 30))
        arr = np.asarray(img)
        self.assertTrue(np.all(arr[:, :, 0] == 255))
        self.assertTrue(np.all(arr[:, :, 1] == 0))
        self.assertTrue(np.all(arr[:, :, 2] == 128))


class TestTopographicGradient(unittest.TestCase):
    def test_topographic_generates_correct_dimensions_and_mode(self):
        from engine.gradient import TopographicGradient
        engine = TopographicGradient(width=160, height=100)
        colors = [(15, 20, 35), (230, 80, 140), (50, 180, 220)]
        img = engine.generate(colors, complexity=4, warp_strength=0.9, seed=42)

        self.assertEqual(img.size, (160, 100))
        self.assertEqual(img.mode, "RGB")

    def test_topographic_is_deterministic_with_seed(self):
        from engine.gradient import TopographicGradient
        engine = TopographicGradient(width=120, height=80)
        colors = [(10, 15, 25), (220, 90, 130), (40, 160, 210)]

        img1 = engine.generate(colors, complexity=5, warp_strength=1.1, seed=333)
        img2 = engine.generate(colors, complexity=5, warp_strength=1.1, seed=333)

        np.testing.assert_array_equal(np.asarray(img1), np.asarray(img2))

    def test_topographic_handles_single_color(self):
        from engine.gradient import TopographicGradient
        engine = TopographicGradient(width=40, height=30)
        img = engine.generate([(100, 150, 200)])
        self.assertEqual(img.size, (40, 30))
        arr = np.asarray(img)
        self.assertTrue(np.all(arr[:, :, 0] == 100))
        self.assertTrue(np.all(arr[:, :, 1] == 150))
        self.assertTrue(np.all(arr[:, :, 2] == 200))


class TestFractalFieldGradient(unittest.TestCase):
    def test_fractal_generates_correct_dimensions_and_mode(self):
        from engine.gradient import FractalFieldGradient
        engine = FractalFieldGradient(width=160, height=100)
        colors = [(15, 20, 35), (230, 80, 140), (50, 180, 220)]
        img = engine.generate(colors, complexity=3, fractal_type="julia", seed=42)

        self.assertEqual(img.size, (160, 100))
        self.assertEqual(img.mode, "RGB")

    def test_fractal_is_deterministic_with_seed(self):
        from engine.gradient import FractalFieldGradient
        engine = FractalFieldGradient(width=120, height=80)
        colors = [(10, 15, 25), (220, 90, 130), (40, 160, 210)]

        img1 = engine.generate(colors, complexity=4, fractal_type="julia", seed=101)
        img2 = engine.generate(colors, complexity=4, fractal_type="julia", seed=101)

        np.testing.assert_array_equal(np.asarray(img1), np.asarray(img2))

    def test_fractal_handles_single_color(self):
        from engine.gradient import FractalFieldGradient
        engine = FractalFieldGradient(width=40, height=30)
        img = engine.generate([(70, 120, 180)])
        self.assertEqual(img.size, (40, 30))
        arr = np.asarray(img)
        self.assertTrue(np.all(arr[:, :, 0] == 70))
        self.assertTrue(np.all(arr[:, :, 1] == 120))
        self.assertTrue(np.all(arr[:, :, 2] == 180))

    def test_fractal_types_julia_and_mandelbrot(self):
        from engine.gradient import FractalFieldGradient
        engine = FractalFieldGradient(width=80, height=60)
        colors = [(10, 15, 25), (220, 90, 130), (40, 160, 210)]

        img_julia = engine.generate(colors, fractal_type="julia", seed=42)
        img_mandel = engine.generate(colors, fractal_type="mandelbrot", seed=42)

        self.assertEqual(img_julia.size, (80, 60))
        self.assertEqual(img_mandel.size, (80, 60))
        self.assertFalse(np.array_equal(np.asarray(img_julia), np.asarray(img_mandel)))

    def test_fractal_phase_and_channel_offsets_alter_output(self):
        from engine.gradient import FractalFieldGradient
        engine = FractalFieldGradient(width=80, height=60)
        colors = [(10, 15, 25), (220, 90, 130), (40, 160, 210)]

        img_base = engine.generate(colors, phase_offset=0.0, channel_offset=0.0, seed=42)
        img_phase = engine.generate(colors, phase_offset=0.5, channel_offset=0.0, seed=42)
        img_chan = engine.generate(colors, phase_offset=0.0, channel_offset=0.2, seed=42)

        self.assertFalse(np.array_equal(np.asarray(img_base), np.asarray(img_phase)))
        self.assertFalse(np.array_equal(np.asarray(img_base), np.asarray(img_chan)))

    def test_fractal_fluid_mode_deterministic_and_unique(self):
        from engine.gradient import FractalFieldGradient
        engine = FractalFieldGradient(width=80, height=60)
        colors = [(10, 15, 25), (220, 90, 130), (40, 160, 210)]

        img1 = engine.generate(colors, fractal_type="fluid", seed=99)
        img2 = engine.generate(colors, fractal_type="fluid", seed=99)
        img_julia = engine.generate(colors, fractal_type="julia", seed=99)

        np.testing.assert_array_equal(np.asarray(img1), np.asarray(img2))
        self.assertFalse(np.array_equal(np.asarray(img1), np.asarray(img_julia)))

    def test_fractal_focal_center_and_angle_alter_output(self):
        from engine.gradient import FractalFieldGradient
        engine = FractalFieldGradient(width=80, height=60)
        colors = [(10, 15, 25), (220, 90, 130), (40, 160, 210)]

        img_center_a = engine.generate(colors, fractal_type="fluid", center=(0.0, 0.0), angle=0.0, seed=42)
        img_center_b = engine.generate(colors, fractal_type="fluid", center=(2.5, -1.8), angle=0.0, seed=42)
        img_angle_c = engine.generate(colors, fractal_type="fluid", center=(0.0, 0.0), angle=1.57, seed=42)

        self.assertFalse(np.array_equal(np.asarray(img_center_a), np.asarray(img_center_b)))
        self.assertFalse(np.array_equal(np.asarray(img_center_a), np.asarray(img_angle_c)))

    def test_fractal_macro_zoom_and_offscreen_focal_point(self):
        from engine.gradient import FractalFieldGradient
        engine = FractalFieldGradient(width=100, height=60)
        colors = [(15, 20, 35), (230, 110, 80), (60, 180, 210)]

        # Off-screen center and macro radius zoom
        img = engine.generate(
            colors,
            fractal_type="fluid",
            zoom=0.45,
            center=(3.0, -2.5),
            angle=0.785,
            seed=123,
        )
        self.assertEqual(img.size, (100, 60))
        arr = np.asarray(img)
        self.assertEqual(arr.shape, (60, 100, 3))
        self.assertFalse(np.isnan(arr).any())
        self.assertGreater(arr.std(), 5.0)

    def test_fractal_random_archetypes_generate_valid_variety(self):
        from engine.gradient import FractalFieldGradient
        engine = FractalFieldGradient(width=64, height=48)
        colors = [(20, 30, 45), (210, 80, 120), (50, 190, 170)]

        images = []
        for s in range(5):
            img = engine.generate(colors, fractal_type="fluid", seed=1000 + s)
            images.append(np.asarray(img))
            self.assertEqual(img.size, (64, 48))

        # Ensure all distinct seeds produce distinct outputs
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                self.assertFalse(np.array_equal(images[i], images[j]))


if __name__ == "__main__":
    unittest.main()


