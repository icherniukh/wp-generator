import unittest
import numpy as np
from PIL import Image

from engine.gradients import (
    DomainWarpGradient,
    FlowFieldGradient,
    FluidMeshGradient,
    FractalFieldGradient,
    MeshGradient,
    RadialBasisGradient,
    SimplexMeshGradient,
    SplineWaveGradient,
    TopographicGradient,
    colors_to_oklab,
    evaluate_decoupled_palette,
    interpolate_color_curve,
    render_gradient,
)
from engine.gradients.registry import (
    GRADIENT_REGISTRY,
    get_gradient_engine_class,
    register_gradient_engine,
)


class TestModularGradients(unittest.TestCase):
    def test_registry_contains_all_core_engines(self):
        expected_keys = [
            "simplex",
            "flow-field",
            "fluid-wave",
            "spline-wave",
            "domain-warp",
            "rbf",
            "radial-basis",
            "topographic",
            "fractal",
            "idw",
        ]
        for key in expected_keys:
            self.assertIn(key, GRADIENT_REGISTRY)

    def test_get_gradient_engine_class_fallback(self):
        self.assertEqual(get_gradient_engine_class("simplex"), SimplexMeshGradient)
        self.assertEqual(get_gradient_engine_class(None), SimplexMeshGradient)
        self.assertEqual(get_gradient_engine_class("nonexistent"), SimplexMeshGradient)

    def test_register_gradient_engine_custom(self):
        class DummyEngine:
            pass

        register_gradient_engine("dummy-test", DummyEngine)
        self.assertEqual(get_gradient_engine_class("dummy-test"), DummyEngine)

    def test_render_gradient_dispatches_cleanly(self):
        colors = [(20, 30, 45), (200, 100, 80), (60, 160, 210)]
        for engine_type in ["simplex", "fluid-wave", "domain-warp", "rbf", "topographic", "fractal", "idw"]:
            img = render_gradient(
                engine_type,
                width=64,
                height=48,
                colors=colors,
                complexity=3,
                seed=42,
            )
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (64, 48))

    def test_interpolate_color_curve_output_shape(self):
        colors = [(10, 20, 30), (220, 90, 120), (50, 180, 220)]
        colors_oklab = colors_to_oklab(colors)
        t_values = np.linspace(0.0, 1.0, 100).reshape((10, 10))

        curve = interpolate_color_curve(t_values, colors_oklab)
        self.assertEqual(curve.shape, (10, 10, 3))
        self.assertFalse(np.isnan(curve).any())

    def test_evaluate_decoupled_palette(self):
        colors = [(10, 20, 30), (220, 90, 120), (50, 180, 220)]
        colors_oklab = colors_to_oklab(colors)
        u_l = np.zeros((8, 8), dtype=np.float32)
        u_c = np.ones((8, 8), dtype=np.float32)

        res = evaluate_decoupled_palette(colors_oklab, u_l, u_c, 8, 8)
        self.assertEqual(res.shape, (8, 8, 3))
        self.assertFalse(np.isnan(res).any())


if __name__ == "__main__":
    unittest.main()
