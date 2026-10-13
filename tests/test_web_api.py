import unittest
import io
from PIL import Image
from fastapi.testclient import TestClient

from web.app import app


class TestWebAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "wp-generator"})

    def test_list_devices(self):
        response = self.client.get("/api/devices")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        keys = [item["key"] for item in data]
        self.assertIn("mbp14", keys)
        self.assertIn("4k", keys)

    def test_list_presets(self):
        response = self.client.get("/api/presets")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        ids = [item["id"] for item in data]
        self.assertIn("nocturne", ids)
        self.assertIn("afterglow", ids)
        # Verify structure
        first = data[0]
        self.assertIn("name", first)
        self.assertIn("color_space", first)
        self.assertIn("sample_palette", first)

    def test_render_get_streams_png(self):
        response = self.client.get(
            "/api/render?preset=nocturne&width=160&height=100&seed=42"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["x-render-seed"], "42")
        self.assertIn("x-render-colors", response.headers)

        img = Image.open(io.BytesIO(response.content))
        self.assertEqual(img.size, (160, 100))
        self.assertEqual(img.format, "PNG")

    def test_render_post_with_custom_colors(self):
        payload = {
            "width": 120,
            "height": 80,
            "colors": ["#112233", "#445566", "#778899"],
            "seed": 99,
        }
        response = self.client.post("/api/render", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(
            response.headers["x-render-colors"], "#112233,#445566,#778899"
        )

        img = Image.open(io.BytesIO(response.content))
        self.assertEqual(img.size, (120, 80))

    def test_render_unknown_preset_returns_404(self):
        response = self.client.get("/api/render?preset=non_existent_preset_xyz")
        self.assertEqual(response.status_code, 404)

    def test_render_invalid_dimensions_returns_422(self):
        response = self.client.get("/api/render?width=5&height=100")
        self.assertEqual(response.status_code, 422)

    def test_render_with_chromatic_aberration(self):
        response = self.client.get(
            "/api/render?preset=nocturne&width=160&height=100&seed=42&aberration=0.02"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-render-aberration"], "0.02")

    def test_rbf_controls_match_between_get_and_post(self):
        parameters = {
            "preset": "nocturne", "engine": "rbf", "width": 160,
            "height": 100, "seed": 42, "warp_strength": 1.2,
            "ambient_weight": 0.5, "dispersion": 0.8,
        }
        get_response = self.client.get("/api/render", params=parameters)
        post_response = self.client.post("/api/render", json=parameters)
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(post_response.status_code, 200)
        self.assertEqual(post_response.content, get_response.content)

    def test_post_rejects_invalid_rbf_controls(self):
        for parameter, value in (("ambient_weight", -0.1), ("dispersion", 0.0)):
            with self.subTest(parameter=parameter):
                response = self.client.post("/api/render", json={
                    "engine": "rbf", "width": 160, "height": 100,
                    parameter: value,
                })
                self.assertEqual(response.status_code, 422)

    def test_post_reports_unrounded_palette_contrast(self):
        response = self.client.post("/api/render", json={
            "colors": ["#ffffff", "#777777"], "harmonize": "none",
            "width": 80, "height": 60, "seed": 42,
        })
        self.assertEqual(response.status_code, 200)
        ratio = float(response.headers["x-render-contrast-ratios"])
        self.assertLess(ratio, 4.5)
        self.assertEqual(response.headers["x-render-contrast-pass"], "false")
        self.assertEqual(response.headers["x-render-colors"], "#ffffff,#777777")

    def test_perceptual_harmonization_matches_between_get_and_post(self):
        parameters = {
            "preset": "nocturne", "harmonize": "perceptual",
            "width": 80, "height": 60, "seed": 42,
        }
        get_response = self.client.get("/api/render", params=parameters)
        post_response = self.client.post("/api/render", json=parameters)
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(post_response.status_code, 200)
        self.assertEqual(get_response.content, post_response.content)
        self.assertEqual(get_response.headers["x-render-contrast-ratios"],
                         post_response.headers["x-render-contrast-ratios"])

    def test_invalid_harmonizer_returns_422_for_both_methods(self):
        parameters = {"harmonize": "unknown", "width": 80, "height": 60}
        self.assertEqual(self.client.get("/api/render", params=parameters).status_code, 422)
        self.assertEqual(self.client.post("/api/render", json=parameters).status_code, 422)

    def test_render_with_contours_header(self):
        parameters = {
            "preset": "tectonic",
            "width": 100,
            "height": 80,
            "seed": 42,
            "contours": 0.8,
            "contour_levels": 20,
        }
        get_res = self.client.get("/api/render", params=parameters)
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.headers["x-render-contours"], "0.8")
        self.assertEqual(get_res.headers["x-render-engine"], "topographic")

        post_res = self.client.post("/api/render", json=parameters)
        self.assertEqual(post_res.status_code, 200)
        self.assertEqual(post_res.headers["x-render-contours"], "0.8")
        self.assertEqual(post_res.headers["x-render-engine"], "topographic")

    def test_render_fractal_preset_and_engine(self):
        parameters = {
            "preset": "fractal",
            "width": 100,
            "height": 80,
            "seed": 42,
            "phase_offset": 0.25,
            "frequency": 1.5,
            "trap_weight": 0.4,
            "fractal_type": "fluid",
        }
        get_res = self.client.get("/api/render", params=parameters)
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.headers["x-render-engine"], "fractal")

        post_res = self.client.post("/api/render", json=parameters)
        self.assertEqual(post_res.status_code, 200)
        self.assertEqual(post_res.headers["x-render-engine"], "fractal")

    def test_invalid_fractal_parameters_rejected(self):
        for param, val in [("phase_offset", 1.5), ("frequency", -1.0), ("trap_weight", 2.0)]:
            res = self.client.post("/api/render", json={
                "engine": "fractal", "width": 80, "height": 60, param: val,
            })
            self.assertEqual(res.status_code, 422)


    def test_render_fractal_center_and_angle_via_get_and_post(self):
        parameters = {
            "preset": "fractal",
            "width": 100,
            "height": 80,
            "seed": 42,
            "fractal_type": "fluid",
            "fractal_zoom": 0.6,
            "fractal_center_x": 1.5,
            "fractal_center_y": -2.0,
            "fractal_angle": 0.785,
        }
        get_res = self.client.get("/api/render", params=parameters)
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.headers["x-render-engine"], "fractal")

        post_payload = {
            "preset": "fractal",
            "width": 100,
            "height": 80,
            "seed": 42,
            "fractal_type": "fluid",
            "fractal_zoom": 0.6,
            "fractal_center": [1.5, -2.0],
            "fractal_angle": 0.785,
        }
        post_res = self.client.post("/api/render", json=post_payload)
        self.assertEqual(post_res.status_code, 200)
        self.assertEqual(post_res.headers["x-render-engine"], "fractal")


if __name__ == "__main__":
    unittest.main()
