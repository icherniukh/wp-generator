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


if __name__ == "__main__":
    unittest.main()
