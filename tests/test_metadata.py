import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
from fastapi.testclient import TestClient

from engine import __version__
from engine.metadata import (
    EXIF_TAG_DATE_TIME,
    EXIF_TAG_IMAGE_DESCRIPTION,
    EXIF_TAG_SOFTWARE,
    EXIF_TAG_USER_COMMENT,
    ImageMetadata,
    attach_metadata_to_image,
    build_image_metadata,
    encode_exif,
    encode_png_info,
    extract_image_metadata,
)
from engine.pipeline import RenderConfiguration, WallpaperPipeline
from web.app import app


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.pipeline = WallpaperPipeline()

    def test_build_image_metadata_captures_engine_and_parameters(self):
        cfg = RenderConfiguration(preset="nocturne", width=120, height=80, seed=777)
        result = self.pipeline.render(cfg)

        metadata = build_image_metadata(result, version="1.2.3")
        self.assertEqual(metadata.version, "1.2.3")
        self.assertEqual(metadata.engine, result.engine)
        self.assertEqual(metadata.seed, 777)
        self.assertEqual(metadata.parameters["width"], 120)
        self.assertEqual(metadata.parameters["height"], 80)
        self.assertEqual(metadata.parameters["colors"], result.color_hexes)
        self.assertIn("contrast", metadata.parameters)

    def test_encode_exif_and_png_info(self):
        meta = ImageMetadata(
            version="0.1.0",
            engine="simplex",
            seed=42,
            parameters={"width": 100, "height": 100, "noise": 0.05},
            created_at="2026:10:25 12:00:00",
        )

        exif = encode_exif(meta)
        self.assertEqual(exif[EXIF_TAG_SOFTWARE], "wp-generator 0.1.0")
        self.assertEqual(exif[EXIF_TAG_DATE_TIME], "2026:10:25 12:00:00")
        desc = json.loads(exif[EXIF_TAG_IMAGE_DESCRIPTION])
        self.assertEqual(desc["version"], "0.1.0")
        self.assertEqual(desc["engine"], "simplex")
        self.assertEqual(desc["parameters"]["noise"], 0.05)

        png_info = encode_png_info(meta)
        self.assertIsNotNone(png_info)

    def test_automatic_metadata_embedding_on_render_png_bytes(self):
        cfg = RenderConfiguration(preset="nocturne", width=100, height=75, seed=123)
        png_bytes, result = self.pipeline.render_png_bytes(cfg)

        extracted = extract_image_metadata(png_bytes)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.version, __version__)
        self.assertEqual(extracted.engine, result.engine)
        self.assertEqual(extracted.seed, 123)
        self.assertEqual(extracted.parameters["width"], 100)
        self.assertEqual(extracted.parameters["height"], 75)
        self.assertEqual(extracted.parameters["colors"], result.color_hexes)

    def test_automatic_metadata_embedding_on_save_file(self):
        cfg = RenderConfiguration(preset="afterglow", width=90, height=60, seed=888)
        result = self.pipeline.render(cfg)

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_out.png"
            result.save(file_path)

            extracted = extract_image_metadata(file_path)
            self.assertIsNotNone(extracted)
            self.assertEqual(extracted.engine, result.engine)
            self.assertEqual(extracted.seed, 888)
            self.assertEqual(extracted.parameters["colors"], result.color_hexes)

            # Test opening with PIL directly
            with Image.open(file_path) as pil_img:
                from_pil = extract_image_metadata(pil_img)
                self.assertIsNotNone(from_pil)
                self.assertEqual(from_pil.seed, 888)

    def test_extract_returns_none_for_plain_image(self):
        plain_img = Image.new("RGB", (30, 30), color=(100, 150, 200))
        buffer = io.BytesIO()
        plain_img.save(buffer, format="PNG")
        extracted = extract_image_metadata(buffer.getvalue())
        self.assertIsNone(extracted)

    def test_to_render_configuration_reconstructs_config(self):
        cfg = RenderConfiguration(preset="nocturne", width=80, height=60, seed=999)
        result = self.pipeline.render(cfg)
        meta = build_image_metadata(result)

        reconstructed_cfg = meta.to_render_configuration()
        self.assertEqual(reconstructed_cfg.seed, 999)
        self.assertEqual(reconstructed_cfg.width, 80)
        self.assertEqual(reconstructed_cfg.height, 60)
        self.assertEqual(reconstructed_cfg.engine, result.engine)

    def test_cli_metadata_inspection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_img = Path(tmp_dir) / "cli_sample.png"
            main_script = Path(__file__).resolve().parents[1] / "main.py"

            gen_proc = subprocess.run([
                sys.executable, str(main_script),
                "--preset", "nocturne", "--res", "64", "48",
                "--output", str(out_img), "--seed", "4321",
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(gen_proc.returncode, 0, gen_proc.stderr)

            # Inspect human-readable
            inspect_proc = subprocess.run([
                sys.executable, str(main_script),
                "--metadata", str(out_img),
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(inspect_proc.returncode, 0, inspect_proc.stderr)
            self.assertIn("Engine:", inspect_proc.stdout)
            self.assertIn("Seed:       4321", inspect_proc.stdout)

            # Inspect JSON
            json_proc = subprocess.run([
                sys.executable, str(main_script),
                "--metadata", str(out_img), "--json",
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(json_proc.returncode, 0, json_proc.stderr)
            parsed = json.loads(json_proc.stdout)
            self.assertEqual(parsed["seed"], 4321)
            self.assertEqual(parsed["version"], __version__)
            self.assertIn("colors", parsed["parameters"])

    def test_web_api_extract_endpoint(self):
        client = TestClient(app)
        cfg = RenderConfiguration(preset="nocturne", width=64, height=48, seed=555)
        png_bytes, _ = self.pipeline.render_png_bytes(cfg)

        response = client.post("/api/metadata/extract", content=png_bytes)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["seed"], 555)
        self.assertEqual(data["version"], __version__)
        self.assertIn("parameters", data)

        # Empty body test
        empty_response = client.post("/api/metadata/extract", content=b"")
        self.assertEqual(empty_response.status_code, 400)

        # Non-wallpaper image test
        plain_buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(plain_buf, format="PNG")
        plain_response = client.post("/api/metadata/extract", content=plain_buf.getvalue())
        self.assertEqual(plain_response.status_code, 404)
