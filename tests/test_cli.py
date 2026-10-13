from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from engine.pipeline import RenderConfiguration, WallpaperPipeline
from main import default_output_path


class TestPresetAndOutputResolution(unittest.TestCase):
    def test_preset_harmonize_override_is_used_when_cli_is_unspecified(self):
        palette = [(101, 46, 193), (82, 208, 191), (205, 101, 25)]
        config = RenderConfiguration(
            width=80, height=60, seed=42,
            preset_data={
                "palette_families": [["#652ec1", "#52d0bf", "#cd6519"]],
                "overrides": {"harmonize": "none"},
            },
        )
        result = WallpaperPipeline().render(config)
        self.assertEqual(result.colors, palette)

    def test_default_output_path_is_unique_per_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            first = default_output_path("afterglow", 42, directory=directory)
            second = default_output_path("afterglow", 43, directory=directory)

        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith("_42.png"))
        self.assertTrue(second.endswith("_43.png"))

    def test_parallel_batch_creates_output_directory_and_previews(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "batch"
            result = subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1] / "main.py"),
                "--preset", "afterglow", "--engine", "rbf", "--res", "80", "60",
                "--count", "3", "--workers", "2", "--seed", "50",
                "--preview-fx", "--out-dir", str(target),
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            paths = list(target.glob("*.png"))
            self.assertEqual(len(paths), 6)
            for seed in (50, 51, 52):
                self.assertEqual(len(list(target.glob(f"*_{seed}.png"))), 1)
            for path in paths:
                with Image.open(path) as image:
                    expected = (160, 60) if path.stem.endswith("_preview") else (80, 60)
                    self.assertEqual(image.size, expected)
                    image.verify()


if __name__ == "__main__":
    unittest.main()
