import argparse
import os
import tempfile
import unittest

from main import default_output_path, resolve_preset_parameter


class TestPresetAndOutputResolution(unittest.TestCase):
    def test_preset_harmonize_override_is_used_when_cli_is_unspecified(self):
        value = resolve_preset_parameter(
            None,
            "harmonize",
            {"harmonize": "none"},
            lambda: "theory",
        )

        self.assertEqual(value, "none")

    def test_default_output_path_is_unique_per_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            first = default_output_path("afterglow", 42, directory=directory)
            second = default_output_path("afterglow", 43, directory=directory)

        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith("_42.png"))
        self.assertTrue(second.endswith("_43.png"))


if __name__ == "__main__":
    unittest.main()
