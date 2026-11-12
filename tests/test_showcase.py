"""The visual guide must show real, isolated, reproducible primitive output."""

import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from engine.filters import WallpaperFilters
from engine.metadata import extract_image_metadata
from tools.render_showcase import FILTERS, ROOT, build_assets, reset_seed


class GuideReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.links = []
        self.ids = set()
        self.effects = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            assert attrs.get("alt"), "Every image needs an accessible label"
            self.images.append(attrs["src"])
        if tag == "a":
            self.links.append(attrs["href"])
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if "data-effect" in attrs:
            self.effects.add(attrs["data-effect"])


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    directory = tmp_path_factory.mktemp("showcase")
    manifest = build_assets(directory, width=160, height=90)
    return directory, manifest


def test_guide_images_and_navigation_are_reproducible(assets):
    directory, manifest = assets
    guide = GuideReferences()
    guide.feed((ROOT / "index.html").read_text())
    assert guide.effects == set(FILTERS)
    for src in guide.images:
        name = src.removeprefix("docs/assets/")
        assert name in manifest, f"No reproducible recipe for {src}"
        with Image.open(directory / name) as image:
            image.verify()
    for link in guide.links:
        if link.startswith("#"):
            assert link[1:] in guide.ids
        elif not link.startswith("https://"):
            assert (ROOT / link).exists(), link


def test_finish_examples_apply_only_the_named_filter(assets):
    directory, manifest = assets
    with Image.open(directory / "demo/fx_base.png") as image:
        base = image.copy()
    original = np.asarray(base).copy()
    for name, (method, kwargs) in FILTERS.items():
        reset_seed()
        expected = getattr(WallpaperFilters, method)(base.copy(), **kwargs).convert("RGB")
        with Image.open(directory / f"demo/fx_{name}.png") as actual:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            assert np.any(np.asarray(actual) != original), f"{name} must visibly change pixels"
        assert manifest[f"demo/fx_{name}.png"]["input"] == "demo/fx_base.png"
    np.testing.assert_array_equal(np.asarray(base), original)


def test_asset_generation_is_deterministic(assets, tmp_path):
    directory, manifest = assets
    again = build_assets(tmp_path, width=160, height=90)
    assert json.loads(json.dumps(again)) == json.loads(json.dumps(manifest))
    # Pixels are reproducible; embedded render creation timestamps may differ.
    for name in manifest:
        with Image.open(directory / name) as before, Image.open(tmp_path / name) as after:
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=name)


def test_worked_example_keeps_embedded_render_metadata(assets):
    directory, manifest = assets
    metadata = extract_image_metadata(directory / "demo/run_final.png")
    assert metadata is not None
    assert metadata.seed == 7
    assert metadata.engine == "idw"
    assert metadata.parameters["preset"] == "afterglow"
    assert metadata.parameters == manifest["demo/run_final.png"]["parameters"]
