"""Contour source (image or field) and placement (frosted or sharp)."""

import numpy as np
import pytest

from engine.metadata import extract_image_metadata
from engine.pipeline import RenderConfiguration, WallpaperPipeline
from engine.stages import StageParams
from engine.stages.finish import finish_stages

PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90), (60, 170, 200)]


def _config(**extra):
    base = dict(colors=PALETTE, harmonize="none", width=200, height=120, seed=4, complexity=5,
                contours=0.8, contour_levels=12, glass=True, chromatic_aberration=0.02, noise=0.0, vignette=0.0)
    base.update(extra)
    return RenderConfiguration(**base)


def _names(**params):
    base = dict(contours=0.8, glass=True, chromatic_aberration=0.02, noise=0.1, vignette=0.2)
    base.update(params)
    return [s.name for s in finish_stages(StageParams(base))]


def test_stage_order():
    assert _names() == ["contours", "glass", "chromatic aberration", "film grain", "vignette"]
    assert _names(contour_placement="sharp") == ["glass", "chromatic aberration", "contours", "film grain", "vignette"]
    assert _names(contour_source="field")[0] == "field contours"


def _render(**extra):
    return np.asarray(WallpaperPipeline().render(_config(**extra)).image.convert("RGB"), np.int16)


def test_sharp_contours_survive_the_glass():
    plain = _render(engine="topographic", contours=0.0)
    frosted = _render(engine="topographic")
    sharp = _render(engine="topographic", contour_placement="sharp")
    assert np.abs(sharp - plain).max() > np.abs(frosted - plain).max() + 40


def test_field_contours_differ_from_image_contours():
    image = _render(engine="simplex", field_stage="spiral", mapping="bands", glass=False)
    field = _render(engine="simplex", field_stage="spiral", mapping="bands", glass=False, contour_source="field")
    assert np.abs(image - field).max() > 40


def test_weight_field_engines_fall_back_to_image_and_say_so():
    result = WallpaperPipeline().render(_config(engine="rbf", contour_source="field"))
    assert result.contour_source == "image"
    assert np.array_equal(
        np.asarray(result.image), np.asarray(WallpaperPipeline().render(_config(engine="rbf")).image)
    )


@pytest.mark.parametrize("extra", [dict(engine="topographic", contour_source="field", contour_placement="sharp"),
                                   dict(engine="simplex", contour_placement="sharp")])
def test_contour_options_replay(extra):
    pipeline = WallpaperPipeline()
    png, original = pipeline.render_png_bytes(_config(**extra))
    replayed = pipeline.render(extract_image_metadata(png).to_render_configuration())
    assert np.array_equal(np.asarray(original.image), np.asarray(replayed.image))


def test_finish_pixel_sizes_scale_with_the_short_side():
    from engine.stages.finish import GLASS, REFERENCE_SHORT_SIDE, _contours

    base = dict(contours=0.5, contour_levels=24, contour_width=1.4, contour_mode="light")
    at_reference = StageParams(dict(base, width=3024, height=REFERENCE_SHORT_SIDE))
    assert GLASS.bind(at_reference)["blur_radius"] == 60
    assert _contours(at_reference)["line_width"] == 1.4
    half = StageParams(dict(base, width=1512, height=REFERENCE_SHORT_SIDE // 2))
    assert GLASS.bind(half)["blur_radius"] == 30
    assert _contours(half)["line_width"] == 0.7
    assert GLASS.bind(StageParams(dict(base, width=40, height=20)))["blur_radius"] == 2
