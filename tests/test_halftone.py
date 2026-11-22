"""The halftone finish stage: dots, cmyk, lines, dither and pixel screens."""

import random

import numpy as np
import pytest
import scipy.ndimage
from PIL import Image

from engine.metadata import extract_image_metadata
from engine.pipeline import PipelineConfigurationError, RenderConfiguration, WallpaperPipeline
from engine.stages.catalog import PlanCompositionError
from engine.stages.finish import HALFTONE, check_finish_stage_params, finish_stages, halftone
from engine.stages.halftone import MODES, dot_coverage, find_paper
from engine.stages.runner import StageParams

PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90), (60, 170, 200)]
WHITE = (255, 255, 255)
ARGS = dict(cell=8.0, angle=45.0, mix=1.0, levels=4)


def gradient(size=(160, 100)):
    w, h = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 30 + 200 * x / w
    g = 40 + 160 * y / h
    b = 220 - 150 * x / w
    return Image.fromarray(np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8))


def grey(level, size=(240, 160)):
    return np.full((size[1], size[0], 3), level * 255.0, dtype=np.float32)


def finish_params(**overrides):
    values = dict(
        contours=0.0, contour_levels=8, contour_width=2.0, contour_mode="lines", contour_source="image",
        contour_placement="frosted", glass=False, chromatic_aberration=0.0, noise=0.0, vignette=0.0,
        shading=0.0, width=240, height=160,
    )
    values.update(overrides)
    return StageParams(values)


def render(**kwargs):
    config = RenderConfiguration(
        width=160, height=100, seed=5, colors=PALETTE, harmonize="none",
        vignette=0.0, contours=0.0, glass=False, chromatic_aberration=0.0, noise=0.0, **kwargs,
    )
    return WallpaperPipeline().render(config)


def arrays(image):
    return np.asarray(image, dtype=np.int16)


# --- dots ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0.25, 0.5, 0.75])
def test_dot_coverage_tracks_the_luminance(level):
    coverage = dot_coverage(grey(level), cell=10.0, angle=45.0, paper=WHITE)
    assert coverage.mean() == pytest.approx(1.0 - level, abs=0.05)


@pytest.mark.parametrize("level", [0.25, 0.5, 0.75])
def test_dots_keep_the_image_brightness(level):
    # Paper and ink in the covered shares average back to the grey the cell had.
    out = np.asarray(halftone(Image.fromarray(grey(level).astype(np.uint8)), mode="dots", paper=WHITE, **ARGS))
    assert out.mean() / 255.0 == pytest.approx(level, abs=0.05)


def test_dots_are_round_dots_anti_aliased_over_one_pixel():
    coverage = dot_coverage(grey(0.6), cell=12.0, angle=0.0, paper=WHITE)
    # Pixels are mostly solid ink or bare paper, with a thin intermediate rim.
    rim = ((coverage > 0.02) & (coverage < 0.98)).mean()
    assert 0.02 < rim < 0.35
    # One dot at the cell centre: ink at its centre, bare paper in the corner between four dots.
    # The grid is centred on the image: the cell centres are half a cell off the middle.
    h, w = coverage.shape
    cy, cx = h // 2 + 6, w // 2 + 6
    assert coverage[cy - 1:cy + 1, cx - 1:cx + 1].min() == 1.0
    assert coverage[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1].max() < 0.5
    # The profile across a dot edge takes about one pixel.
    row = coverage[cy, cx:cx + 12]
    assert 1 <= ((row > 0.02) & (row < 0.98)).sum() <= 2


def test_dots_have_no_black_ink_or_grey_wash_on_a_coloured_image():
    image = gradient((240, 160))
    out = np.asarray(halftone(image, mode="dots", **ARGS), dtype=np.float32)
    paper = find_paper(np.asarray(image, dtype=np.float32))
    # The page is light and stays the paper's own hue; the ink is not pure black or grey.
    assert paper.max() > 230 and float(paper @ [0.2126, 0.7152, 0.0722]) >= 0.89 * 255
    assert (np.abs(out - paper).max(axis=-1) < 1).mean() > 0.05
    darkest = out[out.sum(axis=-1) < np.percentile(out.sum(axis=-1), 10)]
    assert (darkest.max(axis=-1) - darkest.min(axis=-1)).mean() > 10


def test_dots_take_the_cell_hue_for_their_ink():
    red = np.zeros((120, 120, 3), dtype=np.float32)
    red[...] = (200, 60, 60)
    out = np.asarray(halftone(Image.fromarray(red.astype(np.uint8)), mode="dots", **ARGS), dtype=np.float32)
    ink = out[out.sum(axis=-1) < np.percentile(out.sum(axis=-1), 30)]
    assert (ink[:, 0] > ink[:, 1] + 20).all()


def test_dot_edges_have_no_resampling_moire():
    # A uniform tone must give the same ink in every region of a rotated screen: a
    # moire would beat the grid against the pixels and make some areas darker.
    coverage = dot_coverage(grey(0.5, (480, 320)), cell=7.3, angle=45.0, paper=WHITE)
    blocks = coverage[:300, :460].reshape(10, 30, 23, 20).mean(axis=(1, 3))
    assert blocks.std() < 0.03


# --- scale ------------------------------------------------------------------------------------------


def test_the_cell_scales_with_the_image_size():
    small, large = finish_params(halftone="dots", width=900, height=600), finish_params(halftone="dots", width=2700, height=1800)
    assert HALFTONE.bind(large)["cell"] == pytest.approx(3 * HALFTONE.bind(small)["cell"])


def test_an_override_of_the_cell_scales_like_the_default():
    overrides = {"halftone": {"cell": 60.0}}
    small = finish_params(halftone="dots", stage_params=overrides, width=300, height=200)
    large = finish_params(halftone="dots", stage_params=overrides, width=900, height=600)
    (stage_small,), (stage_large,) = finish_stages(small), finish_stages(large)
    assert stage_large.bind(large)["cell"] == pytest.approx(3 * stage_small.bind(small)["cell"])


def test_a_screen_has_the_same_number_of_dots_at_every_size():
    def dot_count(width, height):
        params = finish_params(halftone="dots", width=width, height=height, stage_params={"halftone": {"cell": 60.0}})
        (stage,) = finish_stages(params)
        coverage = dot_coverage(
            grey(0.8, (width, height)), cell=stage.bind(params)["cell"], angle=45.0, paper=WHITE
        )
        return scipy.ndimage.label(coverage > 0.5)[1]

    # The cell is 60 px per 1964, so the count over the picture is the same at any size.
    small, large = dot_count(480, 320), dot_count(960, 640)
    assert large == pytest.approx(small, rel=0.2)


def test_a_tiny_preview_keeps_a_screen_that_can_be_drawn():
    assert HALFTONE.bind(finish_params(halftone="dots", width=96, height=60))["cell"] == 2.0


# --- dither and pixel -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["dither", "pixel"])
@pytest.mark.parametrize("levels", [2, 3, 4, 6])
def test_dither_and_pixel_give_at_most_levels_values_per_channel(mode, levels):
    out = np.asarray(halftone(gradient(), mode=mode, cell=3.0, angle=45.0, mix=1.0, levels=levels))
    for channel in range(3):
        assert len(np.unique(out[..., channel])) <= levels
    assert out.min() == 0 or out.max() == 255 or levels > 2


def test_pixel_cells_are_flat_blocks_of_the_cell_size():
    out = np.asarray(halftone(gradient(), mode="pixel", cell=5.0, angle=45.0, mix=1.0, levels=8))
    assert (out[0:5, 0:5] == out[0, 0]).all()
    assert (out[5:10, 5:10] == out[5, 5]).all()


def test_dither_averages_to_the_original_tone():
    out = np.asarray(halftone(Image.fromarray(grey(0.4).astype(np.uint8)), mode="dither", cell=1.0, angle=0.0, mix=1.0, levels=2))
    assert set(np.unique(out)) == {0, 255}
    assert out.mean() / 255.0 == pytest.approx(0.4, abs=0.03)


# --- mix and off -------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_mix_zero_is_identity(mode):
    image = gradient()
    out = halftone(image, mode=mode, **{**ARGS, "mix": 0.0})
    assert np.array_equal(arrays(out), arrays(image))


@pytest.mark.parametrize("mode", MODES)
def test_mix_blends_between_the_original_and_the_effect(mode):
    image = gradient()
    full = arrays(halftone(image, mode=mode, **ARGS))
    half = arrays(halftone(image, mode=mode, **{**ARGS, "mix": 0.5}))
    assert np.abs(half - (arrays(image) + full) / 2).max() <= 1


def test_off_gives_identical_pixels():
    assert np.array_equal(arrays(render().image), arrays(render(halftone=None).image))
    assert "halftone" not in render().metadata.parameters
    # An override never turns the stage on.
    assert np.array_equal(arrays(render().image), arrays(render(stage_params={"halftone": {"cell": 20.0}}).image))


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_changes_the_image_and_keeps_its_size(mode):
    base, printed = render(), render(halftone=mode)
    assert printed.image.size == base.image.size
    assert not np.array_equal(arrays(base.image), arrays(printed.image))
    assert printed.metadata.parameters["halftone"] == mode


def test_stage_uses_no_rng_and_is_repeatable():
    state = random.getstate()
    first = arrays(halftone(gradient(), mode="cmyk", **ARGS))
    assert random.getstate() == state
    assert np.array_equal(first, arrays(halftone(gradient(), mode="cmyk", **ARGS)))


# --- order, options and metadata ----------------------------------------------------------------


def test_halftone_goes_between_the_sharp_contours_and_the_film_grain():
    params = finish_params(
        contours=0.5, contour_placement="sharp", glass=True, shading=0.5, chromatic_aberration=0.01,
        halftone="dots", noise=0.1, vignette=0.2,
    )
    assert [stage.name for stage in finish_stages(params)] == [
        "glass", "shading", "chromatic aberration", "contours", "halftone", "film grain", "vignette",
    ]


def test_halftone_follows_the_frosted_glass_and_precedes_grain():
    names = [s.name for s in finish_stages(finish_params(contours=0.5, glass=True, halftone="lines", noise=0.1))]
    assert names == ["contours", "glass", "halftone", "film grain"]


def test_overrides_reach_the_stage_and_unknown_arguments_are_rejected():
    params = finish_params(halftone="dots", stage_params={"halftone": {"angle": 15.0, "mode": "lines", "levels": 2}})
    (stage,) = finish_stages(params)
    arguments = stage.bind(params)
    assert (arguments["angle"], arguments["mode"], arguments["levels"]) == (15.0, "lines", 2)
    check_finish_stage_params({"halftone": {"paper": [250, 245, 235]}})
    with pytest.raises(PlanCompositionError, match="no argument bogus"):
        check_finish_stage_params({"halftone": {"bogus": 1}})


def test_unknown_mode_and_bad_arguments_are_errors():
    with pytest.raises(PipelineConfigurationError, match="halftone"):
        render(halftone="stipple")
    image = gradient()
    with pytest.raises(ValueError, match="mode"):
        halftone(image, mode="stipple", **ARGS)
    with pytest.raises(ValueError, match="mix"):
        halftone(image, mode="dots", **{**ARGS, "mix": 1.5})
    with pytest.raises(ValueError, match="levels"):
        halftone(image, mode="dither", **{**ARGS, "levels": 1})
    with pytest.raises(ValueError, match="cell"):
        halftone(image, mode="dots", **{**ARGS, "cell": 0.0})


def test_metadata_stores_the_mode_and_replays_the_image():
    original = render(halftone="dots", stage_params={"halftone": {"cell": 30.0}})
    png, _ = WallpaperPipeline().render_png_bytes(
        RenderConfiguration(
            width=160, height=100, seed=5, colors=PALETTE, harmonize="none", vignette=0.0, contours=0.0,
            glass=False, chromatic_aberration=0.0, noise=0.0, halftone="dots",
            stage_params={"halftone": {"cell": 30.0}},
        )
    )
    metadata = extract_image_metadata(png)
    assert metadata.parameters["halftone"] == "dots"
    replayed = WallpaperPipeline().render(metadata.to_render_configuration())
    assert np.array_equal(arrays(original.image), arrays(replayed.image))


def test_cli_accepts_the_hidden_option_and_rejects_other_modes(tmp_path):
    from click.testing import CliRunner

    from main import main

    runner = CliRunner()
    output = tmp_path / "out.png"
    result = runner.invoke(main, ["--res", "96", "60", "--seed", "3", "--halftone", "lines", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert extract_image_metadata(output.read_bytes()).parameters["halftone"] == "lines"
    assert runner.invoke(main, ["--halftone", "stipple"]).exit_code != 0
