"""Field grain: noise added to the field before color mapping."""

import random

import numpy as np
import pytest

from engine.colors import rgb_to_oklab
from engine.metadata import extract_image_metadata
from engine.pipeline import PipelineConfigurationError, RenderConfiguration, WallpaperPipeline
from engine.stages import STAGED_PLANS, Composition, render_staged
from engine.stages.catalog import PlanCompositionError, compose
from engine.stages.fields import field_grain
from engine.stages.runner import StageParams
from engine.stages.types import Coords, ScalarField, WeightField

PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90), (60, 170, 200)]
TWO_COLORS = [(26, 42, 108), (178, 31, 31)]
ENGINES = sorted(set(STAGED_PLANS) - {"radial-basis"})


def params(**extra):
    return {
        **dict(
            width=96, height=60, colors=PALETTE, complexity=4, fluid_mode="random", warp_strength=None,
            transverse_warp=None, ambient_weight=0.15, dispersion=0.35, fractal_type="fluid", fractal_zoom=None,
            fractal_center=None, fractal_angle=None, phase_offset=0.0, frequency=1.0, trap_weight=0.35,
            channel_offset=0.0, seed=5,
        ),
        **extra,
    }


def render(engine, grain, **extra):
    state = {}
    image = render_staged(engine, params(**extra), composition=Composition(field_grain=grain), keep_state=state)
    return np.asarray(image.convert("RGB")), state


def pipeline_render(**config):
    config = {"width": 160, "height": 100, "seed": 8, "engine": "simplex", "noise": 0.0, **config}
    return WallpaperPipeline().render(RenderConfiguration(**config))


def coords(width, height):
    xv, yv = np.meshgrid(np.linspace(0, 1, 8, dtype=np.float32), np.linspace(0, 1, 6, dtype=np.float32))
    return Coords(x=xv, y=yv, base_x=xv, base_y=yv, output_size=(width, height))


# Off ------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
def test_zero_changes_nothing(engine):
    plain, _ = render(engine, 0.0)
    untouched = np.asarray(render_staged(engine, params()).convert("RGB"))
    assert np.array_equal(plain, untouched)
    assert not Composition(field_grain=0.0)
    stages = STAGED_PLANS[engine].stages
    assert compose(stages, field_grain=0.0) == tuple(stages)


def test_zero_in_the_pipeline_is_pixel_identical():
    plain = pipeline_render()
    zero = pipeline_render(field_grain=0.0)
    assert np.array_equal(np.asarray(plain.image), np.asarray(zero.image))
    assert plain.field_grain == 0.0 and zero.field_grain == 0.0


def test_zero_draws_nothing_from_the_rng_and_grain_draws_once():
    def state_after(composition):
        render_staged("blobs", params(), composition=composition)
        return random.getstate()

    plain = state_after(None)
    assert state_after(Composition(field_grain=0.0)) == plain
    assert state_after(Composition(field_grain=0.2)) != plain


def test_grain_changes_the_image():
    plain, _ = render("simplex", 0.0)
    grained, _ = render("simplex", 0.2)
    assert not np.array_equal(plain, grained)


@pytest.mark.parametrize("engine", ENGINES)
def test_tiny_grain_leaves_the_gradient_where_it_was(engine):
    """Enlarging the field must line up with how the plan enlarges its colors.

    fluid-wave maps a 4 x 12 lattice and enlarges the colors with a corner-aligned
    spline; with PIL's bicubic instead the gradient moved by tens of pixels (mean
    difference 8.5 levels, max 89). Mapping then enlarging and the reverse still
    differ a little on so coarse a lattice and so contrasting a palette."""
    extra = dict(width=320, height=200)
    plain, _ = render(engine, 0.0, **extra)
    tiny, _ = render(engine, 1e-4, **extra)
    difference = np.abs(plain.astype(np.int16) - tiny.astype(np.int16))
    mean_limit, max_limit = (4.0, 30) if engine == "fluid-wave" else (1.5, 25)
    assert difference.mean() < mean_limit, f"{engine}: mean {difference.mean():.2f}"
    assert difference.max() <= max_limit, f"{engine}: max {difference.max()}"


def test_plans_with_a_spline_upscale_get_the_spline_resize():
    def resize(engine):
        grain = next(s for s in compose(STAGED_PLANS[engine].stages, field_grain=0.1) if s.name == "field grain")
        return grain.bind(StageParams(params()))["resize"]

    assert resize("fluid-wave") == "spline"
    assert {resize(e) for e in ENGINES if e != "fluid-wave"} == {"bicubic"}


# Plan ----------------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
def test_stage_sits_between_the_field_and_the_mapping(engine):
    names = [s.name for s in compose(STAGED_PLANS[engine].stages, field_grain=0.1)]
    plain = [s.name for s in STAGED_PLANS[engine].stages]
    at = names.index("field grain")
    assert names[at - 1] == plain[next(i for i, s in enumerate(STAGED_PLANS[engine].stages) if s.phase == "mapping") - 1]
    mapping = next(s for s in compose(STAGED_PLANS[engine].stages, field_grain=0.1) if s.phase == "mapping")
    assert mapping.inputs[0] == "grained_field"


def test_stage_goes_after_a_blend():
    names = [s.name for s in compose(STAGED_PLANS["simplex"].stages, blend="radial:max", field_grain=0.1)]
    assert names[names.index("noisy ramp") + 1:][:3] == ["blend radial", "blend", "field grain"]


def test_stage_param_reaches_the_stage():
    stages = compose(STAGED_PLANS["simplex"].stages, field_grain=0.1, stage_params={"field-grain": {"amount": 0.3}})
    grain = next(s for s in stages if s.name == "field grain")
    assert grain.bind(None)["amount"] == 0.3


def test_strength_is_per_run_of_the_palette():
    """Bands and cosine cycles run the palette several times over the field, which
    would multiply the grain's reach; the stage divides its amount by that."""
    def amount(engine, **kw):
        grain = next(s for s in compose(STAGED_PLANS[engine].stages, field_grain=0.3, **kw) if s.name == "field grain")
        return grain.bind(StageParams(params(frequency=2.0)))["amount"]

    assert amount("simplex") == pytest.approx(0.3)
    assert amount("simplex", mapping_name="bands") == pytest.approx(0.3 / 3.0)
    assert amount("simplex", mapping_name="bands", stage_params={"bands": {"cycles": 6}}) == pytest.approx(0.3 / 6.0)
    assert amount("fractal") == pytest.approx(0.3 / 2.0)
    assert amount("simplex", stage_params={"field-grain": {"amount": 0.2}}) == 0.2


def test_works_with_every_field_and_mapping():
    for field in ("conic", "voronoi", "blob-layers", "strokes"):
        for mapping in ("ramp", "bands"):
            try:
                image = render_staged(
                    "simplex", params(), composition=Composition(field=field, mapping=mapping, field_grain=0.1)
                )
            except PlanCompositionError:
                continue
            assert image.size == (96, 60)


# Determinism -----------------------------------------------------------------

@pytest.mark.parametrize("engine", ["simplex", "blobs", "fractal", "fluid-wave"])
def test_deterministic_per_seed(engine):
    first, _ = render(engine, 0.15)
    second, _ = render(engine, 0.15)
    other, _ = render(engine, 0.15, seed=6)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


# Colors ----------------------------------------------------------------------

def distance_to_ramp(image, colors):
    """Oklab distance of each pixel to the segment between two palette colors."""
    a, b = rgb_to_oklab(np.array(colors[0])), rgb_to_oklab(np.array(colors[1]))
    pixels = rgb_to_oklab(image.reshape(-1, 3))
    direction = b - a
    t = np.clip(((pixels - a) @ direction) / float(direction @ direction), 0.0, 1.0)
    return np.linalg.norm(pixels - (a + t[:, None] * direction), axis=1)


def test_grained_colors_stay_on_the_ramp_unlike_film_grain():
    base = dict(colors=TWO_COLORS, harmonize="none")
    field = np.asarray(pipeline_render(**base, field_grain=0.15).image.convert("RGB"))
    film = np.asarray(pipeline_render(**base, noise=0.1).image.convert("RGB"))
    plain = np.asarray(pipeline_render(**base).image.convert("RGB"))

    on_ramp = distance_to_ramp(field, TWO_COLORS)
    # Plain render sits at 0.003 (8-bit rounding and dither); film grain's median is 0.034.
    assert np.percentile(on_ramp, 99) < 0.006
    film_distance = distance_to_ramp(film, TWO_COLORS)
    assert np.percentile(film_distance, 50) > 0.02
    assert np.percentile(film_distance, 50) > 5 * np.percentile(on_ramp, 99)
    # And it is real grain: neighboring pixels differ far more than in the plain gradient.
    step = lambda image: np.abs(np.diff(image.astype(np.int16), axis=1)).mean()
    assert step(field) > 4 * step(plain)


def test_noise_is_at_output_pixel_scale():
    field = ScalarField(np.full((15, 24), 0.5, dtype=np.float32))
    out = field_grain(field, coords(96, 60), amount=0.1, rng=random.Random(1))
    assert out.t.shape == (60, 96)
    noise = out.t - 0.5
    lag1 = np.corrcoef(noise[:, :-1].ravel(), noise[:, 1:].ravel())[0, 1]
    assert abs(lag1) < 0.1, "noise added at the grid would be smoothed into blotches"
    assert noise.std() == pytest.approx(0.1 * 0.25, rel=0.1)


def test_a_field_at_output_size_is_not_resized():
    t = np.tile(np.linspace(0.0, 1.0, 96, dtype=np.float32), (60, 1))
    out = field_grain(ScalarField(t), coords(96, 60), amount=0.05, rng=random.Random(1))
    assert out.t.shape == (60, 96)
    assert np.abs(out.t - t).mean() < 0.05
    assert out.t.min() >= 0.0 and out.t.max() <= 1.0


def test_scalar_stays_in_range_and_keeps_its_mean():
    t = np.tile(np.linspace(0.0, 1.0, 96, dtype=np.float32), (60, 1))
    out = field_grain(ScalarField(t), coords(96, 60), amount=0.3, rng=random.Random(2)).t
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert abs(float(out.mean()) - 0.5) < 0.02


def test_pipeline_renders_at_the_requested_size_for_a_coarse_lattice():
    result = pipeline_render(engine="fluid-wave", field_grain=0.1, width=200, height=120)
    assert result.image.size == (200, 120)


# Weights ---------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["blobs", "patch-mesh", "rbf", "idw", "flow-field"])
def test_weights_still_sum_to_one(engine):
    _, state = render(engine, 0.3)
    field = state["grained_field"]
    assert isinstance(field, WeightField)
    assert field.weights.shape[1:] == (60, 96)
    assert field.weights.min() >= 0.0
    assert np.allclose(field.weights.sum(axis=0), 1.0, atol=1e-5)
    # The plan's own field is kept as it was, for contours and shading.
    clean = state["field"]
    assert clean.weights.shape[1:] != (60, 96) or engine in ("idw", "flow-field")


def test_weight_noise_moves_weight_between_layers():
    weights = np.zeros((3, 15, 24), dtype=np.float32)
    weights[0] = 1.0
    out = field_grain(WeightField(weights, background=0), coords(96, 60), amount=0.2, rng=random.Random(3))
    assert out.background == 0
    assert out.weights[0].mean() < 1.0
    assert out.weights[1:].max() > 0.0


def test_scalar_field_keeps_the_plan_field_for_contours_and_shading():
    _, with_grain = render("simplex", 0.3)
    _, without = render("simplex", 0.0)
    assert np.array_equal(with_grain["field"].t, without["field"].t)
    assert with_grain["field"].t.shape == (16, 24)  # the grid is a quarter of the output, at least 16 rows
    assert with_grain["grained_field"].t.shape == (60, 96)


# Configuration, CLI and metadata ----------------------------------------------

def test_metadata_round_trips():
    pipeline = WallpaperPipeline()
    config = RenderConfiguration(width=160, height=100, seed=8, engine="blobs", noise=0.0, field_grain=0.2)
    png, original = pipeline.render_png_bytes(config)
    assert original.field_grain == 0.2
    metadata = extract_image_metadata(png)
    assert metadata.to_render_configuration().field_grain == 0.2
    replayed = pipeline.render(metadata.to_render_configuration())
    assert np.array_equal(np.asarray(original.image), np.asarray(replayed.image))


def test_old_metadata_without_field_grain_replays_with_it_off():
    pipeline = WallpaperPipeline()
    png, original = pipeline.render_png_bytes(RenderConfiguration(width=64, height=40, seed=3, noise=0.0))
    metadata = extract_image_metadata(png)
    assert metadata.to_render_configuration().field_grain in (None, 0.0)


@pytest.mark.parametrize("value", [-0.1, 1.5, "a lot", True])
def test_out_of_range_values_are_rejected(value):
    with pytest.raises(PipelineConfigurationError, match="field_grain"):
        pipeline_render(field_grain=value)


def test_cli_option_is_hidden_and_validated(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    main = str(Path(__file__).resolve().parents[1] / "main.py")
    base = [sys.executable, main, "--engine", "simplex", "--res", "64", "40", "--seed", "4"]
    help_text = subprocess.run([*base, "--help"], capture_output=True, text=True, timeout=30).stdout
    assert "--field-grain" not in help_text
    assert "--field-grain" in subprocess.run([*base, "--help-all"], capture_output=True, text=True, timeout=30).stdout
    good = subprocess.run([*base, "--field-grain", "0.2", "--output", str(tmp_path / "a.png")],
                          capture_output=True, text=True, timeout=60)
    assert good.returncode == 0, good.stderr
    bad = subprocess.run([*base, "--field-grain", "2", "--output", str(tmp_path / "b.png")],
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode != 0
