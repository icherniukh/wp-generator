"""Film grain size, color mode and blend."""

import numpy as np
import pytest
from PIL import Image

from engine.filters import WallpaperFilters
from engine.metadata import build_image_metadata, extract_image_metadata
from engine.pipeline import PipelineConfigurationError, RenderConfiguration, WallpaperPipeline
from engine.stages import StageParams
from engine.stages.finish import GRAIN, finish_stages

BLENDS = ["add", "overlay", "soft-light", "multiply"]
PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90), (60, 170, 200)]


def _ramp(width=240, height=160):
    """A smooth diagonal ramp through shadows, mids and highlights, different per channel."""
    x = np.linspace(0.08, 0.92, width)[None, :]
    y = np.linspace(0.0, 0.1, height)[:, None]
    base = np.clip(x + y, 0, 1)
    arr = np.stack([base, 0.3 + 0.5 * base, 1.0 - 0.8 * base], axis=-1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _grain(image, seed=5, **options):
    np.random.seed(seed)
    return np.asarray(WallpaperFilters.apply_film_grain(image, amount=0.06, **options), np.float32)


def _config(**extra):
    base = dict(colors=PALETTE, harmonize="none", width=160, height=100, seed=4, complexity=4,
                noise=0.08, vignette=0.0, glass=False, contours=0.0, chromatic_aberration=0.0)
    base.update(extra)
    return RenderConfiguration(**base)


def _high_frequency_energy(noise):
    """Mean squared difference between neighbouring pixels, per unit of variance."""
    d = np.diff(noise, axis=1)
    return float(np.mean(d * d) / np.var(noise))


def _old_film_grain(image, amount):
    """apply_film_grain as it was before the grain options, copied verbatim."""
    import scipy.ndimage

    img_arr = np.array(image, dtype=np.float32)
    h, w = img_arr.shape[:2]
    channels = img_arr.shape[2] if img_arr.ndim == 3 else 1
    sigma = 255.0 * amount
    fine_grain = np.random.normal(0.0, sigma * 0.7, (h, w)).astype(np.float32)
    down_h, down_w = max(1, h // 2), max(1, w // 2)
    coarse_small = np.random.normal(0.0, sigma * 0.6, (down_h, down_w)).astype(np.float32)
    coarse_grain = scipy.ndimage.zoom(coarse_small, (h / down_h, w / down_w), order=1)[:h, :w]
    combined_grain = fine_grain + coarse_grain
    rgb = img_arr[..., :3] if channels >= 3 else img_arr
    lum = np.mean(rgb, axis=2 if channels >= 3 else -1) / 255.0
    lum_weight = np.clip(4.0 * lum * (1.0 - lum), 0.35, 1.0)
    modulated_grain = combined_grain * lum_weight
    if channels >= 3:
        img_arr[..., :3] += modulated_grain[..., None]
    else:
        img_arr += modulated_grain
    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


@pytest.mark.parametrize("size", [(240, 160), (101, 67)])
def test_defaults_are_pixel_identical_to_the_old_path(size):
    image = _ramp(*size)
    np.random.seed(9)
    old = np.asarray(_old_film_grain(image, 0.06), np.uint8)
    np.random.seed(9)
    default = np.asarray(WallpaperFilters.apply_film_grain(image, amount=0.06), np.uint8)
    np.random.seed(9)
    explicit = np.asarray(
        WallpaperFilters.apply_film_grain(image, amount=0.06, grain_size=1.0, grain_color="mono", grain_blend="add"),
        np.uint8,
    )
    assert np.array_equal(old, default)
    assert np.array_equal(old, explicit)
    # Same number of random draws: the next draw after the grain matches too.
    np.random.seed(9)
    _old_film_grain(image, 0.06)
    after_old = np.random.random()
    np.random.seed(9)
    WallpaperFilters.apply_film_grain(image, amount=0.06)
    assert np.random.random() == after_old


def test_default_stage_arguments_are_untouched():
    params = StageParams(dict(noise=0.07, width=1920, height=1080))
    assert GRAIN.bind(params) == {"amount": 0.07, "grain_size": 1.0, "grain_color": "mono", "grain_blend": "add"}


def test_default_pipeline_render_matches_explicit_defaults():
    pipeline = WallpaperPipeline()
    plain = pipeline.render(_config(noise=0.0)).image
    graded = pipeline.render(_config()).image
    explicit = pipeline.render(_config(grain_size=1.0, grain_color="mono", grain_blend="add")).image
    assert np.array_equal(np.asarray(graded), np.asarray(explicit))
    assert not np.array_equal(np.asarray(graded), np.asarray(plain))


def test_grain_size_scales_with_the_image():
    small = StageParams(dict(noise=0.1, grain_size=4.0, width=982, height=982))
    big = StageParams(dict(noise=0.1, grain_size=4.0, width=3928, height=3928))
    assert GRAIN.bind(small)["grain_size"] == pytest.approx(2.0)
    assert GRAIN.bind(big)["grain_size"] == pytest.approx(8.0)
    tiny = StageParams(dict(noise=0.1, grain_size=2.0, width=160, height=100))
    assert GRAIN.bind(tiny)["grain_size"] == 1.0


@pytest.mark.parametrize("blend", BLENDS)
def test_larger_grain_has_less_high_frequency_energy(blend):
    flat = Image.new("RGB", (320, 200), (128, 128, 128))
    energies = []
    for size in (1.5, 3.0, 6.0):
        grain = _grain(flat, grain_size=size, grain_blend=blend)[..., 1]
        energies.append(_high_frequency_energy(grain))
    default = _high_frequency_energy(_grain(flat)[..., 1])
    assert energies[0] < default
    assert energies[0] > energies[1] > energies[2]


def test_grain_strength_does_not_depend_on_size():
    flat = Image.new("RGB", (400, 300), (128, 128, 128))
    stds = [_grain(flat, grain_size=s)[..., 0].std() for s in (1.0, 2.0, 4.0)]
    assert max(stds) / min(stds) < 1.2


def test_color_grain_has_independent_channels():
    flat = Image.new("RGB", (300, 200), (128, 128, 128))
    mono = _grain(flat)
    color = _grain(flat, grain_color="color")

    def corr(arr, a, b):
        return float(np.corrcoef(arr[..., a].ravel(), arr[..., b].ravel())[0, 1])

    assert corr(mono, 0, 1) > 0.99
    for a, b in ((0, 1), (1, 2), (0, 2)):
        assert abs(corr(color, a, b)) < 0.05
    # Same strength per channel as the mono grain.
    assert color[..., 0].std() == pytest.approx(mono[..., 0].std(), rel=0.1)


@pytest.mark.parametrize("color", ["mono", "color"])
@pytest.mark.parametrize("blend", BLENDS)
def test_blends_keep_mean_brightness(blend, color):
    image = _ramp()
    base = np.asarray(image, np.float32)
    out = _grain(image, grain_size=2.0, grain_color=color, grain_blend=blend)
    assert abs(out.mean() - base.mean()) < 1.0
    for channel in range(3):
        assert abs(out[..., channel].mean() - base[..., channel].mean()) < 1.5


def test_blend_strengths_are_comparable():
    image = _ramp()
    base = np.asarray(image, np.float32)
    stds = {blend: (_grain(image, grain_size=2.0, grain_blend=blend) - base).std() for blend in BLENDS}
    for blend, std in stds.items():
        assert 0.6 < std / stds["add"] < 1.5, (blend, stds)  # multiply follows brightness


@pytest.mark.parametrize("blend", BLENDS)
def test_blends_are_reproducible_and_leave_the_image_shape_alone(blend):
    image = _ramp()
    a = _grain(image, seed=2, grain_size=3.0, grain_color="color", grain_blend=blend)
    b = _grain(image, seed=2, grain_size=3.0, grain_color="color", grain_blend=blend)
    assert np.array_equal(a, b)
    assert a.shape == (image.height, image.width, 3)


def test_unknown_options_are_rejected():
    image = _ramp(20, 20)
    with pytest.raises(ValueError):
        WallpaperFilters.apply_film_grain(image, amount=0.1, grain_blend="screen")
    with pytest.raises(ValueError):
        WallpaperFilters.apply_film_grain(image, amount=0.1, grain_color="rgb")
    for bad in (dict(grain_blend="screen"), dict(grain_color="rgb"), dict(grain_size=0.5), dict(grain_size=9.0)):
        with pytest.raises(PipelineConfigurationError):
            WallpaperPipeline().render(_config(**bad))


def test_grain_stage_stays_in_the_plan_with_options():
    names = [s.name for s in finish_stages(StageParams(dict(
        contours=0.0, glass=False, chromatic_aberration=0.0, noise=0.1, vignette=0.0,
        grain_size=3.0, grain_color="color", grain_blend="soft-light")))]
    assert names == ["film grain"]


def test_options_change_the_render_and_default_metadata_omits_them():
    pipeline = WallpaperPipeline()
    default = pipeline.render(_config())
    assert default.grain_size is None and default.grain_color is None and default.grain_blend is None
    for key in ("grain_size", "grain_color", "grain_blend"):
        assert key not in build_image_metadata(default).parameters

    styled = pipeline.render(_config(grain_color="color", grain_blend="soft-light"))
    assert not np.array_equal(np.asarray(default.image), np.asarray(styled.image))


def test_no_grain_options_are_recorded_without_grain():
    result = WallpaperPipeline().render(_config(noise=0.0, grain_size=3.0, grain_color="color", grain_blend="overlay"))
    assert result.grain_size is None and result.grain_color is None and result.grain_blend is None


@pytest.mark.parametrize("options", [
    dict(grain_size=3.0),
    dict(grain_color="color"),
    dict(grain_blend="multiply"),
    dict(grain_size=6.0, grain_color="color", grain_blend="soft-light"),
])
def test_metadata_round_trips(options):
    pipeline = WallpaperPipeline()
    config = _config(width=640, height=400, **options)
    png, original = pipeline.render_png_bytes(config)
    metadata = extract_image_metadata(png)
    for key, value in options.items():
        assert metadata.parameters[key] == value
    restored = metadata.to_render_configuration()
    for key, value in options.items():
        assert getattr(restored, key) == value
    replayed = pipeline.render(restored)
    assert np.array_equal(np.asarray(original.image), np.asarray(replayed.image))


def test_presets_can_set_grain_options():
    preset = {"preset_id": "t", "version": "1", "color_space": "oklch", "color_count": {"type": "uniform_int", "min": 3, "max": 3},
              "overrides": {"engine": "simplex", "noise": 0.08, "grain_size": 4.0, "grain_color": "color", "grain_blend": "overlay"}}
    result = WallpaperPipeline().render(
        RenderConfiguration(preset="aura", preset_data={**_aura_data(), "overrides": preset["overrides"]},
                            width=640, height=400, seed=3))
    assert (result.grain_size, result.grain_color, result.grain_blend) == (4.0, "color", "overlay")


def _aura_data():
    return WallpaperPipeline().load_preset("aura")


def test_cli_flags(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    main = str(Path(__file__).resolve().parents[1] / "main.py")
    out = tmp_path / "a.png"
    ok = subprocess.run([sys.executable, main, "--preset", "aura", "--res", "640", "400", "--seed", "3",
                         "--grain-size", "4", "--grain-color", "color", "--grain-blend", "soft-light",
                         "--output", str(out)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    metadata = extract_image_metadata(out.read_bytes())
    assert (metadata.parameters["grain_size"], metadata.parameters["grain_color"],
            metadata.parameters["grain_blend"]) == (4.0, "color", "soft-light")
    for flag, value in (("--grain-size", "9"), ("--grain-size", "0.5"), ("--grain-color", "rgb"), ("--grain-blend", "screen")):
        bad = subprocess.run([sys.executable, main, "--preset", "aura", "--res", "64", "40", flag, value,
                              "--output", str(tmp_path / "b.png")], capture_output=True, text=True)
        assert bad.returncode == 2, (flag, value, bad.stderr)
