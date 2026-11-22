"""Finish stages: image effects applied to the exported 8-bit image, in a fixed order.

Sizes in pixels (glass blur, contour width and smoothing) are given for an
image whose short side is REFERENCE_SHORT_SIDE and scale with the image, so a
small preview looks like the full render reduced.

Contours can follow the image's luminance (default) or the engine's scalar
field, and sit under the glass and aberration (frosted, default) or over them
(sharp).

stage_params (see engine.stages.catalog) can override the arguments of the
finish stages too; see finish_stages() for the keys and the rules.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import scipy.ndimage
from PIL import Image

from engine.colors import rgb_to_oklab
from engine.filters import WallpaperFilters
from engine.stages.halftone import MIN_SCREEN_CELL, halftone
from engine.stages.runner import Stage, StageParams
from engine.stages.types import Field, ScalarField


# The default device (MacBook Pro 14", 3024 x 1964), where the pixel sizes below were tuned.
REFERENCE_SHORT_SIDE = 1964


def size_scale(p: StageParams) -> float:
    """The image's short side relative to REFERENCE_SHORT_SIDE."""
    return min(int(p["width"]), int(p["height"])) / REFERENCE_SHORT_SIDE


def contour_lines(
    image: Image.Image,
    *,
    intensity: float,
    levels: int,
    line_width: float,
    mode: str,
    smoothing_radius: float = 2.5,
) -> Image.Image:
    """Isolines of the image's smoothed luminance, drawn over it."""
    return WallpaperFilters.apply_contour_lines(
        image,
        levels=levels,
        line_width=line_width,
        intensity=intensity,
        mode=mode,
        smoothing_radius=smoothing_radius,
    )


def field_elevation(field: ScalarField, size: Tuple[int, int]) -> np.ndarray:
    """The field's values enlarged bicubically to size (width, height), in [0, 1]."""
    t = Image.fromarray(np.asarray(field.t, dtype=np.float32), mode="F")
    if t.size != size:
        t = t.resize(size, Image.Resampling.BICUBIC)
    return np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)


def field_contour_lines(
    image: Image.Image,
    field: ScalarField,
    *,
    intensity: float,
    levels: int,
    line_width: float,
    mode: str,
    smoothing_radius: float = 2.5,
) -> Image.Image:
    """Isolines of the engine's scalar field, so they follow the shape the
    engine computed rather than the colors the palette gave it."""
    return WallpaperFilters.apply_contour_lines(
        image,
        levels=levels,
        line_width=line_width,
        intensity=intensity,
        mode=mode,
        smoothing_radius=smoothing_radius,
        elevation=field_elevation(field, image.size),
    )


def frosted_glass(image: Image.Image, *, blur_radius: int, noise_amount: float) -> Image.Image:
    """Gaussian blur at half size, then a screen-blended speckle."""
    return WallpaperFilters.apply_frosted_glass(image, blur_radius=blur_radius, noise_amount=noise_amount)


def chromatic_aberration(image: Image.Image, *, strength: float) -> Image.Image:
    """Shifts red outward and blue inward, growing toward the edges."""
    return WallpaperFilters.apply_chromatic_aberration(image, strength=strength)


def film_grain(
    image: Image.Image,
    *,
    amount: float,
    grain_size: float = 1.0,
    grain_color: str = "mono",
    grain_blend: str = "add",
) -> Image.Image:
    """Fine plus coarse Gaussian noise, weaker in deep shadows and highlights.

    grain_size enlarges the grain (1.0 is the default grain), grain_color "color" draws
    the noise per channel, and grain_blend applies it by add, overlay, soft-light or multiply.
    """
    return WallpaperFilters.apply_film_grain(
        image, amount=amount, grain_size=grain_size, grain_color=grain_color, grain_blend=grain_blend
    )


def vignette(image: Image.Image, *, strength: float) -> Image.Image:
    """Darkens toward the corners."""
    return WallpaperFilters.apply_vignette(image, strength=strength)


# The steepest surface tilt shade() allows, as rise over run (2.5 is about 68 degrees).
MAX_SLOPE = 2.5
# The shine fades out over dark colors: from nothing at Oklab lightness 0.18 (near black) to
# full at 0.42. A white highlight added to near-black reads as a grey patch, and on a weight
# field the height is the lightness itself, so the slopes of a dark blob's edge would light up.
SHINE_DARK, SHINE_BRIGHT = 0.18, 0.42


def shade(
    final: Image.Image,
    field: Optional[Field],
    *,
    strength: float,
    depth: float,
    light_angle: float,
    light_elevation: float,
    specular: float,
    shininess: float,
    rim: float,
    refraction: float,
    smoothing: float = 0.0,
    tint: float = 0.0,
) -> Image.Image:
    """Lights the image as a relief, for a glossy 3D glass or embossed look.

    The height is the engine's scalar field, or the image's Oklab lightness
    when the field is a weight field (or absent); it is smoothed with a
    Gaussian of sigma `smoothing` pixels. Normals come from the height gradient
    times `depth`, measured per short side of the image, so the relief does not
    depend on the resolution. The highlight and rim fade out over near-black
    colors (see SHINE_DARK), so dark areas get no grey glints. Lambert diffuse is normalized so a flat surface is
    unchanged, the Blinn-Phong highlight is zero on a flat surface, and a
    Schlick-Fresnel rim brightens steep slopes. light_angle (degrees,
    counterclockwise from the right) and light_elevation (degrees above the
    image plane) set the light. With refraction > 0 the image is sampled that
    many pixels along the normal first, like a lens. `tint` (0 to 1) mixes the
    white highlight toward the local base color (0 is white, the default; 1 is
    the base color), so the shine takes the image's colors. Deterministic; no RNG.
    """
    if not 0.0 <= tint <= 1.0:
        raise ValueError(f"tint must be from 0 to 1, not {tint!r}")
    rgb = np.asarray(final.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]

    if isinstance(field, ScalarField):
        height = field_elevation(field, final.size)
    else:
        height = rgb_to_oklab(rgb)[..., 0].astype(np.float32)
    if smoothing > 0.0:
        height = scipy.ndimage.gaussian_filter(height, smoothing, mode="nearest")

    gy, gx = np.gradient(height)
    slope = depth * min(w, h)
    nx, ny = -gx * slope, -gy * slope
    # Soft-limit the tilt: an edge in the height is a near-vertical wall whose rim
    # and highlight would otherwise saturate and break into 8-bit noise.
    steepness = np.hypot(nx, ny)
    limit = np.tanh(steepness / MAX_SLOPE) * MAX_SLOPE / np.maximum(steepness, 1e-6)
    nx, ny = nx * limit, ny * limit
    inverse_length = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    nx, ny, nz = nx * inverse_length, ny * inverse_length, inverse_length

    base = rgb
    if refraction > 0.0:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        coordinates = np.array([y + ny * refraction, x + nx * refraction])
        base = np.stack(
            [scipy.ndimage.map_coordinates(rgb[..., c], coordinates, order=1, mode="nearest") for c in range(3)],
            axis=-1,
        )

    azimuth, elevation = np.radians(light_angle), np.radians(light_elevation)
    lx = np.cos(azimuth) * np.cos(elevation)
    ly = -np.sin(azimuth) * np.cos(elevation)  # image y grows downward
    lz = np.sin(elevation)

    diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0) / max(lz, 1e-3)

    # Half vector between the light and the viewer (looking along z). The
    # highlight is the Blinn-Phong lobe minus its flat-surface value, scaled so
    # the peak is 1: a flat field has no highlight.
    hx, hy, hz = lx, ly, lz + 1.0
    half_length = np.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hy, hz = hx / half_length, hy / half_length, hz / half_length
    facing = nx * hx + ny * hy + nz * hz
    flat = hz**shininess
    highlight = np.clip((np.clip(facing, 0.0, 1.0) ** shininess - flat) / max(1.0 - flat, 1e-6), 0.0, 1.0)
    fresnel = (1.0 - nz) ** 5

    lightness = rgb_to_oklab(base)[..., 0]
    gate = np.clip((lightness - SHINE_DARK) / (SHINE_BRIGHT - SHINE_DARK), 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)
    shine = (gate * (specular * highlight + rim * fresnel))[..., None] * 255.0
    if tint > 0.0:
        shine = shine * (1.0 - tint) + shine / 255.0 * base * tint
    out = base * (1.0 - strength + strength * diffuse[..., None]) + strength * shine
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


# sRGB code -> linear light, and back through a 14-bit table (fine enough for the darkest codes).
_TO_LINEAR = np.array(
    [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in np.arange(256) / 255.0],
    dtype=np.float32,
)
_ENCODE_STEPS = 16384
_FROM_LINEAR = np.round(
    255.0
    * np.array(
        [12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055 for v in np.arange(_ENCODE_STEPS + 1) / _ENCODE_STEPS]
    )
).astype(np.uint8)

# Gains that make strength 1 and rays 1 a strong but not flat-white glow.
BLOOM_GAIN = 1.2
RAYS_GAIN = 3.0


def _bloom_bright_pass(linear: np.ndarray, threshold: float, knee: float) -> np.ndarray:
    """The part of the image above `threshold` in linear luminance, with a quadratic soft knee.

    Below threshold - knee nothing passes; between threshold - knee and threshold + knee the
    contribution rises smoothly; above, the excess over the threshold passes at full weight.
    The pixel keeps its color (its RGB is scaled), so the glow is tinted like the source.
    """
    luma = linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    excess = luma - threshold
    if knee > 0.0:
        soft = np.clip(excess + knee, 0.0, 2.0 * knee)
        soft = soft * soft / (4.0 * knee)
        passed = np.maximum(soft, excess)
    else:
        passed = np.maximum(excess, 0.0)
    return linear * (passed / np.maximum(luma, 1e-6))[..., None]


def _box_reduce(a: np.ndarray, factor: int) -> np.ndarray:
    """Averages factor x factor blocks of an (h, w, c) array whose sides divide by factor."""
    h, w, c = a.shape
    return a.reshape(h // factor, factor, w // factor, factor, c).mean(axis=(1, 3), dtype=np.float32)


def _enlarge(a: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinear enlargement of an (h, w, c) float array to size (width, height)."""
    if (a.shape[1], a.shape[0]) == size:
        return a
    return np.stack(
        [
            np.asarray(Image.fromarray(a[..., c], mode="F").resize(size, Image.Resampling.BILINEAR))
            for c in range(a.shape[2])
        ],
        axis=-1,
    )


def _bloom_glow(bright: np.ndarray, sigmas: List[float]) -> np.ndarray:
    """Gaussian-blurs the bright pass at each sigma (pixels) and sums with falling weights.

    The sum has weights 1, 1/2, 1/4, ... normalized to 1, narrowest radius first. A radius is blurred
    at the size where its sigma is a few pixels, and the layers are combined from the widest down,
    so most of the work happens on small arrays.
    """
    h, w, _ = bright.shape
    order = sorted(range(len(sigmas)), key=lambda i: sigmas[i])
    weights = np.array([0.5**rank for rank in range(len(order))], dtype=np.float32)
    weights /= weights.sum()
    factors = [max(1, 2 ** int(np.floor(np.log2(max(sigmas[i] / 3.0, 1.0))))) for i in order]
    block = max(factors)
    padded = np.pad(bright, ((0, -h % block), (0, -w % block), (0, 0)))
    levels = {1: padded}
    glow: Optional[np.ndarray] = None
    for rank in sorted(range(len(order)), key=lambda r: -factors[r]):
        factor = factors[rank]
        if factor not in levels:
            levels[factor] = _box_reduce(padded, factor)
        blurred = scipy.ndimage.gaussian_filter(levels[factor], (sigmas[order[rank]] / factor,) * 2 + (0,), mode="constant")
        blurred[blurred < 1e-8] = 0.0  # no denormals
        layer = blurred * weights[rank]
        if glow is not None:
            layer = layer + _enlarge(glow, (layer.shape[1], layer.shape[0]))
        glow = layer
    glow = _enlarge(glow, (padded.shape[1], padded.shape[0]))
    return glow[:h, :w]


def _light_rays(bright: np.ndarray, origin: Tuple[float, float], length: float, samples: int) -> np.ndarray:
    """Streaks of the bright pass pointing away from `origin` (pixels, in the array's frame).

    Each pixel averages the bright pass at `length` pixels along the line toward the origin,
    with weights falling toward the far end (about e^-2 at `length`). 2**n samples are taken in
    n passes: each pass averages the result with a copy of itself moved a further 2**j steps along
    its own ray, which is exact because a ray's direction stays the same along it.
    """
    h, w, c = bright.shape
    passes = max(1, int(np.ceil(np.log2(max(samples, 2)))))
    count = 2**passes
    step = length / count
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = x - origin[0], y - origin[1]
    distance = np.maximum(np.hypot(dx, dy), 1e-6)
    ux, uy = dx / distance, dy / distance
    decay = float(np.exp(-2.0 / count))
    out = bright
    for j in range(passes):
        move = step * 2**j
        coordinates = np.array([y - uy * move, x - ux * move])
        weight = decay ** (2**j)
        moved = np.stack(
            [scipy.ndimage.map_coordinates(out[..., k], coordinates, order=1, mode="constant") for k in range(c)],
            axis=-1,
        )
        out = (out + weight * moved) / (1.0 + weight)
    return out


def bloom(
    final: Image.Image,
    *,
    strength: float,
    threshold: float = 0.5,
    knee: float = 0.3,
    radii: Tuple[float, ...] = (6.0, 24.0, 90.0),
    rays: float = 0.0,
    ray_origin: Tuple[float, float] = (0.5, -0.25),
    ray_length: float = 700.0,
    ray_samples: int = 32,
) -> Image.Image:
    """A soft glow around the bright parts of the image, and optionally light rays.

    In linear light, the bright pass keeps what is above `threshold` (relative luminance, 0 to 1)
    with a smooth quadratic knee of half-width `knee`, so nothing below threshold - knee glows.
    It is blurred with a Gaussian at each of `radii` (sigma in pixels, narrow to wide; the wide
    ones are blurred at reduced size), summed with weights 1, 1/2, 1/4, ... and screen-added to the
    image, so the result approaches white without clipping. With `rays` > 0 a streak layer is added
    the same way: the bright pass smeared `ray_length` pixels away from `ray_origin` (image-relative
    (x, y); the default is above the top center, so the shafts fall down through the image), sampled
    `ray_samples` times along the way. `strength` (0 to 1) scales both; `rays` is the amount of
    streaks relative to the bloom. Deterministic; no RNG. strength 0 returns the image unchanged.
    """
    if strength < 0.0 or rays < 0.0:
        raise ValueError(f"strength and rays must not be negative, not {strength!r} and {rays!r}")
    if threshold < 0.0 or knee < 0.0:
        raise ValueError(f"threshold and knee must not be negative, not {threshold!r} and {knee!r}")
    radii = tuple(float(r) for r in radii)
    if not radii or min(radii) <= 0.0:
        raise ValueError(f"radii must be a non-empty list of positive pixel sizes, not {radii!r}")
    if ray_length < 0.0 or int(ray_samples) < 1:
        raise ValueError("ray_length must not be negative and ray_samples must be at least 1")
    source = np.asarray(final.convert("RGB"))
    if strength <= 0.0:
        return Image.fromarray(source)
    h, w = source.shape[:2]
    linear = _TO_LINEAR[source]
    bright = _bloom_bright_pass(linear, threshold, knee)
    if not bright.any():
        return Image.fromarray(source)

    glow = _bloom_glow(bright, list(radii)) * BLOOM_GAIN
    if rays > 0.0 and ray_length > 0.0:
        # Streaks are soft, so make them at about a quarter of the size.
        factor = max(1, min(4, int(min(w, h) // 64)))
        small = _box_reduce(np.pad(bright, ((0, -h % factor), (0, -w % factor), (0, 0))), factor) if factor > 1 else bright
        origin = (ray_origin[0] * w / factor - 0.5, ray_origin[1] * h / factor - 0.5)
        streaks = _light_rays(small, origin, ray_length / factor, int(ray_samples))
        streaks = _enlarge(streaks, (small.shape[1] * factor, small.shape[0] * factor))[:h, :w]
        glow = glow + rays * RAYS_GAIN * streaks
    glow *= strength

    # Screen: 1 - (1 - image)(1 - glow), which stays below white where the image is.
    glow = np.minimum(glow, 1.0)
    out = 1.0 - (1.0 - linear) * (1.0 - glow)
    encoded = _FROM_LINEAR[(out * _ENCODE_STEPS + 0.5).astype(np.int32)]
    touched = (glow > 1e-5).any(axis=-1, keepdims=True)
    return Image.fromarray(np.where(touched, encoded, source))


def _contours(p: StageParams) -> Dict:
    scale = size_scale(p)
    return {
        "intensity": p["contours"],
        "levels": p["contour_levels"],
        "line_width": p["contour_width"] * scale,
        "mode": p["contour_mode"],
        "smoothing_radius": 2.5 * scale,
    }


def _blur_radius(reference: float, scale: float) -> int:
    # frosted_glass blurs at half size with radius blur_radius // 2, so keep at least 2.
    return max(2, int(round(reference * scale)))


def _grain_size(reference: float, scale: float) -> float:
    # The size is given for the reference image and scales with it; the default stays 1 pixel.
    return 1.0 if reference == 1.0 else max(1.0, reference * scale)


def _halftone_cell(reference: float, scale: float) -> float:
    # A screen needs a few pixels per cell, so a small preview keeps a floor.
    return max(MIN_SCREEN_CELL, reference * scale)


def _halftone(p: StageParams) -> Dict:
    # The cell is given for the reference image and scales with it.
    return {"mode": p["halftone"], "cell": _halftone_cell(9.0, size_scale(p)), "angle": 45.0, "mix": 1.0, "levels": 4}


def _glass(p: StageParams) -> Dict:
    return {"blur_radius": _blur_radius(60, size_scale(p)), "noise_amount": 0.15}


def _grain(p: StageParams) -> Dict:
    args: Dict = {"amount": p["noise"]}
    args["grain_size"] = _grain_size(p.get("grain_size", 1.0), size_scale(p))
    args["grain_color"] = p.get("grain_color", "mono")
    args["grain_blend"] = p.get("grain_blend", "add")
    return args


CONTOURS = Stage("contours", "finish", contour_lines, ("final",), "final", _contours)
FIELD_CONTOURS = Stage("field contours", "finish", field_contour_lines, ("final", "field"), "final", _contours)
GLASS = Stage("glass", "finish", frosted_glass, ("final",), "final", _glass)
ABERRATION = Stage("chromatic aberration", "finish", chromatic_aberration, ("final",), "final",
                   lambda p: {"strength": p["chromatic_aberration"]})
HALFTONE = Stage("halftone", "finish", halftone, ("final",), "final", _halftone)
GRAIN = Stage("film grain", "finish", film_grain, ("final",), "final", _grain)
VIGNETTE = Stage("vignette", "finish", vignette, ("final",), "final", lambda p: {"strength": p["vignette"]})


def _shade(p: StageParams) -> Dict:
    # Pixel sizes are for REFERENCE_SHORT_SIDE and scale with the image; depth is
    # per short side, so it needs no scaling. The smoothing is wide because the
    # height of a weight-field image is its lightness, whose 8-bit noise and
    # dark-level steps would otherwise show as grain in the highlights.
    scale = size_scale(p)
    return {
        "strength": p["shading"],
        "depth": 3.5,
        "light_angle": 135.0,
        "light_elevation": 50.0,
        "specular": 0.45,
        "shininess": 10.0,
        "rim": 1.0,
        "refraction": 12.0 * scale,
        "smoothing": 30.0 * scale,
    }


SHADE = Stage("shading", "finish", shade, ("final", "field"), "final", _shade)


def _bloom(p: StageParams) -> Dict:
    # Pixel sizes are for REFERENCE_SHORT_SIDE and scale with the image. The ray origin is
    # relative to the image, so it needs no scaling.
    scale = size_scale(p)
    return {
        "strength": p["bloom"],
        "threshold": 0.5,
        "knee": 0.3,
        "radii": tuple(r * scale for r in (6.0, 24.0, 90.0)),
        "rays": 0.0,
        "ray_origin": (0.5, -0.25),
        "ray_length": 700.0 * scale,
        "ray_samples": 32,
    }


BLOOM = Stage("bloom", "finish", bloom, ("final",), "final", _bloom)


# stage_params keys of the finish stages: the stage's name with spaces as dashes,
# the way engine.stages.catalog.compose names its keys. "contours" reaches both
# contour stages (image and field), whichever the render uses.
FINISH_KEYS: Dict[str, Tuple[Stage, ...]] = {
    "contours": (CONTOURS, FIELD_CONTOURS),
    "glass": (GLASS,),
    "chromatic-aberration": (ABERRATION,),
    "shading": (SHADE,),
    "halftone": (HALFTONE,),
    "bloom": (BLOOM,),
    "film-grain": (GRAIN,),
    "vignette": (VIGNETTE,),
}

# Arguments that are pixel sizes: an override is given for REFERENCE_SHORT_SIDE
# and scaled to the image exactly like the stage's own default.
_PIXEL_ARGUMENTS: Dict[str, Dict[str, Callable[[Any, float], Any]]] = {
    "contours": {"line_width": lambda v, scale: v * scale, "smoothing_radius": lambda v, scale: v * scale},
    "glass": {"blur_radius": _blur_radius},
    "shading": {"refraction": lambda v, scale: v * scale, "smoothing": lambda v, scale: v * scale},
    "halftone": {"cell": _halftone_cell},
    "bloom": {
        "radii": lambda v, scale: tuple(float(r) * scale for r in v),
        "ray_length": lambda v, scale: v * scale,
    },
    "film-grain": {"grain_size": _grain_size},
}


def split_stage_params(
    stage_params: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Splits stage_params into the engine stages' keys and the finish stages' keys."""
    engine: Dict[str, Any] = {}
    finish: Dict[str, Any] = {}
    for key, value in (stage_params or {}).items():
        (finish if key in FINISH_KEYS else engine)[key] = value
    return engine, finish


def _argument_names(stage: Stage) -> Tuple[str, ...]:
    """The keyword arguments of the stage's function (its inputs are positional)."""
    names = list(inspect.signature(stage.fn).parameters)[len(stage.inputs):]
    return tuple(name for name in names if name != "rng")


def check_finish_stage_params(stage_params: Optional[Mapping[str, Any]]) -> None:
    """Raises PlanCompositionError (a ValueError) for an argument a finish stage does not take.

    Keys of engine stages are not looked at here (compose checks those)."""
    from engine.stages.catalog import PlanCompositionError

    for key, overrides in split_stage_params(stage_params)[1].items():
        if not isinstance(overrides, Mapping):
            raise PlanCompositionError(f"stage_params for {key!r} must map argument names to values, not {overrides!r}")
        valid = set.intersection(*(set(_argument_names(stage)) for stage in FINISH_KEYS[key]))
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise PlanCompositionError(
                f"stage {key!r} has no argument {', '.join(unknown)}; arguments: {', '.join(sorted(valid))}"
            )


def _with_finish_overrides(stage: Stage, key: str, overrides: Mapping[str, Any]) -> Stage:
    base_bind = stage.bind
    pixel = _PIXEL_ARGUMENTS.get(key, {})

    def bind(p: StageParams) -> Dict:
        arguments = base_bind(p)
        scale = size_scale(p) if any(name in pixel for name in overrides) else 1.0
        for name, value in overrides.items():
            arguments[name] = pixel[name](value, scale) if name in pixel else value
        return arguments

    return replace(stage, bind=bind)


def finish_stages(p: StageParams) -> Tuple[Stage, ...]:
    """The enabled effects, in their fixed order.

    Contours go first (frosted: the glass softens them) or after the glass and
    aberration (sharp). contour_source "field" draws them from the field. The
    halftone (when p["halftone"] names a mode) comes after the sharp contours and
    before the film grain, so the grain is not printed into dots.

    p["stage_params"] (optional) maps a stage key to argument overrides, like the
    engine stages': contours, glass, chromatic-aberration, shading, bloom, halftone,
    film-grain, vignette (the stage's name with spaces as dashes; "contours" reaches both
    contour stages). Overrides replace the arguments the stage binds from the
    render options, so {"shading": {"light_angle": 45}} turns the light. An
    override never enables a stage: a stage runs only when its own option does
    (shading > 0, noise > 0, ...). An argument the stage does not take raises
    PlanCompositionError. Keys of engine stages are ignored here. Pixel-size
    arguments (line_width, smoothing_radius, blur_radius, refraction, smoothing
    of shading, radii and ray_length of bloom, cell of halftone, grain_size) are given for
    REFERENCE_SHORT_SIDE and scale with the image like the stage's own defaults,
    so a preview matches the full render.

    Order: frosted contours, glass, shading, bloom, chromatic aberration, sharp
    contours, halftone, film grain, vignette.
    """
    with p.reading_for("finish plan"):
        overrides = split_stage_params(p.get("stage_params"))[1]
        check_finish_stage_params(overrides)
        stages: List[Stage] = []
        contours = None
        if p["contours"] > 0.0:
            contours = FIELD_CONTOURS if p.get("contour_source", "image") == "field" else CONTOURS
        sharp = p.get("contour_placement", "frosted") == "sharp"
        if contours is not None and not sharp:
            stages.append(contours)
        if p["glass"]:
            stages.append(GLASS)
        if p.get("shading", 0.0) > 0.0:
            stages.append(SHADE)
        if p.get("bloom", 0.0) > 0.0:
            stages.append(BLOOM)
        if p["chromatic_aberration"] > 0.0:
            stages.append(ABERRATION)
        if contours is not None and sharp:
            stages.append(contours)
        if p.get("halftone"):
            stages.append(HALFTONE)
        if p["noise"] > 0.0:
            stages.append(GRAIN)
        if p["vignette"] > 0.0:
            stages.append(VIGNETTE)
    keys = {stage: key for key, group in FINISH_KEYS.items() for stage in group}
    return tuple(
        _with_finish_overrides(stage, keys[stage], overrides[keys[stage]]) if keys[stage] in overrides else stage
        for stage in stages
    )
