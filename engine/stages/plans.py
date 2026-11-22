"""Stage sequences that reproduce the registry engines.

Each plan is a legacy engine's code split at its stage boundaries, with the
legacy parameter names bound to stage arguments. Output matches the legacy
engine exactly (tests/test_stages.py). A bind function reads only what its
stage uses, so the trace shows which parameters an engine ignores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from engine.parameter_ranges import PARAMETER_RANGES
from engine.stages.export import to_srgb_image
from engine.stages.fields import (
    anchor_points,
    blob_layers,
    elevation,
    folded_noise,
    folded_ramp,
    fractal_field,
    noisy_ramp,
    orbiting_emitters,
    patch_mesh,
    radial_emitters,
    shapes,
    smooth_ramp,
)
from engine.stages.frame import (
    control_lattice,
    fractal_camera,
    full_resolution_grid,
    rotate,
    rotate_about_camera,
    sample_grid,
)
from engine.stages.mapping import bicubic_upscale, blend_srgb, cosine_cycles, map_colors
from engine.stages.palette import make_palette
from engine.stages.runner import Stage, StageParams
from engine.stages.warp import noise_displacement, perlin_displacement, sine_warp

Bind = Callable[[StageParams], Dict]


@dataclass(frozen=True)
class Plan:
    stages: Tuple[Stage, ...]
    # A single color renders as a solid fill before any stage runs.
    solid_if_single_color: bool = False


def _range(key: str) -> Callable[[], float]:
    return lambda: PARAMETER_RANGES.sample(key)


def _palette(p: StageParams) -> Dict:
    return {"colors": p["colors"]}


def _size(p: StageParams) -> Dict:
    return {"width": p["width"], "height": p["height"]}


def _grid(extent: Tuple[float, float], aspect_correct: bool = False) -> Bind:
    return lambda p: {**_size(p), "extent": extent, "aspect_correct": aspect_correct}


def _complexity(p: StageParams) -> Dict:
    return {"complexity": p["complexity"]}


def _no_resize(_: StageParams) -> Dict:
    return {"resize": False}


PALETTE = Stage("palette", "palette", make_palette, (), "palette", _palette)
MAP = Stage("map", "mapping", map_colors, ("field", "palette"), "image")
EXPORT = Stage("export", "export", to_srgb_image, ("image", "coords"), "final")
EXPORT_AS_IS = Stage("export", "export", to_srgb_image, ("image", "coords"), "final", _no_resize)
# The engine classes for domain-warp and topographic accept a fixed angle.
ROTATE = Stage("rotate", "frame", rotate, ("coords",), "coords", lambda p: {"angle": p.get("angle")})


# domain-warp ---------------------------------------------------------------

def _domain_warp_amounts(p: StageParams) -> Tuple[float, float]:
    return (
        float(p.get_or_sample("warp_strength", _range("warp_strength"))),
        float(p.get_or_sample("transverse_warp", _range("transverse_warp"))),
    )


def _domain_warp_warp(p: StageParams) -> Dict:
    warp, transverse = _domain_warp_amounts(p)
    return {"amount_x": warp, "amount_y": transverse, "frequency": 0.5 + (p["complexity"] * 0.08)}


def _domain_warp_field(p: StageParams) -> Dict:
    warp, transverse = _domain_warp_amounts(p)
    return {"warp": warp, "transverse": transverse, "frequency": 0.5 + (p["complexity"] * 0.08)}


DOMAIN_WARP = Plan((
    PALETTE,
    Stage("grid", "frame", sample_grid, (), "coords", _grid((-1.0, 1.0))),
    ROTATE,
    Stage("noise warp", "warp", noise_displacement, ("coords",), "coords", _domain_warp_warp),
    Stage("folded noise", "field", folded_noise, ("coords",), "field", _domain_warp_field),
    MAP,
    EXPORT,
))


# rbf -----------------------------------------------------------------------

def _rbf_warp(p: StageParams) -> Dict:
    value = p["warp_strength"]
    amount = max(0.0, float(1.0 if value is None else value)) * 0.35
    return {
        "amount_x": amount,
        "amount_y": amount,
        "frequency": 0.6 + (max(1, p["complexity"]) * 0.08),
        "skip_when_zero": True,
    }


def _rbf_field(p: StageParams) -> Dict:
    return {
        "complexity": p["complexity"],
        "ambient_weight": p["ambient_weight"],
        "dispersion": p["dispersion"],
    }


RADIAL_BASIS = Plan((
    PALETTE,
    Stage("grid", "frame", sample_grid, (), "coords", _grid((0.0, 1.0))),
    Stage("noise warp", "warp", noise_displacement, ("coords",), "coords", _rbf_warp),
    Stage("radial emitters", "field", radial_emitters, ("coords", "palette"), "field", _rbf_field),
    MAP,
    EXPORT,
), solid_if_single_color=True)


# simplex -------------------------------------------------------------------

SIMPLEX = Plan((
    PALETTE,
    Stage("grid", "frame", sample_grid, (), "coords", _grid((-0.5, 0.5))),
    ROTATE,
    Stage("noisy ramp", "field", noisy_ramp, ("coords",), "field", _complexity),
    MAP,
    EXPORT,
))


# fluid-wave ----------------------------------------------------------------

def _lattice(p: StageParams) -> Dict:
    return {**_size(p), "complexity": p["complexity"]}


FLUID_WAVE = Plan((
    PALETTE,
    Stage("control lattice", "frame", control_lattice, (), "coords", _lattice),
    ROTATE,
    Stage("folded ramp", "field", folded_ramp, ("coords",), "field"),
    MAP,
    Stage("bicubic upscale", "resample", bicubic_upscale, ("image",), "image", _size),
    EXPORT_AS_IS,
))


# spline-wave and flow-field -------------------------------------------------

def _perlin_warp(range_key: str, octaves: int) -> Bind:
    def bind(p: StageParams) -> Dict:
        value = p["warp_strength"]
        strength_range = (
            (float(value), float(value)) if value is not None else PARAMETER_RANGES.get_bounds(range_key)
        )
        return {"strength_range": strength_range, "complexity": p["complexity"], "octaves": octaves}

    return bind


FULL_GRID = Stage("full-resolution grid", "frame", full_resolution_grid, (), "coords", _size)

SPLINE_WAVE = Plan((
    PALETTE,
    FULL_GRID,
    Stage("perlin warp", "warp", perlin_displacement, ("coords",), "coords", _perlin_warp("spline_warp", 2)),
    ROTATE,
    Stage("smooth ramp", "field", smooth_ramp, ("coords",), "field"),
    MAP,
    EXPORT_AS_IS,
))

FLOW_FIELD = Plan((
    PALETTE,
    FULL_GRID,
    Stage("perlin warp", "warp", perlin_displacement, ("coords",), "coords", _perlin_warp("flow_warp", 3)),
    Stage("orbiting emitters", "field", orbiting_emitters, ("coords", "palette"), "field"),
    MAP,
    EXPORT_AS_IS,
))


# topographic ---------------------------------------------------------------

def _elevation(p: StageParams) -> Dict:
    value = p["warp_strength"]
    return {"complexity": p["complexity"], "noise": 0.85 if value is None else value}


TOPOGRAPHIC = Plan((
    PALETTE,
    Stage("grid", "frame", sample_grid, (), "coords", _grid((-1.5, 1.5), aspect_correct=True)),
    ROTATE,
    Stage("elevation", "field", elevation, ("coords",), "field", _elevation),
    MAP,
    EXPORT,
), solid_if_single_color=True)


# idw -----------------------------------------------------------------------

def _idw(p: StageParams) -> Dict:
    # The registry renders idw with power 2.5; MeshGradient can pass its own points, power and softness.
    return {
        "power": p.get("idw_power", 2.5),
        "softness": p.get("idw_softness", 0.12),
        "points": p.get("idw_points"),
    }


IDW = Plan((
    PALETTE,
    FULL_GRID,
    Stage("anchor points", "field", anchor_points, ("coords", "palette"), "field", _idw),
    Stage("blend in sRGB", "mapping", blend_srgb, ("field", "palette"), "image"),
    EXPORT_AS_IS,
))


# patch-mesh ----------------------------------------------------------------

def patch_mesh_size(p: StageParams) -> Dict:
    """Grid size from complexity: 3x2 vertices at 1, 4x3 at 4, 5x4 at 8."""
    complexity = max(1, int(p["complexity"]))
    return {"columns": 3 + complexity // 4, "rows": 2 + complexity // 4, "jitter": 0.3}


PATCH_MESH = Plan((
    PALETTE,
    Stage("grid", "frame", sample_grid, (), "coords", _grid((0.0, 1.0))),
    Stage("patch mesh", "field", patch_mesh, ("coords", "palette"), "field", patch_mesh_size),
    MAP,
    EXPORT,
), solid_if_single_color=True)


# blobs ---------------------------------------------------------------------

def blob_count(p: StageParams) -> Dict:
    """3 blobs at complexity 1, 4 at 3, 6 at 9."""
    return {"blobs": 3 + max(1, int(p["complexity"])) // 3, "blur": 0.5}


def stroke_count(p: StageParams) -> Dict:
    """3 strokes at complexity 1, 4 at 3, 6 at 9."""
    return {"strokes": 3 + max(1, int(p["complexity"])) // 3, "width": 0.9, "blur": 0.5}


def shape_layers(p: StageParams) -> Dict:
    """5 shapes at complexity 1, 6 at 4, 9 at 10, with hard edges (hardness 0.97)."""
    return {"count": min(9, 4 + max(1, int(p["complexity"])) // 2), "hardness": 0.97}


BLOBS = Plan((
    PALETTE,
    Stage("grid", "frame", sample_grid, (), "coords", _grid((0.0, 1.0))),
    Stage("blob layers", "field", blob_layers, ("coords", "palette"), "field", blob_count),
    MAP,
    EXPORT,
), solid_if_single_color=True)


# shapes --------------------------------------------------------------------

# Hard edges need one grid sample per output pixel: a coarser grid enlarged by
# the export would blur the edge and ring around it, so this grid is full size.
SHAPES = Plan((
    PALETTE,
    Stage("grid", "frame", sample_grid, (), "coords", lambda p: {**_grid((0.0, 1.0))(p), "scale": 1.0}),
    Stage("shapes", "field", shapes, ("coords", "palette"), "field", shape_layers),
    MAP,
    EXPORT,
), solid_if_single_color=True)


# fractals ------------------------------------------------------------------

def _fractal_kind(engine: str) -> Callable[[StageParams], str]:
    if engine == "fluid-fractal":
        return lambda p: "fluid"
    if engine == "escape-fractal":
        return lambda p: p["fractal_type"] if p["fractal_type"] in ("julia", "mandelbrot") else "julia"
    return lambda p: p["fractal_type"]


def _fractal_plan(engine: str) -> Plan:
    kind_of = _fractal_kind(engine)

    def camera(p: StageParams) -> Dict:
        return {
            **_size(p),
            "kind": kind_of(p),
            "zoom": p["fractal_zoom"],
            "center": p["fractal_center"],
            "angle": p["fractal_angle"],
        }

    def field(p: StageParams) -> Dict:
        kind = kind_of(p)
        # The fluid field has no orbit trap; the legacy engine ignored trap_weight there.
        trap = p["trap_weight"] if kind != "fluid" and engine != "fluid-fractal" else 0.35
        return {"kind": kind, "complexity": p["complexity"], "trap_weight": trap}

    def cycles(p: StageParams) -> Dict:
        return {
            "frequency": p["frequency"],
            "phase": p["phase_offset"],
            "channel_offset": p["channel_offset"],
        }

    return Plan((
        PALETTE,
        Stage("camera", "frame", fractal_camera, (), "coords", camera),
        Stage("sine warp", "warp", sine_warp, ("coords",), "coords"),
        Stage("into camera frame", "frame", rotate_about_camera, ("coords",), "coords"),
        Stage("fractal field", "field", fractal_field, ("coords",), "field", field),
        Stage("cosine cycles", "mapping", cosine_cycles, ("field", "palette"), "image", cycles),
        EXPORT,
    ), solid_if_single_color=True)


STAGED_PLANS: Dict[str, Plan] = {
    "simplex": SIMPLEX,
    "flow-field": FLOW_FIELD,
    "fluid-wave": FLUID_WAVE,
    "spline-wave": SPLINE_WAVE,
    "domain-warp": DOMAIN_WARP,
    "rbf": RADIAL_BASIS,
    "radial-basis": RADIAL_BASIS,
    "topographic": TOPOGRAPHIC,
    "fractal": _fractal_plan("fractal"),
    "fluid-fractal": _fractal_plan("fluid-fractal"),
    "escape-fractal": _fractal_plan("escape-fractal"),
    "idw": IDW,
    "patch-mesh": PATCH_MESH,
    "blobs": BLOBS,
    "shapes": SHAPES,
}
