"""Stage sequences that reproduce the registry engines ported so far.

Each plan is the legacy engine's code split at its stage boundaries, with the
legacy parameter names bound to stage arguments. Output matches the legacy
engine exactly (tests/test_stages.py).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from engine.parameter_ranges import PARAMETER_RANGES
from engine.stages.export import to_srgb_image
from engine.stages.fields import folded_noise, radial_emitters
from engine.stages.frame import rotate, sample_grid
from engine.stages.mapping import map_colors
from engine.stages.palette import make_palette
from engine.stages.runner import Stage, StageParams
from engine.stages.warp import noise_displacement

Plan = Tuple[Stage, ...]


def _range(key: str) -> Callable[[], float]:
    return lambda: PARAMETER_RANGES.sample(key)


def _palette(p: StageParams) -> Dict:
    return {"colors": p["colors"]}


def _output_size(extent: Tuple[float, float]) -> Callable[[StageParams], Dict]:
    return lambda p: {"width": p["width"], "height": p["height"], "extent": extent}


PALETTE = Stage("palette", "palette", make_palette, (), "palette", _palette)
MAP = Stage("map", "mapping", map_colors, ("field", "palette"), "image")
EXPORT = Stage("export", "export", to_srgb_image, ("image", "coords"), "final")


def _domain_warp_warp(p: StageParams) -> Dict:
    return {
        "amount_x": float(p.get_or_sample("warp_strength", _range("warp_strength"))),
        "amount_y": float(p.get_or_sample("transverse_warp", _range("transverse_warp"))),
        "frequency": 0.5 + (p["complexity"] * 0.08),
    }


def _domain_warp_field(p: StageParams) -> Dict:
    return {
        "warp": float(p.get_or_sample("warp_strength", _range("warp_strength"))),
        "transverse": float(p.get_or_sample("transverse_warp", _range("transverse_warp"))),
        "frequency": 0.5 + (p["complexity"] * 0.08),
    }


DOMAIN_WARP: Plan = (
    PALETTE,
    Stage("grid", "frame", sample_grid, (), "coords", _output_size((-1.0, 1.0))),
    Stage("rotate", "frame", rotate, ("coords",), "coords"),
    Stage("noise warp", "warp", noise_displacement, ("coords",), "coords", _domain_warp_warp),
    Stage("folded noise", "field", folded_noise, ("coords",), "field", _domain_warp_field),
    MAP,
    EXPORT,
)


def _rbf_strength(p: StageParams) -> float:
    value = p["warp_strength"]
    return max(0.0, float(1.0 if value is None else value))


def _rbf_warp(p: StageParams) -> Dict:
    amount = _rbf_strength(p) * 0.35
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


RADIAL_BASIS: Plan = (
    PALETTE,
    Stage("grid", "frame", sample_grid, (), "coords", _output_size((0.0, 1.0))),
    Stage("noise warp", "warp", noise_displacement, ("coords",), "coords", _rbf_warp),
    Stage("radial emitters", "field", radial_emitters, ("coords", "palette"), "field", _rbf_field),
    MAP,
    EXPORT,
)

# A single color renders as a solid fill before any stage runs, as in the legacy engine.
SOLID_IF_SINGLE_COLOR = {"rbf", "radial-basis"}

STAGED_PLANS: Dict[str, Plan] = {
    "domain-warp": DOMAIN_WARP,
    "rbf": RADIAL_BASIS,
    "radial-basis": RADIAL_BASIS,
}
