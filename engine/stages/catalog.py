"""Named stages that can replace or extend phases of an engine's plan.

An engine's plan gives every phase a default. compose() lets a render swap the
field or the mapping, add warps, and override any stage's arguments:

    --warp twist --warp wave     add warps after the engine's own, in order
    --warp none --warp bulge     drop the engine's warps first
    --field voronoi              replace the field
    --mapping bands              replace the mapping
    --blend conic:difference     compute a second field and combine it with the
                                 first (modes: add, multiply, screen, difference,
                                 overlay, min, max; default screen)
    --stage-param twist.amount=3 set an argument of a stage (catalogue name or
                                 plan stage name with spaces as dashes)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from engine.stages import fields, mapping, warp
from engine.stages.runner import Stage, StageParams


@dataclass(frozen=True)
class Choice:
    """A catalogue stage and a one-line description for listings."""

    stage: Stage
    summary: str


class PlanCompositionError(ValueError):
    pass


def _fixed(**arguments: Any) -> Callable[[StageParams], Dict[str, Any]]:
    return lambda p: dict(arguments)


def _complexity_field(p: StageParams) -> Dict[str, Any]:
    return {"complexity": p["complexity"]}


def _patch_mesh_size(p: StageParams) -> Dict[str, Any]:
    from engine.stages.plans import patch_mesh_size

    return patch_mesh_size(p)


def _blob_count(p: StageParams) -> Dict[str, Any]:
    from engine.stages.plans import blob_count

    return blob_count(p)


def _w(name: str, fn: Callable, summary: str, **defaults: Any) -> Choice:
    return Choice(Stage(name, "warp", fn, ("coords",), "coords", _fixed(**defaults)), summary)


def _f(
    name: str,
    fn: Callable,
    summary: str,
    inputs: Tuple[str, ...] = ("coords",),
    bind: Callable[[StageParams], Dict[str, Any]] = _fixed(),
) -> Choice:
    return Choice(Stage(name, "field", fn, inputs, "field", bind), summary)


def _m(name: str, fn: Callable, summary: str, **defaults: Any) -> Choice:
    return Choice(Stage(name, "mapping", fn, ("field", "palette"), "image", _fixed(**defaults)), summary)


WARPS: Dict[str, Choice] = {
    "noise": _w("noise", warp.noise_displacement, "shift by two simplex-noise planes",
                amount_x=0.4, amount_y=0.4, frequency=0.9),
    "sine": _w("sine", warp.sine_warp, "a fixed smooth sinusoidal bend", amplitude=0.18),
    "twist": _w("twist", warp.twist, "swirl around the center", amount=2.0, radius=1.0),
    "bulge": _w("bulge", warp.bulge, "lens magnify (or pinch, amount < 0) at the center", amount=0.5, radius=1.0),
    "wave": _w("wave", warp.wave, "sideways sine ripple", amplitude=0.08, waves=3.0, angle=0.0),
    "sphere": _w("sphere", warp.sphere, "wrap the center disc around a sphere", amount=1.0, radius=1.0),
    "lens": _w("lens", warp.lens, "barrel (amount > 0) or pincushion (< 0) lens distortion", amount=0.3),
}

FIELDS: Dict[str, Choice] = {
    "noisy-ramp": _f("noisy-ramp", fields.noisy_ramp, "a ramp plus simplex noise (simplex)", bind=_complexity_field),
    "folded-ramp": _f("folded-ramp", fields.folded_ramp, "a ramp with one fixed fold (fluid-wave)"),
    "smooth-ramp": _f("smooth-ramp", fields.smooth_ramp, "a percentile-stretched smoothstep ramp (spline-wave)"),
    "folded-noise": _f("folded-noise", fields.folded_noise, "a ramp folded by noise (domain-warp)",
                       bind=_fixed(warp=0.6, transverse=0.3, frequency=0.82)),
    "elevation": _f("elevation", fields.elevation, "interfering sines as terrain (topographic)",
                    bind=lambda p: {"complexity": p["complexity"], "noise": 0.85}),
    "radial-emitters": _f("radial-emitters", fields.radial_emitters, "soft color spots over color 0 (rbf)",
                          inputs=("coords", "palette"),
                          bind=lambda p: {"complexity": p["complexity"], "ambient_weight": 0.15, "dispersion": 0.35}),
    "orbiting-emitters": _f("orbiting-emitters", fields.orbiting_emitters, "inverse-square emitters on a ring (flow-field)",
                            inputs=("coords", "palette")),
    "anchor-points": _f("anchor-points", fields.anchor_points, "inverse distance between random anchors (idw)",
                        inputs=("coords", "palette"), bind=_fixed(power=2.5)),
    "fractal": _f("fractal", fields.fractal_field, "fluid or escape-time fractal (fractal engines)",
                  bind=lambda p: {"kind": "fluid", "complexity": p["complexity"], "trap_weight": 0.35}),
    "conic": _f("conic", fields.conic, "sweep around the center", bind=_fixed(turns=1.0)),
    "spiral": _f("spiral", fields.spiral, "conic sweep advancing with distance", bind=_fixed(turns=1.0, tightness=1.5)),
    "radial": _f("radial", fields.radial, "distance from the center"),
    "voronoi": _f("voronoi", fields.voronoi, "cells around random points (mode: distance, edges, f2, cells)",
                  bind=lambda p: {"cells": max(4, 3 * p["complexity"]), "mode": "distance"}),
    "patch-mesh": _f("patch-mesh", fields.patch_mesh, "colored control grid blended bicubically (patch-mesh)",
                     inputs=("coords", "palette"), bind=lambda p: _patch_mesh_size(p)),
    "blob-layers": _f("blob-layers", fields.blob_layers, "soft blobs stacked over the darkest color (blobs)",
                      inputs=("coords", "palette"), bind=lambda p: _blob_count(p)),
    "ridged": _f("ridged", fields.ridged_noise, "ridged multifractal noise, sharp crests",
                 bind=_fixed(scale=1.5, octaves=4, sharpness=2.0)),
    "billow": _f("billow", fields.billow_noise, "billow noise, rounded puffs with creases",
                 bind=_fixed(scale=1.5, octaves=4)),
    "fbm": _f("fbm", fields.fbm_noise, "plain fractal noise (gain, lacunarity, octaves)",
              bind=_fixed(scale=1.2, octaves=5)),
}

MAPPINGS: Dict[str, Choice] = {
    "ramp": _m("ramp", mapping.map_colors, "palette in order (scalar) or weighted blend (weights), in Oklab"),
    "bands": _m("bands", mapping.bands, "palette repeated in bands (repeat: mirror or repeat)",
                cycles=3.0, repeat="mirror", phase=0.0),
    "cycles": _m("cycles", mapping.cosine_cycles, "cosine cycles with separate lightness (fractals)",
                 frequency=1.0, phase=0.0, channel_offset=0.0),
    "srgb-blend": _m("srgb-blend", mapping.blend_srgb, "weighted blend in sRGB (idw)"),
}

# Field kinds and needs, and mapping inputs, keyed by stage function so engine
# plan stages and catalogue stages are checked alike.
_FIELD_KIND = {
    fields.noisy_ramp: "scalar", fields.folded_ramp: "scalar", fields.smooth_ramp: "scalar",
    fields.folded_noise: "scalar", fields.elevation: "scalar", fields.fractal_field: "scalar",
    fields.conic: "scalar", fields.spiral: "scalar", fields.radial: "scalar", fields.voronoi: "scalar",
    fields.ridged_noise: "scalar", fields.billow_noise: "scalar", fields.fbm_noise: "scalar",
    fields.radial_emitters: "weights", fields.orbiting_emitters: "weights", fields.anchor_points: "weights",
    fields.patch_mesh: "weights", fields.blob_layers: "weights",
}
_FIELD_REQUIRES = {fields.folded_noise: ("displacement",), fields.fractal_field: ("camera",)}
_MAPPING_ACCEPTS = {
    mapping.map_colors: ("scalar", "weights"), mapping.bands: ("scalar",),
    mapping.cosine_cycles: ("scalar",), mapping.blend_srgb: ("weights",),
}


def _provides(stage: Stage) -> Tuple[str, ...]:
    from engine.stages.frame import fractal_camera

    if stage.fn is fractal_camera:
        return ("camera",)
    if stage.fn is warp.noise_displacement:
        return ("displacement",)
    return ()


def slug(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def parse_stage_params(items: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Parses "stage.key=value" items; values are JSON when they parse, else strings."""
    result: Dict[str, Dict[str, Any]] = {}
    for item in items:
        target, sep, raw = item.partition("=")
        stage_name, dot, key = target.rpartition(".")
        if not sep or not dot or not stage_name or not key:
            raise PlanCompositionError(f"stage parameter must look like stage.key=value, not {item!r}")
        try:
            value: Any = json.loads(raw)
        except ValueError:
            value = raw
        result.setdefault(slug(stage_name), {})[key.strip()] = value
    return result


def _with_overrides(stage: Stage, overrides: Mapping[str, Any]) -> Stage:
    base_bind = stage.bind

    def bind(p: StageParams) -> Dict[str, Any]:
        return {**base_bind(p), **overrides}

    return replace(stage, bind=bind)


def parse_blend(blend: str) -> Tuple[str, str]:
    """Splits "FIELD[:MODE]" (mode defaults to screen) and checks both names."""
    name, _, mode = blend.partition(":")
    name, mode = name.strip(), (mode.strip() or "screen")
    if name not in FIELDS:
        raise PlanCompositionError(f"unknown blend field {name!r}; choose from {', '.join(FIELDS)}")
    if mode not in fields.BLEND_MODES:
        raise PlanCompositionError(f"unknown blend mode {mode!r}; choose from {', '.join(fields.BLEND_MODES)}")
    return name, mode


def _check_requires(field_stage: Stage, available: set) -> None:
    missing = [r for r in _FIELD_REQUIRES.get(field_stage.fn, ()) if r not in available]
    if missing:
        hint = {"camera": "use a fractal engine", "displacement": "add --warp noise"}
        raise PlanCompositionError(
            f"field {field_stage.name!r} needs {', '.join(missing)} ({'; '.join(hint[m] for m in missing)})"
        )


def _blend_stages(blend: str, field_stage: Stage, kind: str) -> Tuple[Stage, Stage]:
    """The second field, writing field_b, and the stage that combines it into field.

    The second field's stage is named "blend-<field>" and the combining stage
    "blend", so --stage-param can reach both (blend.mode, blend.amount,
    blend.normalize, blend-conic.turns)."""
    name, mode = parse_blend(blend)
    other = FIELDS[name].stage
    if kind != "scalar" or _FIELD_KIND.get(other.fn, "scalar") != "scalar":
        raise PlanCompositionError(
            f"--blend combines two scalar fields; {field_stage.name!r} gives {kind}, "
            f"{name!r} gives {_FIELD_KIND.get(other.fn, 'scalar')}"
        )
    second = replace(other, name=f"blend {name}", output="field_b")
    combine = Stage("blend", "field", fields.blend_fields, ("field", "field_b"), "field", _fixed(mode=mode))
    return second, combine


def compose(
    stages: Sequence[Stage],
    *,
    warps: Optional[Sequence[str]] = None,
    field: Optional[str] = None,
    mapping_name: Optional[str] = None,
    stage_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    blend: Optional[str] = None,
) -> Tuple[Stage, ...]:
    """The plan's stages with the requested replacements, checked for fit."""
    result: List[Stage] = list(stages)
    field_index = next(i for i, s in enumerate(result) if s.phase == "field")

    if field:
        if field not in FIELDS:
            raise PlanCompositionError(f"unknown field {field!r}; choose from {', '.join(FIELDS)}")
        result[field_index] = FIELDS[field].stage

    if warps:
        additions: List[Stage] = []
        for name in warps:
            if name == "none":
                result = [s for s in result if s.phase != "warp"]
                additions = []
                continue
            if name not in WARPS:
                raise PlanCompositionError(f"unknown warp {name!r}; choose from none, {', '.join(WARPS)}")
            additions.append(WARPS[name].stage)
        field_index = next(i for i, s in enumerate(result) if s.phase == "field")
        result[field_index:field_index] = additions

    field_index = next(i for i, s in enumerate(result) if s.phase == "field")
    field_stage = result[field_index]
    kind = _FIELD_KIND.get(field_stage.fn, "scalar")

    mapping_indexes = [i for i, s in enumerate(result) if s.phase == "mapping"]
    if mapping_name:
        if mapping_name not in MAPPINGS:
            raise PlanCompositionError(f"unknown mapping {mapping_name!r}; choose from {', '.join(MAPPINGS)}")
        choice = MAPPINGS[mapping_name]
        accepts = _MAPPING_ACCEPTS[choice.stage.fn]
        if kind not in accepts:
            raise PlanCompositionError(
                f"mapping {mapping_name!r} needs a {' or '.join(accepts)} field, "
                f"but {field_stage.name!r} gives {kind}"
            )
        result[mapping_indexes[0]] = choice.stage
    else:
        current = result[mapping_indexes[0]]
        if kind not in _MAPPING_ACCEPTS.get(current.fn, ("scalar", "weights")):
            # The engine's own mapping cannot color this field; fall back to the plain ramp/blend.
            result[mapping_indexes[0]] = MAPPINGS["ramp"].stage

    available = {feature for s in result[:field_index] for feature in _provides(s)}
    _check_requires(field_stage, available)

    if blend:
        blend_stages = _blend_stages(blend, field_stage, kind)
        _check_requires(blend_stages[0], available)
        result[field_index + 1:field_index + 1] = blend_stages

    if any(s.fn is mapping.blend_srgb for s in result) and any(s.phase == "resample" for s in result):
        raise PlanCompositionError("srgb-blend cannot be enlarged afterwards; pick another mapping")

    if stage_params:
        known = {slug(s.name) for s in result}
        unknown = sorted(set(stage_params) - known)
        if unknown:
            raise PlanCompositionError(
                f"no stage named {', '.join(unknown)} in this render; stages: {', '.join(sorted(known))}"
            )
        result = [
            _with_overrides(s, stage_params[slug(s.name)]) if slug(s.name) in stage_params else s for s in result
        ]
    return tuple(result)
