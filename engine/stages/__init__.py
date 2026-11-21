"""Stage-based rendering: explicit stages with typed inputs and outputs.

render_staged() renders the engines in STAGED_PLANS through their stages and can
record a Trace of every stage's output and the parameters it read. Engines not
ported yet keep using engine.gradients.registry.
"""

from __future__ import annotations

import random
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from PIL import Image

from engine.gradients.interpolation import validate_palette_colors
from engine.stages.plans import STAGED_PLANS, Plan
from engine.stages.runner import Stage, StageParams, StageRecord, Trace, run_stages
from engine.stages.types import Camera, Coords, Field, OklabImage, Palette, ScalarField, SrgbImage, WeightField


def has_staged_plan(engine: Optional[str]) -> bool:
    return engine in STAGED_PLANS


@dataclass(frozen=True)
class Composition:
    """Changes to an engine's plan: extra warps, a replacement field or mapping,
    and argument overrides per stage (see engine.stages.catalog.compose)."""

    warps: Tuple[str, ...] = ()
    field: Optional[str] = None
    mapping: Optional[str] = None
    stage_params: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.warps or self.field or self.mapping or self.stage_params)


def render_staged(
    engine: str,
    params: Mapping[str, Any],
    *,
    trace: Optional[Trace] = None,
    composition: Optional[Composition] = None,
) -> Image.Image:
    """Renders engine from render_gradient-style params through its stage plan,
    changed by composition if given."""
    plan = STAGED_PLANS[engine]
    stages = plan.stages
    if composition:
        from engine.stages.catalog import compose

        stages = compose(
            stages,
            warps=composition.warps,
            field=composition.field,
            mapping_name=composition.mapping,
            stage_params=composition.stage_params,
        )
    stage_params = StageParams(params)
    if trace is not None:
        trace.watch(stage_params)
    # Reseed from the seed, so the engine's draws depend only on it and not on
    # how many draws the pipeline made before.
    with stage_params.reading_for("seed"):
        seed = stage_params.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if plan.solid_if_single_color:
        with stage_params.reading_for("palette"):
            colors = stage_params["colors"]
        solid = validate_palette_colors(colors, params["width"], params["height"])
        if solid is not None:
            return solid

    state = run_stages(stages, stage_params, rng=random, trace=trace)
    return state["final"]


__all__ = [
    "Camera",
    "Composition",
    "Coords",
    "Field",
    "OklabImage",
    "Palette",
    "Plan",
    "ScalarField",
    "Stage",
    "StageParams",
    "StageRecord",
    "STAGED_PLANS",
    "SrgbImage",
    "Trace",
    "WeightField",
    "has_staged_plan",
    "render_staged",
]
