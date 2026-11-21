"""Stage-based rendering: explicit stages with typed inputs and outputs.

render_staged() renders the engines in STAGED_PLANS through their stages and can
record a Trace of every stage's output and the parameters it read. Engines not
ported yet keep using engine.gradients.registry.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional

import numpy as np
from PIL import Image

from engine.gradients.interpolation import validate_palette_colors
from engine.stages.plans import STAGED_PLANS, Plan
from engine.stages.runner import Stage, StageParams, StageRecord, Trace, run_stages
from engine.stages.types import Camera, Coords, Field, OklabImage, Palette, ScalarField, SrgbImage, WeightField


def has_staged_plan(engine: Optional[str]) -> bool:
    return engine in STAGED_PLANS


def render_staged(
    engine: str,
    params: Mapping[str, Any],
    *,
    trace: Optional[Trace] = None,
) -> Image.Image:
    """Renders engine from render_gradient-style params through its stage plan."""
    plan = STAGED_PLANS[engine]
    stage_params = StageParams(params)
    if trace is not None:
        trace.watch(stage_params)
    if plan.reseeds:
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

    state = run_stages(plan.stages, stage_params, rng=random, trace=trace)
    return state["final"]


__all__ = [
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
    "Trace",
    "WeightField",
    "has_staged_plan",
    "render_staged",
]
