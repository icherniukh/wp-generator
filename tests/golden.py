"""Golden renders of every registry engine and preset through the current pipeline.

The snapshots pin today's pixels so refactors can prove they moved code without
changing output. Regenerate them only for an intended visual change:

    uv run python -m tests.golden --update
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from engine.gradients.registry import GRADIENT_REGISTRY, render_gradient
from engine.pipeline import RenderConfiguration, WallpaperPipeline

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_renders.npz"

SEED = 1234
PALETTE = [(18, 24, 64), (220, 80, 120), (250, 190, 90), (60, 170, 200)]
# No filters, so engine cases show only the gradient itself.
NO_FINISH: Dict[str, Any] = dict(
    noise=0.0, vignette=0.0, contours=0.0, glass=False, chromatic_aberration=0.0
)


@dataclass(frozen=True)
class GoldenCase:
    """A pipeline render, or with direct=True a bare render_gradient call."""

    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    direct: bool = False

    def build(self) -> RenderConfiguration:
        return RenderConfiguration(**self.config)


def _engine_case(name: str, engine: str, width: int = 96, height: int = 60, **extra: Any) -> GoldenCase:
    config = dict(
        engine=engine,
        colors=list(PALETTE),
        complexity=4,
        harmonize="none",
        width=width,
        height=height,
        seed=SEED,
        **NO_FINISH,
    )
    config.update(extra)
    return GoldenCase(name, config)


def golden_cases() -> List[GoldenCase]:
    cases: List[GoldenCase] = []
    for engine in sorted(GRADIENT_REGISTRY):
        cases.append(_engine_case(f"engine/{engine}", engine))
        # Every engine branch the stage plans reproduce: warp off, explicit
        # values, and wide and portrait screens.
        cases.append(_engine_case(f"engine/{engine}/no-warp", engine, warp_strength=0.0, transverse_warp=0.0))
        cases.append(
            _engine_case(
                f"engine/{engine}/explicit",
                engine,
                complexity=6,
                warp_strength=0.9,
                transverse_warp=0.5,
                ambient_weight=0.3,
                dispersion=0.25,
                frequency=1.7,
                phase_offset=0.2,
                channel_offset=0.1,
                trap_weight=0.7,
            )
        )
        cases.append(_engine_case(f"engine/{engine}/32x9", engine, width=128, height=36))
        cases.append(_engine_case(f"engine/{engine}/portrait", engine, width=40, height=90))
    for fractal_type in ("fluid", "julia", "mandelbrot", "orbit_trap"):
        cases.append(_engine_case(f"engine/fractal/{fractal_type}", "fractal", fractal_type=fractal_type))
    cases.append(
        _engine_case(
            "engine/fractal/camera",
            "fractal",
            fractal_type="fluid",
            fractal_zoom=1.4,
            fractal_center=(0.2, -0.1),
            fractal_angle=0.7,
        )
    )
    cases.append(
        _engine_case(
            "engine/escape-fractal/camera",
            "escape-fractal",
            fractal_type="mandelbrot",
            fractal_zoom=12.0,
            fractal_center=(-0.745, 0.11),
            fractal_angle=0.3,
        )
    )
    # Height-field shading from a scalar field and from a weight field's lightness.
    cases.append(_engine_case("shading/simplex", "simplex", width=120, height=80, shading=0.6))
    cases.append(_engine_case("shading/blobs", "blobs", width=120, height=80, shading=0.6))
    # A single color is a solid fill for some engines; the pipeline never passes one.
    for engine in ("rbf", "topographic", "fractal"):
        cases.append(
            GoldenCase(
                f"gradient/{engine}/single-color",
                dict(engine=engine, colors=[(40, 90, 160)], width=32, height=20, seed=SEED),
                direct=True,
            )
        )
    for fractal_type in ("julia", "mandelbrot", "orbit_trap"):
        cases.append(
            GoldenCase(
                f"engine/escape-fractal/{fractal_type}",
                dict(
                    engine="escape-fractal",
                    fractal_type=fractal_type,
                    colors=list(PALETTE),
                    complexity=4,
                    harmonize="none",
                    width=96,
                    height=60,
                    seed=SEED,
                    **NO_FINISH,
                ),
            )
        )
    # Catalogue stages composed onto an engine's plan.
    cases.append(_engine_case("catalog/warp/liquify", "simplex", warps=["liquify"]))
    cases.append(_engine_case("catalog/warp/liquify/blobs", "blobs", warps=["liquify"]))
    cases.append(_engine_case("catalog/field/strokes", "simplex", field_stage="strokes"))
    cases.append(_engine_case("catalog/field/strokes/liquify", "simplex", field_stage="strokes", warps=["liquify"]))
    # Noise on the field before mapping: a scalar field, weight fields, a coarse lattice
    # and a blended field.
    for engine in ("simplex", "blobs", "fluid-wave", "fractal"):
        cases.append(_engine_case(f"field-grain/{engine}", engine, field_grain=0.15))
    cases.append(_engine_case("field-grain/blend", "simplex", field_stage="conic", blend="radial:difference", field_grain=0.1))
    # Film grain options; grain_size is covered in tests/test_grain_controls.py, since it
    # scales with the image and this size is too small for it to change anything.
    for grain_name, grain in (
        ("color", dict(grain_color="color")),
        ("overlay", dict(grain_blend="overlay")),
        ("soft-light", dict(grain_blend="soft-light")),
        ("multiply", dict(grain_blend="multiply")),
        ("color-soft-light", dict(grain_color="color", grain_blend="soft-light")),
    ):
        cases.append(_engine_case(f"grain/{grain_name}", "simplex", noise=0.1, **grain))
    # Presets keep their own sampled parameters and finish, so filters are covered too.
    for preset_path in sorted((Path(__file__).parent.parent / "presets").glob("*.json")):
        if preset_path.name == "parameter_ranges.json":
            continue
        cases.append(
            GoldenCase(
                f"preset/{preset_path.stem}",
                dict(preset=preset_path.stem, width=120, height=80, seed=SEED),
            )
        )
    return cases


def render_case(case: GoldenCase, pipeline: WallpaperPipeline | None = None) -> np.ndarray:
    if case.direct:
        cfg = dict(case.config)
        image = render_gradient(
            cfg.pop("engine"), width=cfg.pop("width"), height=cfg.pop("height"), colors=cfg.pop("colors"), **cfg
        )
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    pipeline = pipeline or WallpaperPipeline()
    return np.asarray(pipeline.render(case.build()).image.convert("RGB"), dtype=np.uint8)


def render_all() -> Dict[str, np.ndarray]:
    pipeline = WallpaperPipeline()
    return {case.name: render_case(case, pipeline) for case in golden_cases()}


def load_golden() -> Dict[str, np.ndarray]:
    with np.load(GOLDEN_PATH) as data:
        return {name: data[name] for name in data.files}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite the golden file")
    args = parser.parse_args(argv)

    renders = render_all()
    if args.update:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN_PATH, **renders)
        print(f"wrote {len(renders)} renders to {GOLDEN_PATH}")
        return 0

    golden = load_golden()
    changed = 0
    for name, image in renders.items():
        ref = golden.get(name)
        if ref is None or ref.shape != image.shape:
            print(f"{name}: missing or shape changed")
            changed += 1
            continue
        diff = np.abs(ref.astype(np.int16) - image.astype(np.int16))
        if diff.max():
            changed += 1
            print(f"{name}: max diff {diff.max()}, {np.mean(diff.max(axis=2) > 0):.2%} of pixels")
    print(f"{changed} of {len(renders)} renders differ")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
