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

from engine.gradients.registry import GRADIENT_REGISTRY
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
    name: str
    config: Dict[str, Any] = field(default_factory=dict)

    def build(self) -> RenderConfiguration:
        return RenderConfiguration(**self.config)


def golden_cases() -> List[GoldenCase]:
    cases: List[GoldenCase] = []
    for engine in sorted(GRADIENT_REGISTRY):
        cases.append(
            GoldenCase(
                f"engine/{engine}",
                dict(
                    engine=engine,
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
