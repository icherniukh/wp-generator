"""Spline-wave engine; runs its stage plan (engine.stages.plans)."""

from typing import Optional, Sequence

from PIL import Image

from engine.gradients.staged import StagedEngine


class SplineWaveGradient(StagedEngine):
    plan = "spline-wave"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        warp_strength: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """A smoothstepped ramp on Perlin-warped coordinates, colored in Oklab."""
        return self._render(colors, complexity=complexity, warp_strength=warp_strength, seed=seed)
