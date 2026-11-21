"""Ramp and anchor engines; each runs its stage plan (engine.stages.plans)."""

from typing import List, Optional, Sequence, Tuple

from PIL import Image

from engine.gradients.staged import StagedEngine


class FluidMeshGradient(StagedEngine):
    plan = "fluid-wave"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        fluid_mode: str = "random",
    ) -> Image.Image:
        """A folded ramp on a coarse lattice, colored and then enlarged bicubically."""
        return self._render(colors, complexity=complexity, fluid_mode=fluid_mode)


class SimplexMeshGradient(StagedEngine):
    plan = "simplex"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        fluid_mode: str = "random",
        seed: Optional[int] = None,
    ) -> Image.Image:
        """A ramp plus simplex noise, colored in Oklab."""
        return self._render(colors, complexity=complexity, fluid_mode=fluid_mode, seed=seed)


class PatchMeshGradient(StagedEngine):
    plan = "patch-mesh"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """A grid of colored control points blended by a bicubic spline, in Oklab."""
        return self._render(colors, complexity=complexity, seed=seed)


class MeshGradient(StagedEngine):
    plan = "idw"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        points: Optional[List[Tuple[float, float]]] = None,
        p: float = 2.0,
        softness: float = 0.12,
    ) -> Image.Image:
        """Inverse distance weighting between one anchor per color, blended in sRGB."""
        return self._render(colors, idw_points=points, idw_power=p, idw_softness=softness)
