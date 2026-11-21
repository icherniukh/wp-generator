"""Field-based engines; each runs its stage plan (engine.stages.plans)."""

from typing import Optional, Sequence

from PIL import Image

from engine.gradients.staged import StagedEngine


class DomainWarpGradient(StagedEngine):
    plan = "domain-warp"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        warp_strength: Optional[float] = None,
        transverse_warp: Optional[float] = None,
        angle: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """A ramp folded by nested simplex-noise domain warping, colored in Oklab."""
        return self._render(
            colors,
            complexity=complexity,
            warp_strength=warp_strength,
            transverse_warp=transverse_warp,
            angle=angle,
            seed=seed,
        )


class RadialBasisGradient(StagedEngine):
    plan = "rbf"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        warp_strength: float = 1.0,
        ambient_weight: float = 0.15,
        dispersion: float = 0.35,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Soft color spots over the first color, on noise-warped coordinates."""
        return self._render(
            colors,
            complexity=complexity,
            warp_strength=warp_strength,
            ambient_weight=ambient_weight,
            dispersion=dispersion,
            seed=seed,
        )


RBFFieldGradient = RadialBasisGradient


class FlowFieldGradient(StagedEngine):
    plan = "flow-field"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        fluid_mode: str = "random",
        warp_strength: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Inverse-square color emitters on Perlin-warped coordinates."""
        return self._render(
            colors, complexity=complexity, fluid_mode=fluid_mode, warp_strength=warp_strength, seed=seed
        )


class TopographicGradient(StagedEngine):
    plan = "topographic"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        warp_strength: float = 0.85,
        angle: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Interfering sine waves as terrain height, mapped through the palette."""
        return self._render(colors, complexity=complexity, warp_strength=warp_strength, angle=angle, seed=seed)
