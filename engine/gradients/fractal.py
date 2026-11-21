"""Fractal engines; each runs its stage plan (engine.stages.plans)."""

from typing import Optional, Sequence, Tuple

from PIL import Image

from engine.gradients.staged import StagedEngine


class FractalFieldGradient(StagedEngine):
    plan = "fractal"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        fractal_type: str = "fluid",
        zoom: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
        angle: Optional[float] = None,
        phase_offset: float = 0.0,
        frequency: float = 1.0,
        trap_weight: float = 0.35,
        channel_offset: float = 0.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """A fluid fBm fractal, or with another fractal_type an escape-time fractal."""
        return self._render(
            colors,
            complexity=complexity,
            fractal_type=fractal_type,
            fractal_zoom=zoom,
            fractal_center=center,
            fractal_angle=angle,
            phase_offset=phase_offset,
            frequency=frequency,
            trap_weight=trap_weight,
            channel_offset=channel_offset,
            seed=seed,
        )


class FluidFractalGradient(FractalFieldGradient):
    """The fluid fBm fractal only."""

    plan = "fluid-fractal"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        zoom: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
        angle: Optional[float] = None,
        phase_offset: float = 0.0,
        frequency: float = 1.0,
        channel_offset: float = 0.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        return super().generate(
            colors,
            complexity=complexity,
            zoom=zoom,
            center=center,
            angle=angle,
            phase_offset=phase_offset,
            frequency=frequency,
            channel_offset=channel_offset,
            seed=seed,
        )


class EscapeTimeFractalGradient(FractalFieldGradient):
    """Julia and Mandelbrot sets (any other fractal_type renders a Julia set)."""

    plan = "escape-fractal"

    def generate(
        self,
        colors: Sequence[Sequence[float]],
        complexity: int = 5,
        fractal_type: str = "julia",
        zoom: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
        angle: Optional[float] = None,
        phase_offset: float = 0.0,
        frequency: float = 1.0,
        trap_weight: float = 0.35,
        channel_offset: float = 0.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        return super().generate(
            colors,
            complexity=complexity,
            fractal_type=fractal_type,
            zoom=zoom,
            center=center,
            angle=angle,
            phase_offset=phase_offset,
            frequency=frequency,
            trap_weight=trap_weight,
            channel_offset=channel_offset,
            seed=seed,
        )


FractalGradient = FractalFieldGradient
