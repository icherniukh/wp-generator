from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type
from PIL import Image

from engine.gradients.field import (
    DomainWarpGradient,
    FlowFieldGradient,
    RadialBasisGradient,
    TopographicGradient,
)
from engine.gradients.fractal import FractalFieldGradient
from engine.gradients.mesh import FluidMeshGradient, MeshGradient, SimplexMeshGradient
from engine.gradients.spline import SplineWaveGradient

GRADIENT_REGISTRY: Dict[str, Type[Any]] = {
    "simplex": SimplexMeshGradient,
    "flow-field": FlowFieldGradient,
    "fluid-wave": FluidMeshGradient,
    "spline-wave": SplineWaveGradient,
    "domain-warp": DomainWarpGradient,
    "rbf": RadialBasisGradient,
    "radial-basis": RadialBasisGradient,
    "topographic": TopographicGradient,
    "fractal": FractalFieldGradient,
    "idw": MeshGradient,
}


def register_gradient_engine(name: str, engine_class: Type[Any]) -> None:
    """Registers or overrides a gradient engine in the registry."""
    GRADIENT_REGISTRY[name] = engine_class


def get_gradient_engine_class(name: Optional[str]) -> Type[Any]:
    """Retrieves the gradient engine class for a given name, defaulting to SimplexMeshGradient."""
    if not name:
        return SimplexMeshGradient
    return GRADIENT_REGISTRY.get(name, SimplexMeshGradient)


def render_gradient(
    engine_type: Optional[str],
    *,
    width: int,
    height: int,
    colors: Sequence[Sequence[float]],
    complexity: int = 5,
    fluid_mode: str = "random",
    warp_strength: float = 0.85,
    ambient_weight: float = 0.15,
    dispersion: float = 0.35,
    fractal_type: str = "fluid",
    fractal_zoom: Optional[float] = None,
    fractal_center: Optional[Tuple[float, float]] = None,
    fractal_angle: Optional[float] = None,
    phase_offset: float = 0.0,
    frequency: float = 1.0,
    trap_weight: float = 0.35,
    channel_offset: float = 0.0,
    seed: Optional[int] = None,
    legacy_idw: bool = False,
) -> Image.Image:
    """Dispatches gradient generation to the registered engine with normalized parameters."""
    if legacy_idw or engine_type == "idw":
        idw_engine = MeshGradient(width=width, height=height)
        return idw_engine.generate(colors, p=2.5)

    if engine_type == "flow-field":
        return FlowFieldGradient(width=width, height=height).generate(
            colors, complexity=complexity, fluid_mode=fluid_mode, seed=seed
        )

    if engine_type == "fluid-wave":
        return FluidMeshGradient(width=width, height=height).generate(
            colors, complexity=complexity, fluid_mode=fluid_mode
        )

    if engine_type == "spline-wave":
        return SplineWaveGradient(width=width, height=height).generate(
            colors, complexity=complexity, seed=seed
        )

    if engine_type == "domain-warp":
        return DomainWarpGradient(width=width, height=height).generate(
            colors, complexity=complexity, warp_strength=warp_strength, seed=seed
        )

    if engine_type in ("rbf", "radial-basis"):
        return RadialBasisGradient(width=width, height=height).generate(
            colors,
            complexity=complexity,
            warp_strength=warp_strength,
            ambient_weight=ambient_weight,
            dispersion=dispersion,
            seed=seed,
        )

    if engine_type == "topographic":
        return TopographicGradient(width=width, height=height).generate(
            colors, complexity=complexity, warp_strength=warp_strength, seed=seed
        )

    if engine_type == "fractal":
        return FractalFieldGradient(width=width, height=height).generate(
            colors,
            complexity=complexity,
            fractal_type=fractal_type,
            zoom=fractal_zoom,
            center=fractal_center,
            angle=fractal_angle,
            phase_offset=phase_offset,
            frequency=frequency,
            trap_weight=trap_weight,
            channel_offset=channel_offset,
            seed=seed,
        )

    # Default to simplex
    return SimplexMeshGradient(width=width, height=height).generate(
        colors, complexity=complexity, fluid_mode=fluid_mode, seed=seed
    )
