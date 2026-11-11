import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type
from PIL import Image

from engine.gradients.field import (
    DomainWarpGradient,
    FlowFieldGradient,
    RadialBasisGradient,
    TopographicGradient,
)
from engine.gradients.fractal import (
    EscapeTimeFractalGradient,
    FluidFractalGradient,
    FractalFieldGradient,
)
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
    "fluid-fractal": FluidFractalGradient,
    "escape-fractal": EscapeTimeFractalGradient,
    "idw": MeshGradient,
}

_CUSTOM_ADAPTERS: Dict[Type[Any], Callable[..., Image.Image]] = {}


def register_gradient_engine(
    name: str,
    engine_class: Type[Any],
    adapter: Optional[Callable[..., Image.Image]] = None,
) -> None:
    """Registers or overrides a gradient engine in the registry."""
    GRADIENT_REGISTRY[name] = engine_class
    if adapter is not None:
        _CUSTOM_ADAPTERS[engine_class] = adapter


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
    warp_strength: Optional[float] = None,
    transverse_warp: Optional[float] = None,
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
    """Dispatches gradient generation to the registered engine with explicit domain parameters."""
    resolved_name = "idw" if legacy_idw else engine_type
    engine_cls = get_gradient_engine_class(resolved_name)
    engine = engine_cls(width=width, height=height)

    if engine_cls is SimplexMeshGradient:
        return engine.generate(
            colors, complexity=complexity, fluid_mode=fluid_mode, seed=seed
        )
    if engine_cls is FlowFieldGradient:
        return engine.generate(
            colors,
            complexity=complexity,
            fluid_mode=fluid_mode,
            warp_strength=warp_strength,
            seed=seed,
        )
    if engine_cls is FluidMeshGradient:
        return engine.generate(
            colors, complexity=complexity, fluid_mode=fluid_mode
        )
    if engine_cls is SplineWaveGradient:
        return engine.generate(
            colors, complexity=complexity, warp_strength=warp_strength, seed=seed
        )
    if engine_cls is DomainWarpGradient:
        return engine.generate(
            colors,
            complexity=complexity,
            warp_strength=warp_strength,
            transverse_warp=transverse_warp,
            seed=seed,
        )
    if engine_cls is RadialBasisGradient:
        return engine.generate(
            colors,
            complexity=complexity,
            warp_strength=1.0 if warp_strength is None else warp_strength,
            ambient_weight=ambient_weight,
            dispersion=dispersion,
            seed=seed,
        )
    if engine_cls is TopographicGradient:
        return engine.generate(
            colors,
            complexity=complexity,
            warp_strength=0.85 if warp_strength is None else warp_strength,
            seed=seed,
        )
    if engine_cls is FluidFractalGradient:
        return engine.generate(
            colors,
            complexity=complexity,
            zoom=fractal_zoom,
            center=fractal_center,
            angle=fractal_angle,
            phase_offset=phase_offset,
            frequency=frequency,
            channel_offset=channel_offset,
            seed=seed,
        )
    if engine_cls is EscapeTimeFractalGradient or issubclass(engine_cls, FractalFieldGradient):
        return engine.generate(
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
    if engine_cls is MeshGradient:
        return engine.generate(colors, p=2.5)

    custom_adapter = _CUSTOM_ADAPTERS.get(engine_cls)
    if custom_adapter is not None:
        return custom_adapter(engine, colors=colors, complexity=complexity, seed=seed)

    params = inspect.signature(engine.generate).parameters
    kwargs: Dict[str, Any] = {}
    if "complexity" in params:
        kwargs["complexity"] = complexity
    if "seed" in params:
        kwargs["seed"] = seed
    return engine.generate(colors, **kwargs)
