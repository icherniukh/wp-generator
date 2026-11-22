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
from engine.gradients.mesh import BlobLayersGradient, FluidMeshGradient, MeshGradient, PatchMeshGradient, SimplexMeshGradient
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
    "patch-mesh": PatchMeshGradient,
    "blobs": BlobLayersGradient,
}

_BUILTIN_ENGINES: Dict[str, Type[Any]] = dict(GRADIENT_REGISTRY)

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
    trace: Optional[Any] = None,
    composition: Optional[Any] = None,
    keep_state: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """Renders a gradient with the named engine.

    Built-in engines run their stage plan (engine.stages.plans) and can fill a
    trace and take a composition (engine.stages.Composition). An unknown name
    falls back to simplex. Engines registered over a built-in name, or under a
    new name, run their own class and cannot take a composition.
    """
    resolved_name = "idw" if legacy_idw else engine_type
    if not resolved_name or resolved_name not in GRADIENT_REGISTRY:
        resolved_name = "simplex"
    engine_cls = GRADIENT_REGISTRY[resolved_name]

    if engine_cls is _BUILTIN_ENGINES.get(resolved_name):
        from engine.stages import render_staged

        params = dict(
            width=width,
            height=height,
            colors=colors,
            complexity=complexity,
            fluid_mode=fluid_mode,
            warp_strength=warp_strength,
            transverse_warp=transverse_warp,
            ambient_weight=ambient_weight,
            dispersion=dispersion,
            fractal_type=fractal_type,
            fractal_zoom=fractal_zoom,
            fractal_center=fractal_center,
            fractal_angle=fractal_angle,
            phase_offset=phase_offset,
            frequency=frequency,
            trap_weight=trap_weight,
            channel_offset=channel_offset,
            seed=seed,
        )
        return render_staged(resolved_name, params, trace=trace, composition=composition, keep_state=keep_state)

    if composition:
        raise ValueError(
            f"engine {resolved_name!r} is a custom class without a stage plan; "
            "--warp, --field, --mapping and --stage-param need a built-in engine"
        )
    engine = engine_cls(width=width, height=height)
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
