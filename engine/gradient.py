"""Backwards-compatibility facade for gradient engines.

All gradient implementations have been modularized under `engine.gradients`.
This module re-exports all engines and interpolation primitives.
"""

from engine.gradients import (
    DomainWarpGradient,
    FlowFieldGradient,
    FluidMeshGradient,
    FractalFieldGradient,
    MeshGradient,
    RBFFieldGradient,
    RadialBasisGradient,
    SimplexMeshGradient,
    SplineWaveGradient,
    TopographicGradient,
    colors_to_oklab,
    interpolate_color_curve,
    interpolate_oklab,
    oklab_image_to_pil,
    random_anchor_points,
)

# Compatibility aliases for private function imports in test suites
_interpolate_oklab = interpolate_oklab
_colors_to_oklab = colors_to_oklab
_oklab_image_to_pil = oklab_image_to_pil
_random_anchor_points = random_anchor_points

__all__ = [
    "DomainWarpGradient",
    "FlowFieldGradient",
    "FluidMeshGradient",
    "FractalFieldGradient",
    "MeshGradient",
    "RadialBasisGradient",
    "RBFFieldGradient",
    "SimplexMeshGradient",
    "SplineWaveGradient",
    "TopographicGradient",
    "_interpolate_oklab",
    "_colors_to_oklab",
    "_oklab_image_to_pil",
    "_random_anchor_points",
    "interpolate_oklab",
    "colors_to_oklab",
    "oklab_image_to_pil",
    "random_anchor_points",
    "interpolate_color_curve",
]
