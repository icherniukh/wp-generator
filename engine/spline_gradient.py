"""Backwards-compatibility facade for SplineWaveGradient.

SplineWaveGradient is now implemented in `engine.gradients.spline`.
"""

from engine.gradients.spline import SplineWaveGradient

__all__ = ["SplineWaveGradient"]
