"""Base class for the built-in engine classes, which run their stage plan."""

from __future__ import annotations

from typing import Any, Sequence

from PIL import Image


class StagedEngine:
    """An engine named by its stage plan in engine.stages.plans.STAGED_PLANS."""

    plan = ""

    def __init__(self, width: int = 5120, height: int = 2880) -> None:
        self.width = width
        self.height = height

    def _render(self, colors: Sequence[Sequence[float]], **params: Any) -> Image.Image:
        from engine.stages import render_staged

        return render_staged(
            self.plan, dict(width=self.width, height=self.height, colors=colors, **params)
        )
