import colorsys
import io
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from engine.colors import (
    ColorTheoryHarmonizer, PaletteContrast, StyleHarmonizer, evaluate_palette_contrast,
)
from engine.filters import WallpaperFilters
from engine.gradient import (
    DomainWarpGradient,
    FlowFieldGradient,
    FluidMeshGradient,
    MeshGradient,
    RadialBasisGradient,
    SimplexMeshGradient,
)
from engine.presets import PresetEngine
from engine.spline_gradient import SplineWaveGradient

DEVICE_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    # Apple MacBooks (14:9 aspect ratios)
    "mbp14": (3024, 1964),   # MacBook Pro 14" (M1/M2/M3/M4)
    "mbp16": (3456, 2234),   # MacBook Pro 16" (M1/M2/M3/M4)
    "mba13": (2560, 1664),   # MacBook Air 13" (M2/M3)
    "mba15": (2880, 1864),   # MacBook Air 15" (M2/M3)
    # Standard Widescreen (16:9 aspect ratios)
    "1080p": (1920, 1080),   # FHD Standard Monitor
    "1440p": (2560, 1440),   # QHD Monitor
    "4k": (3840, 2160),      # UHD Monitor
    "5k": (5120, 2880),      # Apple Studio Display / 5K Monitors
    # Ultrawide Monitors
    "uw": (3440, 1440),      # Standard Ultrawide (21:9)
    "suw": (5120, 1440),     # Super Ultrawide (32:9)
    # Mobile / Portrait
    "mobile": (1080, 1920),  # Generic Smartphone Portrait (9:16)
}

THEME_FALLBACK_PALETTES: Dict[str, List[Tuple[int, int, int]]] = {
    "dark": [(10, 10, 12), (25, 25, 28), (35, 35, 40)],
    "light": [(210, 210, 215), (230, 230, 235), (190, 190, 195)],
    "red": [(15, 10, 10), (180, 30, 30), (35, 15, 15)],
    "orange": [(15, 12, 10), (200, 70, 20), (35, 20, 15)],
    "glass": [(40, 45, 55), (110, 115, 130), (25, 30, 40)],
}


class PipelineConfigurationError(ValueError):
    """Raised when pipeline options or values are invalid."""


class PresetNotFoundError(KeyError):
    """Raised when a requested preset file cannot be located."""


def parse_hex_color(hex_str: str) -> Tuple[int, int, int]:
    """Converts a 6-character hex string to an RGB tuple."""
    cleaned = hex_str.strip().lstrip("#")
    if len(cleaned) != 6:
        raise PipelineConfigurationError(f"Invalid hex color: '{hex_str}'")
    try:
        return tuple(int(cleaned[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as exc:
        raise PipelineConfigurationError(f"Invalid hex color: '{hex_str}'") from exc


def resolve_preset_directory() -> Path:
    """Locates the presets directory relative to the repository structure."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "presets"


@dataclass
class RenderConfiguration:
    width: int = 3024
    height: int = 1964
    preset: Optional[str] = None
    preset_data: Optional[Dict[str, Any]] = None
    colors: Optional[List[Tuple[int, int, int]]] = None
    complexity: Optional[int] = None
    engine: Optional[str] = None
    fluid_mode: Optional[str] = None
    harmonize: Optional[str] = None
    noise: Optional[float] = None
    vignette: Optional[float] = None
    dots: Optional[float] = None
    dots_spacing: Optional[int] = None
    dots_mode: Optional[str] = None
    glass: Optional[bool] = None
    chromatic_aberration: Optional[float] = None
    warp_strength: Optional[float] = None
    ambient_weight: Optional[float] = None
    dispersion: Optional[float] = None
    legacy_idw: bool = False
    seed: Optional[int] = None
    capture_base_image: bool = False


@dataclass
class RenderResult:
    image: Image.Image
    seed: int
    colors: List[Tuple[int, int, int]]
    color_hexes: List[str]
    engine: str
    width: int
    height: int
    complexity: int
    noise: float
    chromatic_aberration: float = 0.0
    warp_strength: Optional[float] = None
    ambient_weight: Optional[float] = None
    dispersion: Optional[float] = None
    base_image: Optional[Image.Image] = None
    palette_contrast: PaletteContrast = field(default_factory=lambda: PaletteContrast(()))


class WallpaperPipeline:
    def __init__(self, preset_dir: Optional[Union[str, Path]] = None):
        self.preset_dir = Path(preset_dir) if preset_dir else resolve_preset_directory()
        self.preset_engine = PresetEngine(preset_dir=str(self.preset_dir))

    def list_presets(self) -> List[Dict[str, Any]]:
        """Returns metadata for all available presets sorted by identifier."""
        if not self.preset_dir.exists():
            return []
        presets = []
        for file_path in sorted(self.preset_dir.glob("*.json")):
            preset_id = file_path.stem
            try:
                data = self.preset_engine.load_preset(preset_id)
                presets.append({
                    "id": preset_id,
                    "name": data.get("name", preset_id.capitalize()),
                    "description": data.get("description", ""),
                    "color_space": data.get("color_space", "hls"),
                    "color_count": data.get("color_count", {}),
                    "overrides": data.get("overrides", {}),
                })
            except Exception:
                continue
        return presets

    def load_preset(self, preset_id: str) -> Dict[str, Any]:
        """Loads and returns preset dictionary by ID."""
        preset_file = self.preset_dir / f"{preset_id}.json"
        if not preset_file.exists():
            raise PresetNotFoundError(f"Preset '{preset_id}' not found in {self.preset_dir}")
        return self.preset_engine.load_preset(preset_id)

    def _sample_dynamic_palette(self, num_colors: int) -> List[Tuple[int, int, int]]:
        palette = []
        is_dark = random.random() < 0.7
        h = random.random()
        s = random.uniform(0.0, 0.25)
        l = random.uniform(0.05, 0.18) if is_dark else random.uniform(0.85, 0.95)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
        for _ in range(num_colors - 1):
            h = random.random()
            s = random.uniform(0.5, 1.0)
            l = random.uniform(0.3, 0.7)
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            palette.append((int(r * 255), int(g * 255), int(b * 255)))
        return palette

    def render(self, config: RenderConfiguration) -> RenderResult:
        """Executes the complete generation pipeline and returns an in-memory RenderResult."""
        if config.width <= 0 or config.height <= 0:
            raise PipelineConfigurationError("Dimensions must be positive integers.")

        active_seed = config.seed if config.seed is not None else random.randint(0, 2_147_483_647)
        random.seed(active_seed)
        np.random.seed(active_seed)

        # Resolve preset data
        preset_data: Dict[str, Any] = {}
        if config.preset_data:
            preset_data = config.preset_data
        elif config.preset and config.preset not in THEME_FALLBACK_PALETTES:
            preset_data = self.load_preset(config.preset)

        overrides = self.preset_engine.get_engine_overrides(preset_data)

        # Resolve parameters: Explicit Config > Preset Override > Default Distribution
        def resolve_value(explicit_val: Any, override_key: str, default_generator: Any) -> Any:
            if explicit_val is not None:
                return explicit_val
            if override_key in overrides:
                val = overrides[override_key]
                if isinstance(val, dict):
                    if val.get("type", "").endswith("int"):
                        return int(self.preset_engine._sample_distribution(val))
                    return self.preset_engine._sample_distribution(val)
                return val
            return default_generator()

        complexity = resolve_value(config.complexity, "complexity", lambda: random.randint(2, 6))
        noise = resolve_value(config.noise, "noise", lambda: round(random.uniform(0.005, 0.05), 3))
        glass_param = resolve_value(config.glass, "glass", lambda: False)
        chromatic_aberration = resolve_value(
            config.chromatic_aberration, "chromatic_aberration", lambda: 0.0
        )
        warp_strength = resolve_value(config.warp_strength, "warp_strength", lambda: 0.85)
        ambient_weight = resolve_value(config.ambient_weight, "ambient_weight", lambda: 0.15)
        dispersion = resolve_value(config.dispersion, "dispersion", lambda: 0.35)
        fluid_mode = resolve_value(config.fluid_mode, "fluid_mode", lambda: "random")
        vignette = resolve_value(config.vignette, "vignette", lambda: 0.0)
        engine_type = resolve_value(config.engine, "engine", lambda: "simplex")

        dots = resolve_value(config.dots, "dots", lambda: 0.0)
        dots_spacing = resolve_value(config.dots_spacing, "dots_spacing", lambda: random.randint(8, 60))
        dots_mode = resolve_value(config.dots_mode, "dots_mode", lambda: random.choice(["size", "alpha"]))

        # Color generation
        if config.colors:
            base_colors = list(config.colors)
        elif config.preset in THEME_FALLBACK_PALETTES:
            base_colors = list(THEME_FALLBACK_PALETTES[config.preset])
        elif preset_data:
            count = complexity // 2 if config.complexity else None
            base_colors = self.preset_engine.generate_palette(
                preset_data, count=count, family_index=active_seed
            )
        else:
            num_colors = max(2, min(5, complexity // 2))
            base_colors = self._sample_dynamic_palette(num_colors)

        while len(base_colors) < 2:
            base_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        # Harmonization precedence: explicit config > preset override > theory fallback
        harmonize = resolve_value(config.harmonize, "harmonize", lambda: None)
        if harmonize not in (None, "none", "nothing", "theory", "perceptual"):
            raise PipelineConfigurationError(f"Unknown harmonizer '{harmonize}'.")
        colors_were_specified = bool(config.colors or config.preset)

        if harmonize == "none":
            final_colors = base_colors
        elif harmonize == "nothing":
            final_colors = StyleHarmonizer.adjust_for_nothing(base_colors)
        elif harmonize in ("theory", "perceptual"):
            bounds = preset_data.get("harmonization", {})
            try:
                final_colors = ColorTheoryHarmonizer.harmonize(
                    base_colors,
                    lightness_bounds=bounds.get("lightness_bounds", (0.12, 0.90)),
                    chroma_bounds=bounds.get("chroma_bounds", (0.0, 0.25)),
                )
            except (TypeError, ValueError, AttributeError) as exc:
                raise PipelineConfigurationError(f"Invalid harmonization bounds: {exc}") from exc
        elif harmonize is None and not colors_were_specified:
            if random.random() < 2 / 3:
                final_colors = ColorTheoryHarmonizer.harmonize(base_colors)
            else:
                final_colors = base_colors
        else:
            final_colors = base_colors

        # Gradient generation
        if config.legacy_idw or engine_type == "idw":
            gradient_engine = MeshGradient(width=config.width, height=config.height)
            image = gradient_engine.generate(final_colors, p=2.5)
        elif engine_type == "flow-field":
            flow_engine = FlowFieldGradient(width=config.width, height=config.height)
            image = flow_engine.generate(final_colors, complexity=complexity, seed=active_seed)
        elif engine_type == "fluid-wave":
            fluid_engine = FluidMeshGradient(width=config.width, height=config.height)
            image = fluid_engine.generate(final_colors, complexity=complexity, fluid_mode=fluid_mode)
        elif engine_type == "spline-wave":
            spline_engine = SplineWaveGradient(width=config.width, height=config.height)
            image = spline_engine.generate(final_colors, complexity=complexity, seed=active_seed)
        elif engine_type == "domain-warp":
            warp_engine = DomainWarpGradient(width=config.width, height=config.height)
            image = warp_engine.generate(
                final_colors,
                complexity=complexity,
                warp_strength=warp_strength,
                seed=active_seed,
            )
        elif engine_type in ("rbf", "radial-basis"):
            rbf_engine = RadialBasisGradient(width=config.width, height=config.height)
            image = rbf_engine.generate(
                final_colors,
                complexity=complexity,
                warp_strength=warp_strength,
                ambient_weight=ambient_weight,
                dispersion=dispersion,
                seed=active_seed,
            )
        else:
            simplex_engine = SimplexMeshGradient(width=config.width, height=config.height)
            image = simplex_engine.generate(
                final_colors, complexity=complexity, fluid_mode=fluid_mode, seed=active_seed
            )

        base_image = image.copy() if config.capture_base_image else None

        # Filter pass
        if dots > 0.0:
            image = WallpaperFilters.apply_dot_matrix(
                image, spacing=dots_spacing, intensity=dots, mode=dots_mode
            )

        if glass_param:
            image = WallpaperFilters.apply_frosted_glass(image, blur_radius=60, noise_amount=0.15)

        if chromatic_aberration > 0.0:
            image = WallpaperFilters.apply_chromatic_aberration(
                image, strength=chromatic_aberration
            )

        if noise > 0.0:
            image = WallpaperFilters.apply_film_grain(image, amount=noise)

        if vignette > 0.0:
            image = WallpaperFilters.apply_vignette(image, strength=vignette)

        color_hexes = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in final_colors]

        return RenderResult(
            image=image,
            seed=active_seed,
            colors=final_colors,
            color_hexes=color_hexes,
            engine=engine_type,
            width=config.width,
            height=config.height,
            complexity=complexity,
            noise=noise,
            chromatic_aberration=chromatic_aberration,
            warp_strength=warp_strength,
            ambient_weight=ambient_weight,
            dispersion=dispersion,
            base_image=base_image,
            palette_contrast=evaluate_palette_contrast(final_colors),
        )

    def render_png_bytes(self, config: RenderConfiguration) -> Tuple[bytes, RenderResult]:
        """Renders wallpaper directly to PNG bytes without disk access."""
        result = self.render(config)
        buffer = io.BytesIO()
        result.image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue(), result
