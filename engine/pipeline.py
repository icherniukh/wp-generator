import colorsys
import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from engine.colors import (
    ColorTheoryHarmonizer,
    PaletteContrast,
    StyleHarmonizer,
    evaluate_palette_contrast,
    hex_to_rgb,
    rgb_to_hex,
)
from engine.filters import WallpaperFilters
from engine.gradients import render_gradient
from engine.metadata import (
    ImageMetadata,
    attach_metadata_to_image,
    build_image_metadata,
    extract_image_metadata,
)
from engine.parameter_ranges import PARAMETER_RANGES
from engine.presets import PresetEngine

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
    try:
        return hex_to_rgb(hex_str)
    except ValueError as exc:
        raise PipelineConfigurationError(str(exc)) from exc


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
    contours: Optional[float] = None
    contour_levels: Optional[int] = None
    contour_width: Optional[float] = None
    contour_mode: Optional[str] = None
    glass: Optional[bool] = None
    chromatic_aberration: Optional[float] = None
    warp_strength: Optional[float] = None
    transverse_warp: Optional[float] = None
    ambient_weight: Optional[float] = None
    dispersion: Optional[float] = None
    phase_offset: Optional[float] = None
    frequency: Optional[float] = None
    trap_weight: Optional[float] = None
    channel_offset: Optional[float] = None
    fractal_type: Optional[str] = None
    fractal_zoom: Optional[float] = None
    fractal_center: Optional[Tuple[float, float]] = None
    fractal_angle: Optional[float] = None
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
    transverse_warp: Optional[float] = None
    ambient_weight: Optional[float] = None
    dispersion: Optional[float] = None
    contours: float = 0.0
    contour_levels: int = 0
    phase_offset: Optional[float] = None
    frequency: Optional[float] = None
    trap_weight: Optional[float] = None
    fractal_type: Optional[str] = None
    fractal_zoom: Optional[float] = None
    fractal_center: Optional[Tuple[float, float]] = None
    fractal_angle: Optional[float] = None
    dots: float = 0.0
    dots_spacing: Optional[int] = None
    dots_mode: Optional[str] = None
    contour_width: Optional[float] = None
    contour_mode: Optional[str] = None
    glass: bool = False
    vignette: float = 0.0
    channel_offset: Optional[float] = None
    fluid_mode: Optional[str] = None
    harmonize: Optional[str] = None
    preset: Optional[str] = None
    preset_version: Optional[str] = None
    base_image: Optional[Image.Image] = None
    palette_contrast: PaletteContrast = field(default_factory=lambda: PaletteContrast(()))
    metadata: Optional[ImageMetadata] = None

    def save(self, fp: Union[str, Path, io.BufferedIOBase], format: str = "PNG", **kwargs: Any) -> None:
        """Saves rendered image to file path or buffer with embedded metadata."""
        self.image.save(fp, format=format, **kwargs)


class WallpaperPipeline:
    def __init__(self, preset_dir: Optional[Union[str, Path]] = None):
        self.preset_dir = Path(preset_dir) if preset_dir else resolve_preset_directory()
        self.preset_engine = PresetEngine(preset_dir=str(self.preset_dir))

    def list_presets(self, include_palette: bool = False) -> List[Dict[str, Any]]:
        """Returns metadata for all available presets sorted by identifier."""
        if not self.preset_dir.exists():
            return []
        presets = []
        rng_state = random.getstate() if include_palette else None
        try:
            for file_path in sorted(self.preset_dir.glob("*.json")):
                preset_id = file_path.stem
                try:
                    data = self.preset_engine.load_preset(preset_id)
                    entry: Dict[str, Any] = {
                        "id": preset_id,
                        "name": data.get("name", preset_id.capitalize()),
                        "version": data.get("version", "1.0.0"),
                        "description": data.get("description", ""),
                        "color_space": data.get("color_space", "hls"),
                        "color_count": data.get("color_count", {}),
                        "overrides": data.get("overrides", {}),
                    }
                    if include_palette:
                        random.seed(42)
                        sample_colors = self.preset_engine.generate_palette(
                            data, count=3, family_index=42
                        )
                        entry["sample_palette"] = [rgb_to_hex(c) for c in sample_colors]
                    presets.append(entry)
                except Exception:
                    continue
        finally:
            if rng_state is not None:
                random.setstate(rng_state)
        return presets

    def load_preset(self, preset_id: str) -> Dict[str, Any]:
        """Loads and returns preset dictionary by ID."""
        try:
            return self.preset_engine.load_preset(preset_id)
        except FileNotFoundError as exc:
            raise PresetNotFoundError(f"Preset '{preset_id}' not found in {self.preset_dir}") from exc

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

    def _resolve_palette(
        self,
        config: RenderConfiguration,
        preset_data: Dict[str, Any],
        complexity: int,
        active_seed: int,
    ) -> List[Tuple[int, int, int]]:
        """Resolves the initial un-harmonized palette from config, preset, or random sampling."""
        if config.colors:
            colors = list(config.colors)
        elif config.preset in THEME_FALLBACK_PALETTES:
            colors = list(THEME_FALLBACK_PALETTES[config.preset])
        elif preset_data:
            count = None if preset_data.get("palette_families") else (complexity // 2 if config.complexity else None)
            colors = self.preset_engine.generate_palette(
                preset_data, count=count, family_index=active_seed
            )
        else:
            num_colors = max(2, min(5, complexity // 2))
            colors = self._sample_dynamic_palette(num_colors)

        while len(colors) < 2:
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return colors

    def _harmonize_palette(
        self,
        base_colors: List[Tuple[int, int, int]],
        harmonize_mode: Optional[str],
        preset_data: Dict[str, Any],
        colors_were_specified: bool,
    ) -> List[Tuple[int, int, int]]:
        """Applies OKLCH perceptual, color theory, or Nothing-style harmonization."""
        if harmonize_mode == "none":
            return base_colors
        if harmonize_mode == "nothing":
            return StyleHarmonizer.adjust_for_nothing(base_colors)
        if harmonize_mode in ("theory", "perceptual"):
            bounds = preset_data.get("harmonization", {})
            harmony_mode = bounds.get("mode", "analogous") if harmonize_mode == "theory" else "perceptual"
            try:
                return ColorTheoryHarmonizer.harmonize(
                    base_colors,
                    mode=harmony_mode,
                    lightness_bounds=bounds.get("lightness_bounds", (0.12, 0.90)),
                    chroma_bounds=bounds.get("chroma_bounds", (0.0, 0.25)),
                )
            except (TypeError, ValueError, AttributeError) as exc:
                raise PipelineConfigurationError(f"Invalid harmonization bounds: {exc}") from exc
        if harmonize_mode is None and not colors_were_specified:
            if random.random() < 2 / 3:
                return ColorTheoryHarmonizer.harmonize(base_colors)
        return base_colors

    def _apply_filters(
        self,
        image: Image.Image,
        *,
        dots: float,
        dots_spacing: int,
        dots_mode: str,
        contours: float,
        contour_levels: int,
        contour_width: float,
        contour_mode: str,
        glass: bool,
        chromatic_aberration: float,
        noise: float,
        vignette: float,
    ) -> Image.Image:
        """Applies configured post-processing filter overlays sequentially."""
        if dots > 0.0:
            image = WallpaperFilters.apply_dot_matrix(
                image, spacing=dots_spacing, intensity=dots, mode=dots_mode
            )
        if contours > 0.0:
            image = WallpaperFilters.apply_contour_lines(
                image,
                levels=contour_levels,
                line_width=contour_width,
                intensity=contours,
                mode=contour_mode,
            )
        if glass:
            image = WallpaperFilters.apply_frosted_glass(image, blur_radius=60, noise_amount=0.15)
        if chromatic_aberration > 0.0:
            image = WallpaperFilters.apply_chromatic_aberration(
                image, strength=chromatic_aberration
            )
        if noise > 0.0:
            image = WallpaperFilters.apply_film_grain(image, amount=noise)
        if vignette > 0.0:
            image = WallpaperFilters.apply_vignette(image, strength=vignette)
        return image

    def render(self, config: RenderConfiguration) -> RenderResult:
        """Executes the complete generation pipeline and returns an in-memory RenderResult."""
        if config.width <= 0 or config.height <= 0:
            raise PipelineConfigurationError("Dimensions must be positive integers.")

        active_seed = config.seed if config.seed is not None else random.randint(0, 2_147_483_647)
        random.seed(active_seed)
        np.random.seed(active_seed)

        preset_data: Dict[str, Any] = {}
        if config.preset_data:
            preset_data = config.preset_data
        elif config.preset and config.preset not in THEME_FALLBACK_PALETTES:
            preset_data = self.load_preset(config.preset)

        preset_version = self.preset_engine.get_preset_version(preset_data) if (config.preset or config.preset_data) else None

        overrides = self.preset_engine.get_engine_overrides(preset_data)

        def resolve(key: str, default: Any = None) -> Any:
            explicit_val = getattr(config, key)
            if explicit_val is not None:
                return explicit_val
            if key in overrides:
                val = overrides[key]
                return self.preset_engine.sample_distribution(val) if isinstance(val, dict) else val
            return default() if callable(default) else default

        complexity = resolve("complexity", lambda: PARAMETER_RANGES.sample("complexity"))
        noise = resolve("noise", lambda: PARAMETER_RANGES.sample("noise"))
        glass_param = resolve("glass", False)
        chromatic_aberration = resolve("chromatic_aberration", 0.0)
        fluid_mode = resolve("fluid_mode", "random")
        vignette = resolve("vignette", 0.0)
        engine_type = resolve("engine", "simplex")

        warp_key_map = {
            "rbf": "rbf_warp",
            "radial-basis": "rbf_warp",
            "topographic": "topo_warp",
            "flow-field": "flow_warp",
            "spline-wave": "spline_warp",
        }
        warp_range_key = warp_key_map.get(engine_type, "warp_strength")
        warp_strength = resolve("warp_strength", lambda: PARAMETER_RANGES.sample(warp_range_key))
        transverse_warp = resolve(
            "transverse_warp", lambda: PARAMETER_RANGES.sample("transverse_warp")
        )
        ambient_weight = resolve(
            "ambient_weight", lambda: PARAMETER_RANGES.sample("ambient_weight")
        )
        dispersion = resolve("dispersion", lambda: PARAMETER_RANGES.sample("dispersion"))

        dots = resolve("dots", 0.0)
        dots_spacing = resolve("dots_spacing", lambda: random.randint(8, 60))
        dots_mode = resolve("dots_mode", lambda: random.choice(["size", "alpha"]))

        contours = resolve("contours", 0.0)
        contour_levels = resolve("contour_levels", 24)
        contour_width = resolve("contour_width", 1.4)
        contour_mode = resolve("contour_mode", "light")

        fractal_type = resolve("fractal_type", "fluid")
        fractal_zoom = resolve("fractal_zoom", None)
        fractal_center = resolve("fractal_center", None)
        fractal_angle = resolve("fractal_angle", None)
        phase_offset = resolve("phase_offset", 0.0)
        frequency = resolve("frequency", lambda: PARAMETER_RANGES.sample("frequency"))
        trap_weight = resolve("trap_weight", lambda: PARAMETER_RANGES.sample("trap_weight"))
        channel_offset = resolve("channel_offset", 0.0)

        base_colors = self._resolve_palette(config, preset_data, complexity, active_seed)
        harmonize = resolve("harmonize", None)
        if harmonize not in (None, "none", "nothing", "theory", "perceptual"):
            raise PipelineConfigurationError(f"Unknown harmonizer '{harmonize}'.")
        colors_were_specified = bool(config.colors or config.preset)
        final_colors = self._harmonize_palette(base_colors, harmonize, preset_data, colors_were_specified)

        image = render_gradient(
            engine_type,
            width=config.width,
            height=config.height,
            colors=final_colors,
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
            seed=active_seed,
            legacy_idw=config.legacy_idw,
        )

        base_image = image.copy() if config.capture_base_image else None

        image = self._apply_filters(
            image,
            dots=dots,
            dots_spacing=dots_spacing,
            dots_mode=dots_mode,
            contours=contours,
            contour_levels=contour_levels,
            contour_width=contour_width,
            contour_mode=contour_mode,
            glass=glass_param,
            chromatic_aberration=chromatic_aberration,
            noise=noise,
            vignette=vignette,
        )

        result = RenderResult(
            image=image,
            seed=active_seed,
            colors=final_colors,
            color_hexes=[rgb_to_hex(c) for c in final_colors],
            engine=engine_type,
            width=config.width,
            height=config.height,
            complexity=complexity,
            noise=noise,
            chromatic_aberration=chromatic_aberration,
            warp_strength=warp_strength,
            transverse_warp=transverse_warp,
            ambient_weight=ambient_weight,
            dispersion=dispersion,
            contours=contours,
            contour_levels=contour_levels,
            contour_width=contour_width,
            contour_mode=contour_mode,
            dots=dots,
            dots_spacing=dots_spacing,
            dots_mode=dots_mode,
            glass=glass_param,
            vignette=vignette,
            frequency=frequency,
            phase_offset=phase_offset,
            trap_weight=trap_weight,
            channel_offset=channel_offset,
            fluid_mode=fluid_mode,
            harmonize=harmonize,
            fractal_type=fractal_type,
            fractal_zoom=fractal_zoom,
            fractal_center=fractal_center,
            fractal_angle=fractal_angle,
            preset=config.preset,
            preset_version=preset_version,
            base_image=base_image,
            palette_contrast=evaluate_palette_contrast(final_colors),
        )
        metadata = build_image_metadata(result)
        result.metadata = metadata
        attach_metadata_to_image(result.image, metadata)
        return result

    def render_png_bytes(self, config: RenderConfiguration) -> Tuple[bytes, RenderResult]:
        """Renders wallpaper directly to PNG bytes without disk access."""
        result = self.render(config)
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return buffer.getvalue(), result
