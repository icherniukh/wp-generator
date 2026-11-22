import colorsys
import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from engine import resolve_preset_directory
from engine.colors import (
    ColorTheoryHarmonizer,
    PaletteContrast,
    StyleHarmonizer,
    evaluate_palette_contrast,
    hex_to_rgb,
    rgb_to_hex,
)
from engine.filters import GRAIN_BLENDS, GRAIN_COLORS, GRAIN_SIZE_RANGE
from engine.gradients import render_gradient
from engine.metadata import (
    ImageMetadata,
    attach_metadata_to_image,
    build_image_metadata,
    extract_image_metadata,
)
from engine.parameter_ranges import PARAMETER_RANGES
from engine.presets import PresetEngine
from engine.stages import Composition, ScalarField, Trace
from engine.stages.finish import check_finish_stage_params, finish_stages, split_stage_params
from engine.stages.runner import StageParams, run_stages
from engine.stages.palette import make_palette

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
    shading: Optional[float] = None
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
    # Changes to the engine's stage plan (engine.stages.catalog): extra warps
    # ("none" drops the engine's own), a replacement field or mapping, and
    # per-stage argument overrides keyed by stage name.
    warps: Optional[List[str]] = None
    field_stage: Optional[str] = None
    mapping: Optional[str] = None
    blend: Optional[str] = None
    # Noise added to the field before color mapping (0 is off), so the grain takes the ramp's colors.
    field_grain: Optional[float] = None
    stage_params: Optional[Dict[str, Dict[str, Any]]] = None
    # Contours from the image's luminance ("image") or the engine's scalar
    # field ("field"); under the glass and aberration ("frosted") or over ("sharp").
    contour_source: Optional[str] = None
    contour_placement: Optional[str] = None
    # Film grain: size (1.0 is the default grain), "mono" or "color", and how it is
    # blended ("add", "overlay", "soft-light", "multiply").
    grain_size: Optional[float] = None
    grain_color: Optional[str] = None
    grain_blend: Optional[str] = None
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
    contour_width: Optional[float] = None
    contour_mode: Optional[str] = None
    contour_source: Optional[str] = None
    contour_placement: Optional[str] = None
    grain_size: Optional[float] = None
    grain_color: Optional[str] = None
    grain_blend: Optional[str] = None
    glass: bool = False
    vignette: float = 0.0
    shading: float = 0.0
    channel_offset: Optional[float] = None
    fluid_mode: Optional[str] = None
    harmonize: Optional[str] = None
    preset: Optional[str] = None
    preset_version: Optional[str] = None
    warps: Optional[List[str]] = None
    field_stage: Optional[str] = None
    mapping: Optional[str] = None
    blend: Optional[str] = None
    field_grain: float = 0.0
    stage_params: Optional[Dict[str, Dict[str, Any]]] = None
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
        contours: float,
        contour_levels: int,
        contour_width: float,
        contour_mode: str,
        glass: bool,
        contour_source: str = "image",
        contour_placement: str = "frosted",
        field: Optional[Any] = None,
        chromatic_aberration: float,
        noise: float,
        grain_size: float = 1.0,
        grain_color: str = "mono",
        grain_blend: str = "add",
        vignette: float,
        shading: float = 0.0,
        stage_params: Optional[Dict[str, Dict[str, Any]]] = None,
        trace: Optional[Trace] = None,
    ) -> Image.Image:
        """Applies the enabled finish effects in their fixed order (engine.stages.finish)."""
        params = StageParams(
            dict(
                contours=contours,
                contour_levels=contour_levels,
                contour_width=contour_width,
                contour_mode=contour_mode,
                contour_source=contour_source,
                contour_placement=contour_placement,
                glass=glass,
                chromatic_aberration=chromatic_aberration,
                noise=noise,
                grain_size=grain_size,
                grain_color=grain_color,
                grain_blend=grain_blend,
                vignette=vignette,
                shading=shading,
                stage_params=stage_params,
                width=image.width,
                height=image.height,
            )
        )
        stages = finish_stages(params)
        state = run_stages(stages, params, rng=random, state={"final": image, "field": field}, trace=trace)
        return state["final"]

    def render(self, config: RenderConfiguration, trace: Optional[Trace] = None) -> RenderResult:
        """Executes the complete generation pipeline and returns an in-memory RenderResult.

        With a trace, every step's output and parameters are recorded into it; the
        image is the same as without one.
        """
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
                # stage_params is a plain mapping of stage arguments, not a distribution.
                if isinstance(val, dict) and key != "stage_params":
                    return self.preset_engine.sample_distribution(val)
                return val
            return default() if callable(default) else default

        complexity = resolve("complexity", lambda: PARAMETER_RANGES.sample("complexity"))
        noise = resolve("noise", lambda: PARAMETER_RANGES.sample("noise"))
        glass_param = resolve("glass", False)
        chromatic_aberration = resolve("chromatic_aberration", 0.0)
        fluid_mode = resolve("fluid_mode", "random")
        vignette = resolve("vignette", 0.0)
        shading = resolve("shading", 0.0)
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

        contours = resolve("contours", 0.0)
        contour_levels = resolve("contour_levels", 24)
        contour_width = resolve("contour_width", 1.4)
        contour_mode = resolve("contour_mode", "light")
        contour_source = resolve("contour_source", "image")
        contour_placement = resolve("contour_placement", "frosted")
        if contour_source not in ("image", "field"):
            raise PipelineConfigurationError(f"contour_source must be image or field, not {contour_source!r}.")
        if contour_placement not in ("frosted", "sharp"):
            raise PipelineConfigurationError(f"contour_placement must be frosted or sharp, not {contour_placement!r}.")
        grain_size = resolve("grain_size", 1.0)
        grain_color = resolve("grain_color", "mono")
        grain_blend = resolve("grain_blend", "add")
        if not isinstance(grain_size, (int, float)) or isinstance(grain_size, bool) or not GRAIN_SIZE_RANGE[0] <= grain_size <= GRAIN_SIZE_RANGE[1]:
            raise PipelineConfigurationError(f"grain_size must be a number from {GRAIN_SIZE_RANGE[0]} to {GRAIN_SIZE_RANGE[1]}, not {grain_size!r}.")
        if grain_color not in GRAIN_COLORS:
            raise PipelineConfigurationError(f"grain_color must be one of {GRAIN_COLORS}, not {grain_color!r}.")
        if grain_blend not in GRAIN_BLENDS:
            raise PipelineConfigurationError(f"grain_blend must be one of {GRAIN_BLENDS}, not {grain_blend!r}.")

        fractal_type = resolve("fractal_type", "fluid")
        fractal_zoom = resolve("fractal_zoom", None)
        fractal_center = resolve("fractal_center", None)
        fractal_angle = resolve("fractal_angle", None)
        phase_offset = resolve("phase_offset", 0.0)
        frequency = resolve("frequency", lambda: PARAMETER_RANGES.sample("frequency"))
        trap_weight = resolve("trap_weight", lambda: PARAMETER_RANGES.sample("trap_weight"))
        channel_offset = resolve("channel_offset", 0.0)

        warps = resolve("warps", None)
        field_stage = resolve("field_stage", None)
        mapping = resolve("mapping", None)
        blend = resolve("blend", None)
        field_grain = resolve("field_grain", 0.0)
        if not isinstance(field_grain, (int, float)) or isinstance(field_grain, bool) or not 0.0 <= field_grain <= 1.0:
            raise PipelineConfigurationError(f"field_grain must be a number from 0 to 1, not {field_grain!r}.")
        stage_params = resolve("stage_params", None)
        # Finish stages' keys are applied after the engine; refuse a bad one before rendering.
        check_finish_stage_params(stage_params)
        composition = Composition(
            warps=tuple(warps or ()),
            field=field_stage,
            mapping=mapping,
            stage_params=split_stage_params(stage_params)[0],
            blend=blend,
            field_grain=float(field_grain),
        )

        base_colors = self._resolve_palette(config, preset_data, complexity, active_seed)
        harmonize = resolve("harmonize", None)
        if harmonize not in (None, "none", "nothing", "theory", "perceptual"):
            raise PipelineConfigurationError(f"Unknown harmonizer '{harmonize}'.")
        colors_were_specified = bool(config.colors or config.preset)
        final_colors = self._harmonize_palette(base_colors, harmonize, preset_data, colors_were_specified)
        if trace is not None:
            trace.add(
                "palette",
                "palette",
                make_palette(colors=final_colors),
                output_name="palette",
                arguments=dict(
                    preset=config.preset,
                    colors="given" if config.colors else "from preset" if config.preset else "random",
                    harmonize=harmonize,
                    before_harmonize=" ".join(rgb_to_hex(c) for c in base_colors),
                ),
            )

        records_before = len(trace.records) if trace is not None else 0
        engine_state: Dict[str, Any] = {}
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
            trace=trace,
            composition=composition or None,
            keep_state=engine_state,
        )
        field = engine_state.get("field")
        if contour_source == "field" and not isinstance(field, ScalarField):
            # Weight fields (rbf, flow-field, idw) and custom engines have no
            # single value per pixel to draw isolines of; use the image.
            contour_source = "image"
        if trace is not None and len(trace.records) == records_before:
            trace.add(
                f"{engine_type} (custom engine, not split into stages)",
                "engine",
                image,
                output_name="image",
                arguments=dict(engine=engine_type, complexity=complexity, warp_strength=warp_strength),
            )

        base_image = image.copy() if config.capture_base_image else None

        image = self._apply_filters(
            image,
            contours=contours,
            contour_levels=contour_levels,
            contour_width=contour_width,
            contour_mode=contour_mode,
            contour_source=contour_source,
            contour_placement=contour_placement,
            field=field,
            glass=glass_param,
            chromatic_aberration=chromatic_aberration,
            noise=noise,
            grain_size=grain_size,
            grain_color=grain_color,
            grain_blend=grain_blend,
            vignette=vignette,
            shading=shading,
            stage_params=stage_params,
            trace=trace,
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
            contour_source=contour_source if contours else None,
            contour_placement=contour_placement if contours else None,
            grain_size=grain_size if noise > 0 and grain_size != 1.0 else None,
            grain_color=grain_color if noise > 0 and grain_color != "mono" else None,
            grain_blend=grain_blend if noise > 0 and grain_blend != "add" else None,
            glass=glass_param,
            vignette=vignette,
            shading=shading,
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
            warps=list(warps) if warps else None,
            field_stage=field_stage,
            mapping=mapping,
            blend=blend,
            field_grain=float(field_grain),
            stage_params=dict(stage_params) if stage_params else None,
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
