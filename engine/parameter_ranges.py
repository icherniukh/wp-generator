from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ParameterRangeSpec:
    key: str
    label: str
    category: str
    description: str
    engine: str
    render_param: str
    hard_min: float
    hard_max: float
    step: float
    selected_min: float
    selected_max: float
    is_int: bool = False
    fixed_params: Dict[str, Any] = field(default_factory=dict)

    def sample(self) -> Union[float, int]:
        if self.is_int:
            return random.randint(int(round(self.selected_min)), int(round(self.selected_max)))
        return round(random.uniform(float(self.selected_min), float(self.selected_max)), 4)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_parameter_specs() -> List[ParameterRangeSpec]:
    return [
        ParameterRangeSpec(
            key="warp_strength",
            label="Primary Domain Warp Strength",
            category="Domain Warp",
            description="Controls primary coordinate displacement along the sweep axis. Excessively high values break smooth gradients into turbulent noise.",
            engine="domain-warp",
            render_param="warp_strength",
            hard_min=0.0,
            hard_max=2.5,
            step=0.05,
            selected_min=0.25,
            selected_max=0.90,
            fixed_params={"transverse_warp": 0.12, "complexity": 4},
        ),
        ParameterRangeSpec(
            key="transverse_warp",
            label="2D Transverse & Nested Field Warp",
            category="Domain Warp",
            description="Controls 2D perpendicular coordinate bending and secondary noise lookup displacement (f(p + q)). High values create muddy, over-folded regions.",
            engine="domain-warp",
            render_param="transverse_warp",
            hard_min=0.0,
            hard_max=1.2,
            step=0.02,
            selected_min=0.04,
            selected_max=0.22,
            fixed_params={"warp_strength": 0.65, "complexity": 4},
        ),
        ParameterRangeSpec(
            key="rbf_warp",
            label="RBF Emitter Domain Warp",
            category="Radial Basis (RBF)",
            description="Warps the 2D coordinate grid before evaluating Gaussian radial basis color emitters.",
            engine="rbf",
            render_param="warp_strength",
            hard_min=0.0,
            hard_max=2.5,
            step=0.05,
            selected_min=0.20,
            selected_max=0.95,
            fixed_params={"ambient_weight": 0.15, "dispersion": 0.35, "complexity": 5},
        ),
        ParameterRangeSpec(
            key="ambient_weight",
            label="RBF Ambient Base Weight",
            category="Radial Basis (RBF)",
            description="Weight of the foundational background color relative to foreground radial emitters.",
            engine="rbf",
            render_param="ambient_weight",
            hard_min=0.01,
            hard_max=0.80,
            step=0.02,
            selected_min=0.08,
            selected_max=0.28,
            fixed_params={"warp_strength": 0.60, "dispersion": 0.35, "complexity": 5},
        ),
        ParameterRangeSpec(
            key="dispersion",
            label="RBF Emitter Dispersion Radius",
            category="Radial Basis (RBF)",
            description="Spread radius (sigma) of each color emitter across the canvas.",
            engine="rbf",
            render_param="dispersion",
            hard_min=0.08,
            hard_max=1.20,
            step=0.04,
            selected_min=0.24,
            selected_max=0.55,
            fixed_params={"warp_strength": 0.60, "ambient_weight": 0.15, "complexity": 5},
        ),
        ParameterRangeSpec(
            key="flow_warp",
            label="Flow Field Curl Warp",
            category="Flow & Spline",
            description="Perlin displacement amplitude applied to inverse-distance color anchors in flow-field.",
            engine="flow-field",
            render_param="warp_strength",
            hard_min=0.0,
            hard_max=1.5,
            step=0.05,
            selected_min=0.18,
            selected_max=0.55,
            fixed_params={"complexity": 4},
        ),
        ParameterRangeSpec(
            key="spline_warp",
            label="Spline Wave Curl Warp",
            category="Flow & Spline",
            description="Perlin displacement amplitude applied to PCHIP monotonic spline sweeps.",
            engine="spline-wave",
            render_param="warp_strength",
            hard_min=0.0,
            hard_max=1.2,
            step=0.04,
            selected_min=0.12,
            selected_max=0.42,
            fixed_params={"complexity": 4},
        ),
        ParameterRangeSpec(
            key="topo_warp",
            label="Topographic Elevation Warp",
            category="Topographic",
            description="Simplex distortion added to harmonic elevation hills before mapping through Oklab.",
            engine="topographic",
            render_param="warp_strength",
            hard_min=0.0,
            hard_max=2.5,
            step=0.05,
            selected_min=0.25,
            selected_max=0.95,
            fixed_params={"complexity": 4, "contours": 0.65, "contour_levels": 20},
        ),
        ParameterRangeSpec(
            key="complexity",
            label="Spatial Complexity & Harmonics",
            category="Global Geometry",
            description="Controls spatial frequency, octave count, and emitter density across all gradient engines.",
            engine="simplex",
            render_param="complexity",
            hard_min=1,
            hard_max=10,
            step=1,
            selected_min=2,
            selected_max=6,
            is_int=True,
        ),
        ParameterRangeSpec(
            key="fractal_zoom",
            label="Fractal Camera Zoom",
            category="Fractal Field",
            description="Camera magnification for fluid and Julia fractal fields (smaller = macro sweep, larger = intricate detail).",
            engine="fractal",
            render_param="fractal_zoom",
            hard_min=0.25,
            hard_max=3.50,
            step=0.05,
            selected_min=0.45,
            selected_max=1.85,
            fixed_params={"fractal_type": "fluid", "complexity": 4},
        ),
        ParameterRangeSpec(
            key="frequency",
            label="Fractal Palette Cycle Frequency",
            category="Fractal Field",
            description="Number of harmonic color cycles across the normalized fractal scalar field.",
            engine="fractal",
            render_param="frequency",
            hard_min=0.25,
            hard_max=3.00,
            step=0.05,
            selected_min=0.65,
            selected_max=1.45,
            fixed_params={"fractal_type": "fluid", "fractal_zoom": 0.85, "complexity": 4},
        ),
        ParameterRangeSpec(
            key="trap_weight",
            label="Escape-Time Orbit Trap Weight",
            category="Fractal Field",
            description="Blend ratio between smooth escape potential and geometric orbit-trap structure in Julia/Mandelbrot fractals.",
            engine="fractal",
            render_param="trap_weight",
            hard_min=0.0,
            hard_max=1.0,
            step=0.05,
            selected_min=0.20,
            selected_max=0.55,
            fixed_params={"fractal_type": "julia", "fractal_zoom": 1.1, "complexity": 5},
        ),
        ParameterRangeSpec(
            key="noise",
            label="Film Grain Intensity",
            category="Post-Processing Filters",
            description="Luminance-aware organic film grain overlay strength.",
            engine="simplex",
            render_param="noise",
            hard_min=0.0,
            hard_max=0.20,
            step=0.005,
            selected_min=0.005,
            selected_max=0.050,
            fixed_params={"complexity": 4},
        ),
        ParameterRangeSpec(
            key="chromatic_aberration",
            label="Chromatic Aberration",
            category="Post-Processing Filters",
            description="Radial red/blue channel dispersion simulating optical prism refraction near edges.",
            engine="fluid-wave",
            render_param="aberration",
            hard_min=0.0,
            hard_max=0.06,
            step=0.002,
            selected_min=0.004,
            selected_max=0.020,
            fixed_params={"contours": 0.7, "contour_levels": 16, "complexity": 4},
        ),
        ParameterRangeSpec(
            key="vignette",
            label="Optical Vignette",
            category="Post-Processing Filters",
            description="Radial corner luminance falloff toward the perimeter of the frame.",
            engine="simplex",
            render_param="vignette",
            hard_min=0.0,
            hard_max=0.65,
            step=0.02,
            selected_min=0.06,
            selected_max=0.25,
            fixed_params={"complexity": 4},
        ),
        ParameterRangeSpec(
            key="dots",
            label="Dot Matrix Intensity",
            category="Post-Processing Filters",
            description="Nothing-style halftone dot matrix overlay opacity.",
            engine="simplex",
            render_param="dots",
            hard_min=0.0,
            hard_max=1.0,
            step=0.05,
            selected_min=0.15,
            selected_max=0.60,
            fixed_params={"dots_spacing": 24, "complexity": 4},
        ),
        ParameterRangeSpec(
            key="contours",
            label="Topographic Contour Lines",
            category="Post-Processing Filters",
            description="Anti-aliased elevation isobar line overlay opacity.",
            engine="topographic",
            render_param="contours",
            hard_min=0.0,
            hard_max=1.0,
            step=0.05,
            selected_min=0.35,
            selected_max=0.85,
            fixed_params={"contour_levels": 22, "contour_width": 1.4, "warp_strength": 0.65},
        ),
    ]


class ParameterRangeStore:
    """Manages configurable [min, max] sampling ranges for engine and filter parameters."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        if config_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            config_path = repo_root / "presets" / "parameter_ranges.json"
        self.config_path = Path(config_path)
        self._specs: Dict[str, ParameterRangeSpec] = {
            spec.key: spec for spec in default_parameter_specs()
        }
        self.load()

    def load(self) -> None:
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key, bounds in saved.items():
                if key in self._specs and isinstance(bounds, dict):
                    spec = self._specs[key]
                    s_min = float(bounds.get("selected_min", spec.selected_min))
                    s_max = float(bounds.get("selected_max", spec.selected_max))
                    s_min = max(spec.hard_min, min(spec.hard_max, s_min))
                    s_max = max(s_min, min(spec.hard_max, s_max))
                    spec.selected_min = int(round(s_min)) if spec.is_int else s_min
                    spec.selected_max = int(round(s_max)) if spec.is_int else s_max
        except (OSError, ValueError, TypeError):
            pass

    def save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: {
                "selected_min": spec.selected_min,
                "selected_max": spec.selected_max,
            }
            for key, spec in self._specs.items()
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def reset_defaults(self) -> List[Dict[str, Any]]:
        self._specs = {spec.key: spec for spec in default_parameter_specs()}
        if self.config_path.exists():
            self.config_path.unlink()
        return self.list_specs()

    def update_ranges(self, updates: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
        for key, (new_min, new_max) in updates.items():
            if key not in self._specs:
                raise KeyError(f"Unknown parameter key '{key}'.")
            spec = self._specs[key]
            if new_min > new_max:
                raise ValueError(f"selected_min ({new_min}) cannot exceed selected_max ({new_max}) for '{key}'.")
            if new_min < spec.hard_min or new_max > spec.hard_max:
                raise ValueError(
                    f"Range [{new_min}, {new_max}] for '{key}' exceeds bounds [{spec.hard_min}, {spec.hard_max}]."
                )
            spec.selected_min = int(round(new_min)) if spec.is_int else round(float(new_min), 4)
            spec.selected_max = int(round(new_max)) if spec.is_int else round(float(new_max), 4)
        self.save()
        return self.list_specs()

    def get(self, key: str) -> ParameterRangeSpec:
        return self._specs[key]

    def get_bounds(self, key: str) -> Tuple[float, float]:
        spec = self._specs[key]
        return float(spec.selected_min), float(spec.selected_max)

    def sample(self, key: str) -> Union[float, int]:
        return self._specs[key].sample()

    def list_specs(self) -> List[Dict[str, Any]]:
        return [spec.to_dict() for spec in self._specs.values()]


PARAMETER_RANGES = ParameterRangeStore()
