import colorsys
import json
import os
import random
from typing import Any, Dict, List, Optional, Union

from engine.colors import hex_to_rgb, oklch_to_rgb_clamped

# Maximum practical sRGB chroma in OKLCH (used to scale preset 0-1 chroma values)
_OKLCH_MAX_CHROMA = 0.37


def hls_to_rgb(h: float, l: float, s: float) -> tuple[int, int, int]:
    h = max(0.0, min(1.0, h))
    l = max(0.0, min(1.0, l))
    s = max(0.0, min(1.0, s))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def oklch_from_preset(h_norm: float, l: float, c_norm: float, h_deg_offset: float = 0.0) -> tuple[int, int, int]:
    """Convert normalized preset values to a gamut-clamped sRGB tuple via OKLCH.

    h_norm: hue in [0, 1] (maps to [0, 360) degrees)
    l:      lightness in [0, 1] (OKLCH L)
    c_norm: chroma in [0, 1] (maps to [0, _OKLCH_MAX_CHROMA])
    h_deg_offset: additional hue offset in degrees
    """
    H = (h_norm * 360.0 + h_deg_offset) % 360.0
    L = max(0.0, min(1.0, l))
    C = max(0.0, c_norm * _OKLCH_MAX_CHROMA)
    return oklch_to_rgb_clamped([L, C, H])


class PresetEngine:
    def __init__(self, preset_dir: str = "presets"):
        self.preset_dir = preset_dir

    def load_preset(self, preset_id: str) -> Dict[str, Any]:
        """Loads a JSON preset file from disk."""
        path = os.path.join(self.preset_dir, f"{preset_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preset file not found: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def get_preset_version(self, preset_data: Dict[str, Any]) -> str:
        """Returns the semantic version of the preset data, defaulting to '1.0.0'."""
        return str(preset_data.get("version", "1.0.0"))

    def sample_distribution(self, config: Dict[str, Any]) -> Union[float, int]:
        """Samples a value based on a distribution configuration."""
        dist_type = config.get("type", "uniform")

        if dist_type == "uniform_int":
            return random.randint(int(config.get("min", 2)), int(config.get("max", 5)))
        if dist_type == "uniform":
            val = random.uniform(config.get("min", 0.0), config.get("max", 1.0))
        elif dist_type == "gaussian":
            val = random.gauss(config.get("mu", 0.5), config.get("sigma", 0.1))
        elif dist_type == "beta":
            val = random.betavariate(config.get("alpha", 1), config.get("beta", 1))
            if "scale" in config:
                val *= config["scale"]
            if "offset" in config:
                val += config["offset"]
        elif dist_type == "bimodal":
            modes = config.get("modes", [])
            roll = random.random()
            cumulative = 0.0
            chosen_mode = modes[-1]
            for mode in modes:
                cumulative += mode.get("weight", 0.5)
                if roll < cumulative:
                    chosen_mode = mode
                    break
            return self.sample_distribution(chosen_mode)
        elif dist_type == "relative":
            return 0.0
        else:
            val = random.random()

        if "clip" in config:
            c_min, c_max = config["clip"]
            val = max(c_min, min(c_max, val))

        return int(round(val)) if dist_type.endswith("int") else val

    _sample_distribution = sample_distribution

    def generate_palette(
        self,
        preset: Dict[str, Any],
        count: Optional[int] = None,
        family_index: Optional[int] = None,
    ) -> List[tuple[int, int, int]]:
        """Generates a color palette based on preset distribution rules."""
        palette_families = preset.get("palette_families", [])
        if palette_families:
            family = (
                palette_families[family_index % len(palette_families)]
                if family_index is not None
                else random.choice(palette_families)
            )
            colors = [hex_to_rgb(color) for color in family]
            if count is None:
                return colors
            return [colors[index % len(colors)] for index in range(count)]

        if count is None:
            count_conf = preset.get("color_count", {"type": "uniform_int", "min": 2, "max": 5})
            count = int(self.sample_distribution({"type": "uniform_int", **count_conf}))

        structure = preset.get("palette_structure", [])
        if not structure:
            return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(count)]

        use_oklch = preset.get("color_space", "hls") == "oklch"
        colors: List[tuple[int, int, int]] = []

        # 1. Generate the Anchor (always first in structure)
        anchor_rule = next((r for r in structure if r.get("role") == "grounding_anchor"), structure[0])
        anchor_hue = float(self.sample_distribution(anchor_rule.get("hue", {})))
        anchor_l = float(self.sample_distribution(anchor_rule.get("lightness", {})))
        anchor_c = float(self.sample_distribution(anchor_rule.get("chroma", {})))
        if use_oklch:
            colors.append(oklch_from_preset(anchor_hue, anchor_l, anchor_c))
        else:
            colors.append(hls_to_rgb(anchor_hue, anchor_l, anchor_c))

        # 2. Generate Accents
        accent_rule = next((r for r in structure if r.get("role") == "accent_core"), structure[-1])
        n_accents = count - 1
        hue_conf = accent_rule.get("hue", {})
        is_relative = hue_conf.get("type") == "relative" and hue_conf.get("target") == "grounding_anchor"

        if is_relative and n_accents > 1:
            base_offset = random.gauss(hue_conf.get("mu_offset", 0), max(5.0, hue_conf.get("sigma", 15) * 0.4))
            step = random.choice([24.0, 30.0, 36.0, -24.0, -30.0, -36.0])
        else:
            base_offset = hue_conf.get("mu_offset", 0)
            step = 0.0

        for idx in range(n_accents):
            if is_relative:
                if n_accents > 1:
                    offset_deg = base_offset + idx * step
                else:
                    offset_deg = random.gauss(hue_conf.get("mu_offset", 0), hue_conf.get("sigma", 15))

                if use_oklch:
                    h = anchor_hue
                    h_offset = offset_deg
                else:
                    h = (anchor_hue + offset_deg / 360.0) % 1.0
                    h_offset = 0.0
            else:
                h = float(self.sample_distribution(hue_conf))
                h_offset = 0.0

            l = float(self.sample_distribution(accent_rule.get("lightness", {})))
            c = float(self.sample_distribution(accent_rule.get("chroma", {})))
            if n_accents > 1:
                t = (idx + 1) / n_accents
                l = max(0.0, min(1.0, l * (0.88 + 0.24 * t)))
                c = max(0.0, min(1.0, c * (0.82 + 0.28 * t)))

            if use_oklch:
                colors.append(oklch_from_preset(h, l, c, h_deg_offset=h_offset))
            else:
                colors.append(hls_to_rgb(h, l, c))

        return colors

    def get_engine_overrides(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts default engine/filter parameters defined by the preset."""
        return preset.get("overrides", {})
