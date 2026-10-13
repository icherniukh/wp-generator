import json
import os
import random
import math
import colorsys
from typing import Dict, Any, List

from engine.colors import oklch_to_rgb_clamped

# Maximum practical sRGB chroma in OKLCH (used to scale preset 0-1 chroma values)
_OKLCH_MAX_CHROMA = 0.37


def hls_to_rgb(h, l, s):
    h = max(0.0, min(1.0, h))
    l = max(0.0, min(1.0, l))
    s = max(0.0, min(1.0, s))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def oklch_from_preset(h_norm, l, c_norm, h_deg_offset=0.0):
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
    def __init__(self, preset_dir="presets"):
        self.preset_dir = preset_dir

    def load_preset(self, preset_id: str) -> Dict[str, Any]:
        """Loads a JSON preset file from disk."""
        path = os.path.join(self.preset_dir, f"{preset_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preset file not found: {path}")
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse preset '{preset_id}': {e}")
            return {}

    def _sample_distribution(self, config: Dict[str, Any]) -> float:
        """Samples a value based on a distribution configuration."""
        dist_type = config.get("type", "uniform")
        
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
            # Pick a mode based on weights
            modes = config.get("modes", [])
            roll = random.random()
            cumulative = 0.0
            chosen_mode = modes[-1] # fallback
            for mode in modes:
                cumulative += mode.get("weight", 0.5)
                if roll < cumulative:
                    chosen_mode = mode
                    break
            return self._sample_distribution(chosen_mode)
        elif dist_type == "relative":
            # To implement relative, we'd need context. 
            # For simplicity in this generator, we handle relative logic explicitly in generate_palette.
            return 0.0
        else:
            val = random.random()
            
        # Apply clipping if defined
        if "clip" in config:
            c_min, c_max = config["clip"]
            val = max(c_min, min(c_max, val))
            
        return val

    def generate_palette(self, preset: Dict[str, Any], count: int = None, family_index: int = None) -> List[tuple]:
        """Generates a color palette based on preset distribution rules."""
        palette_families = preset.get("palette_families", [])
        if palette_families:
            family = (
                palette_families[family_index % len(palette_families)]
                if family_index is not None
                else random.choice(palette_families)
            )
            colors = [
                tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
                for color in family
            ]
            if count is None:
                return colors
            return [colors[index % len(colors)] for index in range(count)]

        if count is None:
            count_conf = preset.get("color_count", {"min": 2, "max": 5})
            count = random.randint(count_conf.get("min", 2), count_conf.get("max", 5))

        structure = preset.get("palette_structure", [])
        if not structure:
            return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(count)]

        use_oklch = preset.get("color_space", "hls") == "oklch"
        colors = []

        # 1. Generate the Anchor (always first in structure)
        anchor_rule = next((r for r in structure if r.get("role") == "grounding_anchor"), structure[0])
        anchor_hue = self._sample_distribution(anchor_rule.get("hue", {}))
        anchor_l = self._sample_distribution(anchor_rule.get("lightness", {}))
        anchor_c = self._sample_distribution(anchor_rule.get("chroma", {}))
        if use_oklch:
            colors.append(oklch_from_preset(anchor_hue, anchor_l, anchor_c))
        else:
            colors.append(hls_to_rgb(anchor_hue, anchor_l, anchor_c))

        # 2. Generate Accents
        accent_rule = next((r for r in structure if r.get("role") == "accent_core"), structure[-1])
        for _ in range(count - 1):
            hue_conf = accent_rule.get("hue", {})
            if hue_conf.get("type") == "relative" and hue_conf.get("target") == "grounding_anchor":
                offset_deg = random.gauss(hue_conf.get("mu_offset", 0), hue_conf.get("sigma", 15))
                if use_oklch:
                    h = anchor_hue
                    h_offset = offset_deg
                else:
                    h = (anchor_hue + offset_deg / 360.0) % 1.0
                    h_offset = 0.0
            else:
                h = self._sample_distribution(hue_conf)
                h_offset = 0.0

            l = self._sample_distribution(accent_rule.get("lightness", {}))
            c = self._sample_distribution(accent_rule.get("chroma", {}))
            if use_oklch:
                colors.append(oklch_from_preset(h, l, c, h_deg_offset=h_offset))
            else:
                colors.append(hls_to_rgb(h, l, c))

        return colors

    def get_engine_overrides(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts default engine/filter parameters defined by the preset."""
        return preset.get("overrides", {})
