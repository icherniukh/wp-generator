from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# OKLCH / OKLab color space (Björn Ottosson, 2020)
# https://bottosson.github.io/posts/oklab/
# L ∈ [0, 1], C ∈ [0, ~0.37], H ∈ [0, 360)
# ---------------------------------------------------------------------------

# sRGB linear ↔ OKLab matrices
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)
_M1_INV = np.array([
    [ 4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010],
], dtype=np.float32)
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float32)
_M2_INV = np.array([
    [1.0000000000,  0.3963377774,  0.2158037573],
    [1.0000000000, -0.1055613458, -0.0638541728],
    [1.0000000000, -0.0894841775, -1.2914855480],
], dtype=np.float32)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c)
    result = np.empty_like(c, dtype=np.result_type(c, np.float32))
    linear_mask = c <= 0.0031308
    result[linear_mask] = 12.92 * c[linear_mask]
    result[~linear_mask] = 1.055 * np.power(c[~linear_mask], 1.0 / 2.4) - 0.055
    return result


def rgb_to_oklab(rgb) -> np.ndarray:
    """Convert one or more sRGB colors to Oklab, preserving the leading shape."""
    rgb = np.asarray(rgb)
    is_int = np.issubdtype(rgb.dtype, np.integer)
    rgb = rgb.astype(np.float32)
    if is_int or (rgb.size and rgb.max() > 1.0):
        rgb = rgb / 255.0
    linear = _srgb_to_linear(rgb)
    lms = linear @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_rgb(oklab) -> np.ndarray:
    """Convert one or more Oklab colors to unclipped sRGB in nominal [0, 1]."""
    oklab = np.asarray(oklab, dtype=np.float32)
    lms_root = oklab @ _M2_INV.T
    linear = (lms_root ** 3) @ _M1_INV.T
    return _linear_to_srgb(linear)


def rgb_to_oklch(rgb) -> np.ndarray:
    """Convert sRGB (0-255 int or 0-1 float) → OKLCH [L, C, H°]."""
    lab = rgb_to_oklab(rgb)
    L, a, b = lab
    C = np.hypot(a, b)
    H = np.degrees(np.arctan2(b, a)) % 360.0
    return np.array([L, C, H])


def oklch_to_rgb(lch) -> np.ndarray:
    """Convert OKLCH [L, C, H°] → sRGB numpy array in [0, 1]. May exceed [0,1] if out of gamut."""
    L, C, H = lch
    h_rad = np.radians(H)
    oklab = np.array([L, C * np.cos(h_rad), C * np.sin(h_rad)])
    return oklab_to_rgb(oklab)


def clamp_oklch(lch, *, lightness_bounds=(0.0, 1.0), chroma_bounds=(0.0, 0.37)) -> np.ndarray:
    """Clamp lightness and chroma in OKLCH, retaining the circular hue."""
    values = np.asarray(lch, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("OKLCH must contain three finite values.")
    l_min, l_max = lightness_bounds
    c_min, c_max = chroma_bounds
    if not (0.0 <= l_min <= l_max <= 1.0):
        raise ValueError("Lightness bounds must satisfy 0 <= min <= max <= 1.")
    if not (0.0 <= c_min <= c_max and np.isfinite(c_max)):
        raise ValueError("Chroma bounds must be finite, nonnegative, and ordered.")
    return np.array([
        np.clip(values[0], l_min, l_max),
        np.clip(values[1], c_min, c_max),
        values[2] % 360.0,
    ])


def oklch_to_rgb_clamped(lch) -> tuple:
    """Convert OKLCH → sRGB (0-255 int tuple), clamping chroma to stay in gamut.

    Bounds L to [0, 1] and C to [0, 0.37], then reduces C until sRGB is in gamut.
    Preserves hue and the bounded lightness before 8-bit rounding.
    """
    L, C, H = clamp_oklch(lch)
    rgb = oklch_to_rgb([L, C, H])
    if np.all((rgb >= 0.0) & (rgb <= 1.0)):
        return tuple(int(round(v * 255)) for v in rgb)

    # Binary-search chroma down to gamut boundary
    lo, hi = 0.0, C
    for _ in range(20):
        mid = (lo + hi) / 2.0
        rgb_mid = oklch_to_rgb([L, mid, H])
        if np.all((rgb_mid >= 0.0) & (rgb_mid <= 1.0)):
            lo = mid
        else:
            hi = mid

    rgb = np.clip(oklch_to_rgb([L, lo, H]), 0.0, 1.0)
    return tuple(int(round(v * 255)) for v in rgb)


def relative_luminance(rgb) -> float:
    """WCAG sRGB relative luminance for an RGB color on the 0-255 scale."""
    channels = np.asarray(rgb, dtype=np.float64)
    if channels.shape != (3,) or not np.all(np.isfinite(channels)) or np.any((channels < 0) | (channels > 255)):
        raise ValueError("RGB must contain three finite channels between 0 and 255.")
    return float(_srgb_to_linear(channels / 255.0) @ [0.2126, 0.7152, 0.0722])


def contrast_ratio(first, second) -> float:
    """Return the WCAG luminance contrast ratio of two RGB colors."""
    light, dark = sorted((relative_luminance(first), relative_luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


@dataclass(frozen=True)
class PaletteContrast:
    """Contrast of each accent against the first (anchor) palette color."""
    ratios: tuple[float, ...]
    threshold: float = 4.5

    @property
    def passes(self) -> bool:
        return bool(self.ratios) and all(ratio >= self.threshold for ratio in self.ratios)


def evaluate_palette_contrast(colors, threshold=4.5) -> PaletteContrast:
    """Evaluate final palette pairs without changing the colors or rounding ratios."""
    if not 1.0 <= threshold <= 21.0:
        raise ValueError("Contrast threshold must be between 1 and 21.")
    ratios = tuple(contrast_ratio(colors[0], color) for color in colors[1:]) if len(colors) > 1 else ()
    return PaletteContrast(ratios, threshold)


class StyleHarmonizer:
    NOTHING_PALETTE = {
        "black": (15, 15, 18),
        "white": (220, 220, 225),
        "dark_gray": (35, 35, 40),
        "red": (190, 40, 40),
        "orange": (210, 80, 30),
        "milky": (210, 210, 215)
    }

    @staticmethod
    def evaluate_fit(colors):
        """
        Evaluates palette contrast using CIELAB space.
        Returns a contrast score.
        """
        lab_colors = [rgb_to_lab(c) for c in colors]
        
        # Calculate Delta E distances (76 or 2000)
        distances = []
        for i in range(len(lab_colors)):
            for j in range(i + 1, len(lab_colors)):
                d_e = np.linalg.norm(lab_colors[i] - lab_colors[j])
                distances.append(d_e)
        
        # In Nothing style, we want high contrast (large Delta E) 
        # or very close monochrome (small Delta E) with 1 pop color (large Delta E).
        return np.mean(distances)

    @staticmethod
    def adjust_for_nothing(colors):
        """Preserve hue while bounding vivid accents and muted grounding tones in OKLCH."""
        adjusted = []
        for color in colors:
            lch = rgb_to_oklch(color)
            if lch[1] > 0.1:
                bounds = {"lightness_bounds": (0.2, 0.85), "chroma_bounds": (0.0, 0.22)}
            else:
                lightness = (0.12, 0.35) if lch[0] < 0.5 else (0.75, 0.88)
                bounds = {"lightness_bounds": lightness, "chroma_bounds": (0.0, 0.015)}
            adjusted.append(oklch_to_rgb_clamped(clamp_oklch(lch, **bounds)))
        return adjusted


def rgb_to_lab(rgb):
    rgb = np.array(rgb, dtype=np.float64)
    if np.max(rgb) > 1:
        rgb = rgb / 255.0

    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]) @ linear
    return xyz_to_lab(xyz)


def lab_to_rgb(lab):
    xyz = lab_to_xyz(lab)
    linear = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]) @ xyz
    linear = np.clip(linear, 0, 1)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * (linear ** (1 / 2.4)) - 0.055,
    )


def xyz_to_lab(xyz):
    white = np.array([0.95047, 1.0, 1.08883])
    ratio = xyz / white
    delta = 6 / 29
    f = np.where(
        ratio > delta ** 3,
        np.cbrt(ratio),
        ratio / (3 * delta ** 2) + 4 / 29,
    )
    return np.array([
        116 * f[1] - 16,
        500 * (f[0] - f[1]),
        200 * (f[1] - f[2]),
    ])


def lab_to_xyz(lab):
    white = np.array([0.95047, 1.0, 1.08883])
    L, a, b = lab
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    delta = 6 / 29
    f = np.array([fx, fy, fz])
    ratio = np.where(
        f > delta,
        f ** 3,
        3 * delta ** 2 * (f - 4 / 29),
    )
    return ratio * white


def lab_to_lch(lab):
    L, a, b = lab
    C = np.hypot(a, b)
    H = np.degrees(np.arctan2(b, a)) % 360
    return np.array([L, C, H])


def lch_to_lab(lch):
    L, C, H = lch
    h_rad = np.radians(H)
    return np.array([L, C * np.cos(h_rad), C * np.sin(h_rad)])


class ColorTheoryHarmonizer:
    """Preserve hues by default; retain explicitly selected classical LCH harmony modes."""

    MODES = ["complementary", "analogous", "triadic", "split_complementary", "tetradic"]

    _HUE_OFFSETS = {
        "complementary":       [0, 180, 30],
        "analogous":           [0, 30, -30],
        "triadic":             [0, 120, 240],
        "split_complementary": [0, 150, 210],
        "tetradic":            [0, 90, 180],
    }

    @classmethod
    def harmonize(cls, colors, mode=None, *, lightness_bounds=(0.12, 0.90), chroma_bounds=(0.0, 0.25)):
        if mode is None or mode == "perceptual":
            return [
                oklch_to_rgb_clamped(clamp_oklch(
                    rgb_to_oklch(color), lightness_bounds=lightness_bounds,
                    chroma_bounds=chroma_bounds,
                ))
                for color in colors
            ]

        offsets = cls._HUE_OFFSETS[mode]

        # Find seed hue from the most chromatic input color
        lchs = [lab_to_lch(rgb_to_lab(c)) for c in colors]
        seed_hue = max(lchs, key=lambda lch: lch[1])[2]

        # Sort inputs by lightness so output preserves dark/light character
        sorted_lchs = sorted(lchs, key=lambda lch: lch[0])

        result = []
        for i, lch in enumerate(sorted_lchs):
            L, C, _ = lch
            # Determine offset based on i
            # If there are more colors than offsets, we vary Lightness and Chroma to create tonal variations
            base_offset = offsets[i % len(offsets)]
            new_H = (seed_hue + base_offset) % 360
            
            # Tonal variation for N > len(offsets)
            cycle = i // len(offsets)
            if cycle > 0:
                L = max(10, min(90, L + (cycle * 10 * (1 if cycle % 2 == 0 else -1))))
                new_C = max(C * (0.8 ** cycle), 10.0)
            else:
                new_C = max(C, 10.0)
                
            lab_new = lch_to_lab(np.array([L, new_C, new_H]))
            rgb_new = lab_to_rgb(lab_new)
            result.append(tuple(int(v) for v in (np.clip(rgb_new, 0, 1) * 255).astype(int)))

        return result


# Backward-compatible alias for older imports.
ColorHarmonizer = StyleHarmonizer

if __name__ == "__main__":
    # Test
    input_colors = [(100, 50, 50), (200, 100, 0)] # Jarring muddy red and orange
    score = ColorHarmonizer.evaluate_fit(input_colors)
    print(f"Initial Harmony Score: {score:.2f}")
    
    adjusted = StyleHarmonizer.adjust_for_nothing(input_colors)
    print(f"Adjusted Colors: {adjusted}")
    
    score_new = ColorHarmonizer.evaluate_fit(adjusted)
    print(f"New Harmony Score: {score_new:.2f}")
