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
])
_M1_INV = np.array([
    [ 4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010],
])
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
])
_M2_INV = np.array([
    [1.0000000000,  0.3963377774,  0.2158037573],
    [1.0000000000, -0.1055613458, -0.0638541728],
    [1.0000000000, -0.0894841775, -1.2914855480],
])


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


def rgb_to_oklch(rgb) -> np.ndarray:
    """Convert sRGB (0-255 int or 0-1 float) → OKLCH [L, C, H°]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    linear = _srgb_to_linear(rgb)
    lms = _M1 @ linear
    lms_ = np.cbrt(np.maximum(lms, 0.0))
    lab = _M2 @ lms_
    L, a, b = lab
    C = np.hypot(a, b)
    H = np.degrees(np.arctan2(b, a)) % 360.0
    return np.array([L, C, H])


def oklch_to_rgb(lch) -> np.ndarray:
    """Convert OKLCH [L, C, H°] → sRGB numpy array in [0, 1]. May exceed [0,1] if out of gamut."""
    L, C, H = lch
    h_rad = np.radians(H)
    oklab = np.array([L, C * np.cos(h_rad), C * np.sin(h_rad)])
    lms_ = _M2_INV @ oklab          # OKLab → LMS'
    lms = lms_ ** 3                  # LMS' → LMS (undo cube-root)
    linear = _M1_INV @ lms           # LMS → linear sRGB
    return _linear_to_srgb(linear)


def oklch_to_rgb_clamped(lch) -> tuple:
    """Convert OKLCH → sRGB (0-255 int tuple), clamping chroma to stay in gamut.

    Preserves hue and lightness; reduces C via binary search until all channels ∈ [0, 1].
    """
    L, C, H = float(lch[0]), float(lch[1]), float(lch[2])
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
        """
        Adjusts colors to fit the 'Nothing' aesthetic.
        Rules: 
        1. Preserve user intent (Hue) but map Lightness and Chroma.
        2. High-chroma colors get contrast-boosted (very bright or deep dark).
        3. Background/low-chroma colors are pushed to strict monochrome extremes.
        """
        adjusted = []
        for c in colors:
            rgb = np.array(c) / 255.0
            lab = rgb_to_lab(rgb)
            
            # Convert to LCH for easier adjustment
            lch = lab_to_lch(lab)
            L, C, H = lch
            
            if C > 30: # Vibrant color
                # Push lightness to extremes to make it 'punchy' but soften them
                if L < 60:
                    L = max(20, L * 0.9) # Deep, rich tone, not pure black
                else:
                    L = min(85, L * 1.0) # Bright tone, not blinding white
                
                # Keep the user's hue, desaturate more significantly to look elegant
                C = C * 0.7
                
            else: # Muted or background color
                # Push towards dark gray or off-white
                if L < 50:
                    L = max(12, L * 0.7) # Make it very dark but not pitch black
                    C = C * 0.1 # Kill saturation
                else:
                    L = min(88, max(75, L * 1.1)) # Make it light, but not pure white
                    C = C * 0.15 # Allow a tiny bit of warmth/coolness
            
            # Reconstruct color
            lch_new = np.array([L, C, H])
            lab_new = lch_to_lab(lch_new)
            rgb_new = lab_to_rgb(lab_new)
            adjusted.append(tuple(int(v) for v in (np.clip(rgb_new, 0, 1) * 255).astype(int)))
        
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
    """Applies classical color harmony rules (hue rotation in LCH space)."""

    MODES = ["complementary", "analogous", "triadic", "split_complementary", "tetradic"]

    _HUE_OFFSETS = {
        "complementary":       [0, 180, 30],
        "analogous":           [0, 30, -30],
        "triadic":             [0, 120, 240],
        "split_complementary": [0, 150, 210],
        "tetradic":            [0, 90, 180],
    }

    @classmethod
    def harmonize(cls, colors, mode=None):
        import random as _rnd
        if mode is None:
            mode = _rnd.choice(cls.MODES)

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
