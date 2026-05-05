import numpy as np

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
