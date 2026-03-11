import numpy as np
import colour
from PIL import Image

class ColorHarmonizer:
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
        Evaluates the fit of the input colors using CIELAB space.
        Returns a 'harmony score'.
        """
        # Convert RGB to CIELAB
        lab_colors = [colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.array(c)/255.0)) for c in colors]
        
        # Calculate Delta E distances (76 or 2000)
        distances = []
        for i in range(len(lab_colors)):
            for j in range(i + 1, len(lab_colors)):
                d_e = colour.delta_E(lab_colors[i], lab_colors[j])
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
            xyz = colour.sRGB_to_XYZ(rgb)
            lab = colour.XYZ_to_Lab(xyz)
            
            # Convert to LCH for easier adjustment
            lch = colour.Lab_to_LCHab(lab)
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
            lab_new = colour.LCHab_to_Lab(lch_new)
            rgb_new = colour.XYZ_to_sRGB(colour.Lab_to_XYZ(lab_new))
            adjusted.append(tuple((np.clip(rgb_new, 0, 1) * 255).astype(int)))
        
        return adjusted

if __name__ == "__main__":
    # Test
    input_colors = [(100, 50, 50), (200, 100, 0)] # Jarring muddy red and orange
    score = ColorHarmonizer.evaluate_fit(input_colors)
    print(f"Initial Harmony Score: {score:.2f}")
    
    adjusted = ColorHarmonizer.adjust_for_nothing(input_colors)
    print(f"Adjusted Colors: {adjusted}")
    
    score_new = ColorHarmonizer.evaluate_fit(adjusted)
    print(f"New Harmony Score: {score_new:.2f}")
