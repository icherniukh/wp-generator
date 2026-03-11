import numpy as np
import colour
from PIL import Image

class ColorHarmonizer:
    NOTHING_PALETTE = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "dark_gray": (30, 30, 30),
        "red": (255, 0, 0),
        "orange": (255, 80, 0),
        "milky": (240, 240, 240)
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
            
            if C > 35: # Vibrant color
                # Push lightness to extremes to make it 'punchy'
                if L < 60:
                    L = max(15, L * 0.8) # Deep, rich tone
                else:
                    L = min(95, L * 1.1) # Bright, icy tone
                
                # Optionally snap exactly to Nothing Red/Orange if it's very close
                if 10 <= H < 42 and C > 50:
                    adjusted.append(ColorHarmonizer.NOTHING_PALETTE["red"])
                    continue
                elif 42 <= H < 75 and C > 50:
                    adjusted.append(ColorHarmonizer.NOTHING_PALETTE["orange"])
                    continue
                
                # Keep the user's hue, slightly desaturate to look 'industrial'
                C = C * 0.9
                
            else: # Muted or background color
                # Push towards pure black or white/light-gray
                if L < 50:
                    L = max(5, L * 0.4) # Make it very dark
                    C = C * 0.1 # Kill saturation
                else:
                    L = min(98, max(85, L * 1.2)) # Make it light
                    C = C * 0.1 # Kill saturation
            
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
