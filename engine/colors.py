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
        1. Reduce saturation (Chroma) for background colors.
        2. Set specific pops (Red/Orange) to high saturation.
        3. Ensure high contrast with black/white.
        """
        adjusted = []
        for c in colors:
            rgb = np.array(c) / 255.0
            xyz = colour.sRGB_to_XYZ(rgb)
            lab = colour.XYZ_to_Lab(xyz)
            
            # Check if it's a "pop" color candidate (High Chroma)
            # Convert to LCH for easier adjustment
            lch = colour.Lab_to_LCHab(lab)
            
            # If color is 'vibrant', snap it to Nothing Red or CMF Orange if it's close.
            # Otherwise, mute it.
            if lch[1] > 40: # High Chroma
                # Check hue for red/orange
                hue = lch[2]
                if 0 <= hue < 42: # Red-ish
                    # Snap to Nothing Red
                    adjusted.append(ColorHarmonizer.NOTHING_PALETTE["red"])
                elif 42 <= hue < 75: # Orange-ish
                    # Snap to CMF Orange
                    adjusted.append(ColorHarmonizer.NOTHING_PALETTE["orange"])
                else:
                    # Too colorful but not Red/Orange? Mute it.
                    lch[1] *= 0.3
                    lab_new = colour.LCHab_to_Lab(lch)
                    rgb_new = colour.XYZ_to_sRGB(colour.Lab_to_XYZ(lab_new))
                    adjusted.append(tuple((np.clip(rgb_new, 0, 1) * 255).astype(int)))
            else:
                # Background color. Ensure it's very dark or very light.
                if lch[0] < 50:
                    lch[0] *= 0.5 # Make it darker
                else:
                    lch[0] = min(100, lch[0] * 1.1) # Make it lighter
                
                lab_new = colour.LCHab_to_Lab(lch)
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
