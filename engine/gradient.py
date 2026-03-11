import numpy as np
from PIL import Image
import random

class MeshGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, points=None, p=2.0):
        """
        Generates a mesh gradient using Inverse Distance Weighting (IDW).
        
        :param colors: List of (R, G, B) tuples.
        :param points: List of (x, y) tuples in [0, 1] range. If None, random points are used.
        :param p: Power parameter for IDW (usually 2.0).
        :return: PIL Image.
        """
        num_colors = len(colors)
        if points is None:
            points = [(random.random(), random.random()) for _ in range(num_colors)]
        
        # Create coordinate grid
        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        
        # Calculate distances from each pixel to each color point
        # distances shape: (num_colors, height, width)
        distances = np.zeros((num_colors, self.height, self.width), dtype=np.float32)
        for i, (px, py) in enumerate(points):
            distances[i] = np.sqrt((xv - px)**2 + (yv - py)**2)
        
        # Add a small epsilon to avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Calculate weights: W = 1 / d^p
        weights = 1.0 / (distances**p)
        
        # Sum of weights
        sum_weights = np.sum(weights, axis=0)
        
        # Normalize weights
        weights /= sum_weights
        
        # Interp colors
        # colors_arr shape: (num_colors, 3)
        colors_arr = np.array(colors, dtype=np.float32)
        
        # Final image: (height, width, 3)
        # We multiply (num_colors, height, width) weights with (num_colors, 3) colors
        # einsum is very efficient for this:
        # i = color index, h = height, w = width, c = color channel
        res = np.einsum('ihw,ic->hwc', weights, colors_arr)
        
        return Image.fromarray(res.astype(np.uint8))

if __name__ == "__main__":
    # Test
    engine = MeshGradient(width=1280, height=720)
    # Nothing Signature Palette: Black, White, Red, Orange
    colors = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (255, 80, 0)]
    img = engine.generate(colors)
    img.save("output/test_gradient.png")
    print("Test gradient saved to output/test_gradient.png")
