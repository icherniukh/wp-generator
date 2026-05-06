import numpy as np
from PIL import Image
import random
import scipy.ndimage

class FluidMeshGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, complexity=5, fluid_mode="random"):
        """
        Generates a fluid mesh gradient using bicubic interpolation of a low-res color grid.
        Produces much more organic, overlapping folds of color than IDW.
        
        :param complexity: Approximates the number of "points" or "directions" of color.
                           We map this to a grid size to ensure the total number of points 
                           roughly matches the requested complexity.
        :param fluid_mode: "random" or "spatial".
        """
        # Determine grid size based on target complexity and aspect ratio.
        # We want (cols * rows) roughly equal to complexity.
        aspect_ratio = self.width / self.height
        
        # Ensure we have at least a 2x2 grid for interpolation to work smoothly
        min_complexity = max(4, complexity)
        
        # Math: cols * rows = min_complexity
        # cols / rows = aspect_ratio -> cols = rows * aspect_ratio
        # rows^2 * aspect_ratio = min_complexity
        rows = max(2, int(np.sqrt(min_complexity / aspect_ratio)))
        cols = max(2, int(rows * aspect_ratio))
        
        # Create a low-res color grid
        grid = np.zeros((rows, cols, 3), dtype=np.float32)
        
        num_colors = len(colors)
        colors_arr = np.array(colors, dtype=np.float32)

        for y in range(rows):
            for x in range(cols):
                if fluid_mode == "spatial":
                    # Spatial mode: map position to color index using sine waves
                    nx = x / max(1, cols - 1)
                    ny = y / max(1, rows - 1)
                    
                    val1 = np.sin(nx * np.pi + ny * np.pi) * 0.5 + 0.5
                    val2 = np.cos(nx * 2 * np.pi - ny * np.pi) * 0.5 + 0.5
                    
                    idx1 = int(val1 * (num_colors - 1))
                    idx2 = int(val2 * (num_colors - 1))
                    
                    c1 = colors_arr[idx1 % num_colors]
                    c2 = colors_arr[idx2 % num_colors]
                    alpha = (val1 + val2) / 2
                else:
                    # Blend 2 random colors for smooth transitions in the base grid
                    c1 = np.array(random.choice(colors), dtype=np.float32)
                    c2 = np.array(random.choice(colors), dtype=np.float32)
                    alpha = random.random()
                
                # Gamma-corrected blending (sqrt of sum of squares)
                grid[y, x] = np.sqrt(c1**2 * alpha + c2**2 * (1 - alpha))
        
        # Scale factors to reach target resolution
        zoom_y = self.height / rows
        zoom_x = self.width / cols
        
        # Apply bicubic interpolation (order=3) for fluid folds
        # mode='nearest' prevents dark borders
        res = scipy.ndimage.zoom(grid, (zoom_y, zoom_x, 1), order=3, mode='nearest')
        
        # Clip to valid RGB range and convert
        res = np.clip(res, 0, 255).astype(np.uint8)
        return Image.fromarray(res)

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
