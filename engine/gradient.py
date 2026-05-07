import numpy as np
from PIL import Image
import random
import scipy.ndimage
try:
    import noise
except ImportError:
    noise = None

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

class SimplexMeshGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, complexity=5, fluid_mode="random", seed=None):
        if noise is None:
            raise ImportError("The 'noise' package is required for SimplexMeshGradient. Install with 'pip install noise'.")
            
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 1. Base gradient: low-res color grid
        aspect_ratio = self.width / self.height
        min_complexity = max(4, complexity)
        rows = max(2, int(np.sqrt(min_complexity / aspect_ratio)))
        cols = max(2, int(rows * aspect_ratio))
        
        grid = np.zeros((rows, cols, 3), dtype=np.float32)
        num_colors = len(colors)
        colors_arr = np.array(colors, dtype=np.float32)

        for y in range(rows):
            for x in range(cols):
                if fluid_mode == "spatial":
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
                    c1 = np.array(random.choice(colors), dtype=np.float32)
                    c2 = np.array(random.choice(colors), dtype=np.float32)
                    alpha = random.random()
                grid[y, x] = np.sqrt(c1**2 * alpha + c2**2 * (1 - alpha))

        # 2. Coordinate mesh
        x = np.linspace(0, cols - 1, self.width, dtype=np.float32)
        y = np.linspace(0, rows - 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        # 3. Generate Simplex Noise warp
        # Calculate noise on a low-res grid for performance, then scale up
        noise_scale = 0.05 * (10 / complexity)
        noise_res_w = max(2, self.width // 20)
        noise_res_h = max(2, self.height // 20)
        warp_x = np.zeros((noise_res_h, noise_res_w), dtype=np.float32)
        warp_y = np.zeros((noise_res_h, noise_res_w), dtype=np.float32)
        
        offset_x = random.random() * 1000
        offset_y = random.random() * 1000

        for ny in range(noise_res_h):
            for nx in range(noise_res_w):
                warp_x[ny, nx] = noise.snoise2(nx * noise_scale + offset_x, ny * noise_scale + offset_y, octaves=3)
                warp_y[ny, nx] = noise.snoise2(nx * noise_scale + offset_x + 100, ny * noise_scale + offset_y + 100, octaves=3)

        # Scale up the noise grids using bilinear interpolation (order=1) for speed
        zoom_y = self.height / noise_res_h
        zoom_x = self.width / noise_res_w
        warp_x = scipy.ndimage.zoom(warp_x, (zoom_y, zoom_x), order=1)
        warp_y = scipy.ndimage.zoom(warp_y, (zoom_y, zoom_x), order=1)

        # Adjust dimensions if zoom didn't perfectly match (e.g. rounding issues)
        if warp_x.shape[0] > self.height or warp_x.shape[1] > self.width:
            warp_x = warp_x[:self.height, :self.width]
            warp_y = warp_y[:self.height, :self.width]
        elif warp_x.shape[0] < self.height or warp_x.shape[1] < self.width:
            # Pad if slightly smaller
            pad_h = max(0, self.height - warp_x.shape[0])
            pad_w = max(0, self.width - warp_x.shape[1])
            warp_x = np.pad(warp_x, ((0, pad_h), (0, pad_w)), mode='edge')
            warp_y = np.pad(warp_y, ((0, pad_h), (0, pad_w)), mode='edge')

        # Warp coordinates
        # Strength based on complexity and dimensions
        warp_strength = min(cols, rows) * 0.4
        xv_warped = xv + warp_x * warp_strength
        yv_warped = yv + warp_y * warp_strength

        # Clip to valid range so map_coordinates doesn't pull outside colors
        xv_warped = np.clip(xv_warped, 0, cols - 1)
        yv_warped = np.clip(yv_warped, 0, rows - 1)

        # 4. Sample colors
        # map_coordinates requires coordinates in shape (dim, ...), so (2, height, width)
        coords = np.stack((yv_warped, xv_warped))
        
        res = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for i in range(3):
            # Bicubic interpolation (order=3) for smooth color transitions
            res[..., i] = scipy.ndimage.map_coordinates(grid[..., i], coords, order=3, mode='nearest')
        
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
