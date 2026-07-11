import numpy as np
from PIL import Image
import random
import scipy.ndimage
from engine.colors import oklab_to_rgb, rgb_to_oklab
try:
    import noise
except ImportError:
    noise = None


def _interpolate_oklab(c1, c2, alpha):
    """Interpolate sRGB colors through perceptually uniform Oklab."""
    lab1 = rgb_to_oklab(c1)
    lab2 = rgb_to_oklab(c2)
    mixed = lab1 * (1.0 - alpha) + lab2 * alpha
    return np.clip(np.rint(oklab_to_rgb(mixed) * 255.0), 0, 255).astype(np.uint8)


def _colors_to_oklab(colors):
    return rgb_to_oklab(np.asarray(colors, dtype=np.float32))


def _oklab_image_to_pil(values):
    srgb = np.clip(oklab_to_rgb(values), 0.0, 1.0)
    return Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))


def _random_anchor_points(count, margin=0.08):
    """Place color anchors inside the canvas instead of pinning them to edges."""
    return [
        (random.uniform(margin, 1.0 - margin), random.uniform(margin, 1.0 - margin))
        for _ in range(count)
    ]

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
        min_complexity = max(4, complexity)
        rows = max(2, int(np.sqrt(min_complexity / aspect_ratio)))
        cols = max(2, int(rows * aspect_ratio))
        
        # Increase grid size slightly to give bicubic more control points
        rows *= 2
        cols *= 2
        
        grid = np.zeros((rows, cols, 3), dtype=np.float32)
        num_colors = len(colors)
        colors_arr = _colors_to_oklab(colors)

        # Random sweep direction
        angle = random.uniform(0, 2 * np.pi)
        
        t_values = np.zeros((rows, cols))
        for y in range(rows):
            for x in range(cols):
                nx = (x / max(1, cols - 1)) - 0.5
                ny = (y / max(1, rows - 1)) - 0.5
                
                # Rotate for random sweep direction
                t = nx * np.cos(angle) + ny * np.sin(angle)
                
                # Add a massive, low-frequency fold so the wave bends organically
                fold = np.sin(nx * 3.14) * np.cos(ny * 3.14) * 0.25
                
                t_values[y, x] = t + fold

        t_min = t_values.min()
        t_max = t_values.max()
        if t_max > t_min:
            t_values = (t_values - t_min) / (t_max - t_min)
            
        # Optional: Ease-in/out to avoid harsh start/end points
        t_values = np.clip(t_values, 0, 1)
        
        # We need a smooth spline mapping over the colors, so we don't
        # create piecewise-linear plateaus. We use PchipInterpolator here.
        from scipy.interpolate import PchipInterpolator
        nodes = np.linspace(0, 1, num_colors)
        interpolator = PchipInterpolator(nodes, colors_arr, axis=0)
        
        for y in range(rows):
            for x in range(cols):
                grid[y, x] = interpolator(t_values[y, x])
        
        zoom_y = self.height / rows
        zoom_x = self.width / cols
        
        # order=3 (bicubic) eliminates the piecewise linear "plateau" and "white paper edges"
        res = scipy.ndimage.zoom(grid, (zoom_y, zoom_x, 1), order=3, mode='nearest')
        srgb = np.clip(oklab_to_rgb(res), 0.0, 1.0)
        return Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))

class SimplexMeshGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, complexity=5, fluid_mode="random", seed=None):
        """
        Assigns colors to a low-res grid using Simplex noise, then upscales using bicubic interpolation.
        This creates completely unpredictable, organic blobs with no corner anchoring and no flat plateaus.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        aspect_ratio = self.width / self.height
        min_complexity = max(4, complexity)
        rows = max(2, int(np.sqrt(min_complexity / aspect_ratio)))
        cols = max(2, int(rows * aspect_ratio))
        
        # Increase grid size slightly to give bicubic more control points
        rows *= 2
        cols *= 2
        
        grid = np.zeros((rows, cols, 3), dtype=np.float32)
        colors_arr = _colors_to_oklab(colors)

        # To make it unpredictable, we assign colors using a Simplex noise map
        try:
            import noise
            has_noise = True
        except ImportError:
            has_noise = False
            
        if has_noise:
            offset_x = random.uniform(0, 1000)
            offset_y = random.uniform(0, 1000)
            scale = random.uniform(0.3, 0.8)
            for y in range(rows):
                for x in range(cols):
                    val = noise.snoise2(x * scale + offset_x, y * scale + offset_y, octaves=1)
                    val = (val + 1) / 2.0
                    idx = int(val * len(colors))
                    idx = min(len(colors)-1, max(0, idx))
                    grid[y, x] = colors_arr[idx]
        else:
            for y in range(rows):
                for x in range(cols):
                    grid[y, x] = colors_arr[random.randint(0, len(colors)-1)]
                    
        # Smooth the grid values so they blend softly BEFORE the huge bicubic upscaling
        grid = scipy.ndimage.gaussian_filter(grid, sigma=0.8, axes=(0,1))
        
        zoom_y = self.height / rows
        zoom_x = self.width / cols
        
        # order=3 (bicubic) eliminates the piecewise linear "plateau" and "white paper edges"
        res = scipy.ndimage.zoom(grid, (zoom_y, zoom_x, 1), order=3, mode='nearest')
        srgb = np.clip(oklab_to_rgb(res), 0.0, 1.0)
        return Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))


class FlowFieldGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, complexity=5, fluid_mode="random", seed=None):
        if noise is None:
            raise ImportError("The 'noise' package is required for FlowFieldGradient.")
            
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        num_colors = len(colors)
        colors_oklab = _colors_to_oklab(colors)
        
        # FlowField respects the aesthetic palette order (Anchor -> Accents)
        color_order = list(range(num_colors))

        # For FlowField, we use IDW on curl-noise warped coordinates
        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        
        res_scale = 0.1
        small_h = max(2, int(self.height * res_scale))
        small_w = max(2, int(self.width * res_scale))
        
        ox = random.uniform(0, 1000)
        oy = random.uniform(0, 1000)
        
        warp_strength = random.uniform(0.2, 0.6)
        scale = random.uniform(0.002, 0.006) * complexity
        
        dx = np.zeros((small_h, small_w), dtype=np.float32)
        dy = np.zeros((small_h, small_w), dtype=np.float32)
        
        for sy in range(small_h):
            for sx in range(small_w):
                nx, ny = sx * scale, sy * scale
                dx[sy, sx] = noise.pnoise2(nx + ox, ny + oy, octaves=3)
                dy[sy, sx] = noise.pnoise2(nx + ox + 500, ny + oy + 500, octaves=3)
                
        dx = scipy.ndimage.zoom(dx, (self.height / small_h, self.width / small_w), order=1)
        dy = scipy.ndimage.zoom(dy, (self.height / small_h, self.width / small_w), order=1)
        
        if dx.shape != xv.shape:
            dx = np.pad(dx, ((0, max(0, xv.shape[0]-dx.shape[0])), (0, max(0, xv.shape[1]-dx.shape[1]))))[:xv.shape[0], :xv.shape[1]]
            dy = np.pad(dy, ((0, max(0, yv.shape[0]-dy.shape[0])), (0, max(0, yv.shape[1]-dy.shape[1]))))[:yv.shape[0], :yv.shape[1]]
            
        xv_warped = xv + dx * warp_strength
        yv_warped = yv + dy * warp_strength
        
        points = []
        for i in range(num_colors):
            angle = i * (2 * np.pi / num_colors) + random.uniform(0, np.pi)
            dist = random.uniform(0.5, 1.2)
            px = 0.5 + np.cos(angle) * dist
            py = 0.5 + np.sin(angle) * dist
            points.append((px, py))
            
        distances = np.zeros((num_colors, self.height, self.width), dtype=np.float32)
        for i, (px, py) in enumerate(points):
            distances[i] = np.sqrt((xv_warped - px)**2 + (yv_warped - py)**2)
            
        distances[0] *= 0.6
        distances = np.maximum(distances, 0.15)
        
        p = 2.0
        weights = 1.0 / (distances**p)
        sum_weights = np.sum(weights, axis=0)
        weights /= sum_weights
        
        res = np.einsum('ihw,ic->hwc', weights, colors_oklab[color_order])
        srgb = np.clip(oklab_to_rgb(res), 0.0, 1.0)
        return Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))

class MeshGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, points=None, p=2.0, softness=0.12):
        """
        Generates a mesh gradient using Inverse Distance Weighting (IDW).
        
        :param colors: List of (R, G, B) tuples.
        :param points: List of (x, y) tuples in [0, 1] range. If None, random points are used.
        :param p: Power parameter for IDW (usually 2.0).
        :param softness: Minimum normalized anchor radius, avoiding point-like singularities.
        :return: PIL Image.
        """
        num_colors = len(colors)
        if points is None:
            points = _random_anchor_points(num_colors)
        
        # Create coordinate grid
        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        
        # Calculate distances from each pixel to each color point
        # distances shape: (num_colors, height, width)
        distances = np.zeros((num_colors, self.height, self.width), dtype=np.float32)
        for i, (px, py) in enumerate(points):
            distances[i] = np.sqrt((xv - px)**2 + (yv - py)**2)
        
        # Treat each anchor as a soft region rather than an exact point. Besides
        # avoiding division by zero, this prevents repetitive pinprick hotspots.
        distances = np.maximum(distances, softness)
        
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
