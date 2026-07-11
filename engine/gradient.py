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
        colors_arr = _colors_to_oklab(colors)

        flip_x = random.random() < 0.5
        flip_y = random.random() < 0.5
        color_order = list(range(num_colors))
        random.shuffle(color_order)

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
                    nx = (1.0 - x / max(1, cols - 1)) if flip_x else x / max(1, cols - 1)
                    ny = (1.0 - y / max(1, rows - 1)) if flip_y else y / max(1, rows - 1)
                    t = (nx + ny) / 2.0
                    scaled = t * (num_colors - 1)
                    lo = int(scaled)
                    hi = min(lo + 1, num_colors - 1)
                    frac = scaled - lo
                    c1 = colors_arr[color_order[lo]]
                    c2 = colors_arr[color_order[hi]]
                    alpha = frac

                grid[y, x] = c1 * (1 - alpha) + c2 * alpha
        
        # Scale factors to reach target resolution
        zoom_y = self.height / rows
        zoom_x = self.width / cols
        
        # Apply bicubic interpolation (order=1) for fluid folds
        # mode='nearest' prevents dark borders
        res = scipy.ndimage.zoom(grid, (zoom_y, zoom_x, 1), order=1, mode='nearest')
        
        return _oklab_image_to_pil(res)

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
        colors_arr = _colors_to_oklab(colors)

        angle = random.uniform(0, 2 * np.pi)
        dx_dir = np.cos(angle)
        dy_dir = np.sin(angle)
        
        color_order = list(range(num_colors))
        random.shuffle(color_order)
        
        t_values = np.zeros((rows, cols), dtype=np.float32)
        for y in range(rows):
            for x in range(cols):
                nx = x / max(1, cols - 1)
                ny = y / max(1, rows - 1)
                t_values[y, x] = nx * dx_dir + ny * dy_dir
                
        t_min, t_max = t_values.min(), t_values.max()
        if t_max > t_min:
            t_values = (t_values - t_min) / (t_max - t_min)
        t_values = t_values * t_values * (3 - 2 * t_values) # Smoothstep

        for y in range(rows):
            for x in range(cols):
                t = t_values[y, x]
                idx_f = t * (num_colors - 1)
                idx0 = int(idx_f)
                idx1 = min(idx0 + 1, num_colors - 1)
                frac = idx_f - idx0
                c1 = colors_arr[color_order[idx0]]
                c2 = colors_arr[color_order[idx1]]
                grid[y, x] = c1 * (1 - frac) + c2 * frac

        # 2. Coordinate mesh
        x = np.linspace(0, cols - 1, self.width, dtype=np.float32)
        y = np.linspace(0, rows - 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        # 3. Generate Simplex Noise warp
        # Calculate noise on a low-res grid for performance, then scale up
        # We randomize noise_scale and warp_strength for varying levels of smoothness per run.
        noise_scale = random.uniform(0.002, 0.015) * (10 / max(1, complexity))
        octaves = random.randint(1, 2)
        
        noise_res_w = max(2, self.width // 10)
        noise_res_h = max(2, self.height // 10)
        warp_x = np.zeros((noise_res_h, noise_res_w), dtype=np.float32)
        warp_y = np.zeros((noise_res_h, noise_res_w), dtype=np.float32)

        offset_x = random.random() * 1000
        offset_y = random.random() * 1000

        for ny in range(noise_res_h):
            for nx in range(noise_res_w):
                warp_x[ny, nx] = noise.snoise2(nx * noise_scale + offset_x, ny * noise_scale + offset_y, octaves=octaves)
                warp_y[ny, nx] = noise.snoise2(nx * noise_scale + offset_x + 100, ny * noise_scale + offset_y + 100, octaves=octaves)

        # Scale up the noise grids using bicubic interpolation (order=1) for smooth warp field
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
        # Keep strength well below 0.5 grid-cells to avoid tearing the field into isolated blobs
        warp_strength = min(cols, rows) * random.uniform(0.05, 0.20)
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
            # Bicubic interpolation (order=1) for smooth color transitions
            res[..., i] = scipy.ndimage.map_coordinates(grid[..., i], coords, order=1, mode='nearest')
        
        return _oklab_image_to_pil(res)


class FlowFieldGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, complexity=5, fluid_mode="random", seed=None):
        if noise is None:
            raise ImportError("The 'noise' package is required for FlowFieldGradient.")
            
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            
        num_colors = len(colors)
        colors_oklab = _colors_to_oklab(colors)

        grid_w, grid_h = max(2, complexity), max(2, complexity)
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
        
        color_order = list(range(num_colors))
        import random
        random.shuffle(color_order)
        angle = random.uniform(0, 2 * np.pi)
        dx_dir = np.cos(angle)
        dy_dir = np.sin(angle)
        
        t_values = np.zeros((grid_h, grid_w), dtype=np.float32)
        for y in range(grid_h):
            for x in range(grid_w):
                nx = x / max(1, grid_w - 1)
                ny = y / max(1, grid_h - 1)
                t_values[y, x] = nx * dx_dir + ny * dy_dir
                
        t_min, t_max = t_values.min(), t_values.max()
        if t_max > t_min:
            t_values = (t_values - t_min) / (t_max - t_min)
        t_values = t_values * t_values * (3 - 2 * t_values)

        for y in range(grid_h):
            for x in range(grid_w):
                t = t_values[y, x]
                idx_f = t * (num_colors - 1)
                idx0 = int(idx_f)
                idx1 = min(idx0 + 1, num_colors - 1)
                frac = idx_f - idx0
                c0 = colors_oklab[color_order[idx0]]
                c1 = colors_oklab[color_order[idx1]]
                grid[y, x] = c0 * (1 - frac) + c1 * frac

        yv, xv = np.mgrid[0:grid_h:self.height*1j, 0:grid_w:self.width*1j]
        
        warp_strength = random.uniform(0.6, 1.8)
        scale = random.uniform(0.003, 0.012)
        
        res_scale = 0.1
        small_h = max(2, int(self.height * res_scale))
        small_w = max(2, int(self.width * res_scale))
        
        ox = random.uniform(0, 1000)
        oy = random.uniform(0, 1000)
        
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
            
        xv_warped = np.clip(xv + dx * warp_strength * grid_w, 0, grid_w - 1)
        yv_warped = np.clip(yv + dy * warp_strength * grid_h, 0, grid_h - 1)
        
        coords = np.stack((yv_warped, xv_warped))
        
        res = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for i in range(3):
            res[..., i] = scipy.ndimage.map_coordinates(grid[..., i], coords, order=1, mode='nearest')
            
        return _oklab_image_to_pil(res)

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
