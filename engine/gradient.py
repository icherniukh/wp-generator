import numpy as np
from PIL import Image
import random
import scipy.ndimage
from scipy.interpolate import PchipInterpolator
from engine.colors import oklab_to_rgb, rgb_to_oklab

try:
    import noise
except ImportError:
    noise = None


def _colors_to_oklab(colors):
    return rgb_to_oklab(np.asarray(colors, dtype=np.float32))

def _random_anchor_points(count, margin=0.08):
    """Place color anchors inside the canvas instead of pinning them to edges."""
    return [
        (random.uniform(margin, 1.0 - margin), random.uniform(margin, 1.0 - margin))
        for _ in range(count)
    ]

class MeshGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, p=2.0, softness=0.12, seed=None):
        """
        Legacy IDW (Inverse Distance Weighting) gradient for soft radial puddles.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        num_colors = len(colors)
        points = _random_anchor_points(num_colors)
        
        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        
        distances = np.zeros((num_colors, self.height, self.width), dtype=np.float32)
        for i, (px, py) in enumerate(points):
            distances[i] = np.sqrt((xv - px)**2 + (yv - py)**2)
        
        distances = np.maximum(distances, softness)
        weights = 1.0 / (distances**p)
        weights /= np.sum(weights, axis=0)
        
        colors_arr = np.array(colors, dtype=np.float32)
        res = np.einsum('ihw,ic->hwc', weights, colors_arr)
        return Image.fromarray(res.astype(np.uint8))


class FluidMeshGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, complexity=5, fluid_mode="random", seed=None):
        """
        Original blocky linear sweep using scipy.ndimage.zoom.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        aspect_ratio = self.width / self.height
        min_complexity = max(4, complexity)
        rows = max(2, int(np.sqrt(min_complexity / aspect_ratio)))
        cols = max(2, int(rows * aspect_ratio))
        
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
        
        zoom_y = self.height / rows
        zoom_x = self.width / cols
        res = scipy.ndimage.zoom(grid, (zoom_y, zoom_x, 1), order=1, mode='nearest')
        srgb = np.clip(oklab_to_rgb(res), 0.0, 1.0)
        return Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))


class FlowFieldGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, complexity=5, seed=None, fluid_mode="random"):
        """
        Uses a random 1D directional sweep over the canvas, warped by Perlin noise,
        and smoothly interpolates colors using a Monotonic Cubic Spline (PCHIP) in Oklab space.
        This produces gorgeous, sweeping waves of color that fold organically.
        """
        if noise is None:
            raise ImportError("The 'noise' package is required.")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        num_colors = len(colors)
        colors_oklab = _colors_to_oklab(colors)

        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        
        res_scale = 0.1
        small_h = max(2, int(self.height * res_scale))
        small_w = max(2, int(self.width * res_scale))
        
        ox = random.uniform(0, 1000)
        oy = random.uniform(0, 1000)
        
        warp_strength = random.uniform(0.1, 0.4)
        scale = random.uniform(0.002, 0.006) * complexity
        
        dx = np.zeros((small_h, small_w), dtype=np.float32)
        dy = np.zeros((small_h, small_w), dtype=np.float32)
        
        for sy in range(small_h):
            for sx in range(small_w):
                nx, ny = sx * scale, sy * scale
                dx[sy, sx] = noise.pnoise2(nx + ox, ny + oy, octaves=2)
                dy[sy, sx] = noise.pnoise2(nx + ox + 500, ny + oy + 500, octaves=2)
                
        dx = scipy.ndimage.zoom(dx, (self.height / small_h, self.width / small_w), order=1)
        dy = scipy.ndimage.zoom(dy, (self.height / small_h, self.width / small_w), order=1)
        
        if dx.shape != xv.shape:
            dx = np.pad(dx, ((0, max(0, xv.shape[0]-dx.shape[0])), (0, max(0, xv.shape[1]-dx.shape[1]))))[:xv.shape[0], :xv.shape[1]]
            dy = np.pad(dy, ((0, max(0, yv.shape[0]-dy.shape[0])), (0, max(0, yv.shape[1]-dy.shape[1]))))[:yv.shape[0], :yv.shape[1]]
            
        xv_warped = xv + dx * warp_strength
        yv_warped = yv + dy * warp_strength
        
        # Random diagonal/directional sweep
        angle = random.uniform(0, 2 * np.pi)
        dx_dir = np.cos(angle)
        dy_dir = np.sin(angle)
        
        # Calculate parametric position 't' across the gradient line
        t_values = xv_warped * dx_dir + yv_warped * dy_dir
        
        # Robust min/max scaling to account for noise pushing coordinates out of bounds
        t_min = np.percentile(t_values, 1)
        t_max = np.percentile(t_values, 99)
        if t_max > t_min:
            t_values = (t_values - t_min) / (t_max - t_min)
            
        # Smoothstep easing
        t_values = np.clip(t_values, 0, 1)
        t_values = t_values * t_values * (3 - 2 * t_values)
        
        # Perfect spline interpolation in Oklab space
        nodes = np.linspace(0, 1, num_colors)
        interpolator = PchipInterpolator(nodes, colors_oklab, axis=0)
        
        res = interpolator(t_values).astype(np.float32)
        srgb = np.clip(oklab_to_rgb(res), 0.0, 1.0)
        return Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))


class SimplexMeshGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, complexity=5, fluid_mode="random", seed=None):
        """
        Similar to FlowFieldGradient but uses Simplex noise (snoise2) for a more bulbous, 
        less directional warp, producing softer organic curves.
        """
        if noise is None:
            raise ImportError("The 'noise' package is required.")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        num_colors = len(colors)
        colors_oklab = _colors_to_oklab(colors)
        
        # Randomize color order slightly for more variety in Simplex
        color_order = list(range(num_colors))
        random.shuffle(color_order)
        colors_oklab = colors_oklab[color_order]

        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        noise_scale = random.uniform(0.002, 0.015) * (10 / max(1, complexity))
        octaves = random.randint(1, 2)
        
        res_scale = 0.1
        noise_res_w = max(2, int(self.width * res_scale))
        noise_res_h = max(2, int(self.height * res_scale))
        
        warp_x = np.zeros((noise_res_h, noise_res_w), dtype=np.float32)
        warp_y = np.zeros((noise_res_h, noise_res_w), dtype=np.float32)

        offset_x = random.random() * 1000
        offset_y = random.random() * 1000

        for ny in range(noise_res_h):
            for nx in range(noise_res_w):
                warp_x[ny, nx] = noise.snoise2(nx * noise_scale + offset_x, ny * noise_scale + offset_y, octaves=octaves)
                warp_y[ny, nx] = noise.snoise2(nx * noise_scale + offset_x + 100, ny * noise_scale + offset_y + 100, octaves=octaves)

        warp_x = scipy.ndimage.zoom(warp_x, (self.height / noise_res_h, self.width / noise_res_w), order=1)
        warp_y = scipy.ndimage.zoom(warp_y, (self.height / noise_res_h, self.width / noise_res_w), order=1)

        if warp_x.shape != xv.shape:
            warp_x = np.pad(warp_x, ((0, max(0, xv.shape[0]-warp_x.shape[0])), (0, max(0, xv.shape[1]-warp_x.shape[1]))))[:xv.shape[0], :xv.shape[1]]
            warp_y = np.pad(warp_y, ((0, max(0, yv.shape[0]-warp_y.shape[0])), (0, max(0, yv.shape[1]-warp_y.shape[1]))))[:yv.shape[0], :yv.shape[1]]

        warp_strength = random.uniform(0.1, 0.35)
        xv_warped = xv + warp_x * warp_strength
        yv_warped = yv + warp_y * warp_strength

        # Random diagonal/directional sweep
        angle = random.uniform(0, 2 * np.pi)
        dx_dir = np.cos(angle)
        dy_dir = np.sin(angle)
        
        t_values = xv_warped * dx_dir + yv_warped * dy_dir
        
        t_min = np.percentile(t_values, 1)
        t_max = np.percentile(t_values, 99)
        if t_max > t_min:
            t_values = (t_values - t_min) / (t_max - t_min)
            
        t_values = np.clip(t_values, 0, 1)
        t_values = t_values * t_values * (3 - 2 * t_values)
        
        nodes = np.linspace(0, 1, num_colors)
        interpolator = PchipInterpolator(nodes, colors_oklab, axis=0)
        
        res = interpolator(t_values).astype(np.float32)
        srgb = np.clip(oklab_to_rgb(res), 0.0, 1.0)
        return Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))
