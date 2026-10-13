import numpy as np
from PIL import Image
import random
import scipy.ndimage
from scipy.interpolate import PchipInterpolator
from engine.colors import rgb_to_oklab, oklab_to_rgb
try:
    import noise
except ImportError:
    noise = None

class SplineWaveGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(self, colors, complexity=5, seed=None):
        """
        Directional sweep using Monotonic Cubic Spline (PCHIP) in Oklab space
        warped by curl noise. This creates perfect folds and majestic sweeps
        without plateau artifacts.
        """
        if noise is None:
            raise ImportError("The 'noise' package is required.")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        num_colors = len(colors)
        colors_oklab = rgb_to_oklab(np.asarray(colors, dtype=np.float32))

        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        res_scale = 0.1
        small_h = max(2, int(self.height * res_scale))
        small_w = max(2, int(self.width * res_scale))

        ox = random.uniform(0, 1000)
        oy = random.uniform(0, 1000)

        warp_strength = random.uniform(0.15, 0.45)
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
