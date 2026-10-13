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
        Generates a continuous scalar field mapped through a color spline.
        Instead of a low-res grid pinned to the corners, it evaluates sweeping Simplex noise
        across a moderate-res map and maps it directly to the palette, producing massive,
        graceful, un-anchored flowing waves (the 'bigger picture').
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        num_colors = len(colors)
        colors_arr = _colors_to_oklab(colors)

        # Generate at a moderate resolution to keep things fast but continuous
        scale_factor = 0.25
        h = max(16, int(self.height * scale_factor))
        w = max(16, int(self.width * scale_factor))

        x = np.linspace(-0.5, 0.5, w, dtype=np.float32)
        y = np.linspace(-0.5, 0.5, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        # Random sweep direction
        angle = random.uniform(0, 2 * np.pi)
        t_base = xv * np.cos(angle) + yv * np.sin(angle)

        try:
            import noise
            has_noise = True
        except ImportError:
            has_noise = False
            
        if has_noise:
            ox = random.uniform(0, 1000)
            oy = random.uniform(0, 1000)
            # Complexity scales the frequency of the waves
            noise_scale = 1.0 + (complexity * 0.2)

            # Vectorized evaluation of noise would be faster, but for 480x270 a loop is too slow.
            # Let's generate a smaller noise map and upsample it to match h, w.
            noise_h, noise_w = max(4, int(h/4)), max(4, int(w/4))
            nx = np.linspace(-0.5, 0.5, noise_w, dtype=np.float32) * noise_scale
            ny = np.linspace(-0.5, 0.5, noise_h, dtype=np.float32) * noise_scale
            noise_grid = np.zeros((noise_h, noise_w), dtype=np.float32)
            for iy in range(noise_h):
                for ix in range(noise_w):
                    noise_grid[iy, ix] = noise.snoise2(nx[ix] + ox, ny[iy] + oy, octaves=2)

            # Upsample noise to moderate resolution
            import scipy.ndimage
            t_noise = scipy.ndimage.zoom(noise_grid, (h / noise_h, w / noise_w), order=3)
            # Trim padding issues from zoom
            t_noise = t_noise[:h, :w]

            # Blend base linear progression with sweeping noise folds
            t_values = t_base * 0.6 + t_noise * 1.5
        else:
            fold = np.sin(xv * 3.14) * np.cos(yv * 3.14) * 0.5
            t_values = t_base + fold

        t_min = t_values.min()
        t_max = t_values.max()
        if t_max > t_min:
            t_values = (t_values - t_min) / (t_max - t_min)

        t_values = np.clip(t_values, 0, 1)

        from scipy.interpolate import PchipInterpolator
        nodes = np.linspace(0, 1, num_colors)
        interpolator = PchipInterpolator(nodes, colors_arr, axis=0)

        # Flatten t_values, interpolate all colors, then reshape back to image dimensions
        flat_t = t_values.flatten()
        flat_colors = interpolator(flat_t)
        grid_oklab = flat_colors.reshape((h, w, 3))

        srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
        img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))

        return img_mod.resize((self.width, self.height), Image.LANCZOS)


class DomainWarpGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def _generate_noise_plane(self, h, w, scale=1.0, octaves=2, offset=(0.0, 0.0)):
        if noise is None:
            y = np.linspace(-1.0, 1.0, h, dtype=np.float32) * scale + offset[1]
            x = np.linspace(-1.0, 1.0, w, dtype=np.float32) * scale + offset[0]
            xv, yv = np.meshgrid(x, y)
            return np.sin(xv * 3.14159) * np.cos(yv * 3.14159)

        sample_h = max(8, h // 4)
        sample_w = max(8, w // 4)
        y = np.linspace(-1.0, 1.0, sample_h, dtype=np.float32) * scale + offset[1]
        x = np.linspace(-1.0, 1.0, sample_w, dtype=np.float32) * scale + offset[0]
        grid = np.zeros((sample_h, sample_w), dtype=np.float32)
        for iy in range(sample_h):
            for ix in range(sample_w):
                grid[iy, ix] = noise.snoise2(x[ix], y[iy], octaves=octaves)
        zoom_y = h / sample_h
        zoom_x = w / sample_w
        return scipy.ndimage.zoom(grid, (zoom_y, zoom_x), order=3)[:h, :w]

    def generate(self, colors, complexity=5, warp_strength=0.85, angle=None, seed=None):
        """
        Generates a continuous gradient using nested domain warping.
        Rather than adding noise to pixel values, the underlying coordinate space
        is deformed by 2D vector noise fields, creating organic fluid swirls,
        river currents, and marble folds without blotchy spot artifacts.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        num_colors = len(colors)
        colors_oklab = _colors_to_oklab(colors)

        scale_factor = 0.25
        h = max(16, int(self.height * scale_factor))
        w = max(16, int(self.width * scale_factor))

        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        if angle is None:
            angle = random.uniform(0, 2 * np.pi)

        u = xv * np.cos(angle) + yv * np.sin(angle)
        v = -xv * np.sin(angle) + yv * np.cos(angle)

        base_freq = 0.5 + (complexity * 0.08)

        ox1, oy1 = random.uniform(0, 500), random.uniform(0, 500)
        ox2, oy2 = random.uniform(500, 1000), random.uniform(500, 1000)
        qx = self._generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox1, oy1))
        qy = self._generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox2, oy2))

        u_warped = u + warp_strength * qx
        v_warped = v + warp_strength * qy

        ox3, oy3 = random.uniform(1000, 1500), random.uniform(1000, 1500)
        rx = self._generate_noise_plane(h, w, scale=base_freq * 1.3, octaves=2, offset=(ox3, oy3))

        t = u_warped + (warp_strength * 0.4) * rx

        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
        t_values = np.clip(t, 0.0, 1.0)

        from scipy.interpolate import PchipInterpolator
        nodes = np.linspace(0, 1, num_colors)
        interpolator = PchipInterpolator(nodes, colors_oklab, axis=0)

        flat_t = t_values.flatten()
        flat_colors = interpolator(flat_t)
        grid_oklab = flat_colors.reshape((h, w, 3))

        srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
        img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))

        return img_mod.resize((self.width, self.height), Image.LANCZOS)


class RadialBasisGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def _generate_noise_plane(self, h, w, scale=1.0, octaves=2, offset=(0.0, 0.0)):
        if noise is None:
            y = np.linspace(-1.0, 1.0, h, dtype=np.float32) * scale + offset[1]
            x = np.linspace(-1.0, 1.0, w, dtype=np.float32) * scale + offset[0]
            xv, yv = np.meshgrid(x, y)
            return np.sin(xv * np.pi) * np.cos(yv * np.pi)

        sample_h = max(8, h // 4)
        sample_w = max(8, w // 4)
        y = np.linspace(-1.0, 1.0, sample_h, dtype=np.float32) * scale + offset[1]
        x = np.linspace(-1.0, 1.0, sample_w, dtype=np.float32) * scale + offset[0]
        grid = np.zeros((sample_h, sample_w), dtype=np.float32)
        for iy in range(sample_h):
            for ix in range(sample_w):
                grid[iy, ix] = noise.snoise2(x[ix], y[iy], octaves=octaves)
        zoom_y = h / sample_h
        zoom_x = w / sample_w
        return scipy.ndimage.zoom(grid, (zoom_y, zoom_x), order=3)[:h, :w]

    def generate(
        self,
        colors,
        complexity=5,
        warp_strength=1.0,
        ambient_weight=0.15,
        dispersion=0.35,
        seed=None,
    ):
        """
        Generates a continuous 2D color field using radial basis functions and coordinate domain warping.
        Breaks the 1D monotonic scalar sequence limitation by treating colors as independent spatial
        emitters placed across 2D space, blended directly in Oklab color space with partition of unity.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        num_colors = len(colors)
        if num_colors == 0:
            raise ValueError("At least one color must be provided.")
        if num_colors == 1:
            c = tuple(int(v) for v in colors[0])
            return Image.new("RGB", (self.width, self.height), c)

        colors_oklab = _colors_to_oklab(colors)

        scale_factor = 0.25
        h = max(16, int(self.height * scale_factor))
        w = max(16, int(self.width * scale_factor))

        y = np.linspace(0.0, 1.0, h, dtype=np.float32)
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        effective_warp = max(0.0, float(warp_strength))
        if effective_warp > 0.0:
            base_freq = 0.6 + (max(1, complexity) * 0.08)
            ox1, oy1 = random.uniform(0, 500), random.uniform(0, 500)
            ox2, oy2 = random.uniform(500, 1000), random.uniform(500, 1000)
            qx = self._generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox1, oy1))
            qy = self._generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox2, oy2))
            warp_amp = effective_warp * 0.35
            xv = xv + warp_amp * qx
            yv = yv + warp_amp * qy

        weights = np.zeros((num_colors, h, w), dtype=np.float32)
        weights[0] = max(0.01, float(ambient_weight))

        num_emitters = max(num_colors - 1, max(1, complexity))
        for i in range(num_emitters):
            c_idx = 1 + (i % (num_colors - 1))
            angle_pos = i * (2.0 * np.pi / num_emitters) + random.uniform(-0.35, 0.35)
            dist_center = random.uniform(0.18, 0.44)
            cx = float(np.clip(0.5 + dist_center * np.cos(angle_pos), 0.1, 0.9))
            cy = float(np.clip(0.5 + dist_center * np.sin(angle_pos), 0.1, 0.9))
            sigma = max(0.05, float(dispersion) * random.uniform(0.75, 1.25))
            aspect = random.uniform(0.7, 1.4)
            flow_angle = random.uniform(0.0, np.pi)

            dx = xv - cx
            dy = yv - cy
            dx_rot = dx * np.cos(flow_angle) + dy * np.sin(flow_angle)
            dy_rot = -dx * np.sin(flow_angle) + dy * np.cos(flow_angle)
            dist = np.sqrt(dx_rot**2 + (dy_rot * aspect)**2)

            w_norm = np.clip(1.0 - (dist / (sigma * 1.8)), 0.0, 1.0)
            w_emit = (w_norm**2) * (3.0 - 2.0 * w_norm)
            weights[c_idx] += w_emit

        weight_sum = np.sum(weights, axis=0, keepdims=True)
        weights /= np.maximum(weight_sum, 1e-6)

        grid_oklab = np.einsum("nhw,nc->hwc", weights, colors_oklab)
        srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
        img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))

        return img_mod.resize((self.width, self.height), Image.LANCZOS)


RBFFieldGradient = RadialBasisGradient


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


class TopographicGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def _generate_noise_plane(self, h, w, scale=1.0, octaves=2, offset=(0.0, 0.0)):
        if noise is None:
            y = np.linspace(-1.0, 1.0, h, dtype=np.float32) * scale + offset[1]
            x = np.linspace(-1.0, 1.0, w, dtype=np.float32) * scale + offset[0]
            xv, yv = np.meshgrid(x, y)
            return np.sin(xv * np.pi) * np.cos(yv * np.pi)

        sample_h = max(8, h // 4)
        sample_w = max(8, w // 4)
        y = np.linspace(-1.0, 1.0, sample_h, dtype=np.float32) * scale + offset[1]
        x = np.linspace(-1.0, 1.0, sample_w, dtype=np.float32) * scale + offset[0]
        grid = np.zeros((sample_h, sample_w), dtype=np.float32)
        for iy in range(sample_h):
            for ix in range(sample_w):
                grid[iy, ix] = noise.snoise2(x[ix], y[iy], octaves=octaves)
        zoom_y = h / sample_h
        zoom_x = w / sample_w
        return scipy.ndimage.zoom(grid, (zoom_y, zoom_x), order=3)[:h, :w]

    def generate(
        self,
        colors,
        complexity=5,
        warp_strength=0.85,
        angle=None,
        seed=None,
    ):
        """
        Generates a continuous hypsometric elevation field using multi-harmonic
        wave synthesis and domain-warped fractal terrain folds mapped through Oklab space.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        num_colors = len(colors)
        if num_colors == 0:
            raise ValueError("At least one color must be provided.")
        if num_colors == 1:
            c = tuple(int(v) for v in colors[0])
            return Image.new("RGB", (self.width, self.height), c)

        colors_oklab = _colors_to_oklab(colors)

        scale_factor = 0.25
        h = max(16, int(self.height * scale_factor))
        w = max(16, int(self.width * scale_factor))

        aspect = w / h
        y = np.linspace(-1.5, 1.5, h, dtype=np.float32)
        x = np.linspace(-1.5 * aspect, 1.5 * aspect, w, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        if angle is None:
            angle = random.uniform(0, 2 * np.pi)

        u = xv * np.cos(angle) + yv * np.sin(angle)
        v = -xv * np.sin(angle) + yv * np.cos(angle)

        base_freq = 0.7 + (complexity * 0.1)

        h1 = np.sin(u * base_freq * 1.4) * np.cos(v * base_freq * 1.1)
        h2 = np.sin((u + v) * base_freq * 2.1 + 1.1) * 0.5
        h3 = np.cos(np.sqrt(u**2 + v**2) * base_freq * 2.8) * 0.25

        raw_elevation = h1 + h2 + h3

        effective_warp = max(0.0, float(warp_strength))
        if effective_warp > 0.0:
            ox = random.uniform(0, 500)
            oy = random.uniform(500, 1000)
            warp_plane = self._generate_noise_plane(h, w, scale=base_freq, octaves=2, offset=(ox, oy))
            warped = raw_elevation + (effective_warp * 0.6) * warp_plane
        else:
            warped = raw_elevation

        e_min, e_max = warped.min(), warped.max()
        if e_max > e_min:
            t_values = (warped - e_min) / (e_max - e_min)
        else:
            t_values = np.zeros_like(warped)

        t_values = np.clip(t_values, 0.0, 1.0)

        from scipy.interpolate import PchipInterpolator
        nodes = np.linspace(0, 1, num_colors)
        interpolator = PchipInterpolator(nodes, colors_oklab, axis=0)

        flat_t = t_values.flatten()
        flat_colors = interpolator(flat_t)
        grid_oklab = flat_colors.reshape((h, w, 3))

        srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
        img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))

        return img_mod.resize((self.width, self.height), Image.LANCZOS)


class FractalFieldGradient:
    def __init__(self, width=5120, height=2880):
        self.width = width
        self.height = height

    def generate(
        self,
        colors,
        complexity=5,
        fractal_type="fluid",
        zoom=None,
        center=None,
        angle=None,
        phase_offset=0.0,
        frequency=1.0,
        trap_weight=0.35,
        channel_offset=0.0,
        seed=None,
    ):
        """
        Generates a continuous complex fractal field using normalized escape time
        potential, geometric orbit traps, and cyclic Oklab palette phase offsets.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        num_colors = len(colors)
        if num_colors == 0:
            raise ValueError("At least one color must be provided.")
        if num_colors == 1:
            c = tuple(int(v) for v in colors[0])
            return Image.new("RGB", (self.width, self.height), c)

        colors_oklab = _colors_to_oklab(colors)

        scale_factor = 0.25
        h = max(16, int(self.height * scale_factor))
        w = max(16, int(self.width * scale_factor))
        aspect = w / h

        if zoom is not None:
            eff_zoom = float(zoom)
        else:
            if fractal_type == "mandelbrot":
                eff_zoom = random.uniform(8.0, 35.0)
            else:
                radius_tier = random.choices(["macro", "medium", "detail"], weights=[0.45, 0.40, 0.15])[0]
                if radius_tier == "macro":
                    eff_zoom = random.uniform(0.40, 0.85)
                elif radius_tier == "medium":
                    eff_zoom = random.uniform(0.85, 1.75)
                else:
                    eff_zoom = random.uniform(1.75, 3.20)

        eff_angle = float(angle) if angle is not None else random.uniform(0.0, 2.0 * np.pi)

        if center is not None:
            cx, cy = float(center[0]), float(center[1])
        else:
            if fractal_type == "mandelbrot":
                mandel_points = [
                    (-0.7436438870371587, 0.1318259042053119),
                    (-0.1011, 0.9563),
                    (-0.75, 0.1),
                    (-0.748, 0.065),
                    (-1.25, 0.0),
                    (0.275, 0.0),
                ]
                cx, cy = random.choice(mandel_points)
            else:
                focal_archetype = random.choices(["offscreen", "corner", "centered"], weights=[0.45, 0.35, 0.20])[0]
                if focal_archetype == "offscreen":
                    theta_off = random.uniform(0.0, 2.0 * np.pi)
                    dist_off = random.uniform(1.4, 2.8) / eff_zoom
                    cx = dist_off * aspect * np.cos(theta_off)
                    cy = dist_off * np.sin(theta_off)
                elif focal_archetype == "corner":
                    sign_x = random.choice([-1.0, 1.0])
                    sign_y = random.choice([-1.0, 1.0])
                    cx = sign_x * random.uniform(0.40, 0.95) * aspect / eff_zoom
                    cy = sign_y * random.uniform(0.35, 0.90) / eff_zoom
                else:
                    cx = random.uniform(-0.35, 0.35) * aspect / eff_zoom
                    cy = random.uniform(-0.35, 0.35) / eff_zoom

        y = np.linspace(-1.3 / eff_zoom, 1.3 / eff_zoom, h, dtype=np.float32)
        x = np.linspace(-1.3 * aspect / eff_zoom, 1.3 * aspect / eff_zoom, w, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        dx = xv - cx
        dy = yv - cy
        rf = np.sqrt(dx**2 + dy**2)

        warp_amp = 0.18
        qx = np.sin(yv * 2.6 + 1.2) * np.cos(xv * 2.0)
        qy = np.cos(xv * 2.6 + 0.8) * np.sin(yv * 2.0)
        xv_warped = xv + warp_amp * qx
        yv_warped = yv + warp_amp * qy

        rot_x = (xv_warped - cx) * np.cos(eff_angle) - (yv_warped - cy) * np.sin(eff_angle)
        rot_y = (xv_warped - cx) * np.sin(eff_angle) + (yv_warped - cy) * np.cos(eff_angle)
        z_grid = rot_x + 1j * rot_y

        safe_complexity = max(1, int(complexity))

        eff_freq = max(0.1, float(frequency))
        eff_phase = float(phase_offset) % 1.0
        eff_trap_weight = np.clip(float(trap_weight), 0.0, 1.0)
        eff_chan_offset = float(channel_offset)

        if fractal_type == "fluid":
            def fbm_field(x_grid, y_grid, octaves=3, lacunarity=1.8, gain=0.45):
                val = np.zeros_like(x_grid)
                freq = 0.65
                amp = 1.0
                for _ in range(octaves):
                    theta = random.uniform(0, np.pi)
                    rx = x_grid * np.cos(theta) - y_grid * np.sin(theta)
                    ry = x_grid * np.sin(theta) + y_grid * np.cos(theta)
                    phase = random.uniform(0, 2 * np.pi)
                    val += amp * (np.sin(rx * freq + phase) * np.cos(ry * freq + phase * 0.7))
                    freq *= lacunarity
                    amp *= gain
                return val

            q_x = fbm_field(rot_x, rot_y, octaves=2)
            q_y = fbm_field(rot_x + 3.8, rot_y + 1.5, octaves=2)

            r_x = fbm_field(rot_x + 1.8 * q_x + 1.4, rot_y + 1.8 * q_y + 2.8, octaves=2)
            r_y = fbm_field(rot_x + 1.8 * q_x + 4.5, rot_y + 1.8 * q_y + 1.6, octaves=2)

            radial_freq = random.uniform(0.35, 0.80)
            radial_carrier = 0.5 + 0.5 * np.cos(rf * radial_freq)

            sweep_angle = eff_angle + random.uniform(-0.5, 0.5)
            ambient_sweep = 0.5 * (np.cos(sweep_angle) * (xv / aspect) + np.sin(sweep_angle) * yv)

            fluid_folds = fbm_field(rot_x + 2.2 * r_x, rot_y + 2.2 * r_y, octaves=max(2, min(5, safe_complexity)))

            field = 0.30 * radial_carrier + 0.25 * ambient_sweep + 0.45 * fluid_folds
            f_min, f_max = field.min(), field.max()
            u_base = (field - f_min) / max(1e-5, (f_max - f_min))

            u = 0.5 - 0.5 * np.cos(np.pi * (u_base * eff_freq + eff_phase))
            u_L = 0.5 - 0.5 * np.cos(np.pi * (u_base * eff_freq + eff_phase + eff_chan_offset))

            from scipy.interpolate import PchipInterpolator

            nodes = np.linspace(0.0, 1.0, num_colors)
            interp_L = PchipInterpolator(nodes, colors_oklab[:, 0])
            interp_ab = PchipInterpolator(nodes, colors_oklab[:, 1:3], axis=0)

            flat_L = interp_L(u_L.flatten())
            flat_ab = interp_ab(u.flatten())
            grid_oklab = np.column_stack([flat_L, flat_ab]).reshape((h, w, 3))

            srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
            img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))
            return img_mod.resize((self.width, self.height), Image.LANCZOS)

        max_iter = max(16, 16 + safe_complexity * 3)

        escape_radius = 16.0
        ln_r = np.log(escape_radius)

        smooth_pot = np.zeros(z_grid.shape, dtype=np.float32)
        trap = np.full(z_grid.shape, 1e6, dtype=np.float32)
        mask = np.ones(z_grid.shape, dtype=bool)

        if fractal_type == "mandelbrot":
            z = np.zeros_like(z_grid)
            c = z_grid
        else:
            theta = random.uniform(0.0, 2.0 * np.pi)
            if random.random() < 0.3:
                c_bound = -1.0 + 0.25 * np.exp(1j * theta)
            else:
                c_bound = 0.5 * np.exp(1j * theta) - 0.25 * np.exp(2j * theta)
            eps = random.uniform(0.015, 0.04) * np.exp(1j * random.uniform(0.0, 2.0 * np.pi))
            c = c_bound + eps
            z = z_grid.copy()

        for i in range(max_iter):
            z[mask] = z[mask] ** 2 + (c[mask] if isinstance(c, np.ndarray) else c)
            abs_z = np.abs(z)

            curr_trap = np.minimum(np.abs(z.real), np.abs(z.imag)) + 0.15 * abs_z
            trap[mask] = np.minimum(trap[mask], curr_trap[mask])

            escaped = abs_z > escape_radius
            newly_escaped = escaped & mask
            if np.any(newly_escaped):
                smooth_pot[newly_escaped] = (
                    i + 1.0 - (np.log(np.log(abs_z[newly_escaped])) - np.log(ln_r)) / np.log(2.0)
                )
            mask = mask & (~escaped)
            if not np.any(mask):
                break

        if np.any(mask):
            final_abs = np.clip(np.abs(z[mask]), 0.0, escape_radius)
            smooth_pot[mask] = max_iter - np.log1p(escape_radius - final_abs)

        smooth_pot = scipy.ndimage.gaussian_filter(smooth_pot, sigma=1.8)
        trap = scipy.ndimage.gaussian_filter(trap, sigma=1.5)

        p_min, p_max = smooth_pot.min(), smooth_pot.max()
        pot_norm = (smooth_pot - p_min) / max(1e-5, (p_max - p_min))

        pot_smooth = 1.0 / (1.0 + np.exp(-4.0 * (pot_norm - 0.5)))
        pot_smooth = (pot_smooth - pot_smooth.min()) / max(1e-5, (pot_smooth.max() - pot_smooth.min()))

        trap_p90 = np.percentile(trap, 90)
        trap_norm = np.clip(trap / max(1e-5, trap_p90), 0.0, 1.0)
        trap_smooth = 1.0 - np.exp(-2.2 * trap_norm)

        ambient_carrier = 0.5 + 0.5 * (np.sin(rot_x * 0.5) * np.cos(rot_y * 0.5))

        eff_freq = max(0.1, float(frequency))
        eff_phase = float(phase_offset) % 1.0
        eff_trap_weight = np.clip(float(trap_weight), 0.0, 1.0)

        u_field = (
            (1.0 - eff_trap_weight * 0.5) * (0.45 * ambient_carrier + 0.55 * pot_smooth)
            + eff_trap_weight * 0.5 * trap_smooth
        )

        eff_chan_offset = float(channel_offset)
        u = 0.5 - 0.5 * np.cos(np.pi * (u_field * eff_freq + eff_phase))
        u_L = 0.5 - 0.5 * np.cos(np.pi * (u_field * eff_freq + eff_phase + eff_chan_offset))
        u_ab = u

        from scipy.interpolate import PchipInterpolator

        nodes = np.linspace(0.0, 1.0, num_colors)
        interp_L = PchipInterpolator(nodes, colors_oklab[:, 0])
        interp_ab = PchipInterpolator(nodes, colors_oklab[:, 1:3], axis=0)

        flat_L = interp_L(u_L.flatten())
        flat_ab = interp_ab(u_ab.flatten())
        grid_oklab = np.column_stack([flat_L, flat_ab]).reshape((h, w, 3))

        srgb = np.clip(oklab_to_rgb(grid_oklab), 0.0, 1.0)
        img_mod = Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8))

        return img_mod.resize((self.width, self.height), Image.LANCZOS)



if __name__ == "__main__":
    # Test
    engine = MeshGradient(width=1280, height=720)
    # Nothing Signature Palette: Black, White, Red, Orange
    colors = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (255, 80, 0)]
    img = engine.generate(colors)
    img.save("output/test_gradient.png")
    print("Test gradient saved to output/test_gradient.png")
