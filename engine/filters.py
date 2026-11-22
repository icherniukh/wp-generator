import numpy as np
import scipy.ndimage
from PIL import Image, ImageFilter

GRAIN_COLORS = ("mono", "color")
GRAIN_BLENDS = ("add", "overlay", "soft-light", "multiply")
# Below 1 the grain would be finer than a pixel, which is what the default already is.
GRAIN_SIZE_RANGE = (1.0, 8.0)

# Weights of the fine and the coarse grain layer when each has unit variance. They give a
# standard deviation of 0.806 * sigma, which is the default grain's (0.7 fine, 0.4 after
# enlarging the 0.6 half-size layer), so grains of any size share its strength.
_GRAIN_FINE, _GRAIN_COARSE = 0.7, 0.4
# Per-blend gain so the grain's strength at the same amount is close to "add" on typical images.
_GRAIN_BLEND_GAIN = {"overlay": 1.25, "soft-light": 2.0, "multiply": 1.1}


class WallpaperFilters:
    @staticmethod
    def apply_dither(image, amount=0.05):
        """
        Adds bounded monochromatic dither before final PNG quantization.
        Scales amplitude in deep shadows/highlights to prevent clipping artifacts (black dots).
        """
        img_arr = np.array(image, dtype=np.float32)

        amplitude = min(15.0, 255.0 * max(0.0, amount))
        dither = np.random.uniform(-amplitude, amplitude, img_arr.shape[:2])

        lum = np.mean(img_arr[..., :3], axis=2) / 255.0
        # Smooth roll-off near 0 and 1 to avoid clipping dots
        mask = np.clip(4.0 * lum * (1.0 - lum), 0.2, 1.0)

        img_arr[..., :3] += (dither * mask)[..., None]
        return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_film_grain(image, amount=0.08, grain_size=1.0, grain_color="mono", grain_blend="add"):
        """
        Applies multi-scale photographic film grain with luminance roll-off.
        Combines fine high-frequency noise with organic clumpiness visible on high-DPI displays.
        :param amount: Grain intensity (0.0 to 1.0).
        :param grain_size: Grain size in pixels; 1.0 is the default grain, larger is coarser.
        :param grain_color: "mono" (one noise for all channels) or "color" (independent per channel).
        :param grain_blend: "add", "overlay", "soft-light" or "multiply".
        """
        if grain_color not in GRAIN_COLORS:
            raise ValueError(f"grain_color must be one of {GRAIN_COLORS}, not {grain_color!r}.")
        if grain_blend not in GRAIN_BLENDS:
            raise ValueError(f"grain_blend must be one of {GRAIN_BLENDS}, not {grain_blend!r}.")
        if amount <= 0.0:
            return image.copy()

        img_arr = np.array(image, dtype=np.float32)
        h, w = img_arr.shape[:2]
        channels = img_arr.shape[2] if img_arr.ndim == 3 else 1

        if (grain_size, grain_color, grain_blend) != (1.0, "mono", "add"):
            return WallpaperFilters._film_grain_controlled(
                img_arr, amount, max(1.0, float(grain_size)), grain_color, grain_blend
            )

        sigma = 255.0 * amount
        fine_grain = np.random.normal(0.0, sigma * 0.7, (h, w)).astype(np.float32)

        down_h, down_w = max(1, h // 2), max(1, w // 2)
        coarse_small = np.random.normal(0.0, sigma * 0.6, (down_h, down_w)).astype(np.float32)
        coarse_grain = scipy.ndimage.zoom(
            coarse_small, (h / down_h, w / down_w), order=1
        )[:h, :w]

        combined_grain = fine_grain + coarse_grain

        rgb = img_arr[..., :3] if channels >= 3 else img_arr
        lum = np.mean(rgb, axis=2 if channels >= 3 else -1) / 255.0
        # Preserve midtone grain while softening extreme shadows and specular highlights
        lum_weight = np.clip(4.0 * lum * (1.0 - lum), 0.35, 1.0)
        modulated_grain = combined_grain * lum_weight

        if channels >= 3:
            img_arr[..., :3] += modulated_grain[..., None]
        else:
            img_arr += modulated_grain

        return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

    @staticmethod
    def _grain_layer(h, w, size):
        """Unit-variance white noise generated at 1/size resolution and enlarged bicubically."""
        lh, lw = max(1, int(round(h / size))), max(1, int(round(w / size)))
        small = np.random.standard_normal((lh, lw)).astype(np.float32)
        if (lh, lw) == (h, w):
            return small
        layer = np.asarray(
            Image.fromarray(small, mode="F").resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32
        )
        return (layer - layer.mean()) / max(float(layer.std()), 1e-6)

    @staticmethod
    def _grain_noise(h, w, size, sigma):
        """Zero-mean grain with the default grain's strength: a fine layer at `size` pixels and
        a coarser clump layer at twice that."""
        noise = _GRAIN_FINE * WallpaperFilters._grain_layer(h, w, size)
        noise += _GRAIN_COARSE * WallpaperFilters._grain_layer(h, w, 2.0 * size)
        return noise * np.float32(sigma)

    @staticmethod
    def _film_grain_controlled(img_arr, amount, size, color, blend):
        """Film grain with a chosen size, color mode and blend; the default settings take the
        original path in apply_film_grain instead."""
        h, w = img_arr.shape[:2]
        sigma = 255.0 * amount
        has_rgb = img_arr.ndim == 3 and img_arr.shape[2] >= 3
        rgb = img_arr[..., :3] if has_rgb else img_arr[..., None]
        planes = 3 if color == "color" and has_rgb else 1
        # Grain in 0-255 units; "color" draws one noise per channel.
        noise = np.stack([WallpaperFilters._grain_noise(h, w, size, sigma) for _ in range(planes)], axis=-1)

        base = rgb / 255.0
        d = noise / 255.0 * _GRAIN_BLEND_GAIN.get(blend, 1.0)  # the noise layer is 0.5 + d, mid-grey centered
        if blend == "add":
            lum = np.mean(rgb, axis=2) / 255.0
            out = base + d * np.clip(4.0 * lum * (1.0 - lum), 0.35, 1.0)[..., None]
        else:
            layer = np.clip(0.5 + d, 0.0, 1.0)
            if blend == "overlay":
                out = np.where(base < 0.5, 2.0 * base * layer, 1.0 - 2.0 * (1.0 - base) * (1.0 - layer))
            elif blend == "soft-light":
                out = (1.0 - 2.0 * layer) * base * base + 2.0 * layer * base
            else:  # multiply, with the layer doubled so mid-grey leaves the image unchanged
                out = base * 2.0 * layer
        if has_rgb:
            result = img_arr.copy()
            result[..., :3] = out * 255.0
        else:
            result = out[..., 0] * 255.0
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    apply_noise = apply_film_grain

    @staticmethod
    def apply_chromatic_aberration(image, strength=0.012, mode="radial"):
        """
        Applies optical chromatic aberration (color fringing).
        Displaces Red and Blue channels radially or directionally while Green remains fixed.
        :param strength: Magnitude of channel displacement (0.0 to 0.05).
        :param mode: 'radial' (lens falloff from optical center) or 'directional'.
        """
        if strength <= 0.0:
            return image.copy()

        img_arr = np.array(image, dtype=np.float32)
        h, w = img_arr.shape[:2]
        channels = img_arr.shape[2] if img_arr.ndim == 3 else 1

        if channels < 3:
            return image.copy()

        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        if mode == "directional":
            shift_px = strength * min(w, h)
            rx = x_coords + shift_px
            ry = y_coords
            bx = x_coords - shift_px
            by = y_coords
        else:
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            norm_x = (x_coords - cx) / cx
            norm_y = (y_coords - cy) / cy
            disp = strength * (norm_x**2 + norm_y**2)

            rx = x_coords + norm_x * cx * disp
            ry = y_coords + norm_y * cy * disp
            bx = x_coords - norm_x * cx * disp
            by = y_coords - norm_y * cy * disp

        img_arr[..., 0] = scipy.ndimage.map_coordinates(
            img_arr[..., 0], np.array([ry, rx]), order=1, mode="nearest"
        )
        img_arr[..., 2] = scipy.ndimage.map_coordinates(
            img_arr[..., 2], np.array([by, bx]), order=1, mode="nearest"
        )

        return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_vignette(image, strength=0.08):
        """Apply a restrained radial edge falloff while preserving image mode."""
        if strength <= 0:
            return image.copy()

        img_arr = np.array(image, dtype=np.float32)
        height, width = img_arr.shape[:2]
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        distance_sq = (x * x + y * y) / 2.0
        falloff = 1.0 - np.clip(strength, 0.0, 1.0) * distance_sq
        img_arr[..., :3] *= falloff[..., None]
        return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_frosted_glass(image, blur_radius=30, noise_amount=0.15):
        """
        Simulates a premium frosted glass texture across the entire image.
        Uses a diffusion blur followed by a specular frost noise overlay.
        """
        width, height = image.size
        # Diffusion blur (downsample for performance and softer blur)
        small = image.resize((max(1, width // 2), max(1, height // 2)), Image.BILINEAR)
        blurred = small.filter(ImageFilter.GaussianBlur(radius=blur_radius // 2))
        final = blurred.resize((width, height), Image.LANCZOS)

        img_arr = np.array(final, dtype=np.float32) / 255.0

        # Specular frost noise with screen blend mode
        frost_noise = np.clip(
            np.random.normal(0.6, noise_amount * 0.2, img_arr.shape[:2]), 0.0, 1.0
        )
        img_arr[..., :3] = 1.0 - (1.0 - img_arr[..., :3]) * (1.0 - frost_noise[..., None] * noise_amount)

        return Image.fromarray(np.clip(img_arr * 255.0, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_contour_lines(
        image,
        levels=24,
        line_width=1.4,
        intensity=0.8,
        mode="light",
        index_interval=5,
        smoothing_radius=2.5,
        elevation=None,
    ):
        """
        Applies vector-grade anti-aliased topographic elevation contour lines (isobars).
        Derives an elevation field from smoothed perceptual luminance, evaluating
        analytical screen-space distance fields with major index contour emphasis.
        :param levels: Number of contour elevation intervals.
        :param line_width: Thickness of minor contour lines in pixels.
        :param intensity: Blending opacity of the contour overlay (0.0 to 1.0).
        :param mode: 'light' (luminous isobars), 'dark' (ink isobars), or 'accent' (tinted from gradient).
        :param index_interval: Interval for bold major/index contours (e.g. 5 = every 5th line).
        :param smoothing_radius: Gaussian filter radius in pixels to prevent 8-bit quantization stepping.
        :param elevation: Optional (height, width) array in [0, 1] to draw isolines of instead of luminance.
        """
        if intensity <= 0.0 or levels <= 0:
            return image.copy()

        img_arr = np.array(image, dtype=np.float32)
        channels = img_arr.shape[2] if img_arr.ndim == 3 else 1
        if channels < 3:
            return image.copy()

        if elevation is not None:
            lum = np.asarray(elevation, dtype=np.float32)
        else:
            lum = (0.2126 * img_arr[..., 0] + 0.7152 * img_arr[..., 1] + 0.0722 * img_arr[..., 2]) / 255.0
        lum_smooth = (
            scipy.ndimage.gaussian_filter(lum, sigma=float(smoothing_radius))
            if smoothing_radius > 0.0
            else lum
        )

        scaled = lum_smooth * float(levels)
        nearest_level = np.round(scaled)
        dist_level = np.abs(scaled - nearest_level)

        grad_y, grad_x = np.gradient(scaled)
        dist_pixels = dist_level / (np.sqrt(grad_x**2 + grad_y**2) + 1e-5)

        if index_interval > 1:
            is_index = (np.rint(nearest_level).astype(int) % int(index_interval) == 0)
            effective_width = np.where(is_index, line_width * 1.8, line_width)
            level_alpha_boost = np.where(is_index, 1.0, 0.7)
        else:
            effective_width = line_width
            level_alpha_boost = 1.0

        alpha_mask = np.clip(1.0 - (dist_pixels - effective_width + 0.5) / 0.8, 0.0, 1.0)
        alpha_mask *= level_alpha_boost * float(np.clip(intensity, 0.0, 1.0))

        if mode == "dark":
            line_color = np.zeros_like(img_arr[..., :3])
        elif mode == "accent":
            max_c = np.maximum.reduce([img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]])
            boost = 255.0 / (max_c + 1e-3)
            line_color = np.clip(img_arr[..., :3] * boost[..., None] * 0.95, 0, 255)
        else:
            line_color = np.full_like(img_arr[..., :3], 245.0)

        alpha_3d = alpha_mask[..., None]
        img_arr[..., :3] = img_arr[..., :3] * (1.0 - alpha_3d) + line_color * alpha_3d
        return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

