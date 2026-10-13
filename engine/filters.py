import numpy as np
import scipy.ndimage
from PIL import Image, ImageFilter

class WallpaperFilters:
    @staticmethod
    def apply_dot_matrix(image, spacing=25, max_radius=4, intensity=0.8, mode="size"):
        """
        Applies a dot-matrix overlay.
        :param mode: 'size' modulates dot radius, 'alpha' modulates dot opacity.
        """
        width, height = image.size
        
        # Convert image to grayscale to get luminance (0.0 to 1.0)
        grayscale = image.convert("L")
        lum_data = np.array(grayscale, dtype=np.float32) / 255.0
        
        # Create coordinate grid
        y, x = np.ogrid[:height, :width]
        
        # Find offset to the nearest dot center
        half_space = spacing // 2
        dx = (x + half_space) % spacing - half_space
        dy = (y + half_space) % spacing - half_space
        dist_sq = dx**2 + dy**2
        
        # Find coordinates of the nearest dot center to sample a uniform cell luminance
        cx = np.clip(x - dx, 0, width - 1)
        cy = np.clip(y - dy, 0, height - 1)
        cell_lum = lum_data[cy, cx]
        
        # Prepare an RGBA overlay (all white, transparent initially)
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[..., :3] = 255
        
        if mode == "size":
            # Modulate radius: Brighter pixels = smaller dots
            radius = max_radius * (1.0 - cell_lum) * intensity
            alpha_val = int(255 * intensity)
            dist = np.sqrt(dist_sq)
            # Smoothstep-like falloff for anti-aliasing
            coverage = np.clip(radius - dist + 0.5, 0, 1)
            # Only apply if radius > 0.5
            coverage = np.where(radius > 0.5, coverage, 0)
            overlay[..., 3] = (coverage * alpha_val).astype(np.uint8)
        else: # alpha mode
            radius = max_radius
            # Darker pixels = higher opacity
            alpha = np.clip(255 * (1.0 - cell_lum) * intensity, 0, 255)
            dist = np.sqrt(dist_sq)
            coverage = np.clip(radius - dist + 0.5, 0, 1)
            final_alpha = coverage * alpha
            overlay[..., 3] = np.where(final_alpha > 5, final_alpha, 0).astype(np.uint8)
        
        overlay_img = Image.fromarray(overlay)
        return Image.alpha_composite(image.convert("RGBA"), overlay_img)

    @staticmethod
    def apply_dither(image, amount=0.05):
        """
        Adds bounded monochromatic dither before final PNG quantization.
        Scales amplitude in deep shadows/highlights to prevent clipping artifacts (black dots).
        """
        img_arr = np.array(image).astype(np.float32)
        channels = img_arr.shape[2]

        amplitude = min(15.0, 255.0 * max(0.0, amount))
        dither = np.random.uniform(-amplitude, amplitude, img_arr.shape[:2])
        
        lum = np.mean(img_arr[..., :3], axis=2) / 255.0
        # Smooth roll-off near 0 and 1 to avoid clipping dots
        mask = 4.0 * lum * (1.0 - lum)
        mask = np.clip(mask, 0.2, 1.0)
        
        dither_masked = dither * mask
        dither_stack = [dither_masked] * 3
        if channels == 4:
            dither_stack.append(np.zeros(img_arr.shape[:2]))

        res = img_arr + np.stack(dither_stack, axis=-1)
        res = np.clip(res, 0, 255).astype(np.uint8)
        return Image.fromarray(res)

    @staticmethod
    def apply_film_grain(image, amount=0.08):
        """
        Applies multi-scale photographic film grain with luminance roll-off.
        Combines fine high-frequency noise with organic clumpiness visible on high-DPI displays.
        :param amount: Grain intensity (0.0 to 1.0).
        """
        if amount <= 0.0:
            return image.copy()

        img_arr = np.asarray(image).astype(np.float32)
        h, w = img_arr.shape[:2]
        channels = img_arr.shape[2] if img_arr.ndim == 3 else 1

        sigma = 255.0 * amount
        fine_grain = np.random.normal(0.0, sigma * 0.7, (h, w)).astype(np.float32)

        down_factor = 2
        down_h, down_w = max(1, h // down_factor), max(1, w // down_factor)
        coarse_small = np.random.normal(0.0, sigma * 0.6, (down_h, down_w)).astype(np.float32)
        coarse_grain = scipy.ndimage.zoom(
            coarse_small, (h / down_h, w / down_w), order=1
        )[:h, :w]

        combined_grain = fine_grain + coarse_grain

        rgb = img_arr[..., :3] if channels >= 3 else img_arr
        lum = np.mean(rgb, axis=2 if channels >= 3 else -1) / 255.0
        # Preserve midtone grain while softening extreme shadows and specular highlights
        lum_weight = 4.0 * lum * (1.0 - lum)
        lum_weight = np.clip(lum_weight, 0.35, 1.0)

        modulated_grain = combined_grain * lum_weight

        out = img_arr.copy()
        if channels >= 3:
            for c in range(3):
                out[..., c] += modulated_grain
        else:
            out += modulated_grain

        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

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

        img_arr = np.asarray(image).astype(np.float32)
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
            r_sq = norm_x**2 + norm_y**2
            disp = strength * r_sq

            rx = x_coords + norm_x * cx * disp
            ry = y_coords + norm_y * cy * disp
            bx = x_coords - norm_x * cx * disp
            by = y_coords - norm_y * cy * disp

        red_coords = np.array([ry, rx])
        blue_coords = np.array([by, bx])

        r_channel = scipy.ndimage.map_coordinates(
            img_arr[..., 0], red_coords, order=1, mode="nearest"
        )
        b_channel = scipy.ndimage.map_coordinates(
            img_arr[..., 2], blue_coords, order=1, mode="nearest"
        )

        out = img_arr.copy()
        out[..., 0] = r_channel
        out[..., 2] = b_channel

        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_vignette(image, strength=0.08):
        """Apply a restrained radial edge falloff while preserving image mode."""
        if strength <= 0:
            return image.copy()

        img_arr = np.asarray(image).astype(np.float32)
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
        small_width = max(1, width // 2)
        small_height = max(1, height // 2)
        small = image.resize((small_width, small_height), Image.BILINEAR)
        blurred = small.filter(ImageFilter.GaussianBlur(radius=blur_radius // 2))
        final = blurred.resize((width, height), Image.LANCZOS)

        img_arr = np.array(final).astype(np.float32) / 255.0

        # Specular frost noise
        frost_noise = np.random.normal(0.6, noise_amount * 0.2, img_arr.shape[:2])
        frost_noise = np.clip(frost_noise, 0.0, 1.0)

        # Screen blend mode for the frost (makes it luminous and translucent)
        frost_stack = np.stack([frost_noise] * 3, axis=-1)

        # Blend factor for the frost
        blend = 0.15 * noise_amount / 0.15
        res_rgb = 1.0 - (1.0 - img_arr[..., :3]) * (1.0 - frost_stack * blend)

        res = img_arr.copy()
        res[..., :3] = res_rgb
        return Image.fromarray(np.clip(res * 255.0, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_contour_lines(
        image,
        levels=24,
        line_width=1.4,
        intensity=0.8,
        mode="light",
        index_interval=5,
        smoothing_radius=2.5,
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
        """
        if intensity <= 0.0 or levels <= 0:
            return image.copy()

        img_arr = np.asarray(image).astype(np.float32)
        h, w = img_arr.shape[:2]
        channels = img_arr.shape[2] if img_arr.ndim == 3 else 1

        if channels < 3:
            return image.copy()

        lum = (0.2126 * img_arr[..., 0] + 0.7152 * img_arr[..., 1] + 0.0722 * img_arr[..., 2]) / 255.0

        if smoothing_radius > 0.0:
            lum_smooth = scipy.ndimage.gaussian_filter(lum, sigma=float(smoothing_radius))
        else:
            lum_smooth = lum

        scaled = lum_smooth * float(levels)
        nearest_level = np.round(scaled)
        dist_level = np.abs(scaled - nearest_level)

        grad_y, grad_x = np.gradient(scaled)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-5

        dist_pixels = dist_level / grad_mag

        if index_interval > 1:
            is_index = (np.rint(nearest_level).astype(int) % int(index_interval) == 0)
            effective_width = np.where(is_index, line_width * 1.8, line_width)
            level_alpha_boost = np.where(is_index, 1.0, 0.7)
        else:
            effective_width = line_width
            level_alpha_boost = 1.0

        feather = 0.8
        alpha_mask = np.clip(1.0 - (dist_pixels - effective_width + 0.5) / feather, 0.0, 1.0)
        alpha_mask *= level_alpha_boost * float(np.clip(intensity, 0.0, 1.0))

        if mode == "dark":
            line_color = np.zeros_like(img_arr[..., :3])
        elif mode == "accent":
            max_c = np.maximum.reduce([img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]])
            boost = 255.0 / (max_c + 1e-3)
            line_color = np.clip(img_arr[..., :3] * boost[..., None] * 0.95, 0, 255)
        else:
            line_color = np.full_like(img_arr[..., :3], 245.0)

        out = img_arr.copy()
        alpha_3d = alpha_mask[..., None]
        out[..., :3] = img_arr[..., :3] * (1.0 - alpha_3d) + line_color * alpha_3d

        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))



# Backward-compatible alias for older imports.
NothingFilters = WallpaperFilters

if __name__ == "__main__":
    # Test
    test_img = Image.new("RGB", (1280, 720), (30, 30, 30))
    filtered = WallpaperFilters.apply_dot_matrix(test_img)
    filtered.save("output/test_filters.png")
    print("Test filter saved to output/test_filters.png")
