import numpy as np
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
        
        overlay_img = Image.fromarray(overlay, mode="RGBA")
        return Image.alpha_composite(image.convert("RGBA"), overlay_img)

    @staticmethod
    def apply_film_grain(image, amount=0.08):
        """
        Applies a high-quality cinematic film grain using Overlay blending.
        This prevents harsh black clipping and respects the underlying luminance.
        """
        if amount <= 0:
            return image

        img_arr = np.array(image).astype(np.float32) / 255.0
        channels = img_arr.shape[2]

        sigma = amount * 0.15
        noise = np.random.normal(0.5, sigma, img_arr.shape[:2])
        noise = np.clip(noise, 0.0, 1.0)
        
        noise_stack = np.stack([noise] * 3, axis=-1)
        
        base = img_arr[..., :3]
        mask = base < 0.5
        
        res_rgb = np.empty_like(base)
        res_rgb[mask] = 2.0 * base[mask] * noise_stack[mask]
        res_rgb[~mask] = 1.0 - 2.0 * (1.0 - base[~mask]) * (1.0 - noise_stack[~mask])
        
        res = img_arr.copy()
        res[..., :3] = res_rgb
        
        return Image.fromarray(np.clip(res * 255.0, 0, 255).astype(np.uint8))

    apply_noise = apply_film_grain
    apply_dither = apply_film_grain

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


# Backward-compatible alias for older imports.
NothingFilters = WallpaperFilters

if __name__ == "__main__":
    # Test
    test_img = Image.new("RGB", (1280, 720), (30, 30, 30))
    filtered = WallpaperFilters.apply_dot_matrix(test_img)
    filtered.save("output/test_filters.png")
    print("Test filter saved to output/test_filters.png")
