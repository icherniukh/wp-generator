import numpy as np
from PIL import Image, ImageDraw, ImageFilter

class NothingFilters:
    @staticmethod
    def apply_dot_matrix(image, spacing=25, max_radius=4, intensity=0.8, mode="size"):
        """
        Applies a signature 'Nothing' dot-matrix overlay.
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
            mask = (dist_sq <= radius**2) & (radius > 0.5)
            overlay[..., 3] = np.where(mask, alpha_val, 0)
        else: # alpha mode
            radius = max_radius
            # Darker pixels = higher opacity
            alpha = np.clip(255 * (1.0 - cell_lum) * intensity, 0, 255).astype(np.uint8)
            mask = dist_sq <= radius**2
            # Only apply alpha if greater than a threshold (e.g. 5)
            overlay[..., 3] = np.where(mask & (alpha > 5), alpha, 0)
        
        overlay_img = Image.fromarray(overlay, mode="RGBA")
        return Image.alpha_composite(image.convert("RGBA"), overlay_img)

    @staticmethod
    def apply_noise(image, amount=0.05):
        """
        Adds monochromatic grain to the image to prevent banding and add texture.
        """
        img_arr = np.array(image).astype(np.float32)
        channels = img_arr.shape[2]
        
        # Create noise for the RGB channels only, ignore Alpha
        noise = np.random.normal(0, 255 * amount, img_arr.shape[:2])
        noise_stack = [noise] * 3
        if channels == 4:
            noise_stack.append(np.zeros(img_arr.shape[:2])) # No noise on alpha
        
        noise = np.stack(noise_stack, axis=-1)
        
        res = img_arr + noise
        res = np.clip(res, 0, 255).astype(np.uint8)
        
        return Image.fromarray(res)

    @staticmethod
    def apply_frosted_glass(image, blur_radius=50, noise_amount=0.02):
        """
        Simulates the 'Glass' look by blurring and adding subtle noise.
        Uses downsampling for performance on large images.
        """
        # Downsample for speed
        width, height = image.size
        small = image.resize((width // 4, height // 4), Image.NEAREST)
        blurred = small.filter(ImageFilter.GaussianBlur(radius=blur_radius // 4))
        # Upscale back to original resolution
        final = blurred.resize((width, height), Image.LANCZOS)
        return NothingFilters.apply_noise(final, amount=noise_amount)

if __name__ == "__main__":
    # Test
    test_img = Image.new("RGB", (1280, 720), (30, 30, 30))
    filtered = NothingFilters.apply_dot_matrix(test_img)
    filtered.save("output/test_filters.png")
    print("Test filter saved to output/test_filters.png")
