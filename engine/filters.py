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
        # Convert image to grayscale to get luminance
        grayscale = image.convert("L")
        lum_data = np.array(grayscale)
        
        # Create a blank overlay
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw dots in a grid
        for y in range(0, height, spacing):
            for x in range(0, width, spacing):
                # Sample luminance at this grid point
                # Safety check for bounds
                lum = lum_data[min(y, height-1), min(x, width-1)] / 255.0
                
                if mode == "size":
                    # Modulate radius: Brighter pixels = smaller dots
                    radius = max_radius * (1.0 - lum) * intensity
                    alpha = int(255 * intensity)
                else: # alpha mode
                    radius = max_radius
                    # Darker pixels = higher opacity
                    alpha = int(255 * (1.0 - lum) * intensity)
                
                if radius > 0.5 and alpha > 5:
                    # Draw a white dot (classic Nothing)
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                                 fill=(255, 255, 255, alpha))
        
        return Image.alpha_composite(image.convert("RGBA"), overlay)

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
