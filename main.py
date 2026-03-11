import argparse
import random
import numpy as np
from engine.gradient import MeshGradient, FluidMeshGradient
from engine.filters import NothingFilters
from engine.colors import ColorHarmonizer
from tools.sampler import ScreenSampler
from PIL import Image

def hex_to_rgb(hex_str):
    try:
        hex_str = hex_str.lstrip('#')
        if len(hex_str) != 6:
            raise ValueError(f"Invalid hex color: {hex_str}")
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except Exception as e:
        print(f"Error parsing color '{hex_str}': {e}. Defaulting to Gray.")
        return (128, 128, 128)

def main():
    parser = argparse.ArgumentParser(description="Nothing-Style Wallpaper Generator for macOS")
    parser.add_argument("--theme", choices=["dark", "light", "red", "orange", "glass"], help="Pre-defined aesthetic themes.")
    parser.add_argument("--colors", nargs="+", help="Custom hex colors (e.g., #000000 #FFFFFF #FF0000).")
    parser.add_argument("--dots", type=float, default=0.8, help="Intensity of the dot matrix filter (0.0 - 1.0).")
    parser.add_argument("--dots-mode", choices=["size", "alpha"], default="size", help="How dots are modulated.")
    parser.add_argument("--dots-spacing", type=int, default=25, help="Pixel spacing between dots.")
    parser.add_argument("--noise", type=float, default=0.03, help="Amount of grain/noise (0.0 - 1.0).")
    parser.add_argument("--output", default="output/wallpaper.png", help="Path to save the generated wallpaper.")
    parser.add_argument("--sample", action="store_true", help="Grab a color from the screen at mouse cursor.")
    parser.add_argument("--no-adjust", action="store_true", help="Disable automatic color harmonization.")
    parser.add_argument("--res", nargs=2, type=int, default=[3024, 1964], help="Output resolution (width height). Defaults to 14-inch MacBook Pro resolution.")
    parser.add_argument("--no-nothing", action="store_true", help="Disable all 'Nothing' aesthetic filters and color snapping.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible gradients and colors.")
    parser.add_argument("--legacy-idw", action="store_true", help="Use the legacy Inverse Distance Weighting (IDW) gradient engine.")
    parser.add_argument("--complexity", type=int, default=5, help="Complexity of the gradient pattern. Lower = simpler, wider sweeps. Higher = more folds (default: 5).")
    
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Handle screen sampling if requested
    if args.sample:
        sampler = ScreenSampler()
        print("Sampling screen color in 3 seconds... move your mouse!")
        import time
        time.sleep(3)
        color = sampler.sample_pixel()
        print(f"Sampled Color: {color} ({sampler.to_hex(color)})")
        # For now, let's just use it as a base if no colors were provided
        if not args.colors:
            args.colors = [sampler.to_hex(color)]

    # Theme definitions
    themes = {
        "dark": [(10, 10, 12), (25, 25, 28), (35, 35, 40)],
        "light": [(210, 210, 215), (230, 230, 235), (190, 190, 195)],
        "red": [(15, 10, 10), (180, 30, 30), (35, 15, 15)],
        "orange": [(15, 12, 10), (200, 70, 20), (35, 20, 15)],
        "glass": [(40, 45, 55), (110, 115, 130), (25, 30, 40)]
    }

    # Prepare Colors
    if args.colors:
        base_colors = [hex_to_rgb(h) for h in args.colors]
    elif args.theme:
        base_colors = themes[args.theme]
    else:
        # Default to dark theme if nothing provided, unless no-nothing is set
        if not args.no_nothing:
            base_colors = list(themes["dark"])
        else:
            print("No colors provided. Generating random palette...")
            base_colors = [
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for _ in range(3)
            ]

    # Add random variations if only 1 color is provided
    while len(base_colors) < 3:
        if args.no_nothing:
            base_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        else:
            # Generate a color that is either very dark or very light to provide contrast
            if random.random() > 0.5:
                base_colors.append((random.randint(0, 30), random.randint(0, 30), random.randint(0, 30)))
            else:
                base_colors.append((random.randint(220, 255), random.randint(220, 255), random.randint(220, 255)))

    # Color Harmonization
    if not args.no_adjust and not args.no_nothing:
        print("Evaluating and adjusting colors for 'Nothing' aesthetic...")
        score = ColorHarmonizer.evaluate_fit(base_colors)
        print(f"Initial Harmony Score (Delta E): {score:.2f}")
        final_colors = ColorHarmonizer.adjust_for_nothing(base_colors)
        print(f"Final Adjusted Colors: {final_colors}")
    else:
        final_colors = base_colors

    # Generate Gradient
    if args.legacy_idw:
        print(f"Generating Legacy IDW Mesh Gradient ({args.res[0]}x{args.res[1]})...")
        engine = MeshGradient(width=args.res[0], height=args.res[1])
        img = engine.generate(final_colors, p=2.5)
    else:
        print(f"Generating Fluid Mesh Gradient ({args.res[0]}x{args.res[1]})... (Complexity: {args.complexity})")
        engine = FluidMeshGradient(width=args.res[0], height=args.res[1])
        img = engine.generate(final_colors, grid_size=(args.complexity, args.complexity))

    # Apply Filters
    if not args.no_nothing:
        print("Applying Nothing aesthetic filters...")
        # Add noise first to prevent banding
        img = NothingFilters.apply_noise(img, amount=args.noise)
        
        # Apply dot matrix
        img = NothingFilters.apply_dot_matrix(img, spacing=args.dots_spacing, intensity=args.dots, mode=args.dots_mode)
        
        if args.theme == "glass":
            print("Applying Frosted Glass effect...")
            img = NothingFilters.apply_frosted_glass(img, blur_radius=80, noise_amount=0.01)
    else:
        # For generic gradients, just add a tiny bit of noise to prevent banding
        print("Nothing style disabled. Applying minimal dither noise...")
        img = NothingFilters.apply_noise(img, amount=0.01)

    # Save
    print(f"Saving wallpaper to {args.output}...")
    img.save(args.output, "PNG")
    print("Done!")

if __name__ == "__main__":
    main()
