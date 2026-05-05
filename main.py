import argparse
import random
import os
from datetime import datetime

def hex_to_rgb(hex_str):
    try:
        hex_str = hex_str.strip().lstrip('#')
        if len(hex_str) != 6:
            raise ValueError(f"Invalid hex color: {hex_str}")
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except Exception as e:
        raise ValueError(f"Invalid hex color '{hex_str}'") from e

def main():
    parser = argparse.ArgumentParser(description="Wallpaper Generator for macOS")
    parser.add_argument("--style", choices=["generic", "nothing"], default="generic", help="Rendering style to apply.")
    parser.add_argument("--nothing", action="store_true", help="Enable Nothing-inspired filters and color harmonization.")
    parser.add_argument("--theme", choices=["dark", "light", "red", "orange", "glass"], help="Pre-defined aesthetic themes.")
    parser.add_argument("--colors", nargs="+", help="Custom hex colors (e.g., #FF0000 #00FF00).")
    parser.add_argument("--dots", type=float, default=None, help="Dot matrix intensity (0.0–1.0). Randomized if omitted.")
    parser.add_argument("--dots-mode", choices=["size", "alpha"], default=None, help="Dot modulation type. Randomized if omitted.")
    parser.add_argument("--dots-spacing", type=int, default=None, help="Pixel spacing between dots. Randomized if omitted.")
    parser.add_argument("--noise", type=float, default=None, help="Grain/noise amount (0.0–1.0). Randomized if omitted.")
    parser.add_argument("--output", default=None, help="Output path. Defaults to wp_<colors>_<time>.png in current dir.")
    parser.add_argument("--sample", action="store_true", help="Grab a color from the screen at mouse cursor.")
    parser.add_argument("--no-adjust", action="store_true", help="Disable automatic color harmonization.")
    parser.add_argument("--res", nargs=2, type=int, default=[3024, 1964], metavar=("W", "H"), help="Output resolution (default: 3024x1964).")
    parser.add_argument("--no-nothing", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible output.")
    parser.add_argument("--legacy-idw", action="store_true", help="Use legacy Inverse Distance Weighting gradient engine.")
    parser.add_argument("--complexity", type=int, default=None, help="Gradient complexity (grid points). Randomized if omitted.")

    args = parser.parse_args()

    nothing_style = args.nothing or args.style == "nothing"
    if args.no_nothing:
        nothing_style = False

    # --- Seed: always use one so every run is fully reproducible ---
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"Seed: {args.seed}  (pass --seed {args.seed} to reproduce)")
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    # --- Randomize omitted parameters (after seed so they're reproducible) ---
    if args.complexity is None:
        args.complexity = random.randint(3, 9)
        print(f"Complexity: {args.complexity}  (randomized)")

    if args.noise is None:
        args.noise = round(random.uniform(0.01, 0.06), 3)
        print(f"Noise: {args.noise}  (randomized)")

    if nothing_style:
        if args.dots is None:
            args.dots = round(random.uniform(0.3, 0.9), 2)
            print(f"Dots intensity: {args.dots}  (randomized)")
        if args.dots_spacing is None:
            args.dots_spacing = random.randint(12, 35)
            print(f"Dots spacing: {args.dots_spacing}  (randomized)")
        if args.dots_mode is None:
            args.dots_mode = random.choice(["size", "alpha"])
            print(f"Dots mode: {args.dots_mode}  (randomized)")
    else:
        args.dots = args.dots if args.dots is not None else 0.8
        args.dots_spacing = args.dots_spacing if args.dots_spacing is not None else 25
        args.dots_mode = args.dots_mode if args.dots_mode is not None else "size"

    # --- Validate ---
    if args.res[0] <= 0 or args.res[1] <= 0:
        parser.error("--res values must be positive")
    if args.dots_spacing <= 0:
        parser.error("--dots-spacing must be positive")
    if args.complexity <= 0:
        parser.error("--complexity must be positive")
    if not 0 <= args.dots <= 1:
        parser.error("--dots must be between 0.0 and 1.0")
    if not 0 <= args.noise <= 1:
        parser.error("--noise must be between 0.0 and 1.0")

    # --- Screen sampling ---
    if args.sample:
        from tools.sampler import ScreenSampler
        sampler = ScreenSampler()
        print("Sampling screen color in 3 seconds... move your mouse!")
        import time
        time.sleep(3)
        color = sampler.sample_pixel()
        print(f"Sampled color: {color} ({sampler.to_hex(color)})")
        if not args.colors:
            args.colors = [sampler.to_hex(color)]

    # --- Theme definitions ---
    themes = {
        "dark":   [(10, 10, 12),   (25, 25, 28),   (35, 35, 40)],
        "light":  [(210, 210, 215),(230, 230, 235), (190, 190, 195)],
        "red":    [(15, 10, 10),   (180, 30, 30),   (35, 15, 15)],
        "orange": [(15, 12, 10),   (200, 70, 20),   (35, 20, 15)],
        "glass":  [(40, 45, 55),   (110, 115, 130), (25, 30, 40)],
    }

    # --- Resolve base colors ---
    if args.colors:
        try:
            base_colors = [hex_to_rgb(h) for h in args.colors]
        except ValueError as e:
            parser.error(str(e))
    elif args.theme:
        base_colors = list(themes[args.theme])
    elif nothing_style:
        base_colors = list(themes["dark"])
    else:
        base_colors = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(3)
        ]
        print(f"Colors: {['#{:02x}{:02x}{:02x}'.format(*c) for c in base_colors]}  (randomized)")

    # Pad to at least 3 colors
    while len(base_colors) < 3:
        if not nothing_style:
            base_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        elif random.random() > 0.5:
            base_colors.append((random.randint(0, 30), random.randint(0, 30), random.randint(0, 30)))
        else:
            base_colors.append((random.randint(220, 255), random.randint(220, 255), random.randint(220, 255)))

    # --- Color harmonization ---
    if not args.no_adjust and nothing_style:
        from engine.colors import StyleHarmonizer
        print("Adjusting colors for Nothing-inspired style...")
        score = StyleHarmonizer.evaluate_fit(base_colors)
        print(f"Contrast score (Delta E): {score:.2f}")
        final_colors = StyleHarmonizer.adjust_for_nothing(base_colors)
        print(f"Adjusted colors: {final_colors}")
    else:
        final_colors = base_colors

    # --- Smart output filename ---
    if args.output is None:
        timestamp = datetime.now().strftime("%H%M%S")
        color_str = "-".join(f"{r:02x}{g:02x}{b:02x}" for r, g, b in final_colors[:3])
        args.output = f"wp_{color_str}_{timestamp}.png"

    # --- Generate gradient ---
    from engine.gradient import MeshGradient, FluidMeshGradient

    if args.legacy_idw:
        print(f"Generating Legacy IDW Mesh Gradient ({args.res[0]}x{args.res[1]})...")
        engine = MeshGradient(width=args.res[0], height=args.res[1])
        img = engine.generate(final_colors, p=2.5)
    else:
        print(f"Generating Fluid Mesh Gradient ({args.res[0]}x{args.res[1]}, complexity {args.complexity})...")
        engine = FluidMeshGradient(width=args.res[0], height=args.res[1])
        img = engine.generate(final_colors, complexity=args.complexity)

    # --- Apply filters ---
    from engine.filters import WallpaperFilters

    if nothing_style:
        print("Applying Nothing-inspired filters...")
        img = WallpaperFilters.apply_noise(img, amount=args.noise)
        img = WallpaperFilters.apply_dot_matrix(img, spacing=args.dots_spacing, intensity=args.dots, mode=args.dots_mode)
        if args.theme == "glass":
            print("Applying Frosted Glass effect...")
            img = WallpaperFilters.apply_frosted_glass(img, blur_radius=80, noise_amount=0.01)
    else:
        img = WallpaperFilters.apply_noise(img, amount=args.noise)

    # --- Save ---
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving to {args.output}...")
    img.save(args.output, "PNG")
    print("Done!")

if __name__ == "__main__":
    main()
