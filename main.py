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


def generate(args, nothing_style, seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    # Randomize any param the user left unset
    complexity  = args.complexity  if args.complexity  is not None else random.randint(3, 9)
    noise       = args.noise       if args.noise       is not None else round(random.uniform(0.01, 0.06), 3)

    if nothing_style:
        dots         = args.dots         if args.dots         is not None else round(random.uniform(0.3, 0.9), 2)
        dots_spacing = args.dots_spacing if args.dots_spacing is not None else random.randint(12, 35)
        dots_mode    = args.dots_mode    if args.dots_mode    is not None else random.choice(["size", "alpha"])
    else:
        dots         = args.dots         if args.dots         is not None else 0.8
        dots_spacing = args.dots_spacing if args.dots_spacing is not None else 25
        dots_mode    = args.dots_mode    if args.dots_mode    is not None else "size"

    themes = {
        "dark":   [(10, 10, 12),    (25, 25, 28),    (35, 35, 40)],
        "light":  [(210, 210, 215), (230, 230, 235), (190, 190, 195)],
        "red":    [(15, 10, 10),    (180, 30, 30),   (35, 15, 15)],
        "orange": [(15, 12, 10),    (200, 70, 20),   (35, 20, 15)],
        "glass":  [(40, 45, 55),    (110, 115, 130), (25, 30, 40)],
    }

    if args.colors:
        base_colors = [hex_to_rgb(h) for h in args.colors]
    elif args.theme:
        base_colors = list(themes[args.theme])
    elif nothing_style:
        base_colors = list(themes["dark"])
    else:
        base_colors = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(3)
        ]

    while len(base_colors) < 3:
        if not nothing_style:
            base_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        elif random.random() > 0.5:
            base_colors.append((random.randint(0, 30), random.randint(0, 30), random.randint(0, 30)))
        else:
            base_colors.append((random.randint(220, 255), random.randint(220, 255), random.randint(220, 255)))

    if not args.no_adjust and nothing_style:
        from engine.colors import StyleHarmonizer
        final_colors = StyleHarmonizer.adjust_for_nothing(base_colors)
    else:
        final_colors = base_colors

    # Build output path
    if args.output is not None:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%H%M%S")
        color_str = "-".join(f"{r:02x}{g:02x}{b:02x}" for r, g, b in final_colors[:3])
        output_path = f"wp_{color_str}_{timestamp}.png"

    color_hex = " ".join(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in final_colors[:3])
    print(f"  seed={seed}  complexity={complexity}  noise={noise}  colors={color_hex}")

    from engine.gradient import MeshGradient, FluidMeshGradient
    if args.legacy_idw:
        engine = MeshGradient(width=args.res[0], height=args.res[1])
        img = engine.generate(final_colors, p=2.5)
    else:
        engine = FluidMeshGradient(width=args.res[0], height=args.res[1])
        img = engine.generate(final_colors, complexity=complexity)

    from engine.filters import WallpaperFilters
    if nothing_style:
        img = WallpaperFilters.apply_noise(img, amount=noise)
        img = WallpaperFilters.apply_dot_matrix(img, spacing=dots_spacing, intensity=dots, mode=dots_mode)
        if args.theme == "glass":
            img = WallpaperFilters.apply_frosted_glass(img, blur_radius=80, noise_amount=0.01)
    else:
        img = WallpaperFilters.apply_noise(img, amount=noise)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img.save(output_path, "PNG")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Wallpaper Generator for macOS")
    parser.add_argument("--style", choices=["generic", "nothing"], default="generic")
    parser.add_argument("--nothing", action="store_true", help="Enable Nothing-inspired filters and color harmonization.")
    parser.add_argument("--theme", choices=["dark", "light", "red", "orange", "glass"])
    parser.add_argument("--colors", nargs="+", metavar="HEX", help="Custom hex colors.")
    parser.add_argument("--dots", type=float, default=None, help="Dot matrix intensity (0.0–1.0).")
    parser.add_argument("--dots-mode", choices=["size", "alpha"], default=None)
    parser.add_argument("--dots-spacing", type=int, default=None, help="Pixel spacing between dots.")
    parser.add_argument("--noise", type=float, default=None, help="Grain/noise amount (0.0–1.0).")
    parser.add_argument("--output", default=None, help="Output path (single image). Auto-named if omitted.")
    parser.add_argument("--sample", action="store_true", help="Grab a color from the screen at mouse cursor.")
    parser.add_argument("--no-adjust", action="store_true", help="Disable color harmonization.")
    parser.add_argument("--res", nargs=2, type=int, default=[3024, 1964], metavar=("W", "H"))
    parser.add_argument("--no-nothing", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility.")
    parser.add_argument("--legacy-idw", action="store_true", help="Use legacy IDW gradient engine.")
    parser.add_argument("--complexity", type=int, default=None, help="Gradient complexity.")
    parser.add_argument("--count", type=int, default=1, help="Number of wallpapers to generate (default: 1).")

    args = parser.parse_args()

    # --- Validate fixed params ---
    if args.res[0] <= 0 or args.res[1] <= 0:
        parser.error("--res values must be positive")
    if args.dots_spacing is not None and args.dots_spacing <= 0:
        parser.error("--dots-spacing must be positive")
    if args.complexity is not None and args.complexity <= 0:
        parser.error("--complexity must be positive")
    if args.dots is not None and not 0 <= args.dots <= 1:
        parser.error("--dots must be between 0.0 and 1.0")
    if args.noise is not None and not 0 <= args.noise <= 1:
        parser.error("--noise must be between 0.0 and 1.0")
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.count > 1 and args.output:
        parser.error("--output cannot be used with --count > 1 (files are auto-named)")

    nothing_style = args.nothing or args.style == "nothing"
    if args.no_nothing:
        nothing_style = False

    # --- Screen sampling (once, before the loop) ---
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

    # --- Validate colors early if provided ---
    if args.colors:
        try:
            [hex_to_rgb(h) for h in args.colors]
        except ValueError as e:
            parser.error(str(e))

    # --- Pick base seed ---
    base_seed = args.seed
    if base_seed is None:
        base_seed = random.randint(0, 2**31 - 1)
        print(f"Base seed: {base_seed}  (pass --seed {base_seed} to reproduce)")

    # --- Generate ---
    total = args.count
    for i in range(total):
        seed = base_seed + i
        if total > 1:
            print(f"[{i+1}/{total}]", end=" ")
        path = generate(args, nothing_style, seed)
        print(f"  -> {path}")

if __name__ == "__main__":
    main()
