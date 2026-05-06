import argparse
import random
import os
from datetime import datetime

SCREEN_PRESETS = {
    # Apple MacBooks (14:9 aspect ratios)
    "mbp14": (3024, 1964),   # MacBook Pro 14" (M1/M2/M3/M4)
    "mbp16": (3456, 2234),   # MacBook Pro 16" (M1/M2/M3/M4)
    "mba13": (2560, 1664),   # MacBook Air 13" (M2/M3)
    "mba15": (2880, 1864),   # MacBook Air 15" (M2/M3)
    
    # Standard Widescreen (16:9 aspect ratios)
    "1080p": (1920, 1080),   # FHD Standard Monitor
    "1440p": (2560, 1440),   # QHD Monitor
    "4k":    (3840, 2160),   # UHD Monitor
    "5k":    (5120, 2880),   # Apple Studio Display / 5K Monitors
    
    # Ultrawide Monitors
    "uw":    (3440, 1440),   # Standard Ultrawide (21:9)
    "suw":   (5120, 1440),   # Super Ultrawide (32:9)
    
    # Mobile / Portrait
    "mobile": (1080, 1920),  # Generic Smartphone Portrait (9:16)
}

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
    complexity  = args.complexity  if args.complexity  is not None else random.randint(2, 18)
    noise       = args.noise       if args.noise       is not None else round(random.uniform(0.005, 0.20), 3)
    fluid_mode  = args.fluid_mode  if getattr(args, 'fluid_mode', None) else "random"

    if nothing_style:
        dots         = args.dots         if args.dots         is not None else round(random.uniform(0.1, 1.0), 2)
        dots_spacing = args.dots_spacing if args.dots_spacing is not None else random.randint(6, 80)
        dots_mode    = args.dots_mode    if args.dots_mode    is not None else random.choice(["size", "alpha"])
    else:
        dots         = args.dots         if args.dots         is not None else round(random.uniform(0.2, 1.0), 2)
        dots_spacing = args.dots_spacing if args.dots_spacing is not None else random.randint(8, 60)
        dots_mode    = args.dots_mode    if args.dots_mode    is not None else random.choice(["size", "alpha"])

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
        import colorsys
        # Scale colors based on complexity, max 5.
        num_colors = max(2, min(5, complexity // 2))
        base_colors = []
        
        # 1. Grounding Anchor (70% chance dark, 30% light, low saturation)
        is_dark = random.random() < 0.7
        h = random.random()
        s = random.uniform(0.0, 0.25)
        l = random.uniform(0.05, 0.18) if is_dark else random.uniform(0.85, 0.95)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        base_colors.append((int(r*255), int(g*255), int(b*255)))
        
        # 2. Accents (Vibrant colors spread across hue space)
        for _ in range(num_colors - 1):
            h = random.random()
            s = random.uniform(0.5, 1.0)
            l = random.uniform(0.3, 0.7)
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            base_colors.append((int(r*255), int(g*255), int(b*255)))

    # No padding needed for max colors, just ensure minimum 2
    while len(base_colors) < 2:
        base_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    harmonize = getattr(args, 'harmonize', None)
    colors_were_specified = bool(args.colors or getattr(args, 'theme', None))

    if getattr(args, 'no_adjust', False) or harmonize == "none":
        final_colors = base_colors
    elif harmonize == "nothing" or (harmonize is None and nothing_style):
        from engine.colors import StyleHarmonizer
        final_colors = StyleHarmonizer.adjust_for_nothing(base_colors)
    elif harmonize == "theory":
        from engine.colors import ColorTheoryHarmonizer
        final_colors = ColorTheoryHarmonizer.harmonize(base_colors)
    elif harmonize is None and not nothing_style and not colors_were_specified:
        # 2x more likely to apply theory harmonizer when no colors specified
        if random.random() < 2/3:
            from engine.colors import ColorTheoryHarmonizer
            final_colors = ColorTheoryHarmonizer.harmonize(base_colors)
        else:
            final_colors = base_colors
    else:
        final_colors = base_colors

    # Build output path
    if args.output is not None:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        color_str = "-".join(f"{r:02x}{g:02x}{b:02x}" for r, g, b in final_colors[:3])
        if len(final_colors) > 3:
            color_str += f"-plus{len(final_colors)-3}"
        output_path = f"wp_{timestamp}_{color_str}.png"

    color_hex = " ".join(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in final_colors[:3])
    if len(final_colors) > 3:
        color_hex += f" (+{len(final_colors)-3} more)"
        
    print(f"  seed={seed}  complexity={complexity}  noise={noise}  colors={color_hex}")

    res = args.res
    if not res:
        res = SCREEN_PRESETS.get(args.preset, SCREEN_PRESETS["mbp14"])

    from engine.gradient import MeshGradient, FluidMeshGradient
    if getattr(args, 'legacy_idw', False):
        engine = MeshGradient(width=res[0], height=res[1])
        img = engine.generate(final_colors, p=2.5)
    else:
        engine = FluidMeshGradient(width=res[0], height=res[1])
        img = engine.generate(final_colors, complexity=complexity, fluid_mode=fluid_mode)

    from engine.filters import WallpaperFilters
    if nothing_style:
        img = WallpaperFilters.apply_noise(img, amount=noise)
        img = WallpaperFilters.apply_dot_matrix(img, spacing=dots_spacing, intensity=dots, mode=dots_mode)
        if getattr(args, 'theme', None) == "glass":
            img = WallpaperFilters.apply_frosted_glass(img, blur_radius=80, noise_amount=0.01)
    else:
        img = WallpaperFilters.apply_noise(img, amount=noise)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img.save(output_path, "PNG")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Wallpaper Generator for macOS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Output Group ---
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output", default=None, help="Output path (single image). Auto-named if omitted.")
    output_group.add_argument("--preset", choices=list(SCREEN_PRESETS.keys()), default="mbp14", help="Standard screen resolution preset.")
    output_group.add_argument("--res", nargs=2, type=int, default=None, metavar=("W", "H"), help="Custom resolution. Overrides preset.")
    output_group.add_argument("--count", type=int, default=1, help="Number of wallpapers to generate.")

    # --- Color Group ---
    color_group = parser.add_argument_group("Colors & Styling")
    color_source = color_group.add_mutually_exclusive_group()
    color_source.add_argument("--colors", nargs="+", metavar="HEX", help="Custom hex colors.")
    color_source.add_argument("--theme", choices=["dark", "light", "red", "orange", "glass"], help="Pre-defined color palette.")
    color_source.add_argument("--sample", action="store_true", help="Grab a color from the screen at mouse cursor.")
    
    color_group.add_argument("--style", choices=["generic", "nothing"], default="generic", help="Overall visual style.")
    color_group.add_argument("--harmonize", choices=["theory", "nothing", "none"], default=None,
                        help="Color harmonizer: 'theory' (classical hue rules), 'nothing' (Nothing-style), 'none' (raw).")
    color_group.add_argument("--no-adjust", action="store_true", help=argparse.SUPPRESS) # Alias for --harmonize none
    
    # --- Engine Group ---
    engine_group = parser.add_argument_group("Rendering Engine")
    engine_group.add_argument("--complexity", type=int, default=None, help="Gradient complexity (folds/points).")
    engine_group.add_argument("--fluid-mode", choices=["random", "spatial"], default="random", help="Fluid mesh generation logic.")
    engine_group.add_argument("--legacy-idw", action="store_true", help="Use legacy IDW gradient engine instead of fluid mesh.")

    # --- Filters Group ---
    filter_group = parser.add_argument_group("Filters & Effects")
    filter_group.add_argument("--nothing", action="store_true", help="Enable Nothing-inspired filters and color harmonization (alias).")
    filter_group.add_argument("--dots", type=float, default=None, help="Dot matrix intensity (0.0-1.0).")
    filter_group.add_argument("--dots-mode", choices=["size", "alpha"], default=None, help="Vary dot 'size' or 'alpha'.")
    filter_group.add_argument("--dots-spacing", type=int, default=None, help="Pixel spacing between dots.")
    filter_group.add_argument("--noise", type=float, default=None, help="Grain/noise amount (0.0-1.0).")

    # --- Misc Group ---
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility.")
    misc_group.add_argument("--no-nothing", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # --- Validate fixed params ---
    if args.res and (args.res[0] <= 0 or args.res[1] <= 0):
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
    if getattr(args, 'no_nothing', False):
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
        import time
        base_seed = int(time.time())
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
