import argparse
import random
import os
from datetime import datetime

from engine.pipeline import (
    DEVICE_RESOLUTIONS,
    PipelineConfigurationError,
    RenderConfiguration,
    WallpaperPipeline,
    parse_hex_color,
)

SCREEN_PRESETS = DEVICE_RESOLUTIONS
hex_to_rgb = parse_hex_color


def default_output_path(preset_name, seed, directory="."):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(directory, f"wp_{preset_name}_{timestamp}_{seed}.png")


def generate(args, nothing_style, seed):
    # Resolve resolution from --res or --device
    if args.res:
        width, height = args.res
    else:
        width, height = DEVICE_RESOLUTIONS.get(args.device, DEVICE_RESOLUTIONS["mbp14"])

    parsed_colors = [parse_hex_color(h) for h in args.colors] if args.colors else None

    # Harmonize override resolution
    harmonize = args.harmonize
    if harmonize is None and nothing_style:
        harmonize = "nothing"

    dots = args.dots
    if dots is None and nothing_style:
        dots = 0.8

    config = RenderConfiguration(
        width=width,
        height=height,
        preset=args.preset,
        colors=parsed_colors,
        complexity=args.complexity,
        engine=args.engine,
        fluid_mode=args.fluid_mode,
        harmonize=harmonize,
        noise=args.noise,
        vignette=args.vignette,
        dots=dots,
        dots_spacing=args.dots_spacing,
        dots_mode=args.dots_mode,
        contours=args.contours,
        contour_levels=args.contour_levels,
        contour_width=args.contour_width,
        contour_mode=args.contour_mode,
        glass=args.glass,
        chromatic_aberration=args.chromatic_aberration,
        warp_strength=args.warp_strength,
        ambient_weight=args.ambient_weight,
        dispersion=args.dispersion,
        phase_offset=args.phase_offset,
        frequency=args.frequency,
        trap_weight=args.trap_weight,
        channel_offset=args.channel_offset,
        fractal_type=args.fractal_type,
        fractal_zoom=args.fractal_zoom,
        legacy_idw=bool(getattr(args, "legacy_idw", False)),
        seed=seed,
        capture_base_image=bool(getattr(args, "preview_fx", False)),
    )

    pipeline = WallpaperPipeline()
    result = pipeline.render(config)

    # Build output path
    if args.output is not None:
        output_path = args.output
    else:
        preset_name = args.preset if getattr(args, "preset", None) else "custom"
        output_path = default_output_path(
            preset_name, seed, directory=getattr(args, "out_dir", None) or "."
        )

    color_hex = " ".join(result.color_hexes[:3])
    if len(result.color_hexes) > 3:
        color_hex += f" (+{len(result.color_hexes)-3} more)"
    print(f"  seed={seed}  complexity={result.complexity}  noise={result.noise}  colors={color_hex}")
    contrast = result.palette_contrast
    ratios = ", ".join(f"{ratio:.2f}:1" for ratio in contrast.ratios)
    print(f"  anchor contrast={ratios}  target={contrast.threshold}:1  all_pass={contrast.passes}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if getattr(args, "preview_fx", False) and result.base_image:
        from PIL import Image, ImageDraw
        preview_img = Image.new("RGB", (result.width * 2, result.height))
        preview_img.paste(result.base_image, (0, 0))
        preview_img.paste(result.image, (result.width, 0))
        draw = ImageDraw.Draw(preview_img)
        draw.text((10, 10), "BASE MESH", fill=(255, 255, 255))
        draw.text((result.width + 10, 10), "FX APPLIED", fill=(255, 255, 255))
        preview_path = output_path.replace(".png", "_preview.png")
        preview_img.save(preview_path, "PNG")
        print(f"  -> Preview saved to {preview_path}")

    result.image.save(output_path, "PNG")
    return output_path


def main():
    import sys
    show_all = "--help-all" in sys.argv

    parser = argparse.ArgumentParser(
        description="Wallpaper Generator for macOS\nRun with --help-all to see advanced engine and filter overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    
    # --- Help Options ---
    help_group = parser.add_argument_group("Help")
    help_group.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this compact help message and exit.")
    help_group.add_argument("--help-all", action="help", default=argparse.SUPPRESS, help="Show full help message including advanced overrides and exit.")

    def adv_help(text):
        return text if show_all else argparse.SUPPRESS

    # --- Core Group ---
    core_group = parser.add_argument_group("Core Settings")
    core_group.add_argument("--device", choices=list(SCREEN_PRESETS.keys()), default="mbp14", help="Standard screen resolution device.")
    core_group.add_argument("--count", type=int, default=1, help="Number of wallpapers to generate.")
    core_group.add_argument("--output", default=None, help="Output path (single image). Auto-named if omitted.")
    core_group.add_argument("--out-dir", default=None, help="Directory to save generated wallpapers.")
    core_group.add_argument("--workers", type=int, default=None, help="Worker count for parallel generation.")

    # --- Color Group ---
    color_group = parser.add_argument_group("Color Palettes")
    color_source = color_group.add_mutually_exclusive_group()
    
    import os
    try:
        # Resolve the directory where main.py lives, so it works globally
        base_dir = os.path.dirname(os.path.abspath(__file__))
        presets_dir = os.path.join(base_dir, "presets")
        available_presets = [f.replace(".json", "") for f in os.listdir(presets_dir) if f.endswith(".json")]
    except FileNotFoundError:
        available_presets = []

    color_source.add_argument("--preset", choices=available_presets, help="Pre-defined aesthetic preset.")
    color_source.add_argument("--colors", nargs="+", metavar="HEX", help="Custom hex colors.")
    color_source.add_argument("--sample", action="store_true", help="Grab a color from the screen at mouse cursor.")
    
    # --- Engine Overrides ---
    engine_title = "Advanced Engine Overrides" if show_all else "Advanced Engine Overrides (hidden, use --help-all)"
    engine_group = parser.add_argument_group(engine_title)
    engine_group.add_argument("--res", nargs=2, type=int, default=None, metavar=("W", "H"), help=adv_help("Custom resolution. Overrides preset."))
    engine_group.add_argument("--complexity", type=int, default=None, help=adv_help("Gradient complexity (folds/points)."))
    engine_group.add_argument("--engine", choices=["simplex", "flow-field", "fluid-wave", "idw", "spline-wave", "domain-warp", "rbf", "topographic", "fractal"], default=None, help=adv_help("Gradient engine to use."))
    engine_group.add_argument("--warp-strength", type=float, default=None, help=adv_help("Domain warping displacement strength (0.0-3.0)."))
    engine_group.add_argument("--ambient-weight", type=float, default=None, help=adv_help("RBF base color anchor weight (0.0-1.0)."))
    engine_group.add_argument("--dispersion", type=float, default=None, help=adv_help("RBF emitter dispersion/reach (0.05-1.5)."))
    engine_group.add_argument("--fractal-type", choices=["julia", "mandelbrot", "orbit_trap"], default=None, help=adv_help("Fractal formula: julia, mandelbrot, or orbit_trap."))
    engine_group.add_argument("--fractal-zoom", type=float, default=None, help=adv_help("Fractal magnification zoom factor."))
    engine_group.add_argument("--phase-offset", type=float, default=None, help=adv_help("Cyclic palette phase offset (0.0-1.0)."))
    engine_group.add_argument("--frequency", type=float, default=None, help=adv_help("Fractal palette frequency cycle multiplier."))
    engine_group.add_argument("--trap-weight", type=float, default=None, help=adv_help("Geometric orbit trap blending weight (0.0-1.0)."))
    engine_group.add_argument("--channel-offset", type=float, default=None, help=adv_help("Decoupled Oklab lightness/chroma channel phase offset."))
    engine_group.add_argument("--fluid-mode", choices=["random", "spatial"], default=None, help=adv_help("Fluid mesh generation logic."))
    engine_group.add_argument("--legacy-idw", action="store_true", default=None, help=adv_help("Use legacy IDW gradient engine instead of fluid mesh."))
    engine_group.add_argument("--harmonize", choices=["perceptual", "theory", "nothing", "none"], default=None, help=adv_help("Color harmonizer: perceptual/theory preserve hues in OKLCH; nothing bounds muted anchors and vivid accents."))

    # --- Filters Overrides ---
    filter_title = "Advanced Filter Overrides" if show_all else "Advanced Filter Overrides (hidden, use --help-all)"
    filter_group = parser.add_argument_group(filter_title)
    filter_group.add_argument("--dots", type=float, default=None, help=adv_help("Dot matrix intensity (0.0-1.0)."))
    filter_group.add_argument("--dots-mode", choices=["size", "alpha"], default=None, help=adv_help("Vary dot 'size' or 'alpha'."))
    filter_group.add_argument("--dots-spacing", type=int, default=None, help=adv_help("Pixel spacing between dots."))
    filter_group.add_argument("--contours", "--contour", type=float, default=None, dest="contours", help=adv_help("Topographic contour line intensity (0.0-1.0)."))
    filter_group.add_argument("--contour-levels", type=int, default=None, help=adv_help("Number of elevation contour intervals."))
    filter_group.add_argument("--contour-width", type=float, default=None, help=adv_help("Pixel line thickness of contour lines."))
    filter_group.add_argument("--contour-mode", choices=["light", "dark", "accent"], default=None, help=adv_help("Contour line styling: 'light', 'dark', or 'accent'."))
    filter_group.add_argument("--noise", type=float, default=None, help=adv_help("Grain/noise amount (0.0-1.0)."))
    filter_group.add_argument("--vignette", type=float, default=None, help=adv_help("Subtle radial edge darkening (0.0-1.0)."))
    filter_group.add_argument("--glass", action="store_true", default=None, help=adv_help("Apply premium frosted glass texture overlay."))
    filter_group.add_argument("--aberration", "--chromatic-aberration", type=float, default=None, dest="chromatic_aberration", help=adv_help("Radial chromatic aberration strength (0.0-0.1)."))
    filter_group.add_argument("--preview-fx", action="store_true", default=None, help=adv_help("Preview FX rendering layers for development."))

    # --- Misc & Legacy ---
    misc_group = parser.add_argument_group("Miscellaneous" if show_all else "Miscellaneous (hidden, use --help-all)")
    misc_group.add_argument("--seed", type=int, default=None, help=adv_help("Base seed for reproducibility."))
    
    parser.add_argument("--style", choices=["generic", "nothing"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--nothing", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-adjust", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-nothing", action="store_true", default=None, help=argparse.SUPPRESS)

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
    if args.contours is not None and not 0 <= args.contours <= 1:
        parser.error("--contours must be between 0.0 and 1.0")
    if args.contour_levels is not None and args.contour_levels <= 0:
        parser.error("--contour-levels must be positive")
    if args.contour_width is not None and args.contour_width <= 0:
        parser.error("--contour-width must be positive")
    if args.noise is not None and not 0 <= args.noise <= 1:
        parser.error("--noise must be between 0.0 and 1.0")
    if args.vignette is not None and not 0 <= args.vignette <= 1:
        parser.error("--vignette must be between 0.0 and 1.0")
    if args.chromatic_aberration is not None and not 0 <= args.chromatic_aberration <= 0.1:
        parser.error("--aberration must be between 0.0 and 0.1")
    if args.ambient_weight is not None and not 0.0 <= args.ambient_weight <= 1.0:
        parser.error("--ambient-weight must be between 0.0 and 1.0")
    if args.dispersion is not None and args.dispersion <= 0:
        parser.error("--dispersion must be positive")
    if args.phase_offset is not None and not 0.0 <= args.phase_offset <= 1.0:
        parser.error("--phase-offset must be between 0.0 and 1.0")
    if args.frequency is not None and args.frequency <= 0:
        parser.error("--frequency must be positive")
    if args.trap_weight is not None and not 0.0 <= args.trap_weight <= 1.0:
        parser.error("--trap-weight must be between 0.0 and 1.0")
    if args.fractal_zoom is not None and args.fractal_zoom <= 0:
        parser.error("--fractal-zoom must be positive")
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
    workers = args.workers
    if workers is None:
        workers = min(total, os.cpu_count() or 1)

    if total > 1 and workers > 1:
        import concurrent.futures
        print(f"Generating {total} wallpapers in parallel using {workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(generate, args, nothing_style, base_seed + i): i + 1
                for i in range(total)
            }
            completed_count = 0
            for future in concurrent.futures.as_completed(futures):
                completed_count += 1
                path = future.result()
                print(f"[{completed_count}/{total}] -> {path}")
    else:
        for i in range(total):
            seed = base_seed + i
            if total > 1:
                print(f"[{i+1}/{total}]", end=" ")
            path = generate(args, nothing_style, seed)
            print(f"  -> {path}")

if __name__ == "__main__":
    main()
