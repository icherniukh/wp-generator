import argparse
import concurrent.futures
from datetime import datetime
import os
import sys
import time

from engine import __version__
from engine.gradients import GRADIENT_REGISTRY
from engine.pipeline import (
    DEVICE_RESOLUTIONS,
    RenderConfiguration,
    WallpaperPipeline,
    parse_hex_color,
    resolve_preset_directory,
)


def default_output_path(preset_name: str, seed: int, directory: str = ".") -> str:
    """Generates an output filename stamped with datetime and seed."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(directory, f"wp_{preset_name}_{timestamp}_{seed}.png")


def build_render_configuration(args: argparse.Namespace, seed: int) -> RenderConfiguration:
    """Translates parsed CLI arguments into a pipeline RenderConfiguration."""
    if args.res:
        width, height = args.res
    else:
        width, height = DEVICE_RESOLUTIONS.get(args.device, DEVICE_RESOLUTIONS["mbp14"])

    parsed_colors = [parse_hex_color(h) for h in args.colors] if args.colors else None

    return RenderConfiguration(
        width=width,
        height=height,
        preset=args.preset,
        colors=parsed_colors,
        complexity=args.complexity,
        engine=args.engine,
        fluid_mode=args.fluid_mode,
        harmonize=args.harmonize,
        noise=args.noise,
        vignette=args.vignette,
        dots=args.dots,
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
        fractal_center=tuple(args.fractal_center) if args.fractal_center is not None else None,
        fractal_angle=args.fractal_angle,
        legacy_idw=bool(args.legacy_idw),
        seed=seed,
        capture_base_image=bool(args.preview_fx),
    )


def save_preview_image(result, output_path: str) -> None:
    """Generates side-by-side comparison between base mesh and filtered wallpaper."""
    if not result.base_image:
        return
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


def generate(args: argparse.Namespace, seed: int) -> str:
    """Renders a single wallpaper instance and writes it to disk."""
    config = build_render_configuration(args, seed)
    result = WallpaperPipeline().render(config)

    if args.output is not None:
        output_path = args.output
    else:
        preset_name = args.preset or "custom"
        output_path = default_output_path(preset_name, seed, directory=args.out_dir or ".")

    color_hex = " ".join(result.color_hexes[:3])
    if len(result.color_hexes) > 3:
        color_hex += f" (+{len(result.color_hexes) - 3} more)"
    print(f"  seed={seed}  complexity={result.complexity}  noise={result.noise}  colors={color_hex}")
    contrast = result.palette_contrast
    ratios = ", ".join(f"{ratio:.2f}:1" for ratio in contrast.ratios)
    print(f"  anchor contrast={ratios}  target={contrast.threshold}:1  all_pass={contrast.passes}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.preview_fx:
        save_preview_image(result, output_path)

    result.save(output_path)
    return output_path


def build_argument_parser(show_all: bool = False) -> argparse.ArgumentParser:
    """Builds the CLI argument parser with compact default view and extended help."""
    parser = argparse.ArgumentParser(
        description="Wallpaper Generator for macOS\nRun with --help-all to see advanced engine and filter overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    help_group = parser.add_argument_group("Help")
    help_group.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show compact help and exit.")
    help_group.add_argument("--help-all", action="help", default=argparse.SUPPRESS, help="Show all options including advanced overrides.")
    help_group.add_argument("-v", "--version", action="version", version=f"wp-generator {__version__}", help="Show generator engine version and exit.")

    def adv_help(text: str) -> str:
        return text if show_all else argparse.SUPPRESS

    core_group = parser.add_argument_group("Core Settings")
    core_group.add_argument("--device", choices=list(DEVICE_RESOLUTIONS.keys()), default="mbp14", help="Standard screen resolution device.")
    core_group.add_argument("--count", type=int, default=1, help="Number of wallpapers to generate.")
    core_group.add_argument("--output", default=None, help="Output path (single image). Auto-named if omitted.")
    core_group.add_argument("--out-dir", default=None, help="Directory to save generated wallpapers.")
    core_group.add_argument("--workers", type=int, default=None, help="Worker count for parallel generation.")
    core_group.add_argument("--metadata", "--inspect", dest="inspect_image", default=None, metavar="PATH", help="Extract and display engine metadata from an image.")
    core_group.add_argument("--json", action="store_true", dest="json_output", help="Format extracted metadata as JSON.")

    color_group = parser.add_argument_group("Color Palettes")
    color_source = color_group.add_mutually_exclusive_group()

    presets_dir = resolve_preset_directory()
    available_presets = sorted(p.stem for p in presets_dir.glob("*.json")) if presets_dir.exists() else []

    color_source.add_argument("--preset", choices=available_presets, help="Pre-defined aesthetic preset.")
    color_source.add_argument("--colors", nargs="+", metavar="HEX", help="Custom hex colors.")
    color_source.add_argument("--sample", action="store_true", help="Grab a color from the screen at mouse cursor.")

    engine_title = "Advanced Engine Overrides" if show_all else "Advanced Engine Overrides (hidden, use --help-all)"
    engine_group = parser.add_argument_group(engine_title)
    engine_group.add_argument("--res", nargs=2, type=int, default=None, metavar=("W", "H"), help=adv_help("Custom resolution. Overrides preset."))
    engine_group.add_argument("--complexity", type=int, default=None, help=adv_help("Gradient complexity (folds/points)."))
    engine_group.add_argument("--engine", choices=list(GRADIENT_REGISTRY.keys()), default=None, help=adv_help("Gradient engine to use."))
    engine_group.add_argument("--warp-strength", type=float, default=None, help=adv_help("Domain warping displacement strength (0.0-3.0)."))
    engine_group.add_argument("--ambient-weight", type=float, default=None, help=adv_help("RBF base color anchor weight (0.0-1.0)."))
    engine_group.add_argument("--dispersion", type=float, default=None, help=adv_help("RBF emitter dispersion/reach (0.05-1.5)."))
    engine_group.add_argument("--fractal-type", choices=["fluid", "julia", "mandelbrot", "orbit_trap"], default=None, help=adv_help("Fractal formula: fluid, julia, mandelbrot, or orbit_trap."))
    engine_group.add_argument("--fractal-zoom", type=float, default=None, help=adv_help("Fractal magnification zoom factor."))
    engine_group.add_argument("--fractal-center", nargs=2, type=float, default=None, metavar=("CX", "CY"), help=adv_help("Fractal focal center coordinates (cx, cy)."))
    engine_group.add_argument("--fractal-angle", type=float, default=None, help=adv_help("Fractal field rotation angle in radians."))
    engine_group.add_argument("--phase-offset", type=float, default=None, help=adv_help("Cyclic palette phase offset (0.0-1.0)."))
    engine_group.add_argument("--frequency", type=float, default=None, help=adv_help("Fractal palette frequency cycle multiplier."))
    engine_group.add_argument("--trap-weight", type=float, default=None, help=adv_help("Geometric orbit trap blending weight (0.0-1.0)."))
    engine_group.add_argument("--channel-offset", type=float, default=None, help=adv_help("Decoupled Oklab lightness/chroma channel phase offset."))
    engine_group.add_argument("--fluid-mode", choices=["random", "spatial"], default=None, help=adv_help("Fluid mesh generation logic."))
    engine_group.add_argument("--legacy-idw", action="store_true", default=None, help=adv_help("Use legacy IDW gradient engine instead of fluid mesh."))
    engine_group.add_argument("--harmonize", choices=["perceptual", "theory", "nothing", "none"], default=None, help=adv_help("Color harmonizer algorithm."))

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
    filter_group.add_argument("--glass", action="store_true", default=None, help=adv_help("Apply frosted glass texture overlay."))
    filter_group.add_argument("--aberration", "--chromatic-aberration", type=float, default=None, dest="chromatic_aberration", help=adv_help("Chromatic aberration strength (0.0-0.1)."))
    filter_group.add_argument("--preview-fx", action="store_true", default=None, help=adv_help("Preview FX rendering layers."))

    misc_group = parser.add_argument_group("Miscellaneous" if show_all else "Miscellaneous (hidden, use --help-all)")
    misc_group.add_argument("--seed", type=int, default=None, help=adv_help("Base seed for reproducibility."))

    return parser


def validate_cli_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validates parameter ranges and conflicts."""
    if getattr(args, "inspect_image", None):
        return

    if args.res and (args.res[0] <= 0 or args.res[1] <= 0):
        parser.error("--res values must be positive")

    positive_checks = [
        ("--dots-spacing", args.dots_spacing),
        ("--complexity", args.complexity),
        ("--contour-levels", args.contour_levels),
        ("--contour-width", args.contour_width),
        ("--dispersion", args.dispersion),
        ("--frequency", args.frequency),
        ("--fractal-zoom", args.fractal_zoom),
    ]
    for flag, val in positive_checks:
        if val is not None and val <= 0:
            parser.error(f"{flag} must be positive")

    range_checks = [
        ("--dots", args.dots, 0.0, 1.0),
        ("--contours", args.contours, 0.0, 1.0),
        ("--noise", args.noise, 0.0, 1.0),
        ("--vignette", args.vignette, 0.0, 1.0),
        ("--aberration", args.chromatic_aberration, 0.0, 0.1),
        ("--ambient-weight", args.ambient_weight, 0.0, 1.0),
        ("--phase-offset", args.phase_offset, 0.0, 1.0),
        ("--trap-weight", args.trap_weight, 0.0, 1.0),
    ]
    for flag, val, lo, hi in range_checks:
        if val is not None and not lo <= val <= hi:
            parser.error(f"{flag} must be between {lo} and {hi}")

    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.count > 1 and args.output:
        parser.error("--output cannot be used with --count > 1 (files are auto-named)")

    if args.colors:
        try:
            [parse_hex_color(h) for h in args.colors]
        except ValueError as exc:
            parser.error(str(exc))


def execute_batch(args: argparse.Namespace, base_seed: int) -> None:
    """Executes single or multi-worker batch generation loop."""
    total = args.count
    workers = args.workers or min(total, os.cpu_count() or 1)

    if total > 1 and workers > 1:
        print(f"Generating {total} wallpapers in parallel using {workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(generate, args, base_seed + i): i + 1
                for i in range(total)
            }
            for completed_count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                path = future.result()
                print(f"[{completed_count}/{total}] -> {path}")
    else:
        for i in range(total):
            seed = base_seed + i
            if total > 1:
                print(f"[{i+1}/{total}]", end=" ")
            path = generate(args, seed)
            print(f"  -> {path}")


def main() -> None:
    show_all = "--help-all" in sys.argv
    parser = build_argument_parser(show_all=show_all)
    args = parser.parse_args()
    validate_cli_arguments(parser, args)

    if args.inspect_image:
        from engine.metadata import extract_image_metadata

        metadata = extract_image_metadata(args.inspect_image)
        if metadata is None:
            print(f"Error: No wallpaper metadata found in '{args.inspect_image}'.", file=sys.stderr)
            sys.exit(1)

        if args.json_output:
            print(metadata.to_json(indent=2))
        else:
            print(f"Engine:     {metadata.engine} (generator v{metadata.version})")
            if metadata.parameters.get("preset"):
                preset_str = str(metadata.parameters["preset"])
                if metadata.parameters.get("preset_version"):
                    preset_str += f" (v{metadata.parameters['preset_version']})"
                print(f"Preset:     {preset_str}")
            print(f"Seed:       {metadata.seed}")
            if metadata.created_at:
                print(f"Created:    {metadata.created_at}")
            print("Parameters:")
            for key, val in sorted(metadata.parameters.items()):
                print(f"  {key}: {val}")
        return

    if args.sample:
        from tools.sampler import ScreenSampler
        sampler = ScreenSampler()
        print("Sampling screen color in 3 seconds... move your mouse!")
        time.sleep(3)
        color = sampler.sample_pixel()
        print(f"Sampled color: {color} ({sampler.to_hex(color)})")
        if not args.colors:
            args.colors = [sampler.to_hex(color)]

    base_seed = args.seed if args.seed is not None else int(time.time())
    if args.seed is None:
        print(f"Base seed: {base_seed}  (pass --seed {base_seed} to reproduce)")

    execute_batch(args, base_seed)


if __name__ == "__main__":
    main()

