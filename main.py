import concurrent.futures
import dataclasses
import multiprocessing
from datetime import datetime
import os
import sys
import time
from types import SimpleNamespace

import click

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


def build_render_configuration(args: SimpleNamespace, seed: int) -> RenderConfiguration:
    """Translates parsed CLI arguments into a pipeline RenderConfiguration."""
    if args.res:
        width, height = args.res
    else:
        width, height = DEVICE_RESOLUTIONS.get(args.device, DEVICE_RESOLUTIONS["mbp14"])

    parsed_colors = [parse_hex_color(h) for h in args.colors] if args.colors else None

    # CLI flag names match RenderConfiguration fields 1:1, so pass through
    # everything the config declares and let anything else stay CLI-only.
    # Bespoke conversions below are the only hand-maintained mapping.
    kwargs = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(RenderConfiguration)
        if hasattr(args, field.name)
    }
    kwargs.update(
        width=width,
        height=height,
        colors=parsed_colors,
        fractal_center=tuple(args.fractal_center) if args.fractal_center is not None else None,
        legacy_idw=bool(args.legacy_idw),
        seed=seed,
        capture_base_image=bool(args.preview_fx),
    )
    return RenderConfiguration(**kwargs)


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


def generate(args: SimpleNamespace, seed: int) -> str:
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


def _show_full_help(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Eager --help-all callback: reveal hidden advanced options, print help, exit."""
    if not value or ctx.resilient_parsing:
        return
    for opt in ctx.command.params:
        if isinstance(opt, click.Option):
            opt.hidden = False
    click.echo(ctx.command.get_help(ctx))
    ctx.exit()


def _available_presets() -> list:
    presets_dir = resolve_preset_directory()
    if not presets_dir.exists():
        return []
    return sorted(p.stem for p in presets_dir.glob("*.json"))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--help-all", is_flag=True, is_eager=True, expose_value=False, callback=_show_full_help, help="Show all options including advanced overrides.")
@click.version_option(__version__, "-v", "--version", prog_name="wp-generator", message="%(prog)s %(version)s", help="Show generator engine version and exit.")
@click.option("--device", type=click.Choice(list(DEVICE_RESOLUTIONS.keys())), default="mbp14", show_default=True, help="Standard screen resolution device.")
@click.option("--count", type=int, default=1, show_default=True, help="Number of wallpapers to generate.")
@click.option("--output", default=None, help="Output path (single image). Auto-named if omitted.")
@click.option("--out-dir", default=None, help="Directory to save generated wallpapers.")
@click.option("--workers", type=int, default=None, help="Worker count for parallel generation.")
@click.option("--metadata", "--inspect", "inspect_image", default=None, metavar="PATH", help="Extract and display engine metadata from an image.")
@click.option("--json", "json_output", is_flag=True, default=False, help="Format extracted metadata as JSON.")
@click.option("--preset", type=click.Choice(_available_presets()), default=None, help="Pre-defined aesthetic preset.")
@click.option("--colors", multiple=True, metavar="HEX", default=None, help="Custom hex colors. Repeat the flag for each color.")
@click.option("--sample", is_flag=True, default=False, help="Grab a color from the screen at mouse cursor.")
@click.option("--res", nargs=2, type=int, default=None, metavar=("W", "H"), hidden=True, help="Custom resolution. Overrides preset.")
@click.option("--complexity", type=int, default=None, hidden=True, help="Gradient complexity (folds/points).")
@click.option("--engine", type=click.Choice(list(GRADIENT_REGISTRY.keys())), default=None, hidden=True, help="Gradient engine to use.")
@click.option("--warp-strength", type=float, default=None, hidden=True, help="Domain warping displacement strength (0.0-3.0).")
@click.option("--ambient-weight", type=float, default=None, hidden=True, help="RBF base color anchor weight (0.0-1.0).")
@click.option("--dispersion", type=float, default=None, hidden=True, help="RBF emitter dispersion/reach (0.05-1.5).")
@click.option("--fractal-type", type=click.Choice(["fluid", "julia", "mandelbrot", "orbit_trap"]), default=None, hidden=True, help="Fractal formula: fluid, julia, mandelbrot, or orbit_trap.")
@click.option("--fractal-zoom", type=float, default=None, hidden=True, help="Fractal magnification zoom factor.")
@click.option("--fractal-center", nargs=2, type=float, default=None, metavar=("CX", "CY"), hidden=True, help="Fractal focal center coordinates (cx, cy).")
@click.option("--fractal-angle", type=float, default=None, hidden=True, help="Fractal field rotation angle in radians.")
@click.option("--phase-offset", type=float, default=None, hidden=True, help="Cyclic palette phase offset (0.0-1.0).")
@click.option("--frequency", type=float, default=None, hidden=True, help="Fractal palette frequency cycle multiplier.")
@click.option("--trap-weight", type=float, default=None, hidden=True, help="Geometric orbit trap blending weight (0.0-1.0).")
@click.option("--channel-offset", type=float, default=None, hidden=True, help="Decoupled Oklab lightness/chroma channel phase offset.")
@click.option("--fluid-mode", type=click.Choice(["random", "spatial"]), default=None, hidden=True, help="Fluid mesh generation logic.")
@click.option("--legacy-idw", is_flag=True, default=None, hidden=True, help="Use legacy IDW gradient engine instead of fluid mesh.")
@click.option("--harmonize", type=click.Choice(["perceptual", "theory", "nothing", "none"]), default=None, hidden=True, help="Color harmonizer algorithm.")
@click.option("--contours", "--contour", "contours", type=float, default=None, hidden=True, help="Topographic contour line intensity (0.0-1.0).")
@click.option("--contour-levels", type=int, default=None, hidden=True, help="Number of elevation contour intervals.")
@click.option("--contour-width", type=float, default=None, hidden=True, help="Pixel line thickness of contour lines.")
@click.option("--contour-mode", type=click.Choice(["light", "dark", "accent"]), default=None, hidden=True, help="Contour line styling: 'light', 'dark', or 'accent'.")
@click.option("--noise", type=float, default=None, hidden=True, help="Grain/noise amount (0.0-1.0).")
@click.option("--vignette", type=float, default=None, hidden=True, help="Subtle radial edge darkening (0.0-1.0).")
@click.option("--glass", is_flag=True, default=None, hidden=True, help="Apply frosted glass texture overlay.")
@click.option("--aberration", "--chromatic-aberration", "chromatic_aberration", type=float, default=None, hidden=True, help="Chromatic aberration strength (0.0-0.1).")
@click.option("--preview-fx", is_flag=True, default=None, hidden=True, help="Preview FX rendering layers.")
@click.option("--seed", type=int, default=None, hidden=True, help="Base seed for reproducibility.")
@click.pass_context
def main(ctx: click.Context, **kwargs: object) -> None:
    """Procedural wallpaper generator: writes PNG wallpapers for any screen. Run with --help-all to see advanced engine and filter overrides."""
    params = dict(ctx.params)

    if sum(bool(params[key]) for key in ("preset", "colors", "sample")) > 1:
        raise click.UsageError("--preset, --colors, and --sample are mutually exclusive")

    _validate_cli_arguments(params)
    args = SimpleNamespace(**params)

    if args.inspect_image:
        _inspect_image(args)
        return

    if args.sample:
        _sample_screen_color(args)

    base_seed = args.seed if args.seed is not None else int(time.time())
    if args.seed is None:
        print(f"Base seed: {base_seed}  (pass --seed {base_seed} to reproduce)")

    execute_batch(args, base_seed)


def _validate_cli_arguments(params: dict) -> None:
    """Validates parameter ranges and conflicts (exit code 2 on violation)."""
    if params.get("inspect_image"):
        return

    res = params.get("res")
    if res and (res[0] <= 0 or res[1] <= 0):
        raise click.UsageError("--res values must be positive")

    positive_checks = [
        ("--complexity", params.get("complexity")),
        ("--contour-levels", params.get("contour_levels")),
        ("--contour-width", params.get("contour_width")),
        ("--dispersion", params.get("dispersion")),
        ("--frequency", params.get("frequency")),
        ("--fractal-zoom", params.get("fractal_zoom")),
    ]
    for flag, val in positive_checks:
        if val is not None and val <= 0:
            raise click.UsageError(f"{flag} must be positive")

    range_checks = [
        ("--contours", params.get("contours"), 0.0, 1.0),
        ("--noise", params.get("noise"), 0.0, 1.0),
        ("--vignette", params.get("vignette"), 0.0, 1.0),
        ("--aberration", params.get("chromatic_aberration"), 0.0, 0.1),
        ("--ambient-weight", params.get("ambient_weight"), 0.0, 1.0),
        ("--phase-offset", params.get("phase_offset"), 0.0, 1.0),
        ("--trap-weight", params.get("trap_weight"), 0.0, 1.0),
    ]
    for flag, val, lo, hi in range_checks:
        if val is not None and not lo <= val <= hi:
            raise click.UsageError(f"{flag} must be between {lo} and {hi}")

    if params["count"] < 1:
        raise click.UsageError("--count must be at least 1")
    if params["count"] > 1 and params["output"]:
        raise click.UsageError("--output cannot be used with --count > 1 (files are auto-named)")

    if params.get("colors"):
        try:
            [parse_hex_color(h) for h in params["colors"]]
        except ValueError as exc:
            raise click.UsageError(str(exc))


def _inspect_image(args: SimpleNamespace) -> None:
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


def _sample_screen_color(args: SimpleNamespace) -> None:
    try:
        from engine.sampler import ScreenSampler
    except ImportError as exc:
        if sys.platform != "darwin":
            raise click.UsageError("--sample is available on macOS only") from exc
        if getattr(sys, "frozen", False):
            raise click.UsageError("--sample is not available in this build") from exc
        raise click.UsageError(
            "--sample needs the 'macos' extra: "
            "uv tool install 'wp-generator[macos] @ git+https://github.com/icherniukh/wp-generator'"
        ) from exc
    sampler = ScreenSampler()
    print("Sampling screen color in 3 seconds... move your mouse!")
    time.sleep(3)
    color = sampler.sample_pixel()
    print(f"Sampled color: {color} ({sampler.to_hex(color)})")
    if not args.colors:
        args.colors = [sampler.to_hex(color)]


def execute_batch(args: SimpleNamespace, base_seed: int) -> None:
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


if __name__ == "__main__":
    # Standalone builds re-launch this executable for --workers child processes;
    # freeze_support() hands those launches to multiprocessing instead of Click.
    multiprocessing.freeze_support()
    main()

