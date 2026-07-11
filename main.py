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

ENGINES = ["simplex-linear", "simplex-wave", "fluid-linear", "fluid-wave", "idw"]

_ENGINE_MAP = {
    "simplex-linear": ("simplex", "random"),
    "simplex-wave":   ("simplex", "spatial"),
    "fluid-linear":   ("fluid",   "random"),
    "fluid-wave":     ("fluid",   "spatial"),
    "idw":            ("idw",     None),
}
# Backward-compat aliases for presets that still use old names
_ENGINE_ALIASES = {"simplex": "simplex-linear", "fluid": "fluid-linear"}

# Parameters that accept comma-separated multi-values to trigger comparison mode
_COMPARABLE = {
    "engine":       {"coerce": str,   "valid": set(ENGINES)},
    "complexity":   {"coerce": int,   "min": 1},
    "noise":        {"coerce": float, "min": 0.0, "max": 1.0},
    "vignette":     {"coerce": float, "min": 0.0, "max": 1.0},
    "seed":         {"coerce": int},
    "dots":         {"coerce": float, "min": 0.0, "max": 1.0},
    "dots_spacing": {"coerce": int,   "min": 1},
    "dots_mode":    {"coerce": str,   "valid": {"size", "alpha"}},
}


def resolve_preset_parameter(arg_value, key, overrides, fallback):
    """Resolve CLI value, then preset override, then a generated fallback."""
    if arg_value is not None:
        return arg_value
    if key in overrides:
        return overrides[key]
    return fallback()


def default_output_path(preset_name, seed, directory="."):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(directory, f"wp_{preset_name}_{timestamp}_{seed}.png")


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

    # --- Preset Engine ---
    from engine.presets import PresetEngine
    preset_engine = PresetEngine()
    
    preset_data = {}
    if getattr(args, 'preset', None) and args.preset not in ["dark", "light", "red", "orange", "glass"]:
        try:
            preset_data = preset_engine.load_preset(args.preset)
        except FileNotFoundError:
            pass # fallback to legacy if missing

    overrides = preset_engine.get_engine_overrides(preset_data)

    # Helper to resolve param: CLI arg > Preset Override > Fallback Random
    def resolve_param(arg_val, override_key, fallback_gen):
        val = resolve_preset_parameter(arg_val, override_key, overrides, fallback_gen)
        if isinstance(val, dict):
            return int(preset_engine._sample_distribution(val)) if val.get("type", "").endswith("int") else preset_engine._sample_distribution(val)
        return val

    complexity  = resolve_param(args.complexity, "complexity", lambda: random.randint(2, 6))
    noise       = resolve_param(args.noise, "noise", lambda: round(random.uniform(0.005, 0.05), 3))
    vignette    = resolve_param(getattr(args, "vignette", None), "vignette", lambda: 0.0)
    engine_name = resolve_param(getattr(args, 'engine', None), "engine", lambda: "simplex-linear")
    engine_name = _ENGINE_ALIASES.get(engine_name, engine_name)
    
    dots         = resolve_param(args.dots, "dots", lambda: round(random.uniform(0.2, 1.0), 2))
    dots_spacing = resolve_param(args.dots_spacing, "dots_spacing", lambda: random.randint(8, 60))
    dots_mode    = resolve_param(args.dots_mode, "dots_mode", lambda: random.choice(["size", "alpha"]))

    themes = {
        "dark":   [(10, 10, 12),    (25, 25, 28),    (35, 35, 40)],
        "light":  [(210, 210, 215), (230, 230, 235), (190, 190, 195)],
        "red":    [(15, 10, 10),    (180, 30, 30),   (35, 15, 15)],
        "orange": [(15, 12, 10),    (200, 70, 20),   (35, 20, 15)],
        "glass":  [(40, 45, 55),    (110, 115, 130), (25, 30, 40)],
    }

    if args.colors:
        base_colors = [hex_to_rgb(h) for h in args.colors]
    elif getattr(args, 'preset', None) in themes:
        base_colors = list(themes[args.preset])
    elif preset_data:
        base_colors = preset_engine.generate_palette(
            preset_data,
            count=complexity // 2 if args.complexity else None,
            family_index=seed,
        )
    else:
        # Fallback dynamic generation if no preset matched
        import colorsys
        num_colors = max(2, min(5, complexity // 2))
        base_colors = []
        is_dark = random.random() < 0.7
        h = random.random()
        s = random.uniform(0.0, 0.25)
        l = random.uniform(0.05, 0.18) if is_dark else random.uniform(0.85, 0.95)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        base_colors.append((int(r*255), int(g*255), int(b*255)))
        for _ in range(num_colors - 1):
            h = random.random()
            s = random.uniform(0.5, 1.0)
            l = random.uniform(0.3, 0.7)
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            base_colors.append((int(r*255), int(g*255), int(b*255)))

    # No padding needed for max colors, just ensure minimum 2
    while len(base_colors) < 2:
        base_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    harmonize = resolve_param(getattr(args, 'harmonize', None), "harmonize", lambda: None)
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
        preset_name = args.preset if getattr(args, 'preset', None) else "custom"
        output_path = default_output_path(
            preset_name,
            seed,
            directory=getattr(args, "output_dir", None) or ".",
        )

    color_hex = " ".join(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in final_colors[:3])
    if len(final_colors) > 3:
        color_hex += f" (+{len(final_colors)-3} more)"
        
    print(f"  seed={seed}  complexity={complexity}  noise={noise}  colors={color_hex}")

    res = args.res or SCREEN_PRESETS["mbp14"]

    from engine.gradient import MeshGradient, FluidMeshGradient, SimplexMeshGradient
    kind, internal_mode = _ENGINE_MAP.get(engine_name, ("simplex", "random"))
    if kind == "idw":
        gradient_engine = MeshGradient(width=res[0], height=res[1])
        img = gradient_engine.generate(final_colors, p=1.6)
    elif kind == "fluid":
        gradient_engine = FluidMeshGradient(width=res[0], height=res[1])
        img = gradient_engine.generate(final_colors, complexity=complexity, fluid_mode=internal_mode)
    else:
        gradient_engine = SimplexMeshGradient(width=res[0], height=res[1])
        img = gradient_engine.generate(final_colors, complexity=complexity, fluid_mode=internal_mode, seed=seed)

    from engine.filters import WallpaperFilters

    if nothing_style:
        img = WallpaperFilters.apply_dither(img, amount=noise)
        img = WallpaperFilters.apply_dot_matrix(img, spacing=dots_spacing, intensity=dots, mode=dots_mode)
        if getattr(args, 'theme', None) == "glass":
            img = WallpaperFilters.apply_frosted_glass(img, blur_radius=80, noise_amount=0.01)
    else:
        img = WallpaperFilters.apply_dither(img, amount=noise)

    img = WallpaperFilters.apply_vignette(img, strength=vignette)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img.save(output_path, "PNG")
    return output_path


def _run_compare(axes, base_args, nothing_style, base_seed, panel_scale=4):
    """Render a labeled contact sheet varying one or two parameters.

    Panels are rendered at 1/panel_scale of device resolution for speed.
    axes: list of (param_attr, [str_val, ...]) tuples.
    """
    import copy, itertools, os, tempfile
    from PIL import Image, ImageDraw, ImageFont

    if not axes:
        return

    def _coerce(v):
        try: return int(v)
        except ValueError:
            try: return float(v)
            except ValueError: return v

    axis_names  = [a[0] for a in axes]
    axis_values = [a[1] for a in axes]
    combinations = list(itertools.product(*axis_values))

    # Determine reduced panel resolution (render small, not scale after)
    base_res = getattr(base_args, 'res', None) or SCREEN_PRESETS['mbp14']
    panel_res = (max(320, base_res[0] // panel_scale), max(200, base_res[1] // panel_scale))

    # Generate one panel per combination at reduced resolution
    tmp_paths = {}
    for combo in combinations:
        args_copy = copy.copy(base_args)
        args_copy.res = panel_res
        seed = base_seed
        for param, val in zip(axis_names, combo):
            coerced = _coerce(val)
            setattr(args_copy, param, coerced)
            if param == "seed":
                seed = int(coerced)
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        args_copy.output = tmp
        generate(args_copy, nothing_style, seed)
        tmp_paths[combo] = tmp

    LABEL_H = 44
    GAP     = 3
    BG      = (12, 12, 12)
    FG      = (190, 190, 190)

    pw, ph = panel_res

    nrows = len(axis_values[0]) if len(axes) == 2 else 1
    ncols = len(axis_values[1]) if len(axes) == 2 else len(axis_values[0])

    grid_w = ncols * pw + (ncols - 1) * GAP
    grid_h = nrows * (ph + LABEL_H) + (nrows - 1) * GAP
    grid = Image.new("RGB", (grid_w, grid_h), BG)
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()

    for idx, combo in enumerate(combinations):
        if len(axes) == 2:
            r = axis_values[0].index(combo[0])
            c = axis_values[1].index(combo[1])
        else:
            r, c = 0, idx

        x = c * (pw + GAP)
        y = r * (ph + LABEL_H + GAP)

        panel = Image.open(tmp_paths[combo])
        grid.paste(panel, (x, y))
        panel.close()
        os.unlink(tmp_paths[combo])

        label = "  ".join(f"{n}={v}" for n, v in zip(axis_names, combo))
        draw.text((x + 6, y + ph + 10), label, fill=FG, font=font)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"wp_compare_{timestamp}.png"
    grid.save(out, "PNG")
    print(f"  -> Compare grid: {out}  ({ncols}×{nrows} panels @ {pw}×{ph})")
    return out


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
    core_group.add_argument("--res", nargs="+", default=None, metavar="ARG",
        help=f"Resolution: device name ({', '.join(SCREEN_PRESETS)}) or W H in pixels. Default: mbp14.")
    core_group.add_argument("--count", type=int, default=1, help="Number of wallpapers to generate.")
    core_group.add_argument("--output", default=None, help="Output path (single image). Auto-named if omitted.")
    core_group.add_argument("--output-dir", default=None, help="Directory for auto-named output files.")

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
    engine_group.add_argument("--complexity", default=None, metavar="INT[,INT...]", help=adv_help("Gradient complexity. Comma-separate values to compare."))
    engine_group.add_argument("--engine", default=None, metavar="|".join(ENGINES), help=adv_help(f"Gradient engine ({', '.join(ENGINES)}). Comma-separate to compare."))
    engine_group.add_argument("--harmonize", choices=["theory", "nothing", "none"], default=None, help=adv_help("Color harmonizer override."))

    # --- Filters Overrides ---
    filter_title = "Advanced Filter Overrides" if show_all else "Advanced Filter Overrides (hidden, use --help-all)"
    filter_group = parser.add_argument_group(filter_title)
    filter_group.add_argument("--dots", default=None, metavar="0-1[,0-1...]", help=adv_help("Dot matrix intensity. Comma-separate to compare."))
    filter_group.add_argument("--dots-mode", default=None, metavar="size|alpha[,...]", help=adv_help("Vary dot 'size' or 'alpha'. Comma-separate to compare."))
    filter_group.add_argument("--dots-spacing", default=None, metavar="INT[,INT...]", help=adv_help("Pixel spacing between dots. Comma-separate to compare."))
    filter_group.add_argument("--noise", default=None, metavar="0-1[,0-1...]", help=adv_help("Grain/noise amount. Comma-separate to compare."))
    filter_group.add_argument("--vignette", default=None, metavar="0-1[,0-1...]", help=adv_help("Subtle radial edge darkening. Comma-separate to compare."))

    # --- Misc & Legacy ---
    misc_group = parser.add_argument_group("Miscellaneous" if show_all else "Miscellaneous (hidden, use --help-all)")
    misc_group.add_argument("--seed", default=None, metavar="INT[,INT...]", help=adv_help("Base seed. Comma-separate to compare."))
    
    parser.add_argument("--style", choices=["generic", "nothing"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--nothing", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-adjust", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-nothing", action="store_true", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # --- Resolve multi-value params (comma-separated → compare mode) ---
    def _validate_multi(attr, spec):
        raw = getattr(args, attr, None)
        if raw is None:
            return []
        parts = [v.strip() for v in str(raw).split(",")]
        out = []
        for p in parts:
            try:
                v = spec["coerce"](p)
            except (ValueError, TypeError):
                parser.error(f"--{attr.replace('_', '-')}: invalid value '{p}'")
            if "valid" in spec and v not in spec["valid"]:
                parser.error(f"--{attr.replace('_', '-')}: '{p}' not in {sorted(spec['valid'])}")
            if "min" in spec and v < spec["min"]:
                parser.error(f"--{attr.replace('_', '-')}: {v} below minimum {spec['min']}")
            if "max" in spec and v > spec["max"]:
                parser.error(f"--{attr.replace('_', '-')}: {v} above maximum {spec['max']}")
            out.append(v)
        return out

    compare_axes = []
    for attr, spec in _COMPARABLE.items():
        values = _validate_multi(attr, spec)
        if not values:
            continue
        if len(values) > 1:
            compare_axes.append((attr, [str(v) for v in values]))
        else:
            setattr(args, attr, values[0])

    # --- Resolve --res (device name or W H) ---
    if args.res is not None:
        if len(args.res) == 1:
            name = args.res[0]
            if name not in SCREEN_PRESETS:
                parser.error(f"--res: unknown device '{name}'. Valid: {', '.join(SCREEN_PRESETS)}")
            args.res = SCREEN_PRESETS[name]
        elif len(args.res) == 2:
            try:
                args.res = (int(args.res[0]), int(args.res[1]))
            except ValueError:
                parser.error("--res: expected a device name or two integers W H")
            if args.res[0] <= 0 or args.res[1] <= 0:
                parser.error("--res: width and height must be positive")
        else:
            parser.error("--res: expected a device name (e.g. mbp14) or W H")

    # --- Static validations ---
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
    seed_raw = getattr(args, 'seed', None)
    # After _validate_multi, single seeds are already coerced to int; multi-value strings are not
    base_seed = seed_raw if isinstance(seed_raw, int) else None
    if base_seed is None:
        import time
        base_seed = int(time.time())
        print(f"Base seed: {base_seed}  (pass --seed {base_seed} to reproduce)")

    # --- Compare mode (any multi-value param triggers it) ---
    if compare_axes:
        _run_compare(compare_axes, args, nothing_style, base_seed)
        return

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
