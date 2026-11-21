"""Rebuild the static visual guide: uv run python -m tools.render_showcase.

Uses the real primitives, not look-alike illustrations. Each comparison resets
both RNGs; all six finish passes receive the exact same unfiltered image.
Recipes, palettes and seeds are saved alongside the PNGs in manifest.json.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine.colors import ColorTheoryHarmonizer, StyleHarmonizer, hex_to_rgb, rgb_to_hex
from engine.filters import WallpaperFilters
from engine.gradients import GRADIENT_REGISTRY, render_gradient
from engine.gradients.interpolation import (
    colors_to_oklab,
    evaluate_decoupled_palette,
    interpolate_color_curve,
    oklab_image_to_pil,
)
from engine.gradients.spatial import (
    generate_noise_plane,
    normalize_scalar_field,
    random_anchor_points,
    trig_fbm_field,
    warp_coordinates_perlin,
)
from engine.pipeline import RenderConfiguration, WallpaperPipeline

ROOT = Path(__file__).resolve().parent.parent
PALETTE = ["#190E32", "#A54399", "#FF7768", "#FFD49B"]
SEED = 7
FILTERS = {
    "contours": ("apply_contour_lines", {"levels": 24, "line_width": 1.4, "intensity": 0.85, "mode": "light"}),
    "glass": ("apply_frosted_glass", {"blur_radius": 60, "noise_amount": 0.15}),
    "aberration": ("apply_chromatic_aberration", {"strength": 0.06, "mode": "radial"}),
    "grain": ("apply_film_grain", {"amount": 0.10}),
    "vignette": ("apply_vignette", {"strength": 0.55}),
    "dither": ("apply_dither", {"amount": 0.05}),
}


def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)


def render_engine(name, width, height, colors=None, **overrides):
    reset_seed()
    params = dict(
        complexity=5, warp_strength=0.35, transverse_warp=0.08,
        ambient_weight=0.15, dispersion=0.35, fractal_type="fluid",
        fractal_zoom=0.85, fractal_center=(0.65, -0.3), fractal_angle=0.4,
        frequency=1.0, trap_weight=0.35, channel_offset=0.0, seed=SEED,
    )
    params.update(overrides)
    colors = colors if colors is not None else [hex_to_rgb(h) for h in PALETTE]
    return render_gradient(name, width=width, height=height, colors=colors, **params), {
        "call": "render_gradient", "engine": name, "colors": [rgb_to_hex(c) for c in colors],
        "width": width, "height": height, **params,
    }


def palette_strip(colors, width, height):
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(10, min(22, width // 35)))
    for i, color in enumerate(colors):
        left, right = i * width // len(colors), (i + 1) * width // len(colors)
        draw.rectangle((left, 0, right, height), fill=tuple(color))
        # Dark caption plate stays legible on light and dark swatches.
        label = rgb_to_hex(color)
        draw.rectangle((left, height - 34, right, height), fill="#131721")
        draw.text((left + 6, height - 28), label, font=font, fill="#e8edf5")
    return image


def build_assets(output_dir, width=800, height=450):
    output_dir = Path(output_dir)
    manifest = {}

    def save(name, image, recipe):
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the pipeline image instance: its save wrapper embeds render metadata.
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        rgb.save(path, "PNG")
        manifest[name] = {"seed": SEED, **recipe}

    def engine(name, filename, **overrides):
        image, recipe = render_engine(name, width, height, **overrides)
        save(filename, image, recipe)
        return image

    # All comparisons change one active parameter, not an accepted-but-inert flag.
    comparisons = {
        "domain-warp": ("warp_strength", 0.1, 1.5),
        "simplex": ("complexity", 2, 9),
        "spline-wave": ("warp_strength", 0.1, 1.0),
        "fluid-wave": ("complexity", 4, 32),
        "rbf": ("dispersion", 0.15, 0.9),
        "radial-basis": ("ambient_weight", 0.05, 0.6),
        "flow-field": ("warp_strength", 0.1, 1.0),
        "topographic": ("warp_strength", 0.2, 1.5),
        "fractal": ("fractal_zoom", 0.5, 1.8),
        "fluid-fractal": ("frequency", 0.7, 2.0),
        "escape-fractal": ("trap_weight", 0.05, 0.8),
    }
    for name, (key, lo, hi) in comparisons.items():
        fixed = {"fractal_type": "julia"} if name == "escape-fractal" else {}
        for suffix, value in (("lo", lo), ("hi", hi)):
            engine(name, f"demo/eng/{name}_{suffix}.png", **fixed, **{key: value})
    for count in (2, 5):
        colors = [hex_to_rgb(h) for h in ["#190E32", "#683A8D", "#A54399", "#FF7768", "#FFD49B"]]
        engine("idw", f"demo/eng/idw_{count}col.png", colors=colors if count == 5 else [colors[0], colors[-1]])

    # An honest overview: identical parameters as well as identical seed/palette.
    tile_w, tile_h, label_h = 320, 180, 32
    grid = Image.new("RGB", (tile_w * 4, (tile_h + label_h) * 3), "#131721")
    draw = ImageDraw.Draw(grid)
    for i, name in enumerate(GRADIENT_REGISTRY):
        # Record Julia explicitly; the escape wrapper also coerces unsupported types to Julia.
        fixed = {"fractal_type": "julia"} if name == "escape-fractal" else {}
        image, _ = render_engine(name, width, height, **fixed)
        x, y = (i % 4) * tile_w, (i // 4) * (tile_h + label_h)
        grid.paste(image.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 12, y + tile_h + 7), name, font=ImageFont.load_default(size=17), fill="#ecedf4")
    save("engine_primitives.png", grid, {"call": "registry overview", "palette": PALETTE, "escape_fractal_type": "julia"})

    base = engine("topographic", "demo/fx_base.png", warp_strength=0.6)
    for name, (method, kwargs) in FILTERS.items():
        reset_seed()
        after = getattr(WallpaperFilters, method)(base.copy(), **kwargs)
        save(f"demo/fx_{name}.png", after, {"input": "demo/fx_base.png", "call": method, "kwargs": kwargs})
        if name in ("grain", "dither"):
            box = (width // 3, height // 3, width * 2 // 3, height * 2 // 3)
            for label, image in (("before", base), ("after", after)):
                crop = image.crop(box).resize((width, height), Image.Resampling.NEAREST)
                save(f"demo/fx_{name}_{label}_detail.png", crop, {"input": f"demo/fx_{name}.png" if label == "after" else "demo/fx_base.png", "crop": box, "resize": "nearest"})
    for levels in (8, 28):
        save(f"demo/eng/contours_{levels}.png", WallpaperFilters.apply_contour_lines(base, levels=levels, intensity=0.85), {"input": "demo/fx_base.png", "call": "apply_contour_lines", "kwargs": {"levels": levels, "intensity": 0.85}})

    pipeline = WallpaperPipeline()
    for preset in pipeline.list_presets():
        reset_seed()
        colors = pipeline.preset_engine.generate_palette(pipeline.load_preset(preset["id"]), family_index=SEED)
        save(f"demo/pal_{preset['id']}.png", palette_strip(colors, width, 100), {"call": "PresetEngine.generate_palette", "preset": preset["id"], "family_index": SEED})

    raw = [hex_to_rgb(h) for h in ["#080B16", "#69808E", "#FF655B", "#FFEBC4"]]
    save("demo/harm_raw.png", palette_strip(raw, width, 100), {"colors": [rgb_to_hex(c) for c in raw]})
    for mode in ("perceptual", "triadic", "nothing"):
        reset_seed()
        colors = StyleHarmonizer.adjust_for_nothing(raw) if mode == "nothing" else ColorTheoryHarmonizer.harmonize(raw, mode=mode)
        save(f"demo/harm_{mode}.png", palette_strip(colors, width, 100), {"input": "demo/harm_raw.png", "call": "StyleHarmonizer.adjust_for_nothing" if mode == "nothing" else "ColorTheoryHarmonizer.harmonize", "mode": mode})

    lab = colors_to_oklab([hex_to_rgb(h) for h in PALETTE])
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

    def mapped(field):
        return oklab_image_to_pil(interpolate_color_curve(field, lab))

    reset_seed()
    field = generate_noise_plane(height, width, scale=1.5, octaves=2, offset=(13, 29))
    save("demo/spatial_noise.png", mapped(normalize_scalar_field(field)), {"call": "generate_noise_plane", "scale": 1.5, "octaves": 2, "offset": [13, 29], "palette": PALETTE, "display": "normalized then palette-mapped"})
    reset_seed()
    wx, wy = warp_coordinates_perlin(width, height, 5, warp_range=(0.5, 0.5))
    for label, value in (("before", x), ("after", wx)):
        save(f"demo/spatial_warp_{label}.png", mapped(value), {"call": "warp_coordinates_perlin", "complexity": 5, "warp_range": [0.5, 0.5], "display": label + " x-coordinate through palette", "palette": PALETTE})
    reset_seed()
    raw_field = trig_fbm_field(x * 4, y * 4) * 0.25 + 0.4
    for label, value in (("before", raw_field), ("after", normalize_scalar_field(raw_field))):
        save(f"demo/spatial_normalize_{label}.png", mapped(value), {"call": "normalize_scalar_field", "input": "trig_fbm_field(x*4, y*4)*0.25+0.4", "display": label, "palette": PALETTE})
    reset_seed()
    anchors = Image.new("RGB", (width, height), "#131721")
    draw = ImageDraw.Draw(anchors)
    for i, (ax, ay) in enumerate(random_anchor_points(8)):
        px, py = ax * width, ay * height
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=PALETTE[i % len(PALETTE)])
    save("demo/spatial_anchors.png", anchors, {"call": "random_anchor_points", "count": 8, "margin": 0.08})
    save("demo/interp_curve.png", mapped(x), {"call": "interpolate_color_curve", "field": "horizontal 0..1", "palette": PALETTE})
    save("demo/interp_decoupled_before.png", mapped(x), {"call": "evaluate_decoupled_palette", "lightness": "x", "chroma": "x", "palette": PALETTE})
    save("demo/interp_decoupled_after.png", oklab_image_to_pil(evaluate_decoupled_palette(lab, x, y, height, width)), {"call": "evaluate_decoupled_palette", "lightness": "x", "chroma": "y", "palette": PALETTE})

    result = pipeline.render(RenderConfiguration(width=width, height=height, preset="afterglow", seed=SEED, capture_base_image=True))
    save("demo/run_palette.png", palette_strip(result.colors, width, 100), {"preset": "afterglow", "colors": result.color_hexes})
    save("demo/run_base.png", result.base_image, {"preset": "afterglow", "stage": "before finish"})
    save("demo/run_final.png", result.image, {"call": "WallpaperPipeline.render", "preset": "afterglow", "parameters": result.metadata.parameters})

    # The poster uses the actual worked example, including its real palette/finish.
    poster = Image.new("RGB", (1440, 620), "#131721")
    draw = ImageDraw.Draw(poster)
    draw.text((40, 26), "One seed. Three stages. A complete wallpaper.", font=ImageFont.load_default(size=32), fill="#f2eee9")
    draw.text((40, 80), "afterglow / seed 7 / actual pipeline output", font=ImageFont.load_default(size=18), fill="#b5adbf")
    stages = [(palette_strip(result.colors, 420, 236), "1 / PALETTE", "Preset recipe -> resolved colors"), (result.base_image, "2 / FIELD", "Softened IDW anchors -> color blend"), (result.image, "3 / FINISH", "Fine film grain -> soft vignette", "(other finish passes disabled)")]
    for i, (image, title, *lines) in enumerate(stages):
        left = 40 + i * 470
        poster.paste(image.resize((420, 236), Image.Resampling.LANCZOS), (left, 146))
        draw.text((left, 408), title, font=ImageFont.load_default(size=22), fill="#ffba9c")
        for j, line in enumerate(lines):
            draw.text((left, 448 + j * 28), line, font=ImageFont.load_default(size=16), fill="#b5adbf")
    draw.text((40, 560), "Explicit config > preset override > built-in default | PNG embeds the resolved render metadata", font=ImageFont.load_default(size=19), fill="#ececf1")
    save("pipeline.png", poster, {"call": "worked-example poster", "preset": "afterglow", "inputs": ["demo/run_palette.png", "demo/run_base.png", "demo/run_final.png"]})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    args = parser.parse_args()
    if args.width < 64 or args.height < 36:
        parser.error("showcase dimensions must be at least 64 x 36")
    manifest = build_assets(args.output_dir, args.width, args.height)
    print(f"Rendered {len(manifest)} assets + manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
