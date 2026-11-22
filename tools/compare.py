"""Labeled contact sheet that compares render variants side by side.

    uv run python tools/compare.py \\
      --variant 'preset=aura' --variant 'preset=aura,field_grain=0.15' \\
      --seeds 1,2,3 --res 640x360 --out sheet.png

Variant syntax
    A variant is a comma-separated list of key=value pairs. Keys are
    RenderConfiguration fields (width, height and seed are set by --res and
    --seeds). A dotted key builds nested stage_params:
    stage_params.shading.tint=0.6. Values are parsed as JSON when possible
    (0.15, true, null, [1,2], {"a":1}), else kept as strings. A value with a
    top-level "+" and no JSON meaning is a list: warps=noise+liquify,
    colors=#112233+#445566.

Layout
    Rows are variants and columns are seeds (--transpose swaps them). Cells are
    shown at render size, 1:1. --crop WxH adds a center crop under each image
    (the crop is a cut of the same pixels, so it is 100% scale). --diff adds a
    tile right of each image: the absolute difference against the first variant
    for the same seed, multiplied by --diff-gain.

Shared palette rule
    Every cell of a seed column uses one seed. A variant that sets neither
    `preset` nor `colors` gets the same palette as the others in its column:
    the final (harmonized) colors of the first variant's render for that seed
    are passed to it as explicit `colors`. A variant that sets `preset` or
    `colors` keeps its own palette. The first variant always renders as written.
    Caveat: with a preset, the palette is drawn after the preset's engine values
    are sampled, so setting a value explicitly (noise=0.1) can skip a draw and
    change the palette. To compare options on identical colors, set the same
    explicit values in each variant (noise=0.1 vs noise=0,field_grain=0.15).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from engine.colors import hex_to_rgb  # noqa: E402
from engine.pipeline import PipelineConfigurationError, PresetNotFoundError, RenderConfiguration, WallpaperPipeline  # noqa: E402

# Set by --res / --seeds, or not meant to be set per variant.
RESERVED_KEYS = {
    "width": "set it with --res WxH",
    "height": "set it with --res WxH",
    "seed": "set seeds with --seeds",
    "preset_data": "not supported in a variant; use preset",
    "capture_base_image": "not a rendering option",
}
VALID_KEYS = [f.name for f in dataclasses.fields(RenderConfiguration) if f.name not in RESERVED_KEYS]

BACKGROUND = (20, 22, 28)
TEXT = (232, 237, 245)
MUTED = (140, 148, 163)
PLATE = (19, 23, 33)
PAD = 12
GAP = 8
FONT_SIZE = 13
ROW_LABEL_WIDTH = 220  # when rows are variants
SEED_LABEL_WIDTH = 70  # when rows are seeds


class VariantError(ValueError):
    """A --variant string, --seeds, --res or --crop value that cannot be used."""


# --- parsing -----------------------------------------------------------------


def split_top_level(text: str, sep: str = ",") -> List[str]:
    """Splits on sep outside of [], {} and double quotes, so JSON values survive."""
    parts: List[str] = []
    depth = 0
    quoted = False
    current: List[str] = []
    for ch in text:
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch in "[{":
            depth += 1
        elif not quoted and ch in "]}":
            depth -= 1
        if ch == sep and depth == 0 and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def parse_scalar(text: str) -> Any:
    """JSON when it parses (numbers, true/false/null, lists, objects), else the string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_value(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    if "+" in text:
        return [parse_scalar(item.strip()) for item in split_top_level(text, "+")]
    return text


def _coerce(key: str, value: Any) -> Any:
    if key == "colors" and value is not None:
        items = value if isinstance(value, list) else [value]
        try:
            return [hex_to_rgb(item) if isinstance(item, str) else tuple(item) for item in items]
        except (ValueError, TypeError) as exc:
            raise VariantError(f"colors must be hex strings joined with '+', e.g. colors=#112233+#445566 ({exc})") from exc
    if key == "fractal_center" and isinstance(value, list):
        return tuple(value)
    if key in ("warps",) and isinstance(value, str):
        return [value]
    return value


def parse_variant(text: str) -> Dict[str, Any]:
    """Turns 'preset=aura,stage_params.shading.tint=0.6' into RenderConfiguration kwargs."""
    overrides: Dict[str, Any] = {}
    if not text.strip():
        return overrides
    for pair in split_top_level(text, ","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise VariantError(f"Variant entry {pair!r} in {text!r} is not key=value.")
        raw_key, raw_value = pair.split("=", 1)
        key = raw_key.strip()
        value = parse_value(raw_value)
        head, *path = key.split(".")
        if head in RESERVED_KEYS:
            raise VariantError(f"{head!r} cannot be set in a variant: {RESERVED_KEYS[head]}.")
        if head not in VALID_KEYS:
            raise VariantError(f"Unknown key {head!r} in variant {text!r}. Valid keys: {', '.join(VALID_KEYS)}.")
        if path:
            if head != "stage_params":
                raise VariantError(f"Dotted keys are only for stage_params, e.g. stage_params.shading.tint=0.6, not {key!r}.")
            if len(path) != 2 or not all(path):
                raise VariantError(f"Use stage_params.<stage>.<arg>=value, not {key!r}.")
            stage, arg = path
            nested = overrides.setdefault("stage_params", {})
            if not isinstance(nested, dict):
                raise VariantError(f"stage_params is set twice in variant {text!r}.")
            nested.setdefault(stage, {})[arg] = value
        else:
            overrides[head] = _coerce(head, value)
    return overrides


def parse_size(text: str, what: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if not match or int(match.group(1)) <= 0 or int(match.group(2)) <= 0:
        raise VariantError(f"{what} must look like 640x360, not {text!r}.")
    return int(match.group(1)), int(match.group(2))


def parse_seeds(text: str) -> List[int]:
    seeds: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        span = re.fullmatch(r"(\d+)-(\d+)", part)
        try:
            if span:
                low, high = int(span.group(1)), int(span.group(2))
                seeds.extend(range(low, high + 1))
            else:
                seeds.append(int(part))
        except ValueError as exc:
            raise VariantError(f"--seeds must be integers like 1,2,3 or 1-3, not {part!r}.") from exc
    if not seeds:
        raise VariantError("--seeds needs at least one seed.")
    return seeds


# --- rendering ---------------------------------------------------------------


@dataclass
class Cell:
    variant: int
    seed: int
    image: Image.Image
    colors: List[str]
    parameters: Dict[str, Any]
    palette_shared: bool = False
    overrides: Dict[str, Any] = field(default_factory=dict)


def fixes_palette(overrides: Dict[str, Any]) -> bool:
    return bool(overrides.get("preset")) or bool(overrides.get("colors"))


def _render_cell(job: Tuple[int, int, Dict[str, Any], int, int, Optional[List[Tuple[int, int, int]]]]) -> Cell:
    variant, seed, overrides, width, height, shared_colors = job
    kwargs = dict(overrides)
    if shared_colors is not None:
        kwargs["colors"] = [tuple(c) for c in shared_colors]
    config = RenderConfiguration(width=width, height=height, seed=seed, **kwargs)
    result = WallpaperPipeline().render(config)
    params = dict(result.metadata.parameters) if result.metadata else {}
    return Cell(
        variant=variant,
        seed=seed,
        image=result.image.convert("RGB"),
        colors=list(result.color_hexes),
        parameters=params,
        palette_shared=shared_colors is not None,
        overrides=overrides,
    )


def _run_jobs(jobs: Sequence[tuple], workers: int) -> List[Cell]:
    if workers > 1 and len(jobs) > 1:
        # Each render seeds random and numpy from its own seed, so processes cannot interfere.
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            return list(pool.map(_render_cell, jobs))
    return [_render_cell(job) for job in jobs]


def render_cells(
    variants: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    width: int,
    height: int,
    workers: int = 1,
    share_palette: bool = True,
) -> List[List[Cell]]:
    """Returns cells[variant][seed_index]. The first variant renders first, since its palette is shared."""
    first = _run_jobs([(0, seed, variants[0], width, height, None) for seed in seeds], workers)
    rest_jobs = []
    for vi in range(1, len(variants)):
        for si, seed in enumerate(seeds):
            shared = None
            if share_palette and not fixes_palette(variants[vi]):
                shared = [hex_to_rgb(h) for h in first[si].colors]
            rest_jobs.append((vi, seed, variants[vi], width, height, shared))
    rest = _run_jobs(rest_jobs, workers)
    grid = [first]
    for vi in range(1, len(variants)):
        grid.append(rest[(vi - 1) * len(seeds):vi * len(seeds)])
    return grid


# --- sheet drawing -----------------------------------------------------------


def load_font(size: int = FONT_SIZE) -> ImageFont.ImageFont:
    """Pillow's built-in font (sized when FreeType is available), so no system fonts are needed."""
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):
        return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int = 6) -> List[str]:
    """Greedy wrap at commas and spaces, hard-breaking long tokens, ending with '...' when cut."""
    tokens = [t for t in re.split(r"(?<=[, ])", text) if t]
    lines: List[str] = []
    current = ""
    for token in tokens:
        while font.getlength(token) > max_width and len(token) > 1:
            fit = len(token) - 1
            while fit > 1 and font.getlength(token[:fit]) > max_width:
                fit -= 1
            if current:
                lines.append(current)
                current = ""
            lines.append(token[:fit])
            token = token[fit:]
        if current and font.getlength(current + token) > max_width:
            lines.append(current)
            current = token
        else:
            current += token
    if current:
        lines.append(current)
    lines = [line.rstrip() for line in lines] or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and font.getlength(last + "...") > max_width:
            last = last[:-1]
        lines[-1] = last + "..."
    return lines


def diff_image(a: Image.Image, b: Image.Image, gain: float = 8.0) -> Image.Image:
    """Absolute per-channel difference, multiplied by gain and clipped."""
    delta = np.abs(np.asarray(a.convert("RGB"), dtype=np.int16) - np.asarray(b.convert("RGB"), dtype=np.int16))
    return Image.fromarray(np.clip(delta * gain, 0, 255).astype(np.uint8), "RGB")


def center_crop(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    cw, ch = min(size[0], image.width), min(size[1], image.height)
    left, top = (image.width - cw) // 2, (image.height - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


@dataclass
class SheetLayout:
    size: Tuple[int, int]
    boxes: Dict[Tuple[int, int], Dict[str, Tuple[int, int, int, int]]]  # (variant, seed_index) -> tile boxes


def make_sheet(
    grid: Sequence[Sequence[Cell]],
    variant_labels: Sequence[str],
    seeds: Sequence[int],
    width: int,
    height: int,
    crop: Optional[Tuple[int, int]] = None,
    diff: bool = False,
    diff_gain: float = 8.0,
    transpose: bool = False,
    date: Optional[str] = None,
) -> Tuple[Image.Image, SheetLayout]:
    font = load_font()
    line_h = int(font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + 4
    n_var, n_seed = len(variant_labels), len(seeds)
    crop_w, crop_h = (min(crop[0], width), min(crop[1], height)) if crop else (0, 0)

    cell_w = width * (2 if diff else 1) + (GAP if diff else 0)
    cell_h = height + ((GAP + crop_h) if crop else 0)

    row_labels = [f"seed {s}" for s in seeds] if transpose else list(variant_labels)
    col_labels = list(variant_labels) if transpose else [f"seed {s}" for s in seeds]
    n_rows, n_cols = len(row_labels), len(col_labels)
    label_w = SEED_LABEL_WIDTH if transpose else ROW_LABEL_WIDTH

    col_lines = [wrap_text(label, font, cell_w) for label in col_labels]
    row_lines = [wrap_text(label or "(defaults)", font, label_w - GAP, max_lines=max(6, cell_h // line_h)) for label in row_labels]
    col_lines = [lines if lines != [""] else ["(defaults)"] for lines in col_lines]
    header_h = max(len(lines) for lines in col_lines) * line_h + GAP

    footer = f"{width}x{height}   {date or datetime.now().strftime('%Y-%m-%d')}"
    if crop:
        footer += f"   crop {crop_w}x{crop_h} at 100%"
    if diff:
        footer += f"   diff vs first variant x{diff_gain:g}"
    footer_h = line_h + GAP

    sheet_w = PAD + label_w + n_cols * cell_w + (n_cols - 1) * GAP + PAD
    sheet_h = PAD + header_h + n_rows * cell_h + (n_rows - 1) * GAP + footer_h + PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    for ci, lines in enumerate(col_lines):
        x = PAD + label_w + ci * (cell_w + GAP)
        for li, line in enumerate(lines):
            draw.text((x, PAD + li * line_h), line, font=font, fill=TEXT)

    boxes: Dict[Tuple[int, int], Dict[str, Tuple[int, int, int, int]]] = {}
    for ri in range(n_rows):
        y = PAD + header_h + ri * (cell_h + GAP)
        for li, line in enumerate(row_lines[ri]):
            draw.text((PAD, y + li * line_h), line, font=font, fill=TEXT)
        for ci in range(n_cols):
            vi, si = (ci, ri) if transpose else (ri, ci)
            x = PAD + label_w + ci * (cell_w + GAP)
            cell = grid[vi][si]
            tiles: Dict[str, Tuple[int, int, int, int]] = {}
            sheet.paste(cell.image, (x, y))
            tiles["image"] = (x, y, x + width, y + height)
            diff_img = diff_image(grid[0][si].image, cell.image, diff_gain) if diff else None
            if diff_img is not None:
                dx = x + width + GAP
                sheet.paste(diff_img, (dx, y))
                tiles["diff"] = (dx, y, dx + width, y + height)
            if crop:
                cy = y + height + GAP
                sheet.paste(center_crop(cell.image, (crop_w, crop_h)), (x, cy))
                tiles["crop"] = (x, cy, x + crop_w, cy + crop_h)
                if diff_img is not None:
                    sheet.paste(center_crop(diff_img, (crop_w, crop_h)), (x + width + GAP, cy))
                    tiles["diff_crop"] = (x + width + GAP, cy, x + width + GAP + crop_w, cy + crop_h)
            boxes[(vi, si)] = tiles

    draw.text((PAD, sheet_h - PAD - line_h), footer, font=font, fill=MUTED)
    return sheet, SheetLayout(size=(sheet_w, sheet_h), boxes=boxes)


# --- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render variants for several seeds and lay them out as a labeled contact sheet.",
        epilog="Variant keys: " + ", ".join(VALID_KEYS),
    )
    parser.add_argument("--variant", action="append", default=[], metavar="K=V,K=V",
                        help="comma-separated RenderConfiguration fields; repeat for more rows. "
                             "stage_params.<stage>.<arg>=v nests, warps=a+b makes a list, values are JSON when possible")
    parser.add_argument("--seeds", default="1,2,3", help="comma list or range of seeds, one column each (default 1,2,3)")
    parser.add_argument("--res", default="640x360", help="render size per cell (default 640x360)")
    parser.add_argument("--out", default="compare.png", help="output PNG path (default compare.png)")
    parser.add_argument("--transpose", action="store_true", help="seeds as rows, variants as columns")
    parser.add_argument("--crop", metavar="WxH", help="add a 100%% scale center crop under each image")
    parser.add_argument("--diff", action="store_true", help="add the amplified absolute difference against the first variant")
    parser.add_argument("--diff-gain", type=float, default=8.0, help="difference amplification (default 8)")
    parser.add_argument("--workers", type=int, default=1, help="parallel render processes (default 1)")
    parser.add_argument("--no-share-palette", action="store_true", help="do not copy the first variant's palette to the others")
    parser.add_argument("--json", action="store_true", help="print the resolved parameters of every cell as JSON")
    return parser


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    if not args.variant:
        raise VariantError("Give at least one --variant (use --variant '' for the defaults).")
    variants = [parse_variant(v) for v in args.variant]
    seeds = parse_seeds(args.seeds)
    width, height = parse_size(args.res, "--res")
    crop = parse_size(args.crop, "--crop") if args.crop else None

    grid = render_cells(variants, seeds, width, height, workers=max(1, args.workers), share_palette=not args.no_share_palette)
    sheet, layout = make_sheet(
        grid, args.variant, seeds, width, height,
        crop=crop, diff=args.diff, diff_gain=args.diff_gain, transpose=args.transpose,
    )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return {
        "output": str(out.resolve()),
        "size": list(layout.size),
        "cells": [
            {
                "variant": args.variant[cell.variant],
                "seed": cell.seed,
                "palette_shared": cell.palette_shared,
                "colors": cell.colors,
                "resolved": cell.parameters,
            }
            for row in grid for cell in row
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    json_mode = "--json" in (argv if argv is not None else sys.argv[1:])
    try:
        report = run(argv)
    except (VariantError, PipelineConfigurationError) as exc:
        print(f"compare: {exc}", file=sys.stderr)
        return 2
    except PresetNotFoundError as exc:
        print(f"compare: preset not found: {exc}", file=sys.stderr)
        return 2
    if json_mode:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(report["output"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
