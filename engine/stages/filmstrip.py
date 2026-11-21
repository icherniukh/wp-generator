"""Draws a Trace as a contact sheet: one frame per step with what it read and made."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine.colors import oklab_to_rgb
from engine.stages.runner import StageRecord, Trace
from engine.stages.types import Coords, OklabImage, Palette, ScalarField, SrgbImage, WeightField

THUMB_W = 360
TEXT_H = 140
PAD = 12
COLUMNS = 4
BG = (24, 24, 28)
FG = (230, 230, 235)
MUTED = (150, 150, 160)
ACCENT = (255, 200, 90)
WARN = (255, 120, 110)
CHECKER_CELLS = 10


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, (list, tuple)) and len(value) > 4:
        return f"[{len(value)} items]"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_fmt(v) for v in value) + ")"
    return str(value)


def _to_rgb_array(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.clip(oklab_to_rgb(values), 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def _fit(image: Image.Image, aspect: float) -> Image.Image:
    h = max(1, int(round(THUMB_W / aspect)))
    return image.convert("RGB").resize((THUMB_W, h), Image.BILINEAR)


def _palette_view(palette: Palette, aspect: float) -> Image.Image:
    h = max(1, int(round(THUMB_W / aspect)))
    img = Image.new("RGB", (THUMB_W, h), BG)
    draw = ImageDraw.Draw(img)
    font = _font(13)
    band = THUMB_W / max(1, palette.size)
    for i, color in enumerate(palette.rgb):
        x0, x1 = int(i * band), int((i + 1) * band)
        draw.rectangle([x0, 0, x1, h], fill=color)
        lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        label = "#%02x%02x%02x" % color
        draw.text((x0 + 6, h - 20), f"{i}: {label}", fill=(0, 0, 0) if lum > 140 else (255, 255, 255), font=font)
    return img


def _coords_view(coords: Coords, aspect: float) -> Image.Image:
    """Checkerboard sampled at the current positions, in units of the unwarped grid."""
    bx, by = coords.base_x, coords.base_y
    x0, x1 = float(bx.min()), float(bx.max())
    y0, y1 = float(by.min()), float(by.max())
    cx = (coords.x - x0) / max(x1 - x0, 1e-9) * CHECKER_CELLS
    cy = (coords.y - y0) / max(y1 - y0, 1e-9) * CHECKER_CELLS
    checker = (np.floor(cx) + np.floor(cy)) % 2
    gray = (70 + 120 * checker).astype(np.uint8)
    return _fit(Image.fromarray(gray, "L"), aspect)


def _scalar_view(field: ScalarField, aspect: float) -> Image.Image:
    return _fit(Image.fromarray(np.clip(field.t * 255.0, 0, 255).astype(np.uint8), "L"), aspect)


def _weights_view(field: WeightField, palette: Optional[Palette], aspect: float) -> Image.Image:
    """One tile per color: its weight in grayscale (white = 1), with the color as a swatch."""
    n = field.weights.shape[0]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h = max(1, int(round(THUMB_W / aspect)))
    tile_w, tile_h = THUMB_W // cols, h // rows
    sheet = Image.new("RGB", (THUMB_W, h), BG)
    draw = ImageDraw.Draw(sheet)
    font = _font(11)
    for i in range(n):
        gray = (np.clip(field.weights[i], 0.0, 1.0) * 255.0).astype(np.uint8)
        tile_img = Image.fromarray(gray, "L").resize((tile_w - 2, tile_h - 2), Image.BILINEAR)
        x, y = (i % cols) * tile_w, (i // cols) * tile_h
        sheet.paste(tile_img, (x + 1, y + 1))
        if palette is not None and i < palette.size:
            draw.rectangle([x + 4, y + 4, x + 18, y + 18], fill=palette.rgb[i], outline=FG)
        label = f"w{i}" + (" background" if field.background == i else "")
        draw.rectangle([x + 20, y + 4, x + 26 + draw.textlength(label, font=font), y + 18], fill=BG)
        draw.text((x + 23, y + 4), label, fill=ACCENT, font=font)
    return sheet


def _describe(value: Any) -> str:
    if isinstance(value, Palette):
        return f"Palette  {value.size} colors"
    if isinstance(value, Coords):
        warped = "moved" if value.x is not value.base_x else "as sampled"
        text = f"Coords  {value.shape[1]}x{value.shape[0]} grid, {warped}"
        if value.camera is not None:
            cam = value.camera
            text += f", zoom {cam.zoom:.2f} angle {cam.angle:.2f} center ({cam.center[0]:.2f}, {cam.center[1]:.2f})"
        return text
    if isinstance(value, ScalarField):
        return f"ScalarField  {value.t.shape[1]}x{value.t.shape[0]}  t in [{value.t.min():.2f}, {value.t.max():.2f}]"
    if isinstance(value, WeightField):
        n, h, w = value.weights.shape
        return f"WeightField  {n} colors, {w}x{h}"
    if isinstance(value, OklabImage):
        h, w, _ = value.values.shape
        return f"OklabImage  {w}x{h} float"
    if isinstance(value, SrgbImage):
        h, w, _ = value.values.shape
        return f"SrgbImage  {w}x{h} float"
    if isinstance(value, Image.Image):
        return f"Image  {value.size[0]}x{value.size[1]} {value.mode} 8-bit"
    return type(value).__name__


def _view(value: Any, palette: Optional[Palette], aspect: float) -> Image.Image:
    if isinstance(value, Palette):
        return _palette_view(value, aspect)
    if isinstance(value, Coords):
        return _coords_view(value, aspect)
    if isinstance(value, ScalarField):
        return _scalar_view(value, aspect)
    if isinstance(value, WeightField):
        return _weights_view(value, palette, aspect)
    if isinstance(value, OklabImage):
        return _fit(Image.fromarray(_to_rgb_array(value.values)), aspect)
    if isinstance(value, SrgbImage):
        return _fit(Image.fromarray(np.clip(value.values, 0, 255).astype(np.uint8)), aspect)
    if isinstance(value, Image.Image):
        return _fit(value, aspect)
    return Image.new("RGB", (THUMB_W, int(THUMB_W / aspect)), BG)


def _visible_records(trace: Trace) -> List[StageRecord]:
    """Drops a staged palette step that only repeats the pipeline's resolved palette."""
    records: List[StageRecord] = []
    for record in trace.records:
        if (
            records
            and isinstance(record.output, Palette)
            and isinstance(records[-1].output, Palette)
            and record.output.rgb == records[-1].output.rgb
        ):
            continue
        records.append(record)
    return records


def _wrap(text: str, font: ImageFont.ImageFont, width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    words, lines, line = text.split(" "), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) <= width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def render_filmstrip(trace: Trace, title: str = "", aspect: Optional[float] = None) -> Image.Image:
    """Lays the trace out as a grid of frames, left to right, top to bottom."""
    records = _visible_records(trace)
    if aspect is None:
        final = next((r.output for r in reversed(records) if isinstance(r.output, Image.Image)), None)
        aspect = final.size[0] / final.size[1] if final is not None else 16 / 10
    thumb_h = int(round(THUMB_W / aspect))
    cols = min(COLUMNS, max(1, len(records)))
    rows = math.ceil(len(records) / cols)
    header_h = 56
    unread = trace.unread_values()
    footer_h = 60 if unread else 0
    width = PAD + cols * (THUMB_W + PAD)
    height = header_h + rows * (thumb_h + TEXT_H + PAD) + footer_h + PAD
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    big, mid, small = _font(20), _font(14), _font(12)

    draw.text((PAD, 14), title or "render trace", fill=FG, font=big)

    palette: Optional[Palette] = None
    for index, record in enumerate(records):
        if isinstance(record.output, Palette):
            palette = record.output
        col, row = index % cols, index // cols
        x = PAD + col * (THUMB_W + PAD)
        y = header_h + row * (thumb_h + TEXT_H + PAD)
        sheet.paste(_view(record.output, palette, aspect), (x, y))
        ty = y + thumb_h + 6
        draw.text((x, ty), f"{index + 1}. {record.name}", fill=ACCENT, font=mid)
        draw.text((x + THUMB_W - draw.textlength(record.phase, font=small), ty + 2), record.phase, fill=MUTED, font=small)
        ty += 20
        flow = f"{', '.join(record.inputs) or '-'} -> {record.output_name}"
        draw.text((x, ty), flow, fill=MUTED, font=small)
        ty += 16
        for line in _wrap(_describe(record.output), small, THUMB_W, draw)[:2]:
            draw.text((x, ty), line, fill=FG, font=small)
            ty += 15
        ty += 3
        if record.arguments:
            args = "  ".join(f"{k}={_fmt(v)}" for k, v in record.arguments.items() if k != "colors")
            for line in _wrap(args, small, THUMB_W, draw)[:3]:
                draw.text((x, ty), line, fill=FG, font=small)
                ty += 15
        if record.params_read and set(record.params_read) != set(record.arguments):
            reads = "reads: " + ", ".join(record.params_read)
            for line in _wrap(reads, small, THUMB_W, draw)[:2]:
                draw.text((x, ty), line, fill=MUTED, font=small)
                ty += 15

    if unread:
        fy = height - footer_h - PAD + 8
        valued = {k: v for k, v in unread.items() if v is not None}
        unset = [k for k, v in unread.items() if v is None]
        draw.text((PAD, fy), "given a value but never read by the engine:", fill=WARN, font=mid)
        text = "  ".join(f"{k}={_fmt(v)}" for k, v in valued.items()) or "-"
        if unset:
            text += "      (also unread, unset: " + ", ".join(unset) + ")"
        for i, line in enumerate(_wrap(text, small, width - 2 * PAD, draw)[:2]):
            draw.text((PAD, fy + 22 + i * 15), line, fill=FG, font=small)
    return sheet
