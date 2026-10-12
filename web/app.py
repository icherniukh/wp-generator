import asyncio
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from engine.pipeline import (
    DEVICE_RESOLUTIONS,
    PipelineConfigurationError,
    PresetNotFoundError,
    RenderConfiguration,
    WallpaperPipeline,
    parse_hex_color,
)

app = FastAPI(
    title="Wallpaper Generator API",
    description="Procedural statistical wallpaper rendering engine API.",
    version="0.1.0",
)

pipeline = WallpaperPipeline()

# Max allowed dimensions for web rendering to prevent resource exhaustion
MAX_DIMENSION = 5120
MAX_PIXELS = 5120 * 2880


class PresetSummary(BaseModel):
    id: str
    name: str
    description: str
    color_space: str
    sample_palette: List[str]


class DeviceSummary(BaseModel):
    key: str
    width: int
    height: int


class RenderRequest(BaseModel):
    preset: Optional[str] = Field(default=None, description="Preset name, e.g. 'nocturne'")
    device: Optional[str] = Field(default=None, description="Standard device key, e.g. 'mbp14'")
    width: Optional[int] = Field(default=None, ge=16, le=MAX_DIMENSION)
    height: Optional[int] = Field(default=None, ge=16, le=MAX_DIMENSION)
    colors: Optional[List[str]] = Field(default=None, description="Custom hex colors")
    complexity: Optional[int] = Field(default=None, ge=1, le=10)
    engine: Optional[str] = Field(default=None, description="simplex, fluid-wave, idw, etc.")
    fluid_mode: Optional[str] = Field(default=None)
    harmonize: Optional[str] = Field(default=None)
    noise: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    vignette: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dots: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dots_spacing: Optional[int] = Field(default=None, ge=4, le=100)
    dots_mode: Optional[str] = Field(default=None)
    glass: Optional[bool] = Field(default=None)
    chromatic_aberration: Optional[float] = Field(default=None, ge=0.0, le=0.1)
    warp_strength: Optional[float] = Field(default=None, ge=0.0, le=3.0)
    seed: Optional[int] = Field(default=None)


def _resolve_dimensions(
    device: Optional[str], width: Optional[int], height: Optional[int]
) -> Tuple[int, int]:
    if width is not None and height is not None:
        if width * height > MAX_PIXELS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Requested resolution {width}x{height} exceeds maximum allowed pixel count.",
            )
        return width, height
    if device is not None:
        if device not in DEVICE_RESOLUTIONS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown device preset '{device}'. Available: {list(DEVICE_RESOLUTIONS.keys())}",
            )
        return DEVICE_RESOLUTIONS[device]
    # Default to MBP 14
    return DEVICE_RESOLUTIONS["mbp14"]


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "wp-generator"}


@app.get("/api/devices", response_model=List[DeviceSummary])
def list_devices() -> List[DeviceSummary]:
    """Returns available display resolution profiles."""
    return [
        DeviceSummary(key=key, width=dims[0], height=dims[1])
        for key, dims in DEVICE_RESOLUTIONS.items()
    ]


@app.get("/api/presets", response_model=List[PresetSummary])
def list_presets() -> List[PresetSummary]:
    """Lists available presets along with metadata and sampled palette colors."""
    catalog = pipeline.list_presets()
    summaries = []
    for item in catalog:
        preset_id = item["id"]
        # Sample preview colors with fixed seed for determinism
        try:
            data = pipeline.load_preset(preset_id)
            colors = pipeline.preset_engine.generate_palette(data, count=3, family_index=42)
            hexes = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
        except Exception:
            hexes = []
        summaries.append(
            PresetSummary(
                id=preset_id,
                name=item["name"],
                description=item["description"],
                color_space=item["color_space"],
                sample_palette=hexes,
            )
        )
    return summaries


@app.get("/api/render")
async def render_get(
    preset: Optional[str] = Query(default="nocturne"),
    device: Optional[str] = Query(default=None),
    width: Optional[int] = Query(default=None, ge=16, le=MAX_DIMENSION),
    height: Optional[int] = Query(default=None, ge=16, le=MAX_DIMENSION),
    seed: Optional[int] = Query(default=None),
    complexity: Optional[int] = Query(default=None, ge=1, le=10),
    engine: Optional[str] = Query(default=None),
    noise: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    vignette: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    dots: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    glass: Optional[bool] = Query(default=None),
    aberration: Optional[float] = Query(default=None, ge=0.0, le=0.1),
    warp_strength: Optional[float] = Query(default=None, ge=0.0, le=3.0),
    ambient_weight: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    dispersion: Optional[float] = Query(default=None, ge=0.05, le=2.0),
) -> Response:
    """Renders wallpaper directly as a streaming PNG image response."""
    w, h = _resolve_dimensions(device, width, height)
    active_seed = seed if seed is not None else random.randint(0, 2_147_483_647)

    config = RenderConfiguration(
        width=w,
        height=h,
        preset=preset,
        complexity=complexity,
        engine=engine,
        noise=noise,
        vignette=vignette,
        dots=dots,
        glass=glass,
        chromatic_aberration=aberration,
        warp_strength=warp_strength,
        ambient_weight=ambient_weight,
        dispersion=dispersion,
        seed=active_seed,
    )

    try:
        png_bytes, result = await run_in_threadpool(pipeline.render_png_bytes, config)
    except PresetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PipelineConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rendering failure: {exc}",
        ) from exc

    headers = {
        "Content-Type": "image/png",
        "X-Render-Seed": str(result.seed),
        "X-Render-Colors": ",".join(result.color_hexes),
        "X-Render-Engine": result.engine,
        "X-Render-Width": str(result.width),
        "X-Render-Height": str(result.height),
    }
    if result.chromatic_aberration > 0.0:
        headers["X-Render-Aberration"] = str(result.chromatic_aberration)

    return Response(content=png_bytes, media_type="image/png", headers=headers)


@app.post("/api/render")
async def render_post(body: RenderRequest) -> Response:
    """Renders wallpaper from detailed JSON request parameters."""
    w, h = _resolve_dimensions(body.device, body.width, body.height)
    active_seed = body.seed if body.seed is not None else random.randint(0, 2_147_483_647)

    parsed_colors = None
    if body.colors:
        try:
            parsed_colors = [parse_hex_color(c) for c in body.colors]
        except PipelineConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

    config = RenderConfiguration(
        width=w,
        height=h,
        preset=body.preset,
        colors=parsed_colors,
        complexity=body.complexity,
        engine=body.engine,
        fluid_mode=body.fluid_mode,
        harmonize=body.harmonize,
        noise=body.noise,
        vignette=body.vignette,
        dots=body.dots,
        dots_spacing=body.dots_spacing,
        dots_mode=body.dots_mode,
        glass=body.glass,
        chromatic_aberration=body.chromatic_aberration,
        warp_strength=body.warp_strength,
        seed=active_seed,
    )

    try:
        png_bytes, result = await run_in_threadpool(pipeline.render_png_bytes, config)
    except PresetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PipelineConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rendering failure: {exc}",
        ) from exc

    headers = {
        "Content-Type": "image/png",
        "X-Render-Seed": str(result.seed),
        "X-Render-Colors": ",".join(result.color_hexes),
        "X-Render-Engine": result.engine,
        "X-Render-Width": str(result.width),
        "X-Render-Height": str(result.height),
    }
    if result.chromatic_aberration > 0.0:
        headers["X-Render-Aberration"] = str(result.chromatic_aberration)

    return Response(content=png_bytes, media_type="image/png", headers=headers)
