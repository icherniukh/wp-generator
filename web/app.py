from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from engine.metadata import extract_image_metadata
from engine import __version__

from engine.parameter_ranges import PARAMETER_RANGES
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
    version=__version__,
)

pipeline = WallpaperPipeline()
CALIBRATE_HTML_PATH = Path(__file__).resolve().parent / "static" / "calibrate.html"

# Max allowed dimensions for web rendering to prevent resource exhaustion
MAX_DIMENSION = 5120
MAX_PIXELS = 5120 * 2880


class PresetSummary(BaseModel):
    id: str
    name: str
    version: str = "1.0.0"
    description: str
    color_space: str
    sample_palette: List[str]


class DeviceSummary(BaseModel):
    key: str
    width: int
    height: int


class ParameterRangeUpdateRequest(BaseModel):
    ranges: Dict[str, Tuple[float, float]]


class RenderRequest(BaseModel):
    preset: Optional[str] = Field(default=None, description="Preset name, e.g. 'nocturne'")
    device: Optional[str] = Field(default=None, description="Standard device key, e.g. 'mbp14'")
    width: Optional[int] = Field(default=None, ge=16, le=MAX_DIMENSION)
    height: Optional[int] = Field(default=None, ge=16, le=MAX_DIMENSION)
    colors: Optional[List[str]] = Field(default=None, description="Custom hex colors")
    complexity: Optional[int] = Field(default=None, ge=1, le=10)
    engine: Optional[str] = Field(default=None, description="simplex, fluid-wave, idw, etc.")
    fluid_mode: Optional[str] = Field(default=None)
    harmonize: Optional[Literal["perceptual", "theory", "nothing", "none"]] = Field(default=None)
    noise: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    vignette: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dots: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dots_spacing: Optional[int] = Field(default=None, ge=4, le=100)
    dots_mode: Optional[str] = Field(default=None)
    contours: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    contour_levels: Optional[int] = Field(default=None, ge=1, le=100)
    contour_width: Optional[float] = Field(default=None, ge=0.1, le=10.0)
    contour_mode: Optional[Literal["light", "dark", "accent"]] = Field(default=None)
    glass: Optional[bool] = Field(default=None)
    chromatic_aberration: Optional[float] = Field(default=None, ge=0.0, le=0.1)
    warp_strength: Optional[float] = Field(default=None, ge=0.0, le=3.0)
    transverse_warp: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    ambient_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dispersion: Optional[float] = Field(default=None, ge=0.05, le=2.0)
    phase_offset: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency: Optional[float] = Field(default=None, gt=0.0)
    trap_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    channel_offset: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    fractal_type: Optional[Literal["fluid", "julia", "mandelbrot", "orbit_trap"]] = Field(default=None)
    fractal_zoom: Optional[float] = Field(default=None, gt=0.0)
    fractal_center: Optional[Tuple[float, float]] = Field(default=None)
    fractal_angle: Optional[float] = Field(default=None)
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


@app.get("/", response_class=HTMLResponse)
@app.get("/calibrate", response_class=HTMLResponse)
def calibration_lab_page() -> HTMLResponse:
    """Serves the interactive effect and parameter range demonstration & calibration lab."""
    return HTMLResponse(content=CALIBRATE_HTML_PATH.read_text(encoding="utf-8"))


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
    return [
        PresetSummary(
            id=item["id"],
            name=item["name"],
            version=item.get("version", "1.0.0"),
            description=item["description"],
            color_space=item["color_space"],
            sample_palette=item.get("sample_palette", []),
        )
        for item in pipeline.list_presets(include_palette=True)
    ]


@app.get("/api/parameter-ranges")
def get_parameter_ranges() -> List[Dict[str, Any]]:
    """Returns all parameter range specifications and their active [selected_min, selected_max] bounds."""
    return PARAMETER_RANGES.list_specs()


@app.put("/api/parameter-ranges")
def update_parameter_ranges(body: ParameterRangeUpdateRequest) -> List[Dict[str, Any]]:
    """Updates and persists active [selected_min, selected_max] sampling bounds."""
    try:
        return PARAMETER_RANGES.update_ranges(body.ranges)
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@app.post("/api/parameter-ranges/reset")
def reset_parameter_ranges() -> List[Dict[str, Any]]:
    """Resets all parameter sampling ranges to factory defaults."""
    return PARAMETER_RANGES.reset_defaults()


async def _execute_render(config: RenderConfiguration) -> Response:
    """Executes pipeline render in a worker thread and formats the HTTP PNG response."""
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
        "X-Render-Contrast-Ratios": ",".join(str(ratio) for ratio in result.palette_contrast.ratios),
        "X-Render-Contrast-Pass": str(result.palette_contrast.passes).lower(),
    }
    if result.chromatic_aberration > 0.0:
        headers["X-Render-Aberration"] = str(result.chromatic_aberration)
    if result.contours > 0.0:
        headers["X-Render-Contours"] = str(result.contours)

    return Response(content=png_bytes, media_type="image/png", headers=headers)


@app.get("/api/render")
async def render_get(
    preset: Optional[str] = Query(default="nocturne"),
    device: Optional[str] = Query(default=None),
    width: Optional[int] = Query(default=None, ge=16, le=MAX_DIMENSION),
    height: Optional[int] = Query(default=None, ge=16, le=MAX_DIMENSION),
    seed: Optional[int] = Query(default=None),
    complexity: Optional[int] = Query(default=None, ge=1, le=10),
    engine: Optional[str] = Query(default=None),
    harmonize: Optional[Literal["perceptual", "theory", "nothing", "none"]] = Query(default=None),
    noise: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    vignette: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    dots: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    dots_spacing: Optional[int] = Query(default=None, ge=4, le=100),
    contours: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    contour_levels: Optional[int] = Query(default=None, ge=1, le=100),
    contour_width: Optional[float] = Query(default=None, ge=0.1, le=10.0),
    contour_mode: Optional[Literal["light", "dark", "accent"]] = Query(default=None),
    glass: Optional[bool] = Query(default=None),
    aberration: Optional[float] = Query(default=None, ge=0.0, le=0.1),
    warp_strength: Optional[float] = Query(default=None, ge=0.0, le=3.0),
    transverse_warp: Optional[float] = Query(default=None, ge=0.0, le=2.0),
    ambient_weight: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    dispersion: Optional[float] = Query(default=None, ge=0.05, le=2.0),
    phase_offset: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    frequency: Optional[float] = Query(default=None, gt=0.0),
    trap_weight: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    channel_offset: Optional[float] = Query(default=None, ge=-1.0, le=1.0),
    fractal_type: Optional[Literal["fluid", "julia", "mandelbrot", "orbit_trap"]] = Query(default=None),
    fractal_zoom: Optional[float] = Query(default=None, gt=0.0),
    fractal_center_x: Optional[float] = Query(default=None),
    fractal_center_y: Optional[float] = Query(default=None),
    fractal_angle: Optional[float] = Query(default=None),
) -> Response:
    """Renders wallpaper directly as a streaming PNG image response."""
    w, h = _resolve_dimensions(device, width, height)
    fractal_center = (
        (fractal_center_x, fractal_center_y)
        if fractal_center_x is not None and fractal_center_y is not None
        else None
    )

    config = RenderConfiguration(
        width=w,
        height=h,
        preset=preset,
        complexity=complexity,
        engine=engine,
        harmonize=harmonize,
        noise=noise,
        vignette=vignette,
        dots=dots,
        dots_spacing=dots_spacing,
        contours=contours,
        contour_levels=contour_levels,
        contour_width=contour_width,
        contour_mode=contour_mode,
        glass=glass,
        chromatic_aberration=aberration,
        warp_strength=warp_strength,
        transverse_warp=transverse_warp,
        ambient_weight=ambient_weight,
        dispersion=dispersion,
        phase_offset=phase_offset,
        frequency=frequency,
        trap_weight=trap_weight,
        channel_offset=channel_offset,
        fractal_type=fractal_type,
        fractal_zoom=fractal_zoom,
        fractal_center=fractal_center,
        fractal_angle=fractal_angle,
        seed=seed,
    )
    return await _execute_render(config)


@app.post("/api/render")
async def render_post(body: RenderRequest) -> Response:
    """Renders wallpaper from detailed JSON request parameters."""
    w, h = _resolve_dimensions(body.device, body.width, body.height)

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
        contours=body.contours,
        contour_levels=body.contour_levels,
        contour_width=body.contour_width,
        contour_mode=body.contour_mode,
        glass=body.glass,
        chromatic_aberration=body.chromatic_aberration,
        warp_strength=body.warp_strength,
        transverse_warp=body.transverse_warp,
        ambient_weight=body.ambient_weight,
        dispersion=body.dispersion,
        phase_offset=body.phase_offset,
        frequency=body.frequency,
        trap_weight=body.trap_weight,
        channel_offset=body.channel_offset,
        fractal_type=body.fractal_type,
        fractal_zoom=body.fractal_zoom,
        fractal_center=body.fractal_center,
        fractal_angle=body.fractal_angle,
        seed=body.seed,
    )
    return await _execute_render(config)


@app.post("/api/metadata/extract")
async def extract_metadata_endpoint(request: Request) -> Dict[str, Any]:
    """Extracts wallpaper engine parameters and version from uploaded image bytes."""
    body_bytes = await request.body()
    if not body_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image payload. Send raw image binary in the request body.",
        )
    metadata = extract_image_metadata(body_bytes)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No wallpaper metadata found in provided image.",
        )
    return metadata.to_dict()



