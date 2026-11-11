from dataclasses import asdict, dataclass
from datetime import datetime
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import ExifTags, Image, PngImagePlugin

from engine import __version__

# Standard TIFF / EXIF tag identifiers
EXIF_TAG_IMAGE_DESCRIPTION = 0x010E  # 270
EXIF_TAG_SOFTWARE = 0x0131           # 305
EXIF_TAG_DATE_TIME = 0x0132          # 306
EXIF_TAG_USER_COMMENT = 0x9286       # 37510


@dataclass
class ImageMetadata:
    """Represents generation parameters, engine type, and version metadata."""

    version: str
    engine: str
    seed: int
    parameters: Dict[str, Any]
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes metadata to a standard dictionary format."""
        return {
            "version": self.version,
            "engine": self.engine,
            "seed": self.seed,
            "created_at": self.created_at,
            "parameters": self.parameters,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes metadata to a JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageMetadata":
        """Instantiates ImageMetadata from a dictionary."""
        version = data.get("version", "unknown")
        engine = data.get("engine", "unknown")
        seed = int(data.get("seed", 0))
        created_at = data.get("created_at")

        parameters = data.get("parameters")
        if parameters is None:
            # Flattened format compatibility
            parameters = {
                k: v
                for k, v in data.items()
                if k not in ("version", "engine", "seed", "created_at")
            }

        return cls(
            version=version,
            engine=engine,
            seed=seed,
            parameters=parameters,
            created_at=created_at,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ImageMetadata":
        """Parses ImageMetadata from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_render_configuration(self) -> Any:
        """Constructs a RenderConfiguration from embedded parameters."""
        from engine.pipeline import RenderConfiguration, parse_hex_color

        params = self.parameters
        colors = None
        if "colors" in params and params["colors"]:
            colors = [parse_hex_color(c) if isinstance(c, str) else tuple(c) for c in params["colors"]]

        fractal_center = None
        if params.get("fractal_center"):
            fc = params["fractal_center"]
            fractal_center = tuple(fc) if isinstance(fc, (list, tuple)) else fc

        return RenderConfiguration(
            width=params.get("width", 3024),
            height=params.get("height", 1964),
            preset=params.get("preset"),
            colors=colors,
            complexity=params.get("complexity"),
            engine=self.engine or params.get("engine"),
            fluid_mode=params.get("fluid_mode"),
            harmonize=params.get("harmonize"),
            noise=params.get("noise"),
            vignette=params.get("vignette"),
            dots=params.get("dots"),
            dots_spacing=params.get("dots_spacing"),
            dots_mode=params.get("dots_mode"),
            contours=params.get("contours"),
            contour_levels=params.get("contour_levels"),
            contour_width=params.get("contour_width"),
            contour_mode=params.get("contour_mode"),
            glass=params.get("glass"),
            chromatic_aberration=params.get("chromatic_aberration"),
            warp_strength=params.get("warp_strength"),
            transverse_warp=params.get("transverse_warp"),
            ambient_weight=params.get("ambient_weight"),
            dispersion=params.get("dispersion"),
            phase_offset=params.get("phase_offset"),
            frequency=params.get("frequency"),
            trap_weight=params.get("trap_weight"),
            channel_offset=params.get("channel_offset"),
            fractal_type=params.get("fractal_type"),
            fractal_zoom=params.get("fractal_zoom"),
            fractal_center=fractal_center,
            fractal_angle=params.get("fractal_angle"),
            legacy_idw=bool(params.get("legacy_idw", False)),
            seed=self.seed,
        )


def build_image_metadata(result: Any, version: Optional[str] = None) -> ImageMetadata:
    """Extracts generation parameters from RenderResult and constructs ImageMetadata."""
    resolved_version = version or __version__
    created_at = datetime.now().strftime("%Y:%m:%d %H:%M:%S")

    contrast_data = None
    if getattr(result, "palette_contrast", None):
        contrast_data = {
            "ratios": [round(float(r), 2) for r in result.palette_contrast.ratios],
            "threshold": result.palette_contrast.threshold,
            "passes": result.palette_contrast.passes,
        }

    parameters: Dict[str, Any] = {
        "engine": result.engine,
        "seed": result.seed,
        "width": result.width,
        "height": result.height,
        "colors": getattr(result, "color_hexes", []),
        "complexity": getattr(result, "complexity", None),
        "noise": getattr(result, "noise", None),
        "chromatic_aberration": getattr(result, "chromatic_aberration", 0.0),
        "warp_strength": getattr(result, "warp_strength", None),
        "transverse_warp": getattr(result, "transverse_warp", None),
        "ambient_weight": getattr(result, "ambient_weight", None),
        "dispersion": getattr(result, "dispersion", None),
        "contours": getattr(result, "contours", 0.0),
        "contour_levels": getattr(result, "contour_levels", 0),
        "contour_width": getattr(result, "contour_width", None),
        "contour_mode": getattr(result, "contour_mode", None),
        "dots": getattr(result, "dots", None),
        "dots_spacing": getattr(result, "dots_spacing", None),
        "dots_mode": getattr(result, "dots_mode", None),
        "glass": getattr(result, "glass", None),
        "vignette": getattr(result, "vignette", None),
        "frequency": getattr(result, "frequency", None),
        "phase_offset": getattr(result, "phase_offset", None),
        "trap_weight": getattr(result, "trap_weight", None),
        "channel_offset": getattr(result, "channel_offset", None),
        "fluid_mode": getattr(result, "fluid_mode", None),
        "harmonize": getattr(result, "harmonize", None),
        "fractal_type": getattr(result, "fractal_type", None),
        "fractal_zoom": getattr(result, "fractal_zoom", None),
        "fractal_center": getattr(result, "fractal_center", None),
        "fractal_angle": getattr(result, "fractal_angle", None),
        "preset": getattr(result, "preset", None),
        "preset_version": getattr(result, "preset_version", None),
        "contrast": contrast_data,
    }

    # Filter out None keys to keep payload concise
    compact_parameters = {k: v for k, v in parameters.items() if v is not None}

    return ImageMetadata(
        version=resolved_version,
        engine=result.engine,
        seed=result.seed,
        parameters=compact_parameters,
        created_at=created_at,
    )


def encode_exif(metadata: ImageMetadata) -> Image.Exif:
    """Builds a PIL Exif container from ImageMetadata."""
    exif = Image.Exif()
    exif[EXIF_TAG_SOFTWARE] = f"wp-generator {metadata.version}"
    exif[EXIF_TAG_IMAGE_DESCRIPTION] = metadata.to_json()
    exif[EXIF_TAG_USER_COMMENT] = f"ASCII\0\0\0{metadata.to_json()}".encode("utf-8")
    if metadata.created_at:
        exif[EXIF_TAG_DATE_TIME] = metadata.created_at
    return exif


def encode_png_info(metadata: ImageMetadata) -> PngImagePlugin.PngInfo:
    """Builds PNG chunk metadata for text chunks."""
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", f"wp-generator {metadata.version}")
    info.add_text("parameters", metadata.to_json())
    info.add_text("Comment", f"wp-generator {metadata.engine} seed={metadata.seed}")
    return info


def attach_metadata_to_image(image: Image.Image, metadata: ImageMetadata) -> Image.Image:
    """Binds automatic EXIF and PNG chunk serialization to the image instance."""
    exif = encode_exif(metadata)
    pnginfo = encode_png_info(metadata)

    original_save = image.save

    def auto_save(*args: Any, **kwargs: Any) -> Any:
        format_val = kwargs.get("format")
        if format_val is None and args:
            if len(args) >= 2 and isinstance(args[1], str):
                format_val = args[1]
            elif isinstance(args[0], (str, Path)):
                suffix = Path(args[0]).suffix.lower()
                if suffix == ".png":
                    format_val = "PNG"

        kwargs.setdefault("exif", exif)
        if format_val in (None, "PNG", "png"):
            kwargs.setdefault("pnginfo", pnginfo)

        return original_save(*args, **kwargs)

    image.save = auto_save  # type: ignore[method-assign]
    image.info["exif"] = exif.tobytes()
    return image


def _extract_from_text(raw_text: str) -> Optional[ImageMetadata]:
    """Attempts to parse ImageMetadata from raw text or string representation."""
    if not raw_text:
        return None
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return ImageMetadata.from_dict(data)
    except Exception:
        pass
    return None


def extract_image_metadata(
    source: Union[str, Path, Image.Image, io.BytesIO, bytes]
) -> Optional[ImageMetadata]:
    """Extracts ImageMetadata from an image file, open Image object, or raw bytes."""
    close_when_done = False
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, (str, Path)):
        image = Image.open(source)
        close_when_done = True
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
        close_when_done = True
    elif isinstance(source, io.BufferedIOBase) or hasattr(source, "read"):
        image = Image.open(source)
        close_when_done = False
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    try:
        # 1. Attempt extraction from EXIF tags
        exif = image.getexif()
        software_val = exif.get(EXIF_TAG_SOFTWARE)
        version = None
        if software_val and isinstance(software_val, str) and software_val.startswith("wp-generator "):
            version = software_val.replace("wp-generator ", "").strip()

        # Check ImageDescription (tag 0x010E)
        desc_val = exif.get(EXIF_TAG_IMAGE_DESCRIPTION)
        if desc_val and isinstance(desc_val, str):
            meta = _extract_from_text(desc_val)
            if meta:
                if version and meta.version == "unknown":
                    meta.version = version
                return meta

        # Check UserComment (tag 0x9286)
        comment_val = exif.get(EXIF_TAG_USER_COMMENT)
        if comment_val:
            raw_comment = ""
            if isinstance(comment_val, bytes):
                # Check for standard 8-byte ASCII prefix
                if comment_val.startswith(b"ASCII\x00\x00\x00"):
                    raw_comment = comment_val[8:].decode("utf-8", errors="ignore")
                elif comment_val.startswith(b"UNICODE\x00"):
                    raw_comment = comment_val[8:].decode("utf-16", errors="ignore")
                else:
                    raw_comment = comment_val.decode("utf-8", errors="ignore")
            elif isinstance(comment_val, str):
                raw_comment = comment_val

            meta = _extract_from_text(raw_comment)
            if meta:
                if version and meta.version == "unknown":
                    meta.version = version
                return meta

        # 2. Attempt extraction from PNG textual chunks
        png_params = image.info.get("parameters")
        if png_params and isinstance(png_params, str):
            meta = _extract_from_text(png_params)
            if meta:
                png_software = image.info.get("Software")
                if png_software and isinstance(png_software, str) and png_software.startswith("wp-generator "):
                    meta.version = png_software.replace("wp-generator ", "").strip()
                return meta

        return None
    finally:
        if close_when_done:
            image.close()
