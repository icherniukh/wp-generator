"""Procedural wallpaper generation engine."""

from pathlib import Path

__version__ = "0.1.0"


def resolve_preset_directory() -> Path:
    """Returns the presets directory: bundled in the package when installed, else the repo's."""
    package_dir = Path(__file__).resolve().parent
    bundled = package_dir / "presets"
    return bundled if bundled.is_dir() else package_dir.parent / "presets"
