# PyInstaller spec for the standalone wp-generator executable (one folder per OS).
# Build: uv run --with pyinstaller pyinstaller packaging/wp-generator.spec --noconfirm
# macOS signing happens afterwards in packaging/macos-sign.sh.
import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    # engine.resolve_preset_directory() looks for presets next to the engine package.
    datas=[(os.path.join(ROOT, "presets"), "engine/presets")],
    hiddenimports=["engine.sampler"] if sys.platform == "darwin" else [],
    excludes=["web", "fastapi", "uvicorn", "starlette", "httpx", "tests", "tools", "tkinter", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="wp-generator",
    console=True,
    strip=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="wp-generator",
)
