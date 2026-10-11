# Wallpaper Generator

A procedural wallpaper generator driven by a statistical preset engine. Each preset defines color palettes as probability distributions in HLS or perceptual OKLCH color space—every run is statistically unique but aesthetically coherent.

## Key Features

- **Statistical Preset Engine:** Presets encode color aesthetics as HLS distributions. The engine samples them at generation time — no two outputs are identical, but all stay within the preset's aesthetic envelope.
- **Fluid Mesh Gradient:** Bicubic mesh interpolation for smooth, high-quality gradients.
- **Style Filters:** Optional dot-matrix overlays and film grain, configurable per preset or via CLI overrides.
- **Color Harmonizers:** CIELAB-based Nothing-inspired mapping or color-theory harmonization.
- **Device Presets:** Common resolutions built-in — MacBook, 4K, ultrawide, mobile.
- **Screen Sampler:** Grab colors directly from the screen at the mouse cursor (macOS).

## Installation

1. Ensure Python 3.12+ is installed.
2. Install `uv`, clone this repository, and navigate to the folder.

### Option 1: Global Install (Recommended)

```bash
./install.sh
```

Installs dependencies and creates a global `wp-generator` command. Requires sudo for `/usr/local/bin` symlink.

### Option 2: Manual

```bash
uv sync
uv run python main.py --preset cupertino
```

## Usage

### Basic Examples

```bash
# Preset-driven generation (statistical engine picks palette on every run)
python main.py --preset cupertino

# Target a specific display
uv run python main.py --preset cyberpunk --res 4k

# Custom hex colors
python main.py --colors "#FF5000" "#1A1A2E"

# Batch — generate 10 variations of the same preset
python main.py --preset pastel --count 10

# Reproducible output
python main.py --preset nothing --seed 42
```

### Available Presets

| Preset | Aesthetic |
| :--- | :--- |
| `balanced` | Wide hue range, dark anchor, vivid accents |
| `cupertino` | Soft, bimodal light/dark with muted tones |
| `cyberpunk` | Neon palette, near-black background |
| `ink` | Monochromatic near-black, minimal chroma |
| `nothing` | Industrial dark with high-contrast accent and dot-matrix overlay |
| `pastel` | Light anchors, desaturated accents |
| `afterglow` | Vivid reference-derived violet, cyan, coral, orange, and magenta fields |
| `sleek` | Luminous neon accents on dark anchor with frosted glass texture |
| `nocturne` | Deep richly-tinted grounding anchors paired with luminous twilight and ember accents |

### Device Presets (`--device`)

| Key | Resolution | Target |
| :--- | :--- | :--- |
| `mbp14` | 3024×1964 | MacBook Pro 14" |
| `mbp16` | 3456×2234 | MacBook Pro 16" |
| `mba13` | 2560×1664 | MacBook Air 13" |
| `mba15` | 2880×1864 | MacBook Air 15" |
| `1080p` | 1920×1080 | FHD Monitor |
| `1440p` | 2560×1440 | QHD Monitor |
| `4k` | 3840×2160 | UHD Monitor |
| `5k` | 5120×2880 | 5K Monitor |
| `uw` | 3440×1440 | Ultrawide (21:9) |
| `suw` | 5120×1440 | Super Ultrawide (32:9) |
| `mobile` | 1080×1920 | Smartphone Portrait |

### Advanced Overrides

Run `--help-all` for advanced flags including `--complexity`, `--noise`, `--dots`, `--dots-spacing`, `--fluid-mode`, `--harmonize`, `--res`, and `--seed`.

```bash
python main.py --help-all
```

### Preview Mechanism (FX Testing)

When developing or evaluating new visual effects (e.g. volumetric lighting, dot matrix overlays), you can isolate the effects from the underlying base mesh. Pass the `--preview-fx` flag to generate an additional side-by-side composite image:

```bash
# Generates both the final wallpaper and a side-by-side "_preview" image
python main.py --preset nothing --preview-fx
```

This output file (`*_preview.png`) places the raw, un-filtered fluid mesh on the left and the fully processed final render on the right, making it convenient to immediately see the impact of effects like noise and dot-matrix patterns.

### Sample Outputs

![Nothing Preset](samples/nothing.png)
![Balanced Preset](samples/balanced.png)
![Cyberpunk Preset](samples/cyberpunk.png)
