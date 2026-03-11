# Nothing-Style Wallpaper Generator (macOS M4 Pro)

A sophisticated procedural wallpaper generator designed for high-resolution (5K/Retina) displays. It captures the "Nothing" (retro-futuristic, minimalist, industrial) aesthetic using mesh gradients, procedural dot-matrix filters, and perceptual color-harmony intelligence.

## 🚀 Key Features
- **Mesh Gradient Engine:** Uses Inverse Distance Weighting (IDW) for organic, high-fidelity color blending.
- **Nothing Filters:** Procedural dot-matrix overlays, monochromatic grain (noise), and frosted glass effects.
- **Color Harmonizer:** Analyzes and "snaps" input colors to the Nothing aesthetic using CIELAB LCH color science.
- **macOS Screen Sampler:** Retina-aware tool to grab colors directly from your screen at the mouse cursor.

## 🛠️ Installation
1. Ensure you have Python 3.12+ installed.
2. Clone this repository and navigate to the folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On macOS, you may need to grant "Screen Recording" permissions to your terminal for the color sampler to work.*
## 📖 Usage
Run the generator via the CLI. By default, it generates a **5120x2880 (5K)** wallpaper.

### Visual CLI Examples

#### 1. The Classic Nothing Look (Dark Theme)
```bash
python main.py --theme dark --dots 0.8 --dots-spacing 25
```
*Effect: Deep charcoal mesh gradient with a technical dot grid modulated by luminance.*

![Dark Theme Sample](samples/dark.png)

#### 2. CMF Orange "Technical" Edition
```bash
python main.py --colors #FF5000 --dots-mode alpha --dots-spacing 15
```
*Effect: Varies dot **opacity** rather than size. Denser grid (15px) for a textured, industrial feel.*

![Alpha Mode Sample](samples/dots_alpha.png)

#### 3. High-Contrast Red (Phone 2a Style)
```bash
python main.py --theme red --noise 0.05 --dots 1.0
```
*Effect: Adds heavy film grain (noise) to prevent banding in deep reds and emphasize the "retro" display look.*

![Red Theme Sample](samples/red.png)

#### 4. The "Glass" Series (Frosted)
```bash
python main.py --theme glass --noise 0.01 --res 2560 1440
```
*Effect: Smooth, blurred translucent gradients with minimal noise, perfect for a clean desktop setup.*

![Glass Theme Sample](samples/glass.png)

#### 5. Generic Mesh Gradients (No Filters)
```bash
python main.py --no-nothing --colors #FF00FF #00FFFF
```
*Effect: Pure, vibrant mesh gradient without dots, grain, or brand-snapping.*

![Generic Sample](samples/generic_gradient.png)

---

### CLI Options Reference
| Argument | Description | Default | Visual Impact |
| :--- | :--- | :--- | :--- |
| `--no-nothing`| **Disable Nothing-style** | `False` | Disables dots, noise, and color snapping. |
| `--theme` | Pre-defined aesthetic themes | `dark` | `dark`, `light`, `red`, `orange`, `glass` |
| `--colors` | Custom hex color list | (Theme) | Harmonized to "Nothing" LCH space |
| `--dots` | Dot intensity (0.0 - 1.0) | `0.8` | `1.0` = Bold/Sharp dots; `0.2` = Subtle texture |
| `--dots-mode` | Modulation type | `size` | `size` (Variable radius) vs `alpha` (Variable opacity) |
| `--dots-spacing`| Grid density (pixels) | `25` | `10` = Dense/Fabric; `50` = Minimal/Blueprint |
| `--noise` | Grain amount (0.0 - 1.0) | `0.03` | `0.05` = "Film" grain; `0.01` = Modern digital |
| `--res` | Resolution (W H) | `5120 2880`| `5120 2880` (5K), `2560 1440` (2K) |
| `--sample` | Screen Color Sampler | `False` | Captures pixel at mouse cursor (3s delay) |
| `--no-adjust` | Raw Color Mode | `False` | Bypasses the Nothing Aesthetic Harmonizer |

---

### 🎨 The Aesthetic Logic
The generator uses a **Luminance-Aware Modulation** system:
- **Dots on Dark Areas:** Larger/More Opaque (emphasizes structure).
- **Dots on Light Areas:** Smaller/More Transparent (prevents clutter).
- **Colors:** Automatically snapped to the **Nothing Red (Hue 40)** or **CMF Orange (Hue 47)** for brand consistency.
## 🧪 Aesthetic Philosophy
The generator doesn't just pick random colors; it uses the **CIELAB color space** to ensure high-contrast or perfectly monochrome results. Vibrant colors are snapped to "Nothing Red" or "CMF Orange," while background colors are muted to maintain an industrial, retro-futuristic vibe.

---
*Created for MacBook M4 Pro.*
