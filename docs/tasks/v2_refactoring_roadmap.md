# Refactoring Roadmap: Wallpaper Generator V2

This document captures the technical feedback and outlines the necessary work items to transition the prototype into a production-grade, highly performant graphical tool tailored for Apple Silicon (M4 Pro).

## Epic 1: Overhaul Gradient Engine (Fluidity over Geometry)
**The Problem:** Inverse Distance Weighting (IDW) creates rigid, sharp "blobs" or starbursts around control points. Modern mesh gradients (like Apple's or Nothing's UI) rely on fluid, overlapping folds of color.
**The Goal:** Replace IDW with a more organic fluid simulation or advanced interpolation.

*   **Task 1.1: Implement Simplex/Perlin Noise Warp**
    *   *Description:* Replace the current `MeshGradient` logic. Generate a smooth base gradient (e.g., bilinear) and use a multi-octave 2D Simplex noise field to "warp" the coordinates. This creates the "folded fabric" or fluid liquid look.
    *   *Requirements:* Add a `noise` or `perlin-numpy` dependency. Keep array types as `float32` for memory efficiency.
*   **Task 1.2: Implement Bicubic Mesh Interpolation (Alternative)**
    *   *Description:* Instead of distance-based weighting, divide the canvas into a grid of control points (Coons Patch) and use `scipy.interpolate.griddata` with 'cubic' interpolation.
    *   *Requirements:* Ensure the edges of the canvas pin to background colors to avoid edge-bleeding.

## Epic 2: High-Performance Filter Pipeline (Vectorization)
**The Problem:** The `apply_dot_matrix` function uses native Python `for` loops and Pillow's `ImageDraw` to render thousands of circles. This is a massive CPU bottleneck for 5K images.
**The Goal:** Shift all pixel-level operations to pure, vectorized NumPy matrix math.

*   **Task 2.1: Vectorize Dot-Matrix Generation**
    *   *Description:* Rewrite `apply_dot_matrix`. 
        1. Generate a canvas-sized `(X, Y)` coordinate grid (`np.meshgrid`).
        2. Use modulo arithmetic (`X % spacing`, `Y % spacing`) to find the distance of every pixel to its nearest grid center.
        3. Create a boolean mask where `distance < threshold(luminance)`.
        4. Apply the mask directly to the alpha channel of a zeroed NumPy array.
    *   *Requirements:* Zero `ImageDraw` usage. Must execute in < 0.5 seconds for a 5K image.

## Epic 3: Revamp Color Harmonizer (Preserve Intent)
**The Problem:** The current `adjust_for_nothing` logic is too draconian. It aggressively destroys user-provided hues if they don't fall into hardcoded Red/Orange buckets, resulting in muddy grays.
**The Goal:** Shift from "Hue Forcing" to "Contrast/Chroma Mapping," allowing a wider variety of colors while maintaining the industrial vibe.

*   **Task 3.1: Implement LCH Chroma/Lightness Clamping**
    *   *Description:* Instead of changing the user's Hue ($h*$), modify the Harmonizer to clamp Lightness ($L*$) to extremes (either < 30 or > 80 for high contrast) and reduce Chroma ($C*$) proportionally.
    *   *Requirements:* If a user passes `#00FFFF` (Cyan), it should return a "Nothing-style" Cyan (e.g., deep slate-cyan or bright icy-cyan), rather than converting it to a gray.
*   **Task 3.2: Dynamic Contrast Ratio Verification**
    *   *Description:* Ensure that the final selected palette has at least one pair of colors that passes a WCAG 4.5:1 contrast ratio to guarantee the "punchy" look.

## Epic 4: Native macOS Integration (Reduce Dependencies)
**The Problem:** Using the `mss` library to grab a single pixel is over-engineered, introduces cross-platform bloat, and triggers aggressive macOS Screen Recording permission prompts.
**The Goal:** Utilize native macOS APIs for a seamless, lightweight color picker.

*   **Task 4.1: Replace `mss` with Native `NSColorSampler` or CoreGraphics**
    *   *Description:* macOS 10.15+ has a built-in `NSColorSampler` (the native magnifying glass color picker). If a silent grab is needed, use `CGWindowListCreateImage` via PyObjC to capture a 1x1 rect around the mouse.
    *   *Requirements:* Remove `mss` from `requirements.txt`. Ensure the new method correctly maps Retina physical pixels to logical coordinates automatically.

## Epic 5: GPU Acceleration (Future Architecture)
**The Problem:** We are doing heavy matrix math on the CPU. The M4 Pro has a massive GPU that is sitting idle during generation.
**The Goal:** Lay the groundwork for moving the entire pipeline to the GPU.

*   **Task 5.1: Proof of Concept - Metal/CoreImage Translation**
    *   *Description:* Investigate porting the core generation script from Python/NumPy to a Swift/Metal script or using Python bindings for Apple's CoreImage framework (`CIColorMatrix`, `CIGaussianBlur`, `CIPixellate`). 
    *   *Requirements:* Document the potential speedup. A CoreImage pipeline could generate a 5K image in ~16ms.

## Epic 6: Hardware Scaling & Multi-Monitor Support
**The Problem:** The tool currently assumes a static, hardcoded default resolution and doesn't dynamically adapt to different MacBook models or external monitors.
**The Goal:** Automatically detect the user's active display(s) and tailor the output generation precisely to their hardware.

*   **Task 6.1: Native Display Resolution Detection**
    *   *Description:* Implement a display detection utility using macOS native APIs (via PyObjC or `AppKit.NSScreen`) to fetch the main screen's physical pixel dimensions (e.g., 3024x1964 for a 14-inch MacBook Pro M4, 3456x2234 for a 16-inch).
    *   *Requirements:* Use this detected resolution as the default CLI output size if `--res` is not provided.
*   **Task 6.2: Multi-Monitor & Aspect Ratio Adaptation**
    *   *Description:* Adapt gradient meshes based on the aspect ratio (e.g., 16:9, 16:10) so circles/dots remain perfectly round and meshes distribute evenly across wider external displays.

## Epic 7: Distribution & Packaging (Homebrew)
**The Problem:** Currently, users must clone the repository, manage Python virtual environments, and manually run the installer script. This is friction.
**The Goal:** Make the tool available via a standard macOS package manager for one-click installation.

*   **Task 7.1: Create a Homebrew Tap**
    *   *Description:* Package the Python script using `pyinstaller` or `shiv` into a standalone binary to eliminate the need for the end-user to manage Python dependencies.
    *   *Requirements:* Create a GitHub repository (e.g., `homebrew-nothing-wp`) and author a `Formula/nothing-wp.rb` file so users can install it globally via `brew install username/tap/nothing-wp`.
