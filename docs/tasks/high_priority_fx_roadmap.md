# High Priority: FX & Visual Upgrades Roadmap

## Context & Requirements
We are upgrading the algorithmic fluid mesh generator to a modular, preset-driven architecture. The goal is to elevate the output to "Cupertino" (Apple-tier) quality.

**Core Requirement:** 
Effects must be defined at the **preset level**. For example, a `cupertino.json` preset might invoke volumetric scattering, while a `nothing.json` preset invokes monochromatic dot-matrix overlays.
*Crucially*, we must implement a dev-friendly preview mechanism (e.g., a CLI toggle or a web inspector) that allows us to rapidly isolate, toggle, and view output examples for each FX application individually. We need to see exactly what each effect does before locking it into a preset.

---

## Hikaru Nakamura's FX Research Report
*(To be evaluated and implemented as previewable engine toggles)*

> "Look, objectively, the current algorithmic fluid mesh is fine. It's solid. But if we want it to be completely winning at the top level—if we're talking premium, high-end aesthetics—we need to be concrete about what's missing. The engine says we're sitting at +0.5 when we could easily be +5.0. Let's look at the forcing lines.
> 
> The first move here is literally **volumetric lighting**. Just straight up raymarching some subsurface scattering through the fluid. The thing is, standard Phong shading on a mesh is just losing. It looks flat. If you add volumetric caustics—light actually refracting through the density of the noise fields—it instantly becomes objectively better. It's like finding a mate in three. You get these deep, glowing pockets that make the mesh look like liquid glass or silk.
> 
> Then we have to look at the geometry itself. Currently, it's just a continuous undulating mesh. Adding **geometric fragmentation or instanced micro-structures** is the forcing continuation here. Imagine the fluid mesh breaking apart into microscopic Voronoi shards at the crests of the noise. Or adding metallic, floating primitive objects—like tiny gold spheres or wireframe octahedrons—caught in the fluid's vector field. The contrast between the smooth, continuous fluid and sharp, rigid micro-geometry creates this crazy optical illusion of macro-photography. It's completely crushing.
> 
> What about the noise variations? Everyone plays the standard Simplex or Perlin noise openings. It's booked out. To get an edge, you have to play something like **curl noise layered with domain warping**. If you feed the output of one noise function into the coordinates of another, the fluid stops looking like a generic screensaver and starts folding in on itself like real turbulence. It creates these sharp, infinite-resolution creases.
> 
> And we can't forget **chromatic aberration and iridescence**. At the end of the day, thin-film interference on the edges of the mesh is a free move that improves your position instantly. You map the view angle to a color ramp, and suddenly the fluid has this oily, holographic sheen. Add a touch of depth-of-field blur on the background layers, and it’s basically a forced win for the aesthetic."

## Implementation Pipeline
1. Build the preset architecture (`cupertino`, `nothing`, etc.) handling basic colors and noise.
2. Implement a `--preview-fx` CLI argument to isolate rendering layers.
3. Begin integrating Hikaru's suggestions one by one, verifying visual impact against the previewer.