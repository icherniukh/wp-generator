# Agent Instructions

## Beads

This repo uses `bd` for issue tracking.

- Start with `bd ready` and `bd list --status=in_progress`.
- Claim work with `bd update <id> --claim`.
- Create new persistent tasks with `bd create`.
- Close finished work with `bd close <id>`.
- Use `bd remember` for durable project notes.

Do not use markdown TODO files or duplicate the same task outside beads.

## Environment

- Runtime is `uv`. Environment lives in project-local `.venv`; never install into the system interpreter.
- `uv sync` for the default env. Web and macOS extras are opt-in: `uv sync --extra web --extra macos`.
- Run everything through `uv run` (e.g. `uv run python main.py ...`, `uv run python -m pytest tests/`).
- `uv.lock` is the source of truth. `requirements.txt` is a legacy pip fallback; do not add deps to it.

## Layout

- `main.py` — Click CLI. Flag names match `RenderConfiguration` fields 1:1; the builder in `main.py` passes them through, so new config fields need no mapping code.
- `engine/` — pipeline, palette/harmonizer logic, gradient engines (`engine/gradients/`, 15 registered names), filters, parameter ranges, embedded-metadata support.
- `web/` — FastAPI app and the `/calibrate` lab. Run: `uv run uvicorn web.app:app`.
- `presets/*.json` — OKLCH/HLS sampling distributions plus engine overrides.
- `tests/` — pytest suite. Run green before finishing any behavior change.
- `samples/*.png` — committed gallery renders (see README). Regenerate with `main.py` when the engine changes their look.
- `output/` — local renders only, gitignored except `.gitkeep`.

## Conventions

- Conventional Commits. No AI attribution in commits, PRs, or issues.
- Keep the CLI surface minimal: advanced engine/filter flags stay `hidden=True` (compact `--help`, full `--help-all`).
- Validate user-facing behavior by running it; keep 119+ tests passing.
- Do not commit agent tooling state (`.turbocharging/`, `.codex/`, `.claude/`, `.serena/`), experiment reports, or planning docs — all gitignored or intentionally untracked.

## Comparing variants

Use `tools/compare.py` instead of ad hoc PIL scripts to look at render variants side by side. Rows are variants, columns are seeds (`--transpose` swaps them):

```bash
uv run python tools/compare.py \
  --variant 'preset=aura' --variant 'preset=aura,noise=0,field_grain=0.15' \
  --variant 'engine=blobs,warps=noise+liquify,stage_params.shading.tint=0.6' \
  --seeds 1,2,3 --res 640x360 --crop 300x200 --diff --workers 4 --out sheet.png
```

- A variant is `key=value` pairs; keys are `RenderConfiguration` fields (unknown keys list the valid ones). `stage_params.<stage>.<arg>=v` nests, `a+b` makes a list, values are JSON when they parse.
- Each column renders one seed. A variant without `preset` or `colors` reuses the first variant's final palette for that seed, so only the changed option differs. A variant that sets `preset` keeps its own palette, which can shift when you set an engine value the preset would otherwise sample (use `noise=0` rather than leaving `noise` unset).
- `--crop WxH` adds a 100% center crop under each image (grain, halftone), `--diff` adds the amplified difference against the first variant (`--diff-gain`), `--json` prints the resolved parameters per cell.
- Write sheets outside the repo (scratchpad or `output/`), then view them.

## Shell

Use non-interactive forms for file operations:

```bash
cp -f source dest
mv -f source dest
rm -f file
rm -rf directory
cp -rf source dest
```

Avoid commands that wait for confirmation prompts.
