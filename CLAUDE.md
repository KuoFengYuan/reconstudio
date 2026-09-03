# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"          # panel + ruff / mypy / pytest
pytest                           # offline unit tests (no colmap/ffmpeg/GPU/network needed)
pytest tests/test_colmap_run.py  # one module
pytest -k loop_detection         # one test by name
ruff check .                     # lint (line-length 100, rules E/F/I/UP/B/SIM)
mypy pipeline/config.py          # type check — only this module is held to a strict bar
./run.sh                         # start the panel (reads local.env; 127.0.0.1 by default)
./run.sh --doctor                # same report as /doctor, to the terminal; exit code reflects FAILs
```

CI (`.github/workflows/ci.yml`) runs exactly ruff + mypy + pytest, in that order.

Editing `.py` needs a panel restart; editing `templates/` only needs a page reload.

If you run tests without `pip install -e .`, prefix with `PYTHONPATH=.` — otherwise
collection fails with `ModuleNotFoundError: No module named 'pipeline'`.

## Architecture

**The panel is torch-free, and that is load-bearing.** `pipeline/` never imports a
trainer, colmap, or torch — it only spawns them as subprocesses in *their* own conda
envs. This is what makes deployment `pip install -r requirements.txt` while the heavy
CUDA envs stay per-machine prerequisites that `pipeline/backends.py` merely locates and
health-checks. Keep new heavy dependencies behind a subprocess boundary or a lazy import
(see `jobs._run_blocksplit`, which keeps numpy out of the eager import chain).

**Layers**, outermost first:

- `app.py` — app factory only: build FastAPI, mount static, start the job manager, include routers.
- `web/routers/` — HTTP handlers returning htmx fragments (`pages`, `browse`, `create`, `jobs`, `viz`, `viewer`, `measure`, `doctor`).
- `web/services/` — pure request-side logic: `forms.py` (form → validated params), `models.py` (job → filesystem paths). HTTP concerns like 404s stay in the routers.
- `web/shared.py` — the Jinja handle, `_page()`, and the UPPERCASE form-prefill defaults that mirror the pipeline defaults.
- `jobs.py` — `JobManager` (queue, workers, cancel/delete), the `RUN_FUNCS` kind→function dispatch, and the per-kind log parsers.
- `pipeline/` — the actual orchestration. Torch-free.

**Log lines are a contract, not just output.** `jobs.py` parses subprocess stdout to
drive the UI's progress (COLMAP's `=== [HH:MM:SS] <stage> ===` / `skip <stage>` banners;
frames' `######## [j/N]`; depth's `Images: N under …` / `[i/N] name` / `Done. processed=N skipped=N`).
Changing a banner's wording breaks the progress display silently, and a new engine for an
existing job kind should *reproduce* the existing markers rather than add a second parser
(that is why `tools/moge3_preprocess.py` prints LichtFeld's exact wording).

**COLMAP orchestration** (`pipeline/colmap/_run.py`) resolves one `_Ctx` (all config plus
mutable shared state such as `img_root`, which the NESTED staging and the FullHD resize
rebase), then calls `_stage_*` functions in order. Each stage is a thin wrapper around one
colmap sub-command, guarded by its own sentinel file or output check via `c.need(...)`, so
runs are idempotent and re-runnable with `FORCE=1`. Stage names live in `COLMAP_STAGES`
and are individually skippable.

**Adding a COLMAP tunable touches four layers** — miss one and the knob silently never
reaches colmap:

1. `pipeline/colmap/_run.py` → `COLMAP_DEFAULTS` (lowercase key; promote to a `_Ctx` field in `_setup` if a stage needs it as an attribute)
2. `web/shared.py` → `COLMAP_DEFAULTS` (UPPERCASE key, form prefill)
3. `web/services/forms.py` → `build_colmap_params` (map it, then add it to the matching validation loop; blank-means-tool-default values go in the "optional numerics" loop)
4. `templates/index.html` → the input, plus its `.mgrp`/`.pgrp` visibility group in `syncMatcher()`/`syncMapper()`

**Only pass a subprocess flag when it deviates from that tool's own default.** An unknown
option is a hard parse error for colmap, not a warning, so an unconditional flag breaks
the stage against older builds. Phrase a boolean knob as the *opt-out* when the tool's
default is the desirable one — a checkbox absent from a POST body reads as `False`, so an
opt-in default would silently disable a good default for programmatic callers. See
`gm_single_model` and `seq_loop_min_index_distance`.

**Backends are data, not code** (`pipeline/backends.py`). `backends.json` shallow-merges
over `BUILTIN_BACKENDS`, so a per-machine entry only lists the keys that differ. Adding a
trainer is a config edit. Interpreter resolution walks: explicit `"python"` →
`$CONDA_ROOT/envs/<env>/bin/python` → envs dir derived from `sys.prefix` →
`conda info --base` → common install locations.

**Depth/normal generation has two interchangeable engines** under the one `depth` job kind,
dispatched by `jobs._run_depth_engine` on the `engine` param:

- `pipeline/depth.py` — LichtFeld-Studio's `preprocess` (MoGe-2), sharing the compiled binary with the `lichtfeld-mrnf` training backend.
- `pipeline/moge3.py` → `tools/moge3_preprocess.py` — PyTorch MoGe-3 in its own `moge3` conda env.

Both write `depth/`(1-channel PNG) and `normals/`(3-channel) mirroring each image's
relative path. `pipeline/moge3_encode.py` holds the shared encoding — a line-by-line port
of LichtFeld's `build_depth_png`/`build_normals_png`. **Getting that encoding wrong does
not raise; it silently feeds the training losses garbage**, so it is pinned by
`tests/test_moge3_encode.py`. Two rules are easy to invert: depth is per-image min-max
normalised into `[0.02, 1.0]` with **0** as the no-data sentinel, while normals use
`n*0.5+0.5` with the **neutral mid value** as their sentinel (0 is a legitimate normal).
`moge3_encode.py` deliberately imports nothing but numpy so both the foreign `moge3` env
and CI can load it; `moge3.py` next door pulls in `.runner`/`.config` and cannot be
imported from that env.

The depth stage is **non-destructive**: it only creates `depth/`/`normals/` next to
`images/` and never touches the source images. That convention holds across the pipeline.

**`_stage_undistort` sanitizes EXIF before every `image_undistorter` call** (both the
image pass and, when `MASKS_DIR` is set, the masks pass). Canon bodies write
empty-string `Artist`/`Copyright` tags rather than omitting them; OIIO 2.4.17's
IPTC-IIM encoder asserts a non-null pointer for every tag it re-encodes, so
`image_undistorter` SIGABRTs the instant it writes one of these images back out.
`_sanitize_exif` (`pipeline/colmap/_run.py`) shells out to `exiftool -if '$Artist eq ""
or $Copyright eq ""' -Artist= -Copyright=`, which is a no-op for any camera that
doesn't do this and pixel-lossless where it fires. Missing `exiftool` degrades to a
logged warning, not a failure — most datasets never hit the bug, so its absence
shouldn't block them (`preflight.exiftool_check()` surfaces it as `warn` on `/doctor`).

## Testing

Tests run fully offline. `tests/test_colmap_run.py` is the model to follow for pipeline
work: a `FakeRunner` records the argv of every `colmap <subcommand>` call and simulates
just enough on-disk output (a minimal sqlite DB, `sparse/0/*.bin`, the undistorted
dataset) for the next stage's existence checks to pass, with the image-reading and
re-encoding helpers monkeypatched out. Assert on the recorded argv and on
skip/sentinel/error paths.

When a test asserts the *absence* of a flag, also add one that asserts its presence when
enabled — an absence-only test passes even if the feature was never implemented.

`pipeline/vendor/` and `tools/colmap_read_write_model.py` are vendored and excluded from
lint.

## Conventions

- Numbers stay strings through the form → params → pipeline path; the pipeline converts at
  the point of use. Only genuine booleans become `bool`.
- Blank/empty string means "use the tool's own default" for optional knobs, and the flag is
  then omitted entirely.
- User-facing errors are Traditional Chinese (`ValueError` messages surface as the
  `_error.html` fragment); code comments and docstrings are English.
- Comments explain *why*, often naming the concrete failure that motivated the code. Match
  that density — it is the house style, not decoration.
- New endpoint → add a handler in `web/routers/`. New training backend → edit
  `backends.json`, not code. New setting → add a field to `pipeline/config.py`.
