# AGENTS.md — Recon Studio

Instructions for coding agents (Codex, Claude Code, others). `CLAUDE.md` imports
this file, so this is the single source; keep agent rules here, not in copies.

Recon Studio is a local FastAPI + htmx panel that turns photos or video into a
COLMAP reconstruction, a 3D Gaussian Splatting model and, with a supporting
backend, a mesh. It **orchestrates** external tools (ffmpeg, COLMAP, trainers in
their own conda envs); it does not train anything in-process.

## 1. Read only what the task needs

The repository carries long docs and vendored bundles and points at large
datasets; context is the scarce resource.

1. Read this file, then **one** language of the agent guide —
   [English](docs/en/agent-guide.md) or [繁體中文](docs/zh-TW/agent-guide.md) —
   and its context router ([English](docs/en/context-router.md) /
   [繁體中文](docs/zh-TW/context-router.md)). Never load both translations.
2. Open only the files the router points to; read bounded line ranges and search
   explicit files (`rg -n 'TRAIN_DEFAULTS' pipeline/train.py`).
3. **Forbidden:** repository-wide scans — root-level `rg --files`, `rg PATTERN .`,
   `grep -R`, `find .`, `tree` or equivalent scripts — and inventorying datasets,
   checkpoints, `static/supersplat*/` bundles or sibling trainer repositories. If
   the router lacks an entry, inspect one identified folder non-recursively, then
   add the entry to **both** language routers.
4. The detailed implementation contracts (COLMAP stages, matte/depth encodings,
   LAN deployment traps) live in
   [docs/en/agent-guide.md § Implementation reference](docs/en/agent-guide.md#implementation-reference).
   Read only the subsection you need.

## 2. Commands

```bash
pip install -e '.[dev]'          # panel + ruff / mypy / pytest
pytest                           # offline: no colmap/ffmpeg/GPU/network needed
pytest tests/test_colmap_run.py  # one module;  pytest -k name  for one test
ruff check .                     # line-length 100, rules E/F/I/UP/B/SIM
mypy pipeline/config.py          # the one module held to strict typing
./run.sh --doctor                # environment report; exit code reflects FAILs
./run.sh                         # start the panel (reads local.env)
```

Without `pip install -e .`, prefix tests with `PYTHONPATH=.`. CI
(`.github/workflows/ci.yml`, job `lint-type-test`) runs ruff + mypy + pytest on
PRs and pushes to `main`; it never deploys.

## 3. Runtime facts that change how you work

- **Jobs run inside the panel process.** Editing `.py` takes effect only after a
  panel restart, and a restart kills every running job and its trainer
  subprocesses. Before restarting a live panel, check 任務紀錄 (or
  `GET /api/jobs`) and restart only when nothing runs. Templates and `static/`
  are re-read per request: reload the page.
- **Never test against the live panel's data.** For UI or runtime checks start a
  second panel on another port with its own scratch `RECON_STUDIO_DATA`, e.g.
  `RECON_STUDIO_DATA=/tmp/rs-data RECON_STUDIO_PORT=8091 python -m uvicorn app:app --port 8091`.
  Two processes sharing one jobs directory corrupt each other's state.
- **The panel is torch-free, on purpose.** `pipeline/` never imports a trainer,
  COLMAP or torch; it spawns them in their own envs. Keep heavy dependencies
  behind a subprocess or a lazy import.
- **Log lines are a contract.** `jobs.py` parses subprocess output for progress
  (`=== [HH:MM:SS] <stage> ===`, `skip <stage>`, `######## [j/N]`, `[i/N] name`).
  Rewording a banner silently breaks the progress display.
- **Only pass a tool flag when it deviates from the tool's default.** An unknown
  option is a hard parse error for COLMAP, so unconditional flags break older
  builds. Blank form values mean "tool default" and omit the flag entirely.
- **A new COLMAP tunable touches four layers**: `pipeline/colmap/_run.py`
  `COLMAP_DEFAULTS` → `web/shared.py` `COLMAP_DEFAULTS` → `web/services/forms.py`
  `build_colmap_params` (+ its validation loop) → `templates/index.html` input and
  its `.mgrp`/`.pgrp` visibility group. Miss one and the knob never reaches COLMAP.
- **The panel binds 127.0.0.1.** It has no login and spawns processes; LAN access
  goes through nginx (`scripts/deploy-nginx-lan.sh`), never `HOST=0.0.0.0`.
- Baselines: `FRAMES_DEFAULTS` (`pipeline/frames.py`), `COLMAP_DEFAULTS`
  (`pipeline/colmap/_run.py`), `TRAIN_DEFAULTS` (`pipeline/train.py`); backend
  schemas in `pipeline/backends.py`, machine overrides in `backends.json`. Never
  change a baseline for one experiment; use distinct output paths.

## 4. Code conventions

- Numbers stay strings from form to pipeline; convert at the point of use. Only
  real booleans become `bool`. Phrase a boolean as the opt-out when the tool's
  default is the good one (a checkbox missing from a POST reads as `False`).
- User-facing errors are Traditional Chinese (`ValueError` → `_error.html`);
  comments and docstrings are English and explain *why*, often naming the failure
  that motivated the code. Match that density.
- New endpoint → `web/routers/`. New setting → `pipeline/config.py`. New trainer
  → `backends.json`, not code. New job kind → a `run_*` in `pipeline/`, exported
  from `pipeline/__init__.py`, registered in `jobs.RUN_FUNCS` **and**
  `jobs.PARSERS`, a `POST /ui/<kind>` in `web/routers/create.py`, a branch in
  `templates/_jobstatus.html`, a form in `templates/index.html` (add it to
  `showForm`'s list). `matte` is the most recent worked example of all six.
- Tests: follow `tests/test_colmap_run.py` (a `FakeRunner` records argv). When a
  test asserts a flag's absence, add one asserting its presence when enabled.
  `pipeline/vendor/` and `tools/colmap_read_write_model.py` are vendored.

## 5. UI rules

- **Keep the current look.** The owner has rejected a full visual redesign;
  change layout, palette or labels only when asked, and show a preview first.
- The shared palette is the `:root` variables in `templates/index.html`'s
  `<style>` blocks; `static/css/workspace.css` holds layout and interaction
  states and overrides them. Fragments use `var(--accent/--ok/--err/--mut/--line…)`
  inline, so those names are a contract. Standalone viewers (`viz.html`,
  `mesh_*.html`, `gallery.html`, `matte_*.html`) and `doctor.html` /
  `verify.html` carry their own dark sheets.
- Preserve element ids, form field names, `hx-*` endpoints and data attributes;
  JS and tests depend on them (`tests/test_matte_template.py` reads `index.html`).
- Check desktop (≥1280 px) and a 390 px viewport, keyboard focus order, and the
  affected interactions in the sandbox panel. State which checks ran in a
  simulated browser and which still need a physical device.

## 6. Branch, commit, PR, merge, cleanup

**Branches.** One short-lived task branch: `Feature/<desc>` (new user-facing
functionality), `Bugfix/<desc>` (defects), or `Enhance/<desc>` (improvements,
performance, UX, docs, maintenance); case-sensitive prefix, lowercase hyphenated
description. Never `codex/`, lowercase prefixes, or extra long-lived branches.
Preserve unrelated changes; use an isolated worktree when the checkout holds
another task's work.

**Commits.** A specific English subject starting with `feat:`, `fix:`,
`enhance:` or `docs:`.

**Pull request.** English first: the reported symptom, established cause,
resulting behaviour and implementation boundaries (PR #56 is the reference for
level of detail), then separate `Tradeoffs`, `Validation`,
`Physical-device checks pending` and `繁體中文摘要` sections. Record concrete
checks, outcomes, representative inputs and limitations; distinguish simulated
browsers from physical devices. Link any earlier PR whose behaviour this changes.
If the application provides task-attachment tools, attach the PR to the task.

**Validation and merge.** Run the relevant checks (CI runs `ruff check .`,
`mypy pipeline/config.py`, `pytest`). Confirm mergeability, required checks and
reviews; fix failures and update the PR. Merge the verified HEAD, normally by
squash, keeping the PR number in the subject (`fix: describe the change (#123)`;
`gh pr merge --subject` drops GitHub's suffix, so add it). Never push to `main`,
use administrator bypass, disable protections or skip required reviews. If an
external gate blocks the merge, report its exact name and leave the PR and branch
intact. Do not rewrite published `main` history.

**Cleanup after a confirmed merge.** Confirm the PR merged the verified HEAD and
the task worktree is clean; delete only this task's branch (remote and local;
`git branch -D` is fine after a squash merge once those checks pass);
fast-forward local `main` and prune. Never delete unrelated branches, worktrees
or unmerged work; if the original checkout is dirty, leave it intact.

## 7. Boundaries

`templates/`, `static/css/`, `static/js/` are the UI; `web/` handles HTTP and
form processing; `pipeline/` orchestrates external tools; `jobs.py` owns the
queue and log parsers. Keep the web panel and pipeline imports free of heavy
training dependencies. Maintain the English and Traditional Chinese docs
together.
