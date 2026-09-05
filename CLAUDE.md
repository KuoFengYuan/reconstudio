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
sudo scripts/deploy-nginx-lan.sh # put the panel on the LAN behind nginx (https://recon.venraas.tw, TLS + basic auth)
systemctl --user restart reconstudio   # if the optional user service is installed
```

`./run.sh` prints the URLs that actually work — loopback, plus the nginx one once
that is installed — and refuses to start on a port something else already holds,
because the survivor on that port would be the OLD panel serving old code, which
is indistinguishable from "my change did nothing". `HOST`/`PORT` come from
local.env only; see the LAN section below for why they are never inherited.

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
- `web/routers/` — HTTP handlers returning htmx fragments (`pages`, `browse`, `create`, `jobs`, `viz`, `viewer`, `measure`, `matte`, `doctor`).
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

**Background removal (`matte`) follows the depth stage's shape, with three
deliberate differences.** `pipeline/matte.py` (torch-free) resolves the `sam`
conda env and shells out to `tools/sam_matte.py`; the pure maths lives in
`pipeline/matte_encode.py` (numpy only, loadable both from the foreign env and
from CI, like `moge3_encode.py`) and is pinned by `tests/test_matte_encode.py`.
The differences:

1. **Outputs land inside the folder you picked** (`resolve_matte_dataset`), under
   a single `no_bg/` root (`OUTPUT_ROOT`). `depth.resolve_dataset` returns the
   PARENT for a loose photo folder because LichtFeld scans for `depth/` next to
   `images/`; nothing downstream requires that here, and the parent of a photo
   folder is usually a folder of *other* datasets that would then share one
   `cutout/`. The `no_bg/` level exists so a plain photo directory does not get
   two loose output folders dropped in beside the originals.
   The `resize` option moves the whole run: `matte.resize_target` sends it to
   `images_<cap>/` (COLMAP's resize naming, and literally its
   `resize_to_fullhd`), and since `output_path_for` always writes to
   `dataset_root/no_bg/…`, making that copy its own dataset_root is what keeps
   each resolution's cut-outs separate instead of overwriting each other.
   `create_matte` must call `resize_target` too — `job.meta` is written once at
   submit and never rewritten, so the live strip and result links would
   otherwise point at the un-resized folder.
2. **Two output folders, and every COLMAP/trainer consumer wants `masks/`.**
   `cutout/` is for looking at and for tools outside this pipeline.
   `_run.mask_source_dir` is the one place that knows this — a `cutout/` path
   resolves to its `masks/` sibling — and both mask consumers go through it:
   * `MASK_FEATURES` (`--ImageReader.mask_path`) makes SIFT skip the background
     entirely, and the extractor reads its mask as **greyscale**, so an RGBA
     cut-out would mask out every dark part of the subject.
   * `_stage_undistort`'s 5b pushes the masks through the same cameras as the
     images and cleans them with `large_scene.make_mask_uint8`. Inria's original
     reads `img[..., -1]` because it was fed the cut-outs, but this COLMAP's
     `Bitmap::Read` **drops alpha unconditionally** (4→3 channels, 2→1,
     `sensor/bitmap.cc`), so alpha never survives the undistort. `_mask_plane`
     therefore reads whichever plane actually carries the mask; a zero-mask
     export now fails the run instead of finishing with an empty `masks/`.

   `MASK_FEATURES` changes what gets *reconstructed*, not just what gets
   written — right for a single object shot in two orientations (the room is not
   rigid with respect to a subject that was picked up and flipped), wrong for a
   scene. `CUTOUT_IMAGES` then bakes the exported masks into `<dataset>/images/`
   in place (`large_scene.apply_masks_to_images`), for the trainers/viewer/mesh
   tools that only read `images/`; it is idempotent, so a redo cannot compound.
   `_mask_geometry_check` fails the run up front when masks and images are
   at different longest sides, which used to surface only as `image_undistorter`
   dying on a CHECK after the mapper had already run.
3. **A new SAM variant is a new `MaskRunner` subclass** plus one line in
   `build_runner()`. Everything version-specific — package name, checkpoint
   format, prompt encoding, output layout — stays inside the subclass;
   `masks_for_boxes()` takes an `(N, 4)` array as its only shape so that
   "one `set_image`, one batched decode" cannot be written wrong.

Three failure modes in this stage are silent — they produce a valid PNG that is
simply wrong — so each has a guard and a test rather than a comment:

- **Edge colour.** A feathered alpha leaves the *background's* RGB under the
  semi-transparent band, which composites as the dark fringe people blame on the
  segmentation. `compose_rgba` bleeds foreground colour outward first, and runs
  the diffusion until the band is covered rather than for a fixed count (a fixed
  count leaves the outermost, most visible rim un-bled).
- **EXIF orientation.** `cv2.imread` applies it, PIL does not, and SAM 2's video
  reader is PIL — so a rotated JPEG is tracked sideways and every mask comes back
  turned 90° against the photo it is composited onto. `_image_size` reports the
  cv2-visible size and `_stage_jpegs` bakes the rotation in rather than
  symlinking.
- **Mixed geometry.** SAM 2's video predictor takes a *video*: it resizes the
  sequence to the first frame's shape. `run_track` therefore groups frames by
  size and runs one session per group; a group with no box of its own is
  reported, never inferred from another group's masks.

`SKIP_DIR_NAMES` (this tool's outputs plus COLMAP's subtrees) and `OUTPUT_ROOT`
are duplicated in `pipeline/matte.py` and `matte_encode.py` — the first cannot
import numpy, the second cannot be imported from the `sam` env — and the pairs
are pinned equal by a test.

**The GPU is not this stage's bottleneck — writing is.** On a 16 MP frame SAM
costs a few hundred ms and `finish_image` costs ~5.3 s, 4.3 s of it in
`compose_rgba`'s single-threaded colour bleed. So `WriteFanout` runs
`finish_image` on a thread pool (measured 4.7× at 8 workers; capped below the
core count because cv2's own decode/encode is already multi-threaded), workers
re-read each frame from disk rather than queueing decoded pixels (bounds memory
to `workers` frames — the producer runs 10-20× ahead), and **`WriteFanout` is
the only thing allowed to print a `[i/N]` line**: `jobs._parse_depth` sets the
panel's progress from the last one it saw, so a second printer or a raw loop
index would make the bar jump backwards. That is why `handle_empty` returns its
warning text instead of printing it. Inference batching across *photos*
(`--image-batch`, sam2 + json/full/auto only, `batchable()`) is a smaller,
separate win; its measured bf16 edge-noise caveat is in
`Sam2Runner.masks_for_image_batch`.

**Reviewing and repairing is part of the stage, not a separate tool.** A running
job polls `/ui/matte_live` for the newest cut-outs (a poll, not SSE: the images
are written by a subprocess, so there is no event to forward). `/matte_review`
grids them on switchable backdrops, and `/matte_repair` re-cuts a single frame
from a box plus **+/- point prompts** — the information a box cannot carry, since
the box is what produced the bad result in the first place. Repair submits an
ordinary `matte` job whose boxes file carries `only: [rel]`, so it reuses the
queue, the log parser and the overwrite rules rather than growing a second code
path. Both the review and live views serve PNG thumbnails through
`/api/matte/imagefile`, never the gallery's JPEG ones: JPEG has no alpha, and a
cut-out flattened onto black looks exactly like the fringe defect these pages
exist to find.

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

**The panel binds 127.0.0.1 and that is load-bearing too.** It has no login of its
own, browses `RECON_STUDIO_BROWSE_ROOT`, and spawns processes, so LAN access goes
through nginx (`scripts/deploy-nginx-lan.sh` renders
`infra/nginx/reconstudio.conf.template` to `https://recon.venraas.tw/`, TLS +
basic auth, plus an `--alt-port` IP fallback for when DNS doesn't cooperate)
rather than through `HOST=0.0.0.0`. Several things in that path are non-obvious,
and each has a test in `tests/test_lan_deploy.py`:

- **:443 is shared with another site by SNI, so the vhost there claims only its
  own `server_name`.** Two blocks claiming one name on one address:port is not an
  error — nginx picks a winner and the loser silently stops answering, which is
  how you take down an unrelated service while "only adding a site". The bare IP
  and `localhost` belong to that other site, so this config puts them on a second
  vhost on `--alt-port` (8443) instead, which doubles as the way in when DNS is
  broken. The installer refuses to shadow a name another enabled site already
  claims, and rolls the symlink back if `nginx -t` fails.
- **The two vhosts carry different certificates**, which is why `ssl_certificate`
  is in neither the shared snippet nor a single variable: no public CA signs a
  bare `192.168.x.x`, so the name-based vhost holds a Let's Encrypt cert
  (`scripts/issue-letsencrypt-cert.sh`, DNS-01 via Cloud DNS — which never needs
  Let's Encrypt to reach this host) while the IP fallback keeps mkcert's.
  `deploy-nginx-lan.sh --cert auto` switches the moment a real cert exists, and
  issuance stores a `--deploy-hook` so unattended renewal reloads nginx; without
  it the panel serves the old cert until it expires.
- **DNS-01 runs through the gcloud CLI, not a service account.**
  `certbot-dns-google` takes only a service-account key and only looks for the
  managed zone inside that key's own project — and granting a service account
  access to the zone is `setIamPolicy`, which `roles/editor` has at neither the
  project nor the zone level. So `scripts/acme-gcloud-hook.sh` is a certbot
  `--manual` hook that writes the TXT record as the operator, whose editor role
  already allows it. `certbot renew` keeps only the hook command, never the
  environment, so the hook reads `/etc/letsencrypt/acme-gcloud.env`. Zone lookup
  matches the parent's **live NS delegation**, not just `dnsName`: this domain has
  two stale zones with the same name in another project, and a challenge written
  into one of those is invisible to every resolver.
- **Until then the cert is mkcert's, so the padlock is red until a client
  installs the root CA.** That is a trust question, not a config bug — the SAN and the realm are
  what to check when someone reports "不安全" or a login that won't take (a 401
  you cannot get past usually means the URL landed on a different site sharing
  :443). README's 「網址列的紅色『不安全』」 has both ways out, including the
  Let's Encrypt DNS-01 route that works despite the private IP.
- **The upgrade map must not be `$connection_upgrade`.** `sirocco.conf` — an
  unrelated site on the same box — defines that variable at http context, and a
  second definition is `[emerg] duplicate`, which stops *every* site on the machine.
- **`proxy_read_timeout` must be long.** `/ws` and the SSE endpoints send nothing
  while the queue is idle, so nginx's 60s default drops the job bus and the page
  silently stops updating mid-run.
- **The installer reads `PORT` from local.env, never asks.** A proxy pointed at the
  panel's *previous* port 502s while `ss` still shows the panel listening — the
  exact shape of "the port is up but nobody can connect". `preflight.lan_proxy_check`
  reports the drift on /doctor, and run.sh's banner reports it at startup.
- **`HOST`/`PORT` are never inherited from the ambient environment.** conda's
  compiler activation exports `HOST=x86_64-conda-linux-gnu`; plenty of tooling
  exports `PORT`. run.sh unsets both before reading local.env and re-exports the
  resolved values as `RECON_STUDIO_HOST`/`RECON_STUDIO_PORT`, which is what
  preflight reads. The default port is `8077` in run.sh, setup.sh and
  local.env.example, and a test pins the three equal.

## Testing

Tests run fully offline. `tests/test_colmap_run.py` is the model to follow for pipeline
work: a `FakeRunner` records the argv of every `colmap <subcommand>` call and simulates
just enough on-disk output (a minimal sqlite DB, `sparse/0/*.bin`, the undistorted
dataset) for the next stage's existence checks to pass, with the image-reading and
re-encoding helpers monkeypatched out. Assert on the recorded argv and on
skip/sentinel/error paths.

When a test asserts the *absence* of a flag, also add one that asserts its presence when
enabled — an absence-only test passes even if the feature was never implemented.

`tests/test_matte_template.py` is the one suite that reads `templates/`. The box
picker is plain JS in `index.html`, where a stale copy of a helper is not a
syntax error anywhere — it just wins hoisting and silently replaces the live one
(which is how the picker once drew nothing while reporting "0 框"). It also pins
the form's `<option>` values against the pipeline's accepted values, since those
two lists live in different files.

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
- New job kind → a `run_*` in `pipeline/`, exported from `pipeline/__init__.py`,
  registered in `jobs.RUN_FUNCS` **and** `jobs.PARSERS`, a `POST /ui/<kind>` in
  `web/routers/create.py`, a branch in `templates/_jobstatus.html`, and a form +
  toolbar button in `templates/index.html` (whose `showForm` list must gain the
  name). `matte` is the most recent worked example of all six.
