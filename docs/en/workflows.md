# Development, Deployment, and Daily Use

[Documentation](README.md) · [Context router](context-router.md) · [繁體中文](../zh-TW/workflows.md)

Commands run from the repository root unless stated otherwise. Read one language
version; use the context router to open only the component needed for the task.

## Choose a workflow

| Workflow | Starting point | Result and verification |
| :--- | :--- | :--- |
| Development | Isolated task branch/worktree and development environment | Reviewed PR; `lint-type-test` passes on the merged HEAD |
| Deployment | Clean checkout of verified `main`, machine config and external tools | Running panel, optional persistent service, authenticated HTTPS proxy |
| Daily use | Existing deployment and a ready backend | Tracked experiment outputs and quality checks |

The FastAPI panel (`app.py`, `web/`, `jobs.py`) manages forms, job state, and logs.
`pipeline/` launches external tools/trainers; CUDA training dependencies belong in
the backend's own environment. nginx proxies the panel; the optional systemd user
service keeps it running. CI validates code and **does not deploy the server**.

## 1. Develop a change

1. Read [claude.md](../../claude.md), the [agent guide](agent-guide.md), then the
   [context router](context-router.md). Follow [AGENT.md](../../AGENT.md).
   Use a short-lived `Feature/`, `Bugfix/`, or `Enhance/` branch. Keep unrelated
   changes intact; use an isolated worktree when the checkout is dirty.
2. Create or select a Python 3.10+ development environment, then install:

   ```bash
   python -m pip install -e '.[dev]'
   ```

   For local panel testing, configure that checkout's `local.env` with the intended
   `CONDA_ROOT`/`CONDA_ENV`, an unused `PORT`, and separate `RECON_STUDIO_DATA` and
   experiment output paths. `run.sh` chooses its configured interpreter; activating
   a different shell environment alone does not select it. Run `./run.sh --doctor`
   and `./run.sh` when runtime validation is needed. Do not share the production job
   directory with a second panel process.
3. Edit only the routed files and preserve endpoint, form, and log contracts.
   Run relevant checks during iteration and the repository checks before merge:

   ```bash
   ruff check .
   mypy pipeline/config.py
   pytest
   ```

   These checks run without GPU training. UI changes also need desktop/narrow layout,
   keyboard, and affected-interaction checks; report physical-device gaps explicitly.
4. Commit with a concrete English `feat:`, `fix:`, `enhance:`, or `docs:` subject.
   Push the task branch and open a detailed PR: English behavior, tradeoffs,
   validation and pending physical-device checks, followed by a Traditional Chinese
   summary. CI runs on PRs and pushes to `main`; its job is `lint-type-test`.
5. Check required reviews/checks and merge the verified HEAD, normally by squash
   with `(#PR)` in the subject. Never push directly to `main` or bypass protection.
   Confirm the merge before deleting this task's local/remote branch; fast-forward
   `main` and prune. Deployment is the separate procedure below.

## 2. Deploy or update the workstation

### First installation

1. Obtain the repository, Git, Conda, and external tools required for the intended
   stages (ffmpeg, COLMAP, selected GPU trainer). Run `./setup.sh` to create/reuse
   the default `rec` panel environment and install panel requirements. Existing
   `local.env` is preserved; proposed settings may be written to `local.env.detected`.
   Setup does not install trainer builds, nginx, or the user service.
2. Review `local.env`: tool paths, `CONDA_ROOT`/`CONDA_ENV`, `HOST=127.0.0.1`, persistent
   `PORT`, storage/browse roots, and concurrency. Use `backends.json` only for machine
   overrides; built-in backends live in `pipeline/backends.py`.
3. Run `./run.sh --doctor`. Exit `0` means required checks passed; optional backend
   warnings can remain, so confirm the **chosen** backend is ready. `--fast` skips
   backend CUDA probing. Doctor does not start the server or build SuperSplat.
4. Start `./run.sh` for a foreground trial and open the printed URL. Stop the trial
   with Ctrl+C before enabling a service on the same port.

### Persistent panel and LAN access

For a persistent deployment, install the provided user service as the service user:

```bash
mkdir -p ~/.config/systemd/user
cp infra/systemd/reconstudio.service ~/.config/systemd/user/
```

Before enabling it, edit the installed unit's `WorkingDirectory` and `ExecStart`
if the checkout is not at `~/repo/reconstudio`. It must point to the deployment
checkout, not a temporary development worktree. Then run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now reconstudio
sudo loginctl enable-linger "$USER"
systemctl --user status reconstudio
```

Configure authenticated HTTPS LAN access separately:

```bash
sudo scripts/deploy-nginx-lan.sh
./run.sh --doctor
```

The script installs/updates nginx configuration, TLS and basic authentication,
validates with `nginx -t`, then reloads nginx. It **does not start or install the
panel service**. The domain defaults to `recon.venraas.tw`, the HTTPS port to `443`,
and the IP fallback to `8443`; choose the intended domain with `--domain` and
configure its DNS. Certificate selection defaults to `auto`: an existing
Let's Encrypt certificate is preferred, otherwise mkcert is used. Clients must
trust the mkcert CA where applicable. See the
[certificate instructions (Traditional Chinese)](../zh-TW/user-guide.md#網址列的紅色不安全).

Keep the panel on loopback. The proxy reads numeric `PORT` from `local.env`
(default `8077`), or accepts `--panel-port`; rerun the installer after changing
that port. A shell-only override of the panel port does not reconfigure nginx.

### Update an existing deployment

1. Let active jobs finish or cancel them intentionally in the panel and confirm
   they have stopped; clear the queue before restarting. On startup, `jobs.py`
   treats saved `queued`/`running` jobs as failed, not automatically resumed jobs.
   Record the current commit and preserve machine configs, job records and outputs.
2. In a **clean deployment checkout**, stop the service (or foreground process),
   then update only from verified `main`:

   ```bash
   systemctl --user stop reconstudio &&
   git switch main &&
   git pull --ff-only origin main &&
   conda run -n rec python -m pip install -r requirements.txt &&
   ./run.sh --doctor &&
   systemctl --user start reconstudio
   ```

   Replace `rec` with the configured panel environment. Resolve relevant failures
   before starting. Preserve dirty work in its existing worktree rather than
   resetting or overwriting it. For foreground operation, omit systemctl and use
   `./run.sh` after stopping the old process.
3. If proxy/domain/port settings changed, rerun the deployment script. Verify the
   local panel, authenticated LAN page, live job updates, and a representative
   model on the intended client; inspect relevant service/build logs on failure.
4. If rollback is needed, stop after draining jobs, use a clean checkout of the
   recorded known-good commit, reinstall its requirements, and point the user
   service at that checkout. Reuse compatible machine settings/data deliberately;
   verify before resuming work. Preserve failed-release logs and outputs. Do not
   reset dirty work or rewrite published `main` history.

### Change-to-action reference

| Change | Apply and verify |
| :--- | :--- |
| Python or `local.env` | Drain jobs, restart the panel, rerun doctor |
| Panel dependencies | Install requirements in the configured environment, then restart |
| Templates, `static/js/`, `static/css/` | Refresh the browser and exercise the changed interaction |
| SuperSplat patches/version | Build the editor, refresh, reopen a model, verify displayed version |
| nginx templates, domain, panel port | Rerun deploy script, validate/reload proxy, test authenticated URL |
| systemd unit paths/settings | Update installed unit, `daemon-reload`, restart, inspect status |

SuperSplat normally checks the latest stable release in the background at panel
startup. This requires git/Node/npm and network access for uncached dependencies.
Until a successful build finishes, the previous bundle remains in use; a failed
build leaves it intact. Inspect `<RECON_STUDIO_DATA>/supersplat_build.log` and
`static/supersplat/.version`. Pin `SUPERSPLAT_VER` for reproducible deployments;
`SUPERSPLAT_AUTOUPDATE=0` disables startup synchronization. For an explicit manual
latest build, run `SUPERSPLAT_VER=latest ./tools/build_supersplat.sh`; this script
does not source `local.env`, so pass the intended version/environment explicitly.

## 3. Use the deployed panel

For button-by-button instructions and a first photo-to-model run, see the
[after-startup walkthrough](usage.md).

1. If the service is already running, open its URL; do not launch a duplicate
   `run.sh` on the same port. For an ad hoc session, run `./run.sh` and keep its
   terminal open. Use doctor when a required tool/backend is unavailable.
2. Select local images/video or transfer GCS inputs to local storage. For video,
   extract frames. Optional masking/fusion precedes COLMAP; depth/normals used for
   training must match the training images. Use separate outputs for comparisons.
3. Run COLMAP, inspect the reconstruction, then train using its undistorted
   output and a ready backend. Scene partitioning is optional. Extract a mesh only
   with a supporting backend. Follow the panel's next-stage path handoff.
4. Track the job ID, parameters, status, output location, and relevant log lines.
   A completed job needs quality evaluation: follow [Evaluate](README.md#evaluate)
   for geometric checks and the backend workflow for rendering metrics.
5. Open the result in the appropriate viewer. SuperSplat v3 requires a secure
   context and a usable WebGPU adapter; the panel offers an explicit legacy WebGL
   choice when needed. Report browser/GPU and loading errors separately from
   model-quality findings.

## Diagnose the correct layer

| Symptom | Targeted check |
| :--- | :--- |
| Port already occupied | Existing service/process and configured port; do not start another panel |
| LAN 502 or WebSocket 502 | `systemctl --user status reconstudio`, `./run.sh --doctor`, proxy target and `journalctl --user -u reconstudio -n 80` |
| Authentication/certificate error | Intended hostname, nginx credentials, DNS and certificate trust |
| Viewer shows an old version | Background build log, deployed `.version`, then refresh/reopen |
| SuperSplat startup failure | Browser console, secure context/WebGPU adapter, explicit legacy choice |
| Job failed | That job's `job.json`/`console.log`, source/output paths and selected backend |

Panel `HOST`/`PORT` precedence is `local.env` → `RECON_STUDIO_HOST/PORT` → eligible
inherited bare values → `127.0.0.1:8077`. `run.sh` accepts bare HOST values matching
its address check and numeric PORT values; invalid compiler HOST strings are ignored.
The `8078` examples in the operator manual describe this workstation's override.
Pin deployment settings in `local.env` and use the actual startup/doctor output.
