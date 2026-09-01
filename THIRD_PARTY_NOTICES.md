# Third-Party Notices

Recon Studio orchestrates external software that is installed separately (and,
for the frontend, vendored under `static/`). Each component is under its own
license — you are responsible for complying with all of them.

| Component | Role in Recon Studio | License |
|-----------|----------------------|---------|
| **GS-2M** ([ndming/GS-2M](https://github.com/ndming/GS-2M)) + its CUDA submodules (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`, `nvdiffrast`, `render-utils`) | training + mesh backend | **Gaussian-Splatting License (Inria & MPII) — non-commercial, research & evaluation only** |
| **3D Gaussian Splatting** ([graphdeco-inria](https://github.com/graphdeco-inria/gaussian-splatting)) | basis of GS-2M | **Gaussian-Splatting License (Inria & MPII) — non-commercial** |
| **LichtFeld Studio** ([MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio)) — optional training backend (MR-NF strategy) | training only (no mesh) | **GPL-3.0-or-later** |
| **COLMAP** | Structure-from-Motion / reconstruction | New BSD |
| **FFmpeg** | frame extraction + FullHD resize | LGPL-2.1+ / GPL (depends on the build) |
| **FastAPI, Uvicorn, Jinja2, python-multipart** | web panel runtime | MIT / BSD |
| **three.js, htmx** (vendored under `static/`) | 3D viewer / frontend | MIT |
| **SuperSplat** ([playcanvas/supersplat](https://github.com/playcanvas/supersplat)) (built + vendored under `static/supersplat/`; patched via `tools/supersplat-reconstudio.patch`) | in-browser 3DGS editor for background removal (去背) | MIT |
| **PlayCanvas engine / PCUI** (bundled into the SuperSplat build) | SuperSplat rendering + UI | MIT |

> The non-commercial restriction of the Gaussian-Splatting license effectively
> governs the **training** and **mesh** stages of the pipeline (which use GS-2M /
> 3DGS). The frames and COLMAP stages rely on BSD/LGPL tools. For commercial
> licensing of the Gaussian-Splatting technology, contact Inria — see GS-2M's
> `LICENSE.md`.
>
> **LichtFeld Studio (GPL-3.0)** is invoked only as a *separate compiled binary*
> (a subprocess; Recon Studio never links its code), so it is mere aggregation —
> it does not relicense Recon Studio. Recon Studio does not bundle or distribute
> the binary; it is built per-machine (like GS-2M) and located via `backends.json`.
> If you redistribute that binary, comply with GPL-3.0.
