"""Validate the map's metric scale against the EXIF GPS the model was aligned to.

Pairwise camera-to-camera distance ratios (model / GPS) isolate SCALE from the
translation+rotation of the alignment and average per-camera GPS noise down over
long baselines; binning by baseline length exposes scale drift (block bending).
Per-camera residuals after the alignment give the absolute-accuracy picture, and
a per-folder breakdown separates systematic rig offsets (lever arm on oblique
cameras) from random GPS noise.

Three things make the naive "median of model/GPS distance ratios" read low, and
all three are corrected here — without them a *zero-error* model reports several
percent short whenever GPS noise is a meaningful fraction of the site extent:

1. Noise inflates the reference distance. E[d_gps^2] = d_true^2 + 2*sum(sigma^2):
   per-camera noise can only ever make a measured separation LONGER, so every
   ratio is biased below 1. Taking a median does not help — the bias lives inside
   each individual d_gps. Instead of a closed-form moment correction (which the
   baseline-length gate itself skews), we CALIBRATE: run the identical estimator
   on (GPS, GPS + synthetic noise at the measured sigma), where the true ratio is
   1.000 by construction, and subtract whatever it reports.
2. Binning on the noisy side dilutes. Pairs landing in the longest-baseline bin
   are preferentially those whose noise stretched them, so that bin reads extra
   short and fakes "block bending". Gating and binning therefore use the MODEL
   distance (the low-noise side), never the GPS distance.
3. A near-noise vertical axis is pure ballast. Drone GPS height is the worst axis
   (and worse still under a bridge deck); when its noise rivals the flight's true
   vertical extent, including U in a 3D distance adds noise with no signal, so
   the verdict falls back to the horizontal estimate.

This is a CONSISTENCY check against the same GPS the aligner used — it cannot see
a bias shared by the GPS itself; report-grade accuracy still needs independent
check points. The result dict says so (`note`).
"""
from __future__ import annotations

import math
import statistics as st
from pathlib import Path

import numpy as np

from ..model import read_images
from ._gps import image_gps_latlonalt

_WGS_A = 6378137.0                  # WGS84 semi-major axis
_WGS_E2 = 0.00669437999014          # WGS84 first eccentricity squared
_MIN_CAMS = 8                       # below this the estimate is meaningless
_MIN_PAIRS = 30
_MAX_CAMS = 600                     # stride-subsample above this to bound pair count
_N_CALIB = 5                        # noise-baseline trials (median = bias, spread = floor)
_CALIB_SEED = 20260804              # fixed: the same model must score the same twice
_VERT_SNR = 2.0                     # need true vertical extent > this * sigma_U to trust U


def enu_factors(lat0: float, alt0: float) -> tuple[float, float]:
    """Metres per degree of (lat, lon) at the cameras' ellipsoidal height.

    Uses the curvature radii AT alt0 (M+h, N+h): at h=1600 m the sea-level radii
    are ~0.025 % short, which would show up as a fake scale error of exactly the
    size this check is trying to measure."""
    s2 = math.sin(math.radians(lat0)) ** 2
    m_rad = _WGS_A * (1 - _WGS_E2) / (1 - _WGS_E2 * s2) ** 1.5
    n_rad = _WGS_A / (1 - _WGS_E2 * s2) ** 0.5
    return (math.pi * (m_rad + alt0) / 180.0,
            math.pi * (n_rad + alt0) * math.cos(math.radians(lat0)) / 180.0)


def _pair_dists(P: np.ndarray, dims: tuple[int, ...],
                iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Distances for the given index pairs, over the selected axes only."""
    Q = P[:, dims]
    return np.linalg.norm(Q[iu[0]] - Q[iu[1]], axis=1)


def _scale_est(src: np.ndarray, dst: np.ndarray, dims: tuple[int, ...],
               iu: tuple[np.ndarray, np.ndarray], n_bins: int = 3,
               gate: tuple[float, float] | None = None) -> dict | None:
    """median(d_src / d_dst) over long pairs, gated AND binned by d_src.

    `src` must be the low-noise side (the model). Gating/binning on the noisy side
    selects pairs whose noise stretched them and drags the ratio down — see the
    module docstring. `gate` pins (floor, dmax) to another run's values so the two
    sets of bins cover the same baseline ranges and can be read side by side.
    Returns None when too few pairs survive the gate."""
    ds = _pair_dists(src, dims, iu)
    dd = _pair_dists(dst, dims, iu)
    if gate:
        floor, dmax = gate
    else:
        dmax = float(ds.max()) if ds.size else 0.0
        floor = max(30.0, 0.15 * dmax)
    keep = (ds >= floor) & (ds <= dmax) & (dd > 0)
    if int(keep.sum()) < _MIN_PAIRS:
        return None
    r = ds[keep] / dd[keep]
    base = ds[keep]
    med = float(np.median(r))
    bins = []
    step = (dmax - floor) / n_bins or 1.0
    for k in range(n_bins):
        lo, hi = floor + k * step, floor + (k + 1) * step
        m = (base >= lo) & ((base < hi) if k < n_bins - 1 else (base <= hi))
        if m.any():
            bins.append({"lo_m": round(lo), "hi_m": round(hi),
                         "median": round(float(np.median(r[m])), 5), "n": int(m.sum())})
    q1, q3 = (float(v) for v in np.percentile(r, [25, 75]))
    return {"ratio": med, "err_pct": 100.0 * (med - 1.0),
            "floor": floor, "dmax": dmax, "n_pairs": int(keep.sum()),
            "bins": bins, "iqr": [round(q1, 5), round(q3, 5)]}


def _calibrate(gps: np.ndarray, sigma: np.ndarray, dims: tuple[int, ...],
               iu: tuple[np.ndarray, np.ndarray],
               gate: tuple[float, float]) -> tuple[float, float, list]:
    """What does this estimator report for a PERFECT model at this noise level?

    src = clean GPS, dst = GPS + N(0, sigma): the true ratio is exactly 1.000, so
    anything the estimator returns is its own bias. Returns
    (median bias %, half-spread across trials %, the median trial's bins)."""
    rng = np.random.default_rng(_CALIB_SEED)
    errs, bins = [], []
    for _ in range(_N_CALIB):
        noisy = gps + rng.normal(0.0, 1.0, gps.shape) * sigma
        e = _scale_est(gps, noisy, dims, iu, gate=gate)
        if e:
            errs.append(e["err_pct"])
            bins.append(e["bins"])
    if not errs:
        return 0.0, 0.0, []
    order = sorted(range(len(errs)), key=lambda i: errs[i])
    mid = order[len(order) // 2]
    spread = (max(errs) - min(errs)) / 2.0
    return errs[mid], spread, bins[mid]


def analyze(cams: list[tuple[str, tuple[float, float, float],
                             tuple[float, float, float]]]) -> dict:
    """Core math, file-free for testability.

    cams: (image name, model camera center XYZ, GPS (lat, lon, alt)) per camera,
    in the SAME frame the model was aligned to (ENU metres)."""
    if len(cams) < _MIN_CAMS:
        raise ValueError(f"有 GPS 的相機只有 {len(cams)} 台（至少需 {_MIN_CAMS}），無法驗證。")
    lat0 = st.mean(c[2][0] for c in cams)
    lon0 = st.mean(c[2][1] for c in cams)
    alt0 = st.mean(c[2][2] for c in cams)
    m_lat, m_lon = enu_factors(lat0, alt0)
    names = [c[0] for c in cams]
    mdl = np.array([c[1] for c in cams], dtype=float)
    gps = np.array([[(c[2][1] - lon0) * m_lon, (c[2][0] - lat0) * m_lat, c[2][2] - alt0]
                    for c in cams], dtype=float)
    n = len(cams)

    # --- residuals after the alignment (mean offset removed) ---------------- #
    res = (mdl - mdl.mean(0)) - (gps - gps.mean(0))
    norms = np.sort(np.linalg.norm(res, axis=1))
    med3d = float(norms[n // 2])
    # Per-axis RMS residual is our GPS noise estimate. It is an UPPER bound (it
    # also contains whatever model error exists), so the calibration below can
    # over-correct; that is why the corrected figure is always reported next to
    # its uncertainty band rather than on its own.
    sigma = np.sqrt((res ** 2).mean(0))

    # --- scale: pairwise ratios, 3D and horizontal, each noise-calibrated --- #
    stride = max(1, math.ceil(n / _MAX_CAMS))
    sub_mdl, sub_gps = mdl[::stride], gps[::stride]
    iu = np.triu_indices(len(sub_mdl), 1)

    e3 = _scale_est(sub_mdl, sub_gps, (0, 1, 2), iu)
    if e3 is None:
        floor = max(30.0, 0.15 * float(_pair_dists(sub_mdl, (0, 1, 2), iu).max()))
        raise ValueError(f"基線 >{floor:.0f}m 的相機對太少，"
                         "場景太小或相機太少，無法驗證尺度。")
    eh = _scale_est(sub_mdl, sub_gps, (0, 1), iu)

    b3, s3, cb3 = _calibrate(sub_gps, sigma, (0, 1, 2), iu, (e3["floor"], e3["dmax"]))
    bh, sh, cbh = (_calibrate(sub_gps, sigma, (0, 1), iu, (eh["floor"], eh["dmax"]))
                   if eh else (0.0, 0.0, []))

    # Vertical is only signal if the flight's true height spread beats its noise.
    u_spread = float(mdl[:, 2].std())
    vert_ok = u_spread > _VERT_SNR * float(sigma[2])
    use_h = bool(eh) and not vert_ok
    head, bias, scatter, cbins = ((eh, bh, sh, cbh) if use_h else (e3, b3, s3, cb3))
    head_dims = (0, 1) if use_h else (0, 1, 2)
    corrected = head["err_pct"] - bias

    # per-folder systematic bias (rig lever arm shows up here, not as map error)
    folders: dict[str, list[int]] = {}
    for i, nm in enumerate(names):
        folders.setdefault(nm.split("/")[0] if "/" in nm else "(root)", []).append(i)
    fo_rows = []
    if len(folders) >= 2:
        for fo, idx in sorted(folders.items()):
            if len(idx) < 3:
                continue
            mu = res[idx].mean(0)
            after = np.sort(np.linalg.norm(res[idx] - mu, axis=1))
            fo_rows.append({"folder": fo, "n": len(idx),
                            "bias_enu_m": [round(float(v), 2) for v in mu],
                            "bias_m": round(float(np.linalg.norm(mu)), 2),
                            "spread_m": round(float(after[len(idx) // 2]), 2)})
    # Two flights with different fixed offsets stretch every cross-flight pair, so
    # their differential over the site extent is a scale error we cannot rule out.
    # Measured on the HEADLINE axes only: a vertical offset cannot fake a scale
    # error in a horizontal-only estimate, and here it is the noisiest axis by far.
    fo_floor = 0.0
    if len(fo_rows) >= 2:
        mus = [np.array(f["bias_enu_m"])[list(head_dims)] for f in fo_rows]
        dmax_bias = max(float(np.linalg.norm(a - b)) for a in mus for b in mus)
        fo_floor = 100.0 * dmax_bias / head["dmax"] if head["dmax"] else 0.0

    sens = max(scatter, fo_floor, 0.05)

    # Drift (block bending) = how much MORE the bins spread than the noise baseline
    # says they should. The raw spread on its own says nothing: at this sigma even a
    # perfect model's bins wobble, and that wobble is exactly `bins_calib`.
    def _spread(bs: list) -> float:
        vals = [b["median"] for b in bs]
        return 100.0 * (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    drift = max(0.0, _spread(head["bins"]) - _spread(cbins))

    out = {
        "n_cams": n, "baseline_floor_m": round(head["floor"]),
        "baseline_max_m": round(head["dmax"]), "n_pairs": head["n_pairs"],
        # raw (uncalibrated) figures, kept for continuity with older reports
        "scale_ratio_median": round(e3["ratio"], 5),
        "scale_err_pct": round(e3["err_pct"], 3),
        "scale_err_pct_h": round(eh["err_pct"], 3) if eh else None,
        "scale_iqr": e3["iqr"],
        # the noise baseline: what a ZERO-error model scores at this sigma
        "noise_bias_pct": round(b3, 3),
        "noise_bias_pct_h": round(bh, 3) if eh else None,
        "gps_sigma_enu_m": [round(float(v), 2) for v in sigma],
        # headline: calibrated, on whichever axes carry signal
        "headline_dims": "h" if use_h else "3d",
        "headline_err_pct": round(corrected, 3),
        "sensitivity_pct": round(sens, 3),
        "calib_scatter_pct": round(scatter, 3),
        "folder_floor_pct": round(fo_floor, 3),
        "vertical_usable": vert_ok,
        "vert_spread_m": round(u_spread, 2), "vert_sigma_m": round(float(sigma[2]), 2),
        "bins": head["bins"], "bins_calib": cbins,
        "drift_excess_pct": round(drift, 3),
        "resid_median_m": round(med3d, 2),
        "resid_p90_m": round(float(norms[int(0.9 * n)]), 2),
        "resid_max_m": round(float(norms[-1]), 2),
        "resid_axis_median_m": [round(float(np.median(np.abs(res[:, k]))), 2)
                                for k in range(3)],
        "folders": fo_rows,
        "note": "與 EXIF GPS 的一致性檢查（對齊用同一批 GPS）；報告級精度請用獨立檢核點。",
    }
    if abs(corrected) > 5.0:                 # judged on the CALIBRATED figure, so
        out["warning"] = ("比值離 1 太遠——模型可能還沒做 GPS 對齊"   # noise alone can no
                          "（align 階段未跑完？），或 GPS 異常。")     # longer trigger it
    if not vert_ok:
        out["vert_note"] = (f"GPS 高程噪聲 ±{sigma[2]:.1f}m ≥ 實際垂直變化 "
                            f"{u_spread:.1f}m，垂直方向無有效訊號，尺度改用水平估計。")
    return out


def scale_check(model_dir: Path, img_roots: list[Path]) -> dict:
    """Read camera centers from the (GPS-aligned) sparse model and their GPS from
    the ORIGINAL images (the resized TIFF copies don't carry EXIF), then analyze.
    img_roots are tried in order per image (image_root, then ws/staging)."""
    imgs = read_images(model_dir / "images.bin")
    cams = []
    miss = 0
    for im in imgs:
        lla = None
        for root in img_roots:
            p = root / im["name"]
            if str(root) and p.is_file():
                lla = image_gps_latlonalt(p)
                break
        if lla and lla[2] is not None:
            cams.append((im["name"], tuple(im["center"]), lla))
        else:
            miss += 1
    if not cams:
        raise ValueError("原始影像讀不到 EXIF GPS（找不到檔案，或不含 GPS/高度）。")
    out = analyze(cams)
    out["n_no_gps"] = miss
    return out
