"""Validate the map's metric scale against the EXIF GPS the model was aligned to.

Pairwise camera-to-camera distance ratios (model / GPS) isolate SCALE from the
translation+rotation of the alignment and average per-camera GPS noise down over
long baselines; binning by baseline length exposes scale drift (block bending).
Per-camera residuals after the alignment give the absolute-accuracy picture, and
a per-folder breakdown separates systematic rig offsets (lever arm on oblique
cameras) from random GPS noise.

This is a CONSISTENCY check against the same GPS the aligner used — it cannot see
a bias shared by the GPS itself; report-grade accuracy still needs independent
check points. The result dict says so (`note`).
"""
from __future__ import annotations

import math
import statistics as st
from pathlib import Path

from ..model import read_images
from ._gps import image_gps_latlonalt

_WGS_A = 6378137.0                  # WGS84 semi-major axis
_WGS_E2 = 0.00669437999014          # WGS84 first eccentricity squared
_MIN_CAMS = 8                       # below this the estimate is meaningless
_MIN_PAIRS = 30
_MAX_CAMS = 600                     # stride-subsample above this to bound pair count


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
    pts = [(name, ctr, ((lla[1] - lon0) * m_lon, (lla[0] - lat0) * m_lat, lla[2] - alt0))
           for name, ctr, lla in cams]

    # --- scale: pairwise distance ratios over long baselines ---------------- #
    sub = pts[::max(1, math.ceil(len(pts) / _MAX_CAMS))]
    dmax = 0.0
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            dmax = max(dmax, math.dist(sub[i][2], sub[j][2]))
    floor = max(30.0, 0.15 * dmax)
    ratios: list[tuple[float, float]] = []                  # (model/GPS ratio, baseline)
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            dg = math.dist(sub[i][2], sub[j][2])
            if dg >= floor:
                ratios.append((math.dist(sub[i][1], sub[j][1]) / dg, dg))
    if len(ratios) < _MIN_PAIRS:
        raise ValueError(f"基線 >{floor:.0f}m 的相機對只有 {len(ratios)} 對，"
                         "場景太小或相機太少，無法驗證尺度。")
    rs = sorted(r for r, _ in ratios)
    med = st.median(rs)
    bins = []
    step = (dmax - floor) / 3 or 1.0
    for k in range(3):
        lo, hi = floor + k * step, floor + (k + 1) * step
        grp = [r for r, d in ratios if lo <= d < hi or (k == 2 and d == dmax)]
        if grp:
            bins.append({"lo_m": round(lo), "hi_m": round(hi),
                         "median": round(st.median(grp), 5), "n": len(grp)})

    # --- residuals after the alignment (mean offset removed) ---------------- #
    mc = [st.mean(p[1][k] for p in pts) for k in range(3)]
    mg = [st.mean(p[2][k] for p in pts) for k in range(3)]
    res = {name: tuple(ctr[k] - mc[k] - (g[k] - mg[k]) for k in range(3))
           for name, ctr, g in pts}
    norms = sorted(math.dist(r, (0.0, 0.0, 0.0)) for r in res.values())
    n = len(norms)
    med3d = norms[n // 2]

    # sensitivity floor of the scale estimate: per-camera noise vs network extent
    spread = math.sqrt(st.mean(
        (p[2][0] - mg[0]) ** 2 + (p[2][1] - mg[1]) ** 2 for p in pts))
    sens_pct = 100.0 * med3d / (spread * math.sqrt(n)) if spread > 0 else 0.0

    # per-folder systematic bias (rig lever arm shows up here, not as map error)
    folders: dict[str, list[str]] = {}
    for name, _, _ in pts:
        folders.setdefault(name.split("/")[0] if "/" in name else "(root)",
                           []).append(name)
    fo_rows = []
    if len(folders) >= 2:
        for fo, ns in sorted(folders.items()):
            if len(ns) < 3:
                continue
            mu = [st.mean(res[x][k] for x in ns) for k in range(3)]
            after = sorted(math.dist(res[x], mu) for x in ns)
            fo_rows.append({"folder": fo, "n": len(ns),
                            "bias_enu_m": [round(v, 2) for v in mu],
                            "bias_m": round(math.dist(mu, (0, 0, 0)), 2),
                            "spread_m": round(after[len(ns) // 2], 2)})

    out = {
        "n_cams": n, "baseline_floor_m": round(floor), "baseline_max_m": round(dmax),
        "n_pairs": len(ratios),
        "scale_ratio_median": round(med, 5),
        "scale_err_pct": round(100 * (med - 1), 3),
        "scale_iqr": [round(rs[len(rs) // 4], 5), round(rs[3 * len(rs) // 4], 5)],
        "sensitivity_pct": round(sens_pct, 3),
        "bins": bins,
        "resid_median_m": round(med3d, 2),
        "resid_p90_m": round(norms[int(0.9 * n)], 2),
        "resid_max_m": round(norms[-1], 2),
        "resid_axis_median_m": [round(sorted(abs(r[k]) for r in res.values())[n // 2], 2)
                                for k in range(3)],
        "folders": fo_rows,
        "note": "與 EXIF GPS 的一致性檢查（對齊用同一批 GPS）；報告級精度請用獨立檢核點。",
    }
    if abs(med - 1) > 0.05:
        out["warning"] = ("比值離 1 太遠——模型可能還沒做 GPS 對齊"
                          "（align 階段未跑完？），或 GPS 異常。")
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
