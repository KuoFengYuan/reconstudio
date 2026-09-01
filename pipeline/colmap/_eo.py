"""Exterior-orientation (EO) pose priors from a surveyor-supplied CSV.

Aerial vendors ship the adjusted EO of the nadir head as a CSV of
``ID, EASTING, NORTHING, ELLIPSOID HEIGHT, OMEGA, PHI, KAPPA`` — position in a
projected CRS plus the photogrammetric rotation angles. This module turns that
into rows COLMAP can actually consume.

What COLMAP's DB can hold (``pose_priors``): position + position_covariance +
gravity + coordinate_system. **There is no rotation column** — so ω/φ/κ cannot
be written in as an orientation prior. Two of its three DOF survive as the
`gravity` vector (down, expressed in the *sensor* frame), which
``global_mapper --GlobalMapper.ra_use_gravity 1`` uses to constrain rotation
averaging; the heading (κ) is dropped. To use the full ω/φ/κ you need a
known-pose model + ``point_triangulator``, not the database.

Coordinate handling: the CSV's projected E/N are inverse-projected back to
WGS84 lat/lon and written as ``coordinate_system=0`` (same layout `_gps.py`
writes for EXIF), so COLMAP's own ``ConvertPosePriorsToENU`` builds the local
metric frame and handles earth curvature. Writing projected metres straight in
as CARTESIAN is also supported (``crs="cartesian"``) but bakes in the grid scale
factor (~100 ppm for TWD97 TM2) and up to ~1 m of curvature droop across a
multi-km block. COLMAP hard-fails on a DB that mixes coordinate systems
(``DatabaseCache::ConvertPosePriorsToENU``), so whichever is chosen must be used
for *every* prior in the DB.

Rig images: the CSV covers one head, but a 5-head rig exposes all cameras at the
same station. Non-nadir images are matched to a CSV station by nearest EXIF GPS
(stations are hundreds of metres apart, so this is unambiguous), which is safer
than parsing the filename convention — the IDs are not unique on the
strip/index part alone. Only the exactly-name-matched images get gravity,
since the oblique heads sit at fixed mount angles the CSV doesn't describe;
COLMAP's stratified rotation averaging explicitly handles a mix of images with
and without gravity.
"""
from __future__ import annotations

import csv
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

from ..runner import Runner
from ._gps import image_gps_latlonalt

# Transverse-Mercator presets: (semi-major a, inverse flattening 1/f, k0,
# central meridian, false easting, false northing).
TM_PRESETS: dict[str, tuple[float, float, float, float, float, float]] = {
    # TWD97 / TM2 zone 121 (EPSG:3826) — Taiwan main island, GRS80.
    "twd97_tm2_121": (6378137.0, 298.257222101, 0.9999, 121.0, 250000.0, 0.0),
    # TWD97 / TM2 zone 119 (EPSG:3825) — Penghu.
    "twd97_tm2_119": (6378137.0, 298.257222101, 0.9999, 119.0, 250000.0, 0.0),
    # UTM north zones, WGS84 (add more as needed).
    "utm51n": (6378137.0, 298.257223563, 0.9996, 123.0, 500000.0, 0.0),
}
# spellings accepted for the same preset
TM_ALIASES = {"epsg:3826": "twd97_tm2_121", "3826": "twd97_tm2_121",
              "epsg:3825": "twd97_tm2_119", "3825": "twd97_tm2_119",
              "twd97": "twd97_tm2_121", "epsg:32651": "utm51n"}


def resolve_crs(name: str) -> str:
    """Normalise a CRS spelling to a TM_PRESETS key, or 'cartesian'. Raises on
    anything unknown — a silently-wrong projection would put the priors km away."""
    key = (name or "").strip().lower().replace(" ", "").replace("-", "_")
    key = TM_ALIASES.get(key, key)
    if key in ("cartesian", "none", "local", "xyz"):
        return "cartesian"
    if key not in TM_PRESETS:
        raise ValueError(f"unknown POSE_PRIOR_CRS: {name!r} "
                         f"(known: {', '.join(sorted(TM_PRESETS))}, or 'cartesian')")
    return key


def tm_to_wgs84(easting: float, northing: float, preset: str) -> tuple[float, float]:
    """Inverse Transverse Mercator -> (lat, lon) in decimal degrees.

    Standard Snyder series (USGS PP1395), good to well under a millimetre inside
    a TM2/UTM zone — far tighter than any airborne EO. Pure stdlib so the COLMAP
    pipeline keeps its no-pyproj dependency footprint."""
    a, inv_f, k0, lon0_deg, fe, fn = TM_PRESETS[preset]
    f = 1.0 / inv_f
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    m = (northing - fn) / k0
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    mu = m / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    phi1 = (mu
            + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
            + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
            + (151 * e1**3 / 96) * math.sin(6 * mu)
            + (1097 * e1**4 / 512) * math.sin(8 * mu))
    sin1, cos1, tan1 = math.sin(phi1), math.cos(phi1), math.tan(phi1)
    c1 = ep2 * cos1 * cos1
    t1 = tan1 * tan1
    n1 = a / math.sqrt(1 - e2 * sin1 * sin1)
    r1 = a * (1 - e2) / (1 - e2 * sin1 * sin1) ** 1.5
    d = (easting - fe) / (n1 * k0)
    lat = phi1 - (n1 * tan1 / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ep2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ep2 - 3 * c1**2) * d**6 / 720)
    lon = math.radians(lon0_deg) + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ep2 + 24 * t1**2) * d**5 / 120) / cos1
    return math.degrees(lat), math.degrees(lon)


@dataclass(frozen=True)
class EO:
    """One CSV row: image stem + projected position + photogrammetric angles (deg)."""
    stem: str
    easting: float
    northing: float
    height: float
    omega: float
    phi: float
    kappa: float


def parse_eo_csv(path: Path) -> list[EO]:
    """Read the vendor EO CSV. Column names are matched case/space-insensitively
    so `ELLIPSOID HEIGHT` / `Ellipsoid_Height` / `HEIGHT` all work."""
    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    wanted = {"id": ("id", "name", "image", "imageid", "photo"),
              "e": ("easting", "x", "e"),
              "n": ("northing", "y", "n"),
              "h": ("ellipsoidheight", "height", "z", "alt", "altitude", "h"),
              "om": ("omega", "o"), "ph": ("phi", "p"), "ka": ("kappa", "k")}
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        rdr = csv.reader(fh)
        try:
            header = [norm(x) for x in next(rdr)]
        except StopIteration:
            raise ValueError(f"empty EO CSV: {path}") from None
        idx: dict[str, int] = {}
        for key, cands in wanted.items():
            for cand in cands:
                if cand in header:
                    idx[key] = header.index(cand)
                    break
        missing = [k for k in wanted if k not in idx]
        if missing:
            raise ValueError(f"EO CSV {path} is missing column(s) for {missing}; "
                             f"header was {header}")
        rows: list[EO] = []
        for lineno, rec in enumerate(rdr, start=2):
            if not rec or not rec[idx["id"]].strip():
                continue
            try:
                rows.append(EO(
                    stem=Path(rec[idx["id"]].strip()).stem,
                    easting=float(rec[idx["e"]]), northing=float(rec[idx["n"]]),
                    height=float(rec[idx["h"]]), omega=float(rec[idx["om"]]),
                    phi=float(rec[idx["ph"]]), kappa=float(rec[idx["ka"]])))
            except ValueError as exc:
                raise ValueError(f"EO CSV {path}:{lineno}: {exc}") from None
    if not rows:
        raise ValueError(f"EO CSV has no data rows: {path}")
    return rows


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def cam_from_world_rotation(omega: float, phi: float, kappa: float) -> list[list[float]]:
    """Photogrammetric ω/φ/κ (degrees) -> COLMAP's cam_from_world rotation matrix.

    Convention (Australis/BINGO, which this vendor's calibration report cites):
    ``R_object_from_image = Rz(κ)·Ry(φ)·Rx(ω)`` with the image frame x-right,
    y-**up**, z-out-of-image (away from the scene) and the object frame East /
    North / Up. COLMAP's camera frame is x-right, y-**down**, z-forward (into the
    scene), i.e. ``R_cam_from_image = diag(1,-1,-1)``, so

        R_cam_from_world = diag(1,-1,-1) · R_object_from_imageᵀ

    Sanity check for a nadir frame (ω=φ=0, κ=180): this yields diag(-1,1,-1),
    whose viewing direction in world is (0,0,-1) — straight down. If a vendor
    uses the other common convention (y-down image frame), the reconstruction
    comes out upside-down; verify against a trial run before trusting it.
    """
    o, p, k = map(math.radians, (omega, phi, kappa))
    rx = [[1, 0, 0], [0, math.cos(o), -math.sin(o)], [0, math.sin(o), math.cos(o)]]
    ry = [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
    rz = [[math.cos(k), -math.sin(k), 0], [math.sin(k), math.cos(k), 0], [0, 0, 1]]
    r = _mat_mul(rz, _mat_mul(ry, rx))                 # object_from_image
    sign = (1.0, -1.0, -1.0)
    return [[sign[i] * r[j][i] for j in range(3)] for i in range(3)]   # diag·Rᵀ


def gravity_in_camera(omega: float, phi: float, kappa: float) -> tuple[float, float, float]:
    """Unit "down" vector expressed in the camera frame — what `pose_priors.gravity`
    stores. World down in the East/North/Up object frame is (0,0,-1), so this is
    just the negated third column of `cam_from_world_rotation`. A nadir frame gives
    (0,0,1): gravity along the optical axis, as it should be for a downward-looking
    camera. COLMAP's `GravityAlignedRotation` asserts the vector is normalised."""
    r = cam_from_world_rotation(omega, phi, kappa)
    g = [-r[i][2] for i in range(3)]
    norm = math.sqrt(sum(x * x for x in g)) or 1.0
    return (g[0] / norm, g[1] / norm, g[2] / norm)


def map_names_to_eo(names: list[str], rows: list[EO], img_root: str, crs: str,
                    rig_match: bool, r: Runner, rig_max_dist: float = 5.0,
                    orig_names: dict[str, str] | None = None
                    ) -> dict[str, tuple[EO, bool]]:
    """Map image-list entries (paths relative to `img_root`) to CSV rows.

    Two passes: exact filename-stem match against the CSV `ID`, then — when
    `rig_match` is on and the CRS is projected — the remaining images are matched
    to the nearest CSV station by their own EXIF GPS, within `rig_max_dist` metres.
    That covers the other heads of a multi-camera rig, which share the exposure
    station but carry their own serial in the filename.

    Returns ``{relative_name: (EO, is_exact_name_match)}``; the flag gates gravity,
    which is only valid for the head the angles actually describe.

    `orig_names` maps a relative name back to the name it had before any restaging
    renamed it (the rig staging normalises filenames so COLMAP can group frames).
    Exact matching uses that original stem, because the CSV IDs are the vendor's
    filenames; without it every image would fall through to the fuzzy GPS pass and
    nothing would qualify for gravity.
    """
    by_stem = {row.stem: row for row in rows}
    orig_names = orig_names or {}
    out: dict[str, tuple[EO, bool]] = {}
    unmatched: list[str] = []
    for rel in names:
        stem = Path(orig_names.get(rel, rel)).stem
        row = by_stem.get(stem)
        if row is not None:
            out[rel] = (row, True)
        else:
            unmatched.append(rel)
    n_exact = len(out)

    if unmatched and rig_match and crs != "cartesian":
        # station table in WGS84 so it can be compared against EXIF directly
        stations = [(row, *tm_to_wgs84(row.easting, row.northing, crs)) for row in rows]
        lat0 = sum(s[1] for s in stations) / len(stations)
        m_per_deg_lat = 111132.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
        n_rig = 0
        for rel in unmatched:
            gps = image_gps_latlonalt(Path(img_root) / rel)
            if not gps:
                continue
            lat, lon, _ = gps
            best, best_d = None, float("inf")
            for row, slat, slon in stations:
                d = math.hypot((lat - slat) * m_per_deg_lat, (lon - slon) * m_per_deg_lon)
                if d < best_d:
                    best, best_d = row, d
            if best is not None and best_d <= rig_max_dist:
                out[rel] = (best, False)
                n_rig += 1
        r.log(f"EO CSV: {n_exact} exact name matches, {n_rig} rig-mate matches via EXIF "
              f"(<={rig_max_dist} m), {len(names) - len(out)} images without an EO prior")
    else:
        r.log(f"EO CSV: {n_exact}/{len(names)} images matched by filename stem"
              + ("" if rig_match else " (rig matching off)"))
    return out


def inject_eo_priors(db_path: Path, name_to_eo: dict[str, tuple[EO, bool]], crs: str,
                     std_xyz: tuple[float, float, float], with_gravity: bool,
                     r: Runner, overwrite: bool = True) -> tuple[int, int, int]:
    """Write the mapped EO rows into the DB `pose_priors` table.

    Row layout matches what COLMAP writes for EXIF JPEG GPS, so every consumer
    (spatial_matcher / pose_prior_mapper / model_aligner / the in-BA prior term)
    reads them identically: pose_prior_id = corr_data_id = image_id,
    corr_sensor_id = camera_id, corr_sensor_type = 0.

    `crs` decides the position encoding — a TM preset writes (lat, lon, ellipsoid
    height) with coordinate_system=0 (WGS84) and lets COLMAP do the ENU
    conversion; "cartesian" writes (E, N, h) with coordinate_system=1. Either way
    `std_xyz` (and thus the covariance) is in metres, which is how COLMAP
    interprets `position_covariance` regardless of the position encoding.

    `overwrite=True` replaces any prior COLMAP already derived from EXIF for these
    images — the adjusted EO is strictly better than the EXIF fix it was rounded
    into. Returns (n_written, n_gravity, n_images_without_eo).
    """
    sx, sy, sz = std_xyz
    cov = struct.pack("<9d", sx * sx, 0.0, 0.0, 0.0, sy * sy, 0.0, 0.0, 0.0, sz * sz)
    nan3 = struct.pack("<3d", *([float("nan")] * 3))
    coord_sys = 1 if crs == "cartesian" else 0

    db = sqlite3.connect(str(db_path))
    try:
        tbls = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "pose_priors" not in tbls or "images" not in tbls:
            r.log("eo_inject: DB has no pose_priors/images table — skipping")
            return 0, 0, 0
        n_written = n_grav = n_missing = 0
        for image_id, name, camera_id in db.execute(
                "SELECT image_id, name, camera_id FROM images").fetchall():
            hit = name_to_eo.get(name)
            if hit is None:
                n_missing += 1
                continue
            eo, exact = hit
            if coord_sys == 0:
                lat, lon = tm_to_wgs84(eo.easting, eo.northing, crs)
                pos = struct.pack("<3d", lat, lon, eo.height)
            else:
                pos = struct.pack("<3d", eo.easting, eo.northing, eo.height)
            grav = nan3
            if with_gravity and exact:
                grav = struct.pack("<3d", *gravity_in_camera(eo.omega, eo.phi, eo.kappa))
                n_grav += 1
            if not overwrite and db.execute(
                    "SELECT 1 FROM pose_priors WHERE corr_data_id=?", (image_id,)).fetchone():
                continue
            db.execute(
                "INSERT OR REPLACE INTO pose_priors(pose_prior_id, corr_data_id, "
                "corr_sensor_id, corr_sensor_type, position, position_covariance, "
                "gravity, coordinate_system) VALUES(?,?,?,?,?,?,?,?)",
                (image_id, image_id, camera_id, 0, pos, cov, grav, coord_sys))
            n_written += 1
        db.commit()
        return n_written, n_grav, n_missing
    finally:
        db.close()
