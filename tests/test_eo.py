"""Tests for pipeline.colmap._eo — surveyor exterior-orientation (EO) CSV priors.

Three things are worth pinning here, because each is silently wrong-able:
  * the ω/φ/κ -> COLMAP convention (a sign slip flips the model upside-down),
  * the inverse Transverse Mercator (a wrong preset puts priors kilometres off),
  * the DB row layout (COLMAP hard-fails on mixed coordinate systems, and reads
    gravity/covariance as raw little-endian doubles).
No colmap binary and no images are needed: the DB is built with plain sqlite in
COLMAP's own schema, and rig matching is driven through a stubbed EXIF reader.
"""
from __future__ import annotations

import math
import sqlite3
import struct
from pathlib import Path

import pytest

from pipeline.colmap import _eo

CSV_TEXT = (
    "ID,EASTING,NORTHING,ELLIPSOID HEIGHT,OMEGA,PHI,KAPPA\n"
    "N-1_0-61214,215160.245,2648079.265,1610.448,0.00877,-0.00039,-178.98278\n"
    "N-1_1-61220,215165.716,2646497.972,1610.145,0.00658,-0.03191,179.48189\n"
)

# COLMAP's pose_priors / images / cameras tables, verbatim enough for the inject.
_SCHEMA = """
CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER,
                      height INTEGER, params BLOB, prior_focal_length INTEGER);
CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                     camera_id INTEGER NOT NULL);
CREATE TABLE pose_priors (pose_prior_id INTEGER PRIMARY KEY NOT NULL,
                          corr_data_id INTEGER NOT NULL, corr_sensor_id INTEGER NOT NULL,
                          corr_sensor_type INTEGER NOT NULL, position BLOB,
                          position_covariance BLOB, gravity BLOB,
                          coordinate_system INTEGER NOT NULL);
"""


class FakeRunner:
    def __init__(self) -> None:
        self.logs: list[str] = []

    def log(self, msg: str = "") -> None:
        self.logs.append(str(msg))


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "eo.csv"
    p.write_text(CSV_TEXT, encoding="utf-8")
    return p


def _make_db(tmp_path: Path, names: list[str]) -> Path:
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.executescript(_SCHEMA)
    con.execute("INSERT INTO cameras VALUES(1,2,100,100,X'',1)")
    for i, name in enumerate(names, 1):
        con.execute("INSERT INTO images VALUES(?,?,1)", (i, name))
    con.commit()
    con.close()
    return db


def _priors(db: Path) -> dict[str, tuple]:
    con = sqlite3.connect(db)
    out = {}
    for name, pos, cov, grav, cs in con.execute(
            "SELECT i.name, p.position, p.position_covariance, p.gravity, "
            "p.coordinate_system FROM pose_priors p "
            "JOIN images i ON i.image_id = p.corr_data_id"):
        out[name] = (struct.unpack("<3d", pos), struct.unpack("<9d", cov),
                     struct.unpack("<3d", grav), cs)
    con.close()
    return out


# --------------------------------------------------------------------------- #
# CRS resolution + inverse projection
# --------------------------------------------------------------------------- #
def test_resolve_crs_accepts_aliases_and_rejects_unknown():
    assert _eo.resolve_crs("EPSG:3826") == "twd97_tm2_121"
    assert _eo.resolve_crs("TWD97-TM2-121") == "twd97_tm2_121"
    assert _eo.resolve_crs("cartesian") == "cartesian"
    with pytest.raises(ValueError, match="unknown POSE_PRIOR_CRS"):
        _eo.resolve_crs("epsg:4326")


def test_tm_inverse_matches_the_exif_fix_of_the_same_photo():
    # ground truth: the EXIF GPS this vendor wrote into N-1_0-61214.tif, which is the
    # same exposure station rounded to 1e-6 deg (~0.1 m).
    lat, lon = _eo.tm_to_wgs84(215160.245, 2648079.265, "twd97_tm2_121")
    assert lat == pytest.approx(23.936918333, abs=1e-7)
    assert lon == pytest.approx(120.657725, abs=2e-5)


def test_tm_inverse_is_exact_at_the_projection_origin():
    lat, lon = _eo.tm_to_wgs84(250000.0, 0.0, "twd97_tm2_121")
    assert (lat, lon) == pytest.approx((0.0, 121.0), abs=1e-9)


# --------------------------------------------------------------------------- #
# ω/φ/κ -> COLMAP rotation / gravity
# --------------------------------------------------------------------------- #
def test_nadir_frame_looks_straight_down():
    # ω=φ=0, κ=180: the camera z-axis (3rd row of cam_from_world) in world coords.
    r = _eo.cam_from_world_rotation(0.0, 0.0, 180.0)
    assert [r[2][i] for i in range(3)] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def test_rotation_is_orthonormal_and_right_handed():
    r = _eo.cam_from_world_rotation(-1.87952, 8.75117, 164.04769)
    for i in range(3):
        assert sum(r[i][k] ** 2 for k in range(3)) == pytest.approx(1.0)
        for j in range(i + 1, 3):
            assert sum(r[i][k] * r[j][k] for k in range(3)) == pytest.approx(0.0, abs=1e-12)
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
           - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    assert det == pytest.approx(1.0)


def test_gravity_of_a_nadir_frame_is_the_optical_axis():
    # a downward-looking camera sees gravity along its own +Z (viewing direction)
    assert _eo.gravity_in_camera(0.0, 0.0, 180.0) == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)
    assert _eo.gravity_in_camera(0.0, 0.0, 0.0) == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)


def test_gravity_tilts_by_the_tilt_angle_and_stays_normalised():
    g = _eo.gravity_in_camera(0.0, 8.75117, 0.0)
    assert math.sqrt(sum(x * x for x in g)) == pytest.approx(1.0)
    # angle off the optical axis == the φ tilt (COLMAP asserts a unit vector)
    assert math.degrees(math.acos(g[2])) == pytest.approx(8.75117, abs=1e-9)


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #
def test_parse_csv_reads_all_seven_fields(csv_path):
    rows = _eo.parse_eo_csv(csv_path)
    assert [r.stem for r in rows] == ["N-1_0-61214", "N-1_1-61220"]
    assert rows[0].easting == 215160.245
    assert rows[0].height == 1610.448
    assert rows[0].kappa == -178.98278


def test_parse_csv_tolerates_header_case_and_an_extension_in_the_id(tmp_path):
    p = tmp_path / "eo.csv"
    p.write_text("Image,X,Y,Z,Omega,Phi,Kappa\nA-1.tif,1,2,3,0,0,0\n", encoding="utf-8")
    assert _eo.parse_eo_csv(p)[0].stem == "A-1"


def test_parse_csv_rejects_a_missing_column(tmp_path):
    p = tmp_path / "eo.csv"
    p.write_text("ID,EASTING,NORTHING\nA,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        _eo.parse_eo_csv(p)


# --------------------------------------------------------------------------- #
# name -> EO mapping (exact + rig)
# --------------------------------------------------------------------------- #
def test_exact_stem_match_wins_and_is_flagged_exact(csv_path):
    rows = _eo.parse_eo_csv(csv_path)
    m = _eo.map_names_to_eo(["nadir/N-1_0-61214.tif"], rows, "/nowhere",
                            "twd97_tm2_121", rig_match=False, r=FakeRunner())
    assert m["nadir/N-1_0-61214.tif"][1] is True


def test_rig_mates_match_by_exif_gps_and_are_not_flagged_exact(csv_path, monkeypatch):
    rows = _eo.parse_eo_csv(csv_path)
    station = _eo.tm_to_wgs84(rows[0].easting, rows[0].northing, "twd97_tm2_121")
    # the other heads of the rig carry the same EXIF fix as the nadir exposure
    monkeypatch.setattr(_eo, "image_gps_latlonalt",
                        lambda p: (station[0], station[1], 1610.0))
    m = _eo.map_names_to_eo(["nadir/N-1_0-61214.tif", "forward/F-1_0-61294.tif"],
                            rows, "/nowhere", "twd97_tm2_121",
                            rig_match=True, r=FakeRunner())
    assert m["forward/F-1_0-61294.tif"][0].stem == "N-1_0-61214"
    assert m["forward/F-1_0-61294.tif"][1] is False        # no gravity for this one


def test_rig_match_ignores_images_far_from_every_station(csv_path, monkeypatch):
    rows = _eo.parse_eo_csv(csv_path)
    monkeypatch.setattr(_eo, "image_gps_latlonalt", lambda p: (24.5, 121.5, 0.0))
    m = _eo.map_names_to_eo(["other/X.tif"], rows, "/nowhere", "twd97_tm2_121",
                            rig_match=True, r=FakeRunner())
    assert m == {}


# --------------------------------------------------------------------------- #
# DB injection
# --------------------------------------------------------------------------- #
def test_inject_writes_wgs84_rows_with_metric_covariance(csv_path, tmp_path):
    rows = _eo.parse_eo_csv(csv_path)
    names = ["nadir/N-1_0-61214.tif"]
    db = _make_db(tmp_path, names)
    m = _eo.map_names_to_eo(names, rows, "/nowhere", "twd97_tm2_121",
                            rig_match=False, r=FakeRunner())
    n_written, n_grav, n_missing = _eo.inject_eo_priors(
        db, m, "twd97_tm2_121", (0.05, 0.05, 0.10), True, FakeRunner())
    assert (n_written, n_grav, n_missing) == (1, 1, 0)
    pos, cov, grav, cs = _priors(db)[names[0]]
    assert cs == 0                                     # WGS84 -> COLMAP does the ENU
    assert pos[0] == pytest.approx(23.936918333, abs=1e-7)
    assert pos[2] == 1610.448                          # ellipsoid height passes through
    # covariance is diag(std²) in METRES regardless of the position encoding
    assert cov == pytest.approx((0.0025, 0, 0, 0, 0.0025, 0, 0, 0, 0.01))
    assert grav == pytest.approx(_eo.gravity_in_camera(rows[0].omega, rows[0].phi,
                                                       rows[0].kappa))


def test_cartesian_mode_writes_projected_metres_unchanged(csv_path, tmp_path):
    rows = _eo.parse_eo_csv(csv_path)
    names = ["nadir/N-1_0-61214.tif"]
    db = _make_db(tmp_path, names)
    m = _eo.map_names_to_eo(names, rows, "/nowhere", "cartesian",
                            rig_match=False, r=FakeRunner())
    _eo.inject_eo_priors(db, m, "cartesian", (1.0, 1.0, 1.0), False, FakeRunner())
    pos, _cov, grav, cs = _priors(db)[names[0]]
    assert cs == 1
    assert pos == (215160.245, 2648079.265, 1610.448)
    assert all(math.isnan(x) for x in grav)            # gravity off


def test_rig_mates_get_a_position_but_no_gravity(csv_path, tmp_path, monkeypatch):
    rows = _eo.parse_eo_csv(csv_path)
    station = _eo.tm_to_wgs84(rows[0].easting, rows[0].northing, "twd97_tm2_121")
    monkeypatch.setattr(_eo, "image_gps_latlonalt",
                        lambda p: (station[0], station[1], 1610.0))
    names = ["nadir/N-1_0-61214.tif", "forward/F-1_0-61294.tif"]
    db = _make_db(tmp_path, names)
    m = _eo.map_names_to_eo(names, rows, "/nowhere", "twd97_tm2_121",
                            rig_match=True, r=FakeRunner())
    n_written, n_grav, _ = _eo.inject_eo_priors(
        db, m, "twd97_tm2_121", (0.1, 0.1, 0.2), True, FakeRunner())
    assert (n_written, n_grav) == (2, 1)
    got = _priors(db)
    assert got[names[0]][0] == pytest.approx(got[names[1]][0])   # same station
    assert all(math.isnan(x) for x in got[names[1]][2])          # oblique: no gravity


def test_inject_overwrites_an_existing_exif_prior(csv_path, tmp_path):
    rows = _eo.parse_eo_csv(csv_path)
    names = ["nadir/N-1_0-61214.tif"]
    db = _make_db(tmp_path, names)
    con = sqlite3.connect(db)                       # a stale EXIF-derived prior
    con.execute("INSERT INTO pose_priors VALUES(1,1,1,0,?,?,?,0)",
                (struct.pack("<3d", 1.0, 2.0, 3.0), struct.pack("<9d", *([0.0] * 9)),
                 struct.pack("<3d", *([float("nan")] * 3))))
    con.commit()
    con.close()
    m = _eo.map_names_to_eo(names, rows, "/nowhere", "twd97_tm2_121",
                            rig_match=False, r=FakeRunner())
    _eo.inject_eo_priors(db, m, "twd97_tm2_121", (0.1, 0.1, 0.2), True, FakeRunner())
    got = _priors(db)
    assert len(got) == 1                            # replaced, not duplicated
    assert got[names[0]][0][0] == pytest.approx(23.936918333, abs=1e-7)


def test_all_injected_rows_share_one_coordinate_system(csv_path, tmp_path, monkeypatch):
    # COLMAP's DatabaseCache::ConvertPosePriorsToENU THROW_CHECKs on a mixed DB.
    rows = _eo.parse_eo_csv(csv_path)
    station = _eo.tm_to_wgs84(rows[0].easting, rows[0].northing, "twd97_tm2_121")
    monkeypatch.setattr(_eo, "image_gps_latlonalt",
                        lambda p: (station[0], station[1], 1610.0))
    names = ["nadir/N-1_0-61214.tif", "forward/F-1_0-61294.tif", "left/L-1_0-50329.tif"]
    db = _make_db(tmp_path, names)
    m = _eo.map_names_to_eo(names, rows, "/nowhere", "twd97_tm2_121",
                            rig_match=True, r=FakeRunner())
    _eo.inject_eo_priors(db, m, "twd97_tm2_121", (0.1, 0.1, 0.2), True, FakeRunner())
    assert {p[3] for p in _priors(db).values()} == {0}
