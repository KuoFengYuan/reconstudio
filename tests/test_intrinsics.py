"""Vendor calibration certificate -> COLMAP camera parameters.

The value here is the unit conversion and the discovery contract: a certificate
is found by parsing candidate documents rather than by matching filenames, and
only the parts that convert unambiguously (focal length, principal point, sensor
size) are used — distortion is left to the bundle on purpose.
"""
from __future__ import annotations

import json

from pipeline.colmap._intrinsics import (
    Calibration,
    calibrations_from_file,
    discover_calibrations,
    match_to_cameras,
    parse_australis_report,
)

# The shape pdftotext -layout produces for an Australis certificate.
REPORT = """
 3. Nadir Camera Calibration Report:
 Camera:                        PhaseOne iXM RS 150F Nadir (SN: MH012066)
 Sensor Resolution:             x 14204 * y 10652 (150 MP)
 Pixel Size:                    3.76 um
 Camera Variable    Initial Value    Total Adjustment   Final Value
 C                  89.8040          0.00000            89.8040
 XP                   0.0005         0.00000              0.0005
 YP                 - 0.4069         0.00000            - 0.4069
 K1                   7.57269e-06    0.000e-000           7.57269e-06

 4. Forward Camera Calibration Report
 Camera:                        PhaseOne iXM RS 100 Forward (SN: ML011073)
 Sensor Resolution:             x 11664 * y 8750 (100 MP)
 Pixel Size:                    3.76 um
 C                  108.8209        0.00000             108.8209
 XP                   0.2707        0.00000               0.2707
 YP                   0.3616        0.00000               0.3616
"""


def test_parses_each_head_with_its_own_sensor_and_focal():
    cals = {c.name: c for c in parse_australis_report(REPORT)}
    assert set(cals) == {"Nadir", "Forward"}
    # the two heads genuinely differ — one shared EXIF guess cannot serve both
    assert (cals["Nadir"].width, cals["Nadir"].height) == (14204, 10652)
    assert (cals["Forward"].width, cals["Forward"].height) == (11664, 8750)
    assert cals["Nadir"].c_mm == 89.8040
    assert cals["Forward"].c_mm == 108.8209


def test_focal_length_converts_from_mm_and_pixel_pitch():
    cal = next(c for c in parse_australis_report(REPORT) if c.name == "Nadir")
    assert cal.pixel_size_mm == 0.00376          # 3.76 um
    assert round(cal.focal_px, 1) == 23884.0     # 89.8040 / 0.00376


def test_negative_yp_reads_as_below_centre_in_image_coordinates():
    # the certificate's y points up, COLMAP's image y points down, so a negative
    # YP has to come out as a LARGER row index than the centre
    cal = next(c for c in parse_australis_report(REPORT) if c.name == "Nadir")
    cx, cy = cal.principal_point_px()
    assert round(cx, 1) == 7102.1                # 14204/2 + 0.0005/0.00376
    assert cy > cal.height / 2
    assert round(cy, 1) == 5434.2                # 10652/2 + 0.4069/0.00376
    # and the opposite convention is available for certificates already in image coords
    assert cal.principal_point_px(flip_pp_y=False)[1] < cal.height / 2


def test_colmap_params_follow_the_model_order_with_distortion_zeroed():
    cal = next(c for c in parse_australis_report(REPORT) if c.name == "Nadir")
    fx, fy, cx, cy, k1, k2, p1, p2 = cal.colmap_params("OPENCV")
    assert fx == fy == cal.focal_px
    assert (round(cx, 1), round(cy, 1)) == (7102.1, 5434.2)
    # deliberately not converted: the certificate's K/P/B are per-mm and use the
    # opposite sign convention, and OPENCV cannot express K3/B1/B2 at all
    assert (k1, k2, p1, p2) == (0.0, 0.0, 0.0, 0.0)
    assert len(cal.colmap_params("SIMPLE_RADIAL")) == 4
    assert len(cal.colmap_params("FULL_OPENCV")) == 12


def test_colmap_params_can_fall_back_to_the_image_centre():
    cal = next(c for c in parse_australis_report(REPORT) if c.name == "Nadir")
    _, _, cx, cy, *_ = cal.colmap_params("OPENCV", use_pp=False)
    assert (cx, cy) == (14204 / 2, 10652 / 2)


def test_heads_match_camera_ids_case_insensitively():
    cals = parse_australis_report(REPORT)
    hit, missed = match_to_cameras(cals, ["nadir", "forward", "left"])
    assert set(hit) == {"nadir", "forward"}
    assert missed == ["left"]                    # reported, not silently ignored


def test_discovery_finds_a_certificate_by_content_not_by_filename(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "readme.txt").write_text("nothing to see here", encoding="utf-8")
    # an unhelpful name on purpose: discovery must not depend on it
    (tmp_path / "images" / "scan_0001.txt").write_text(REPORT, encoding="utf-8")

    cals, src = discover_calibrations(tmp_path)
    assert src is not None and src.name == "scan_0001.txt"
    assert {c.name for c in cals} == {"Nadir", "Forward"}


def test_discovery_is_quiet_when_the_dataset_ships_no_calibration(tmp_path):
    (tmp_path / "a.txt").write_text("just notes", encoding="utf-8")
    assert discover_calibrations(tmp_path) == ([], None)


def test_a_hand_written_json_is_accepted_as_an_escape_hatch(tmp_path):
    p = tmp_path / "calib.json"
    p.write_text(json.dumps([{"name": "cam0", "width": 4000, "height": 3000,
                              "pixel_size_um": 2.4, "c_mm": 24.0}]), encoding="utf-8")
    cals = calibrations_from_file(p)
    assert len(cals) == 1
    assert round(cals[0].focal_px, 1) == 10000.0     # 24 / 0.0024


def test_a_malformed_json_row_is_skipped_not_fatal(tmp_path):
    p = tmp_path / "calib.json"
    p.write_text(json.dumps([{"name": "ok", "width": 100, "height": 50,
                              "pixel_size_um": 1.0, "c_mm": 10.0},
                             {"name": "broken"}]), encoding="utf-8")
    assert [c.name for c in calibrations_from_file(p)] == ["ok"]


def test_unsupported_camera_model_is_refused():
    cal = Calibration("x", 100, 50, 0.001, 10.0)
    try:
        cal.colmap_params("FISHEYE")
    except ValueError as exc:
        assert "FISHEYE" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_a_sibling_datasets_certificate_is_not_picked_up(tmp_path):
    """The parent is searched because vendors deliver the certificate one level up,
    but recursing there would reach sibling datasets and silently apply another
    survey's calibration — the worst kind of wrong, because it still runs."""
    (tmp_path / "survey_a").mkdir()
    (tmp_path / "survey_b").mkdir()
    (tmp_path / "survey_a" / "cert.txt").write_text(REPORT, encoding="utf-8")

    assert discover_calibrations(tmp_path / "survey_b") == ([], None)
    # ...while survey_a still finds its own
    cals, src = discover_calibrations(tmp_path / "survey_a")
    assert src is not None and len(cals) == 2


def test_a_certificate_one_level_above_the_dataset_is_still_found(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "cert.txt").write_text(REPORT, encoding="utf-8")
    cals, src = discover_calibrations(tmp_path / "images")
    assert src is not None and src.name == "cert.txt"
    assert len(cals) == 2
