"""Scale-check math on synthetic camera networks (no model files needed)."""
import math

import pytest

from pipeline.colmap._scale_check import analyze, enu_factors

LAT0, LON0, ALT0 = 23.93, 120.66, 1610.0


def _cams(scale=1.0, bias=None, noise=0.0, n_side=10, spacing=100.0):
    """Grid of cameras at ~1.6 km ellipsoidal height. GPS is exact; the model
    centers are GPS*scale (+ per-folder bias, + deterministic pseudo-noise)."""
    m_lat, m_lon = enu_factors(LAT0, ALT0)
    cams = []
    for i in range(n_side):
        for j in range(n_side):
            e, nn, u = i * spacing, j * spacing, 10.0 * math.sin(i + j)
            lla = (LAT0 + nn / m_lat, LON0 + e / m_lon, ALT0 + u)
            folder = "fwd" if (i + j) % 2 else "bwd"
            b = (bias or {}).get(folder, (0.0, 0.0, 0.0))
            jit = noise * math.sin(7 * i + 13 * j)
            ctr = (e * scale + b[0] + jit, nn * scale + b[1] - jit, u * scale + b[2])
            cams.append((f"{folder}/img_{i}_{j}.tif", ctr, lla))
    return cams


def test_perfect_scale_and_uniform_bins():
    r = analyze(_cams())
    assert r["scale_err_pct"] == pytest.approx(0.0, abs=0.01)
    assert r["resid_median_m"] == pytest.approx(0.0, abs=0.01)
    assert "warning" not in r
    assert all(b["median"] == pytest.approx(1.0, abs=1e-4) for b in r["bins"])


def test_detects_known_scale_error():
    r = analyze(_cams(scale=1.002, noise=1.0))
    assert r["scale_err_pct"] == pytest.approx(0.2, abs=0.05)


def test_height_correction_matters():
    # sea-level radii would be ~alt0/R ≈ 0.025 % off; the corrected factors must
    # round-trip ENU<->LLA to well under the check's sensitivity.
    m_lat, m_lon = enu_factors(LAT0, ALT0)
    m_lat0, _ = enu_factors(LAT0, 0.0)
    assert (m_lat / m_lat0 - 1) == pytest.approx(ALT0 / 6.37e6, rel=0.1)


def test_per_folder_bias_separated_from_map_error():
    r = analyze(_cams(bias={"fwd": (2.0, 0.0, 1.0), "bwd": (-2.0, 0.0, -1.0)}))
    rows = {f["folder"]: f for f in r["folders"]}
    assert rows["fwd"]["bias_m"] == pytest.approx(math.dist((2, 0, 1), (0, 0, 0)), abs=0.05)
    assert rows["fwd"]["spread_m"] == pytest.approx(0.0, abs=0.05)   # pure bias, no scatter
    # opposite folder biases must NOT register as a scale error
    assert abs(r["scale_err_pct"]) < 0.05


def test_scale_way_off_warns():
    assert "warning" in analyze(_cams(scale=2.0))


def test_too_few_cameras_rejected():
    with pytest.raises(ValueError):
        analyze(_cams(n_side=2))


def test_tiny_site_rejected():
    with pytest.raises(ValueError):                       # all baselines under the floor
        analyze(_cams(n_side=3, spacing=1.0))
