"""Scale-check math on synthetic camera networks (no model files needed)."""
import math
import random

import pytest

# The noise-calibrated estimator needs numpy, which the panel keeps out of its
# base/CI install — skip this module when absent, mirroring test_blocksplit.py.
# Runs in full locally (the `rec` env has it).
pytest.importorskip("numpy")

from pipeline.colmap._scale_check import analyze, enu_factors  # noqa: E402

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


def _noisy_gps_cams(sigma=(3.0, 3.0, 3.0), n_side=10, spacing=6.0, vert=12.0, seed=7):
    """Model is EXACT (scale error is zero by construction); the GPS is what carries
    the noise. `spacing` is deliberately small so sigma is a large fraction of the
    site extent — the regime where the naive ratio reads several percent short."""
    rnd = random.Random(seed)
    m_lat, m_lon = enu_factors(LAT0, ALT0)
    cams = []
    for i in range(n_side):
        for j in range(n_side):
            e, nn, u = i * spacing, j * spacing, vert * math.sin(i + j)
            ge = e + rnd.gauss(0, sigma[0])
            gn = nn + rnd.gauss(0, sigma[1])
            gu = u + rnd.gauss(0, sigma[2])
            lla = (LAT0 + gn / m_lat, LON0 + ge / m_lon, ALT0 + gu)
            cams.append((f"f{(i + j) % 2}/img_{i}_{j}.tif", (e, nn, u), lla))
    return cams


def test_noise_calibration_recovers_zero_error():
    """The headline figure must read ~0 for an exact model even when GPS noise makes
    the raw ratio read badly short. This is the whole point of the calibration."""
    r = analyze(_noisy_gps_cams())
    assert r["scale_err_pct"] < -1.0                       # raw is badly biased low
    assert r["noise_bias_pct"] < -1.0                      # and the baseline sees it
    # calibrated headline lands within the reported sensitivity band of the truth
    assert abs(r["headline_err_pct"]) <= 2 * r["sensitivity_pct"]
    assert "warning" not in r                              # must not cry wolf


def test_vertical_dropped_when_noisier_than_the_flight():
    """Height noise >= true height spread -> vertical carries no signal, so the
    verdict has to fall back to horizontal instead of averaging noise in."""
    r = analyze(_noisy_gps_cams(sigma=(0.2, 0.2, 8.0), vert=1.0))
    assert r["vertical_usable"] is False
    assert r["headline_dims"] == "h"
    assert "vert_note" in r
    # and with clean, informative height the 3D estimate is kept
    r2 = analyze(_noisy_gps_cams(sigma=(0.2, 0.2, 0.2), vert=12.0))
    assert r2["vertical_usable"] is True
    assert r2["headline_dims"] == "3d"


def test_bins_gated_on_model_not_gps():
    """Binning on the noisy side makes the longest-baseline bin read short (pairs
    land there because noise stretched them). Gated on the model, a zero-error
    model's bins must stay flat — that is what makes real drift detectable."""
    r = analyze(_noisy_gps_cams())
    vals = [b["median"] for b in r["bins"]]
    assert len(vals) >= 2
    assert max(vals) - min(vals) < 0.02                    # flat, no fake bending
    assert r["drift_excess_pct"] == pytest.approx(0.0, abs=0.5)


def test_real_scale_error_survives_calibration():
    """Calibration must not eat a genuine error: a 3 % shrink still reports ~-3 %."""
    cams = [(n, tuple(c * 0.97 for c in ctr), lla)
            for n, ctr, lla in _noisy_gps_cams(sigma=(0.2, 0.2, 0.2))]
    r = analyze(cams)
    assert r["headline_err_pct"] == pytest.approx(-3.0, abs=0.5)


def test_too_few_cameras_rejected():
    with pytest.raises(ValueError):
        analyze(_cams(n_side=2))


def test_tiny_site_rejected():
    with pytest.raises(ValueError):                       # all baselines under the floor
        analyze(_cams(n_side=3, spacing=1.0))
