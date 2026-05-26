"""Settings reads the environment with the historical var names + sane fallbacks."""
from pathlib import Path

from pipeline.config import Settings


def test_defaults():
    s = Settings()
    assert s.colmap_bin == "colmap"
    assert s.ffmpeg_bin == "ffmpeg"
    assert s.ffmpeg_hwaccel == "cuda"
    assert s.max_jobs == 4
    assert s.gcs_root == ""


def test_legacy_prefixes(monkeypatch):
    # The two historical prefixes (COLMAP_PANEL_* / RECON_STUDIO_*) + bare names
    # must keep working, so old local.env files don't break.
    monkeypatch.setenv("COLMAP_PANEL_MAX_JOBS", "8")
    monkeypatch.setenv("RECON_STUDIO_BROWSE_ROOT", "/Disk0")
    monkeypatch.setenv("COLMAP_BIN", "/opt/colmap/colmap")
    s = Settings()
    assert s.max_jobs == 8
    assert s.browse_root == Path("/Disk0")
    assert s.colmap_bin == "/opt/colmap/colmap"


def test_max_jobs_clamped_to_at_least_one(monkeypatch):
    monkeypatch.setenv("COLMAP_PANEL_MAX_JOBS", "0")
    assert Settings().max_jobs == 1


def test_gcs_root_stripped(monkeypatch):
    monkeypatch.setenv("RECON_STUDIO_GCS_ROOT", "  gs://b/  ")
    assert Settings().gcs_root == "gs://b/"


def test_resize_workers_fallback_is_cpu_bound():
    # unset -> a polite CPU-count cap, always >= 1
    assert Settings().resolved_resize_workers() >= 1


def test_resize_workers_explicit(monkeypatch):
    monkeypatch.setenv("COLMAP_PANEL_RESIZE_WORKERS", "12")
    assert Settings().resolved_resize_workers() == 12


def test_jobs_dir_derives_from_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RECON_STUDIO_DATA", str(tmp_path / "rs"))
    s = Settings()
    assert s.jobs_dir == tmp_path / "rs" / "jobs"
