"""Single source of truth for runtime configuration.

Every tunable that used to be a scattered ``os.environ.get(...)`` across app.py /
jobs.py / pipeline/* now lives here as one typed `Settings` object. `run.sh` still
loads `local.env` into the process environment (`set -a`); this just reads, types,
and validates those variables in one place.

Env-var names are kept **exactly as before** (via `validation_alias`) so existing
`local.env` files keep working — note the two historical prefixes:
`RECON_STUDIO_*` (storage/browse) and `COLMAP_PANEL_*` (tuning), plus the bare
binary names (`COLMAP_BIN`, `FFMPEG_BIN`, …). Adding a setting = add one field here.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent  # reconstudio/


class Settings(BaseSettings):
    """Process-wide config, read once from the environment at import time."""

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore", case_sensitive=False)

    # --- external tool binaries (PATH by default; override for odd installs) ---
    colmap_bin: str = Field("colmap", validation_alias="COLMAP_BIN")
    ffmpeg_bin: str = Field("ffmpeg", validation_alias="FFMPEG_BIN")
    ffmpeg_hwaccel: str = Field("cuda", validation_alias="FFMPEG_HWACCEL")
    gsutil_bin: str = Field("gsutil", validation_alias="GSUTIL_BIN")

    # --- conda / backends ---
    conda_root: str | None = Field(None, validation_alias="CONDA_ROOT")
    backends_file: Path = Field(
        default_factory=lambda: REPO_ROOT / "backends.json",
        validation_alias="COLMAP_PANEL_BACKENDS",
    )

    # --- storage / browsing ---
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".recon_studio",
        validation_alias="RECON_STUDIO_DATA",
    )
    browse_root: Path = Field(Path("/"), validation_alias="RECON_STUDIO_BROWSE_ROOT")
    dest_root: Path = Field(Path("/"), validation_alias="RECON_STUDIO_DEST_ROOT")
    gcs_root: str = Field("", validation_alias="RECON_STUDIO_GCS_ROOT")

    # --- concurrency / performance ---
    max_jobs: int = Field(4, validation_alias="COLMAP_PANEL_MAX_JOBS")
    resize_workers: int | None = Field(None, validation_alias="COLMAP_PANEL_RESIZE_WORKERS")

    @field_validator("browse_root", "dest_root")
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        return v.resolve()

    @field_validator("gcs_root")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("max_jobs")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        return max(1, v)

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def resolved_resize_workers(self) -> int:
        """FullHD-resize parallelism: the explicit setting, else a polite CPU-count cap."""
        if self.resize_workers and self.resize_workers > 0:
            return self.resize_workers
        return max(1, min((os.cpu_count() or 8), 32))


settings = Settings()
