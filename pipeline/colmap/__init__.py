"""COLMAP pipeline (package).

Public surface:
    COLMAP_STAGES   ordered list of skippable stage names
    COLMAP_DEFAULTS form-default values seeding the UI / CLI
    run_colmap(p, r)  the orchestrator (Runner-driven, with sentinel-based resume)

Internal layout (all `_`-prefixed = package-private):
    _run.py     run_colmap orchestrator + module constants
    _layout.py  image-folder layout detection (single / multi / nested)
    _gps.py     EXIF/GPS read (JPEG + TIFF) + APP1 grafting + pose_prior injection
    _resize.py  FullHD ffmpeg resize (parallel, EXIF-preserving)
"""
from ._run import COLMAP_DEFAULTS, COLMAP_STAGES, run_colmap

__all__ = ["COLMAP_DEFAULTS", "COLMAP_STAGES", "run_colmap"]
