"""Pure-Python ports of extract_frames.sh and colmap_pipeline.sh.

External binaries (ffmpeg, colmap) are still invoked as subprocesses; only the
orchestration logic — input expansion, blur cutoff, NESTED_LAYOUT staging,
sentinels, stage skipping — lives in Python. Log output mirrors the original
shell scripts' format so the panel's progress parsers keep working.
"""
from .backends import available_backends, build_cli, doctor, get_backend, list_gpus  # noqa: F401
from .colmap import COLMAP_DEFAULTS, COLMAP_STAGES, run_colmap  # noqa: F401
from .config import settings  # noqa: F401
from .depth import DEPTH_DEFAULTS, depth_ready, resolve_dataset, run_depth  # noqa: F401
from .frames import FRAMES_DEFAULTS, run_frames  # noqa: F401
from .gcs import (  # noqa: F401
    default_dest,
    gcs_ls,
    gcs_multi_plan,
    gcs_parent,
    run_gcs_sync,
    run_gcs_upload,
)
from .matte import (  # noqa: F401
    MATTE_DEFAULTS,
    matte_ready,
    resolve_matte_dataset,
    run_matte,
)
from .moge3 import MOGE3_DEFAULTS, moge3_ready  # noqa: F401
from .runner import Cancelled, Runner  # noqa: F401
from .train import TRAIN_DEFAULTS, run_mesh, run_train  # noqa: F401
