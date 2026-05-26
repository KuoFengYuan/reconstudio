"""gs:// path helpers used by the GCS browser/downloader (pure, no network)."""
from pathlib import Path

from pipeline.gcs import default_dest, gcs_parent


def test_gcs_parent_walks_up_one_level():
    assert gcs_parent("") is None                      # already at the bucket list
    assert gcs_parent("gs://bucket") == ""             # bucket -> the bucket list
    assert gcs_parent("gs://bucket/a") == "gs://bucket/"
    assert gcs_parent("gs://bucket/a/b") == "gs://bucket/a/"
    assert gcs_parent("gs://bucket/a/b/") == "gs://bucket/a/"  # trailing slash ignored


def test_default_dest_mirrors_gs_path_under_root():
    assert default_dest("gs://bucket/a/c", "/data") == str(Path("/data/bucket/a/c"))
    assert default_dest("gs://bucket", "/data") == str(Path("/data/bucket"))
    assert default_dest("", "/data") == str(Path("/data"))
