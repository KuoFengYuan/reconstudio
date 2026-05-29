"""Planning + listing-cache internals of the GCS browser/downloader (pure, offline).

These two pieces decide *where bytes land* and *how snappy the picker feels*, so
their exact semantics matter even though neither touches the network:

  * ``gcs_multi_plan`` maps the gs:// folders a user multi-selected to LOCAL
    dest paths. The contract is subtle: it flattens away everything *above* the
    deepest folder common to every selection (the "last folder the user entered")
    but preserves the structure *below* it, so a cross-level pick still mirrors
    correctly and two leaves that share a name don't collide.
  * ``_ls_cache_put`` / ``_ls_cache_get`` are the in-memory result cache that
    makes navigating back up / revisiting a folder instant. Entries expire after
    ``_LS_TTL`` seconds measured by ``time.monotonic``, so a stale listing is
    never served. We monkeypatch the module's ``time`` reference to make both the
    fresh-hit and the expired-miss deterministic.

The cache is process-global module state, so each test clears it first to stay
isolated from the others (and from any real listing).
"""
from pathlib import Path

import pytest

import pipeline.gcs as gcs
from pipeline.gcs import gcs_multi_plan


# --------------------------------------------------------------------------- #
# gcs_multi_plan
# --------------------------------------------------------------------------- #
def test_multi_plan_empty_list_returns_empty():
    """No selections -> nothing to download (caller raises on the empty plan)."""
    assert gcs_multi_plan([], "/data") == []


def test_multi_plan_drops_blank_and_whitespace_only_srcs():
    """Blank / whitespace-only entries are filtered out before planning; here
    that leaves nothing, so the plan is empty rather than a bogus mapping."""
    assert gcs_multi_plan(["", "   "], "/data") == []


def test_multi_plan_keeps_only_shared_parent_folder():
    """The documented case: siblings under one folder map to
    <dest_root>/<that folder>/<leaf>, discarding the deep gs:// prefix above it."""
    plan = gcs_multi_plan(
        ["gs://bucket/project/2026/0520_ITRI/CAM_a",
         "gs://bucket/project/2026/0520_ITRI/CAM_b"],
        "/data",
    )
    assert plan == [
        ("gs://bucket/project/2026/0520_ITRI/CAM_a", str(Path("/data/0520_ITRI/CAM_a"))),
        ("gs://bucket/project/2026/0520_ITRI/CAM_b", str(Path("/data/0520_ITRI/CAM_b"))),
    ]


def test_multi_plan_strips_trailing_slashes_from_src_and_dest():
    """Trailing slashes are stripped so `cp -r` copies <parent>/<leaf> instead of
    dumping the folder's flattened contents; the returned src is the cleaned form."""
    plan = gcs_multi_plan(["gs://bucket/a/b/leaf/"], "/data")
    # single selection -> its own parent ('b') is the deepest shared folder.
    assert plan == [("gs://bucket/a/b/leaf", str(Path("/data/b/leaf")))]
    src, dest = plan[0]
    assert not src.endswith("/")
    assert not dest.endswith("/")


def test_multi_plan_single_src_uses_its_own_parent_as_container():
    """A lone selection has no peers to diverge from, so its immediate parent
    folder becomes the container and only the leaf is mirrored under dest_root."""
    assert gcs_multi_plan(["gs://bucket/leaf"], "/data") == [
        ("gs://bucket/leaf", str(Path("/data/bucket/leaf"))),
    ]


def test_multi_plan_cross_level_preserves_structure_below_shared_parent():
    """When selections sit at different depths the shared parent is the deepest
    folder common to *every* parent path, and structure *below* it is kept:
    'a' is common, so the deeper 'y/z' is preserved under <dest>/a."""
    plan = gcs_multi_plan(
        ["gs://bucket/a/x", "gs://bucket/a/y/z"],
        "/data",
    )
    assert plan == [
        ("gs://bucket/a/x", str(Path("/data/a/x"))),
        ("gs://bucket/a/y/z", str(Path("/data/a/y/z"))),
    ]


def test_multi_plan_no_common_parent_mirrors_full_path():
    """Selections from different buckets share nothing, so container is empty and
    each full gs:// body (bucket included) is mirrored under dest_root verbatim."""
    plan = gcs_multi_plan(["gs://b1/a", "gs://b2/c"], "/data")
    assert plan == [
        ("gs://b1/a", str(Path("/data/b1/a"))),
        ("gs://b2/c", str(Path("/data/b2/c"))),
    ]


def test_multi_plan_dest_root_is_honored():
    """dest_root is the local staging root every dest is joined under."""
    plan = gcs_multi_plan(["gs://bucket/p/leaf"], "/tmp/stage")
    assert plan == [("gs://bucket/p/leaf", str(Path("/tmp/stage/p/leaf")))]


# --------------------------------------------------------------------------- #
# _ls_cache_get / _ls_cache_put  (TTL keyed off time.monotonic)
# --------------------------------------------------------------------------- #
@pytest.fixture
def frozen_clock(monkeypatch):
    """Replace the module's monotonic clock with a hand-cranked one and start
    every cache test from an empty cache, so hits/expiry are fully deterministic."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(gcs.time, "monotonic", lambda: clock["now"])
    gcs._ls_cache.clear()
    yield clock
    gcs._ls_cache.clear()


def test_cache_put_then_get_returns_stored_dict(frozen_clock):
    """A put followed by a get for the SAME key (before TTL elapses) returns the
    exact dict that was stored."""
    data = {"prefix": "gs://bucket/a/", "dirs": [{"name": "x"}], "nfiles": 3}
    gcs._ls_cache_put("gs://bucket/a/", data)
    assert gcs._ls_cache_get("gs://bucket/a/") is data


def test_cache_get_unknown_key_returns_none(frozen_clock):
    """A key that was never stored is a miss, not a crash."""
    gcs._ls_cache_put("gs://bucket/a/", {"v": 1})
    assert gcs._ls_cache_get("gs://bucket/other/") is None


def test_cache_get_within_ttl_is_a_fresh_hit(frozen_clock):
    """Advancing the clock by less than _LS_TTL still serves the cached entry."""
    gcs._ls_cache_put("k", {"v": 1})
    frozen_clock["now"] += gcs._LS_TTL - 0.001       # still inside the window
    assert gcs._ls_cache_get("k") == {"v": 1}


def test_cache_get_after_ttl_is_an_expired_miss(frozen_clock):
    """Once more than _LS_TTL has elapsed the entry is treated as stale -> None,
    so the picker re-lists instead of showing an out-of-date folder."""
    gcs._ls_cache_put("k", {"v": 1})
    frozen_clock["now"] += gcs._LS_TTL + 1.0
    assert gcs._ls_cache_get("k") is None


def test_cache_get_at_exact_ttl_boundary_is_a_miss(frozen_clock):
    """Expiry uses a strict `now < exp` comparison, so the instant the TTL is
    reached the entry already counts as expired."""
    gcs._ls_cache_put("k", {"v": 1})
    frozen_clock["now"] += gcs._LS_TTL                # now == stored expiry time
    assert gcs._ls_cache_get("k") is None


def test_cache_put_overwrites_and_refreshes_expiry(frozen_clock):
    """Re-putting the same key replaces the value and restarts its TTL window."""
    gcs._ls_cache_put("k", {"v": "old"})
    frozen_clock["now"] += gcs._LS_TTL - 1.0          # old entry nearly stale
    gcs._ls_cache_put("k", {"v": "new"})              # refresh
    frozen_clock["now"] += gcs._LS_TTL - 1.0          # would have expired the old one
    assert gcs._ls_cache_get("k") == {"v": "new"}
