"""Characterization tests for pipeline.colmap._run.run_colmap.

run_colmap shells out to COLMAP ~9 times across stages; it can't be run for real
in CI (no colmap binary, no dataset). These tests instead drive it with a fake
Runner that RECORDS the argv of every `colmap <subcommand>` call (and simulates
just enough of COLMAP's on-disk outputs — database.db, sparse/0/*.bin, the
undistorted dataset — for the next stage's existence checks to pass). The heavy
helpers that read images (gps_coverage) or re-encode them (resize_to_fullhd) are
monkeypatched out, so everything runs offline with no colmap/ffmpeg/GPU/network.

They pin the OBSERVABLE behaviour — which colmap sub-commands run, in what order,
with which key flags, plus the skip/sentinel and error paths — so the run_colmap
refactor (splitting the monolith into per-stage functions) is provably
behaviour-preserving: these pass identically before and after.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pipeline.colmap import _run

# The slice of COLMAP's DB schema the pose-prior injectors touch, so the fake
# feature_extractor can hand the later stages a database they can actually write to.
_DB_SCHEMA = """
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


# --------------------------------------------------------------------------- #
# fake Runner: records colmap calls + simulates their outputs
# --------------------------------------------------------------------------- #
class FakeRunner:
    cancelled = False                         # the prior injectors poll this per image

    def __init__(self) -> None:
        self.calls: list[list[str]] = []      # every argv passed to .run()
        self.banners: list[str] = []
        self.logs: list[str] = []

    def log(self, msg: str = "") -> None:
        self.logs.append(str(msg))

    def banner(self, msg: str) -> None:
        self.banners.append(str(msg))

    def run(self, argv, **kw) -> int:
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        self._simulate(argv)
        return 0

    # produce the files the NEXT stage's `need(...)` / "missing" guards look for
    @staticmethod
    def _arg_after(argv: list[str], flag: str):
        return argv[argv.index(flag) + 1] if flag in argv else None

    def _simulate(self, argv: list[str]) -> None:
        sub = argv[1] if len(argv) > 1 else ""
        if sub == "feature_extractor":
            db = self._arg_after(argv, "--database_path")
            if db:
                # a real (if minimal) COLMAP DB: enough schema + one images row per
                # entry of image_list.txt for the pose-prior injectors to work against.
                con = sqlite3.connect(db)
                con.executescript(_DB_SCHEMA)
                con.execute("INSERT INTO cameras VALUES(1,2,100,100,X'',1)")
                lst = self._arg_after(argv, "--image_list_path")
                names = Path(lst).read_text().split() if lst else []
                for i, name in enumerate(names, 1):
                    con.execute("INSERT INTO images VALUES(?,?,1)", (i, name))
                con.commit()
                con.close()
        elif sub in ("global_mapper", "mapper", "pose_prior_mapper", "hierarchical_mapper"):
            out = self._arg_after(argv, "--output_path")        # ws/sparse
            if out:
                d0 = Path(out) / "0"
                d0.mkdir(parents=True, exist_ok=True)
                for f in ("cameras.bin", "images.bin", "points3D.bin"):
                    (d0 / f).write_bytes(b"")
        elif sub == "image_undistorter":
            out = self._arg_after(argv, "--output_path")        # dense_dir
            if out:
                sp = Path(out) / "sparse"
                sp.mkdir(parents=True, exist_ok=True)
                for f in ("cameras.bin", "images.bin", "points3D.bin"):
                    (sp / f).write_bytes(b"")
                (Path(out) / "images").mkdir(parents=True, exist_ok=True)

    # convenience views over the recorded calls
    def subcommands(self) -> list[str]:
        return [c[1] for c in self.calls if len(c) > 1]

    def argv_for(self, sub: str) -> list[str]:
        for c in self.calls:
            if len(c) > 1 and c[1] == sub:
                return c
        raise AssertionError(f"{sub} was not called; got {self.subcommands()}")


@pytest.fixture
def patched(monkeypatch):
    """Neutralise the image-reading / re-encoding / binary-probe side effects so
    run_colmap exercises pure control flow. Returns a small knob object."""
    class Knobs:
        gps = (0, 0)          # (n_gps, n_total) returned by gps_coverage; (0,0)->no GPS
        which = "/usr/bin/colmap"

    k = Knobs()

    def fake_gps_coverage(img_root, lines, r, workers):
        if k.gps == "full":
            return (len(lines), len(lines))
        return (0, len(lines))

    monkeypatch.setattr(_run, "gps_coverage", fake_gps_coverage)
    monkeypatch.setattr(_run, "resize_to_fullhd",
                        lambda img_root, lines, ws, force, r, preserve_exif=False,
                        max_size="1920": img_root)
    monkeypatch.setattr(_run.shutil, "which", lambda _name: k.which)
    return k


def _make_images(d: Path, n: int = 3) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"img_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


@pytest.fixture
def imgroot(tmp_path):
    root = tmp_path / "images"
    _make_images(root, 3)
    return root


def _params(imgroot: Path, ws: Path, vocab: Path, **over) -> dict:
    p = {
        "image_root": str(imgroot),
        "workspace": str(ws),
        "vocab_tree": str(vocab),     # exists -> the download path is never taken
    }
    p.update(over)
    return p


@pytest.fixture
def vocab(tmp_path):
    v = tmp_path / "vocab.bin"
    v.write_bytes(b"x")
    return v


# --------------------------------------------------------------------------- #
# happy path + stage ordering
# --------------------------------------------------------------------------- #
def test_default_pipeline_runs_all_stages_in_order(patched, imgroot, vocab, tmp_path):
    ws = tmp_path / "ws"
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, ws, vocab), r)
    # default: matcher=both, mapper=global, simplify/align/reorient OFF
    assert r.subcommands() == [
        "feature_extractor",
        "sequential_matcher", "vocab_tree_matcher",
        "view_graph_calibrator",
        "global_mapper",
        "image_undistorter",
    ]
    # no optional stages fired
    assert "image_deleter" not in r.subcommands()     # simplify off
    assert "model_aligner" not in r.subcommands()      # align off
    # final completion banner
    assert any("done. workspace=" in b for b in r.banners)


def test_dense_dir_and_paths(patched, imgroot, vocab, tmp_path):
    ws = tmp_path / "ws"
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, ws, vocab), r)
    fe = r.argv_for("feature_extractor")
    assert FakeRunner._arg_after(fe, "--database_path") == str(ws / "database.db")
    und = r.argv_for("image_undistorter")
    # dense_dir = <ws>/<dataset_name>_<mapper>_mapper, default training_dataset / global
    assert FakeRunner._arg_after(und, "--output_path") == str(ws / "training_dataset_global_mapper")


# --------------------------------------------------------------------------- #
# matcher variants
# --------------------------------------------------------------------------- #
def test_matcher_both_runs_sequential_then_vocab_without_loop_flags(patched, imgroot, vocab, tmp_path):
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, matcher="both"), r)
    seq = r.argv_for("sequential_matcher")
    assert "--SequentialMatching.loop_detection" not in seq   # loop flags only for pure 'sequential'
    assert "vocab_tree_matcher" in r.subcommands()


def test_matcher_sequential_adds_loop_detection(patched, imgroot, vocab, tmp_path):
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, matcher="sequential"), r)
    seq = r.argv_for("sequential_matcher")
    assert "--SequentialMatching.loop_detection" in seq
    assert "vocab_tree_matcher" not in r.subcommands()


def test_matcher_vocab_only(patched, imgroot, vocab, tmp_path):
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, matcher="vocab"), r)
    assert "sequential_matcher" not in r.subcommands()
    assert "vocab_tree_matcher" in r.subcommands()


def test_matcher_spatial_requires_gps_and_runs_spatial(patched, imgroot, vocab, tmp_path):
    patched.gps = "full"      # spatial needs 100% GPS coverage
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, matcher="spatial"), r)
    assert "spatial_matcher" in r.subcommands()


# --------------------------------------------------------------------------- #
# mapper variants
# --------------------------------------------------------------------------- #
def test_global_mapper_skips_no_calibrate(patched, imgroot, vocab, tmp_path):
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, mapper="global"), r)
    assert "view_graph_calibrator" in r.subcommands()
    assert "global_mapper" in r.subcommands()


def test_incremental_mapper_no_calibrate_and_gpu_ba(patched, imgroot, vocab, tmp_path):
    r = FakeRunner()
    # ba_gpu defaults on; incremental mapper should get --Mapper.ba_use_gpu
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, mapper="incremental"), r)
    assert "view_graph_calibrator" not in r.subcommands()   # global-only stage
    mp = r.argv_for("mapper")
    assert "--Mapper.ba_use_gpu" in mp


def test_pose_prior_mapper_needs_gps_and_emits_prior_flags(patched, imgroot, vocab, tmp_path):
    patched.gps = "full"
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, mapper="pose_prior"), r)
    mp = r.argv_for("pose_prior_mapper")
    assert "--prior_position_std_x" in mp
    assert "--overwrite_priors_covariance" in mp


# --------------------------------------------------------------------------- #
# idempotency / sentinels / force
# --------------------------------------------------------------------------- #
def test_existing_match_sentinel_skips_matching(patched, imgroot, vocab, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".match.done").touch()
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, ws, vocab), r)
    assert "sequential_matcher" not in r.subcommands()
    assert "vocab_tree_matcher" not in r.subcommands()


def test_existing_database_skips_extract(patched, imgroot, vocab, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "database.db").write_bytes(b"")
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, ws, vocab), r)
    assert "feature_extractor" not in r.subcommands()


def test_force_reruns_extract_despite_existing_database(patched, imgroot, vocab, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "database.db").write_bytes(b"")
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, ws, vocab, force=True), r)
    assert "feature_extractor" in r.subcommands()


# --------------------------------------------------------------------------- #
# validation / error paths (raise before any colmap call)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    {"mapper": "nope"},
    {"matcher": "nope"},
    {"camera_mode": "nope"},
])
def test_invalid_enum_raises_value_error(patched, imgroot, vocab, tmp_path, bad):
    r = FakeRunner()
    with pytest.raises(ValueError):
        _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, **bad), r)
    assert r.calls == []


def test_missing_image_root_raises(patched, vocab, tmp_path):
    r = FakeRunner()
    with pytest.raises(FileNotFoundError):
        _run.run_colmap(_params(tmp_path / "nope", tmp_path / "ws", vocab), r)


def test_no_images_raises(patched, vocab, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("x")      # a dir with no images and no subdirs
    r = FakeRunner()
    with pytest.raises(FileNotFoundError):
        _run.run_colmap(_params(empty, tmp_path / "ws", vocab), r)


def test_missing_colmap_binary_raises(patched, imgroot, vocab, tmp_path):
    patched.which = None        # shutil.which -> None
    r = FakeRunner()
    with pytest.raises(RuntimeError):
        _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab), r)


def test_gps_option_without_coverage_aborts(patched, imgroot, vocab, tmp_path):
    # pose_prior is a GPS option; with no EXIF GPS (default patched.gps) it must abort
    # BEFORE running any colmap stage.
    r = FakeRunner()
    with pytest.raises(RuntimeError):
        _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, mapper="pose_prior"), r)
    assert r.calls == []


# --------------------------------------------------------------------------- #
# exterior-orientation (EO) CSV priors
# --------------------------------------------------------------------------- #
def _eo_csv(tmp_path: Path, stems: list[str]) -> Path:
    """An EO CSV whose IDs are the (fake) image stems, positioned inside TWD97 TM2 121."""
    p = tmp_path / "eo.csv"
    rows = "\n".join(f"{s},215160.{i},2648079.{i},1610.4,0.0,0.0,180.0"
                     for i, s in enumerate(stems))
    p.write_text("ID,EASTING,NORTHING,ELLIPSOID HEIGHT,OMEGA,PHI,KAPPA\n" + rows + "\n")
    return p


def test_eo_csv_injects_priors_and_satisfies_the_gps_coverage_gate(
        patched, imgroot, vocab, tmp_path):
    # patched.gps stays at "no EXIF GPS anywhere", so pose_prior would normally abort —
    # the CSV alone must be enough to cover every image.
    csv = _eo_csv(tmp_path, [f"img_{i:03d}" for i in range(3)])
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, mapper="pose_prior",
                            pose_prior_csv=str(csv)), r)
    assert "pose_prior_mapper" in r.subcommands()
    assert any("EO pose priors: wrote 3" in b for b in r.banners)
    # the priors really landed in the DB COLMAP was handed
    con = sqlite3.connect(tmp_path / "ws" / "database.db")
    assert con.execute("SELECT count(*) FROM pose_priors").fetchone()[0] == 3
    assert [x[0] for x in con.execute(
        "SELECT DISTINCT coordinate_system FROM pose_priors")] == [0]
    con.close()


def test_eo_csv_that_matches_nothing_aborts_before_any_colmap_call(
        patched, imgroot, vocab, tmp_path):
    csv = _eo_csv(tmp_path, ["not_an_image"])
    r = FakeRunner()
    with pytest.raises(RuntimeError, match="matched none"):
        _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab,
                                pose_prior_csv=str(csv)), r)
    assert r.calls == []


def test_bad_eo_crs_aborts_before_any_colmap_call(patched, imgroot, vocab, tmp_path):
    csv = _eo_csv(tmp_path, ["img_000"])
    r = FakeRunner()
    with pytest.raises(ValueError, match="unknown POSE_PRIOR_CRS"):
        _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab,
                                pose_prior_csv=str(csv), pose_prior_crs="epsg:4326"), r)
    assert r.calls == []


def test_ra_use_gravity_adds_the_global_mapper_flag(patched, imgroot, vocab, tmp_path):
    csv = _eo_csv(tmp_path, [f"img_{i:03d}" for i in range(3)])
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, mapper="global",
                            pose_prior_csv=str(csv), ra_use_gravity=True), r)
    argv = r.argv_for("global_mapper")
    assert "--GlobalMapper.ra_use_gravity" in argv
    assert argv[argv.index("--GlobalMapper.ra_use_gravity") + 1] == "1"


def test_ra_use_gravity_composes_with_the_caspar_backend(patched, imgroot, vocab, tmp_path):
    csv = _eo_csv(tmp_path, [f"img_{i:03d}" for i in range(3)])
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, mapper="global",
                            pose_prior_csv=str(csv), ra_use_gravity=True,
                            ba_backend="caspar"), r)
    argv = r.argv_for("global_mapper")
    assert "--GlobalMapper.ra_use_gravity" in argv       # not clobbered by the backend flag
    assert "--GlobalMapper.ba_backend" in argv


def test_no_eo_csv_leaves_the_pipeline_untouched(patched, imgroot, vocab, tmp_path):
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab), r)
    assert not any("EO pose priors" in b for b in r.banners)
    assert "--GlobalMapper.ra_use_gravity" not in r.argv_for("global_mapper")
