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

import json
import sqlite3
from pathlib import Path

import pytest

from pipeline.colmap import _run
from pipeline.runner import PipelineError

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
    # default: matcher=vocab, mapper=global, rig/simplify/align/reorient OFF
    assert r.subcommands() == [
        "feature_extractor",
        "vocab_tree_matcher",
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


# --------------------------------------------------------------------------- #
# region subset: re-run the pipeline over an area picked in the viewer
# --------------------------------------------------------------------------- #
# The reference model is built so all three selection paths are exercised at once:
#   img_000  centre INSIDE  the region                  -> phase 1 (camera centre)
#   img_001  centre outside, sees only in-region points  -> phase 2 (visibility 1.0)
#   img_002  centre outside, sees only far-away points   -> dropped (in-region hull 0)
REGION_IN = "-5,-5,15,15"


def _ref_model(model_dir: Path) -> Path:
    """A 3-image / 6-point COLMAP model for the region tests (names match `imgroot`).

    Writing a model needs numpy, which the panel deliberately keeps out of its base/CI
    install (region selection imports it lazily for the same reason). Skip per-test
    rather than per-module here: the rest of this file is numpy-free and must keep
    running in CI. Mirrors test_blocksplit.py / test_scale_check.py, which skip whole.
    """
    np = pytest.importorskip("numpy")

    from pipeline.vendor.read_write_model import (
        Camera,
        Image,
        Point3D,
        write_cameras_binary,
        write_images_binary,
        write_points3D_binary,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    qvec = np.array([1.0, 0.0, 0.0, 0.0])              # identity R -> centre = -tvec
    near = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 10.0, 0.0)]     # inside REGION_IN
    far = [(100.0, 100.0, 0.0), (110.0, 100.0, 0.0), (105.0, 110.0, 0.0)]
    pts, tri = {}, np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 100.0]])   # non-collinear
    for i, xyz in enumerate(near + far, start=1):
        pts[i] = Point3D(id=i, xyz=np.array(xyz), rgb=np.array([1, 2, 3]), error=1.0,
                         image_ids=np.array([1]), point2D_idxs=np.array([0]))
    spec = [("img_000.jpg", (5.0, 5.0, 50.0), [1, 2, 3]),      # centre inside
            ("img_001.jpg", (100.0, 100.0, 50.0), [1, 2, 3]),  # outside, sees in-region
            ("img_002.jpg", (200.0, 200.0, 50.0), [4, 5, 6])]  # outside, sees far only
    images = {}
    for iid, (name, centre, ids) in enumerate(spec, start=1):
        images[iid] = Image(id=iid, qvec=qvec, tvec=-np.array(centre), camera_id=1,
                            name=name, xys=tri.copy(),
                            point3D_ids=np.array(ids, dtype=np.int64))
    write_cameras_binary({1: Camera(id=1, model="PINHOLE", width=64, height=64,
                                    params=np.array([50.0, 50.0, 32.0, 32.0]))},
                         str(model_dir / "cameras.bin"))
    write_images_binary(images, str(model_dir / "images.bin"))
    write_points3D_binary(pts, str(model_dir / "points3D.bin"))
    return model_dir


def test_region_narrows_the_image_list_by_both_phases(patched, imgroot, vocab, tmp_path):
    model = _ref_model(tmp_path / "refmodel")
    ws = tmp_path / "ws_region"
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, ws, vocab, region=REGION_IN,
                            region_model=str(model)), r)
    # img_002 is the only one neither test keeps
    assert (ws / "image_list.txt").read_text().split() == ["img_000.jpg", "img_001.jpg"]
    assert any("框內相機 1" in m and "另收 1" in m for m in r.logs)
    assert any("影像清單 3 → 2 張" in m for m in r.logs)


def test_region_still_extracts_only_the_subset(patched, imgroot, vocab, tmp_path):
    """The filter must reach COLMAP, not just image_list.txt — extraction is driven
    by --image_list_path, so the subset propagates to every downstream stage."""
    model = _ref_model(tmp_path / "refmodel")
    ws = tmp_path / "ws_region"
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, ws, vocab, region=REGION_IN,
                            region_model=str(model)), r)
    argv = r.argv_for("feature_extractor")
    listed = Path(argv[argv.index("--image_list_path") + 1]).read_text().split()
    assert listed == ["img_000.jpg", "img_001.jpg"]


def test_region_buffer_widens_the_visibility_test_only(patched, imgroot, vocab, tmp_path):
    """buffer grows the point mask, never the phase-1 rectangle (blocksplit's rule).
    img_002's points sit 85+ units out, so a buffer that reaches them pulls it in via
    phase 2 while the camera-centre count stays at 1."""
    model = _ref_model(tmp_path / "refmodel")
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws_region", vocab, region=REGION_IN,
                            region_model=str(model), region_buffer="200"), r)
    assert any("框內相機 1" in m for m in r.logs)          # phase 1 unchanged
    assert (tmp_path / "ws_region" / "image_list.txt").read_text().split() == [
        "img_000.jpg", "img_001.jpg", "img_002.jpg"]


def test_region_refuses_a_workspace_that_already_has_a_database(patched, imgroot, vocab,
                                                               tmp_path):
    """Re-running into the original workspace would skip extract/mapper on their
    sentinels and hand back the FULL-dataset model as if the region had applied."""
    model = _ref_model(tmp_path / "refmodel")
    ws = tmp_path / "ws_used"
    ws.mkdir()
    (ws / "database.db").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="必須用一個新的 workspace"):
        _run.run_colmap(_params(imgroot, ws, vocab, region=REGION_IN,
                                region_model=str(model)), FakeRunner())


def test_region_refuses_a_model_inside_the_workspace(patched, imgroot, vocab, tmp_path):
    ws = tmp_path / "ws_region"
    model = _ref_model(ws / "sparse")           # the run would overwrite its own reference
    with pytest.raises(ValueError, match="在 workspace"):
        _run.run_colmap(_params(imgroot, ws, vocab, region=REGION_IN,
                                region_model=str(model)), FakeRunner())


@pytest.mark.parametrize("over, msg", [
    ({"region": REGION_IN}, "region_model 是空的"),
    ({"region_model": "x"}, "region 是空的"),
])
def test_region_and_model_must_travel_together(patched, imgroot, vocab, tmp_path, over, msg):
    with pytest.raises((ValueError, FileNotFoundError), match=msg):
        _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab, **over), FakeRunner())


def test_region_reports_a_wholesale_name_mismatch(patched, vocab, tmp_path):
    """A model built from different files selects names this run has none of. That must
    say so, not read as "the region is empty" and send you hunting the wrong bug."""
    other = tmp_path / "other"
    _make_images(other, 2)
    for p in sorted(other.iterdir()):
        p.rename(other / f"zz_{p.name}")        # names that cannot match the model's
    model = _ref_model(tmp_path / "refmodel")
    with pytest.raises(FileNotFoundError, match="沒有一張出現在這次的影像清單裡"):
        _run.run_colmap(_params(other, tmp_path / "ws_region", vocab, region=REGION_IN,
                                region_model=str(model)), FakeRunner())


def test_no_region_leaves_the_image_list_whole(patched, imgroot, vocab, tmp_path):
    ws = tmp_path / "ws"
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, ws, vocab), r)
    assert len((ws / "image_list.txt").read_text().split()) == 3
    assert not any("region 選片" in m for m in r.logs)


# --- multi-camera rig ------------------------------------------------------
def _make_rig_images(root: Path, cams=("nadir", "forward", "backward"), n: int = 3) -> None:
    """ROOT/<camera>/<CAM>-1_<index>-<serial>.jpg — a real oblique block's shape,
    where the serial differs per body so only <strip>_<index> is shared."""
    for c, base in zip(cams, (61214, 61294, 70924), strict=False):
        d = root / c
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"{c[0].upper()}-1_{i}-{base + i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


def test_rig_stage_groups_images_that_live_in_camera_subfolders(
        patched, vocab, tmp_path):
    """Regression: the stage listed img_root non-recursively and without folder
    prefixes, so a multi layout yielded zero images and the rig always aborted."""
    root = tmp_path / "rig_images"
    _make_rig_images(root)
    ws = tmp_path / "ws"
    r = FakeRunner()

    _run.run_colmap(_params(root, ws, vocab, rig_enable=True, layout="multi"), r)

    assert "rig_configurator" in r.subcommands()
    argv = r.argv_for("rig_configurator")
    cfg = Path(argv[argv.index("--rig_config_path") + 1])
    prefixes = [c["image_prefix"] for c in json.loads(cfg.read_text())[0]["cameras"]]
    assert sorted(prefixes) == ["backward/", "forward/", "nadir/"]

    # feature extraction must read the restaged tree, not the originals
    fe = r.argv_for("feature_extractor")
    assert fe[fe.index("--image_path") + 1] == str(ws / "rig_images")
    # every camera exposes the same stems, which is the whole point of restaging
    # (here the index field alone separates exposures, so the key is just "0"..)
    for cam in ("nadir", "forward", "backward"):
        assert (ws / "rig_images" / cam / "0.jpg").is_symlink()
    assert (ws / "rig_images" / "nadir" / "0.jpg").resolve() == (
        root / "nadir" / "N-1_0-61214.jpg").resolve()


def test_rig_is_skipped_entirely_when_disabled(patched, imgroot, vocab, tmp_path):
    r = FakeRunner()
    _run.run_colmap(_params(imgroot, tmp_path / "ws", vocab), r)
    assert "rig_configurator" not in r.subcommands()


def test_rig_aborts_when_only_one_camera_can_be_identified(patched, imgroot, vocab,
                                                           tmp_path):
    # imgroot is one flat folder of img_000.jpg.. — one folder and one filename
    # prefix, so there is no second camera to constrain against.
    with pytest.raises(PipelineError, match="no frame is covered by every camera"):
        _run.run_colmap(
            _params(imgroot, tmp_path / "ws", vocab, rig_enable=True), FakeRunner())


def test_rig_splits_cameras_by_filename_prefix_when_there_are_no_folders(
        patched, vocab, tmp_path):
    """A flat dataset is still a rig when the bodies stamp a filename prefix.
    Camera ids come from the data, not from any assumed folder naming."""
    root = tmp_path / "flat"
    root.mkdir()
    for cam, base in (("alpha", 500), ("bravo", 900)):
        for i in range(3):
            (root / f"{cam}_{i}-{base + i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    r = FakeRunner()
    _run.run_colmap(_params(root, tmp_path / "ws", vocab,
                            rig_enable=True, layout="single"), r)

    argv = r.argv_for("rig_configurator")
    cfg = Path(argv[argv.index("--rig_config_path") + 1])
    prefixes = [c["image_prefix"] for c in json.loads(cfg.read_text())[0]["cameras"]]
    assert sorted(prefixes) == ["alpha/", "bravo/"]
