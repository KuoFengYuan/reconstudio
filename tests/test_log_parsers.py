"""The five log parsers turn raw pipeline stdout into structured progress.

The web UI's status chips and progress bars are driven *entirely* by the side
effects these parsers have on a Job (job.meta, job.current_stage,
job.stage_status). If a regex drifts or a stage transition stops marking the
previous stage "done", the UI silently lies about what the pipeline is doing.
So we lock down the real line-by-line behavior here: banner/skip/done handling
for COLMAP, video accumulation for frames, the several training-bar dialects
(including the "Loading train cameras" false-positive we must NOT count),
mesh extraction/scaling, and gsutil copy/completion counting.

The parsers MUTATE the job in place and return None; we never call job.save()
(that would touch the filesystem), so a bare Job dataclass is all we need.
"""
from jobs import (
    PARSERS,
    Job,
    _parse_colmap,
    _parse_depth,
    _parse_frames,
    _parse_gcs,
    _parse_mesh,
    _parse_train,
)


def _job(kind="colmap"):
    """A fresh job with only the required fields; meta/stage_status start empty."""
    return Job(id="t", kind=kind, title="x", subtitle="y")


def _feed(parser, job, lines):
    """Feed several lines through a parser the way the worker streams them."""
    for ln in lines:
        parser(job, ln)


# --------------------------------------------------------------------------- #
# _parse_colmap
# --------------------------------------------------------------------------- #
def test_colmap_banner_sets_running_stage():
    job = _job()
    _parse_colmap(job, "=== [12:00:00] feature_extractor (--SiftExtraction) ===")
    assert job.current_stage == "extract"
    assert job.stage_status["extract"] == "running"


def test_colmap_returns_none():
    job = _job()
    assert _parse_colmap(job, "=== [12:00:00] feature_extractor ===") is None


def test_colmap_skip_marks_stage_skipped():
    # A leading-whitespace "skip <stage>" line marks that stage skipped and
    # returns early without touching current_stage.
    job = _job()
    _parse_colmap(job, "  skip extract")
    assert job.stage_status["extract"] == "skipped"
    assert job.current_stage is None


def test_colmap_skip_only_for_known_stages():
    # _SKIP_RE matches, but the captured word must be a real COLMAP stage.
    job = _job()
    _parse_colmap(job, "skip nonsense")
    assert job.stage_status == {}


def test_colmap_non_banner_line_is_ignored():
    # Plain log lines without the "=== [HH:MM:SS]" banner do nothing.
    job = _job()
    _parse_colmap(job, "Reading images from /tmp/foo")
    assert job.current_stage is None
    assert job.stage_status == {}


def test_colmap_new_stage_marks_previous_running_done():
    job = _job()
    _parse_colmap(job, "=== [12:00:00] feature_extractor ===")
    _parse_colmap(job, "=== [12:01:00] colmap mapper  --foo ===")
    assert job.stage_status["extract"] == "done"
    assert job.stage_status["mapper"] == "running"
    assert job.current_stage == "mapper"


def test_colmap_full_pipeline_progression():
    # Walk several real _RUN_MARKERS; every superseded stage ends "done",
    # only the last one stays "running".
    job = _job()
    _feed(_parse_colmap, job, [
        "=== [12:00:00] feature_extractor ===",
        "=== [12:01:00] sequential_matcher ===",
        "=== [12:02:00] colmap mapper  data ===",
        "=== [12:03:00] image_undistorter ===",
        "=== [12:04:00] auto_reorient ===",
    ])
    assert job.stage_status == {
        "extract": "done",
        "match": "done",
        "mapper": "done",
        "undistort": "done",
        "reorient": "running",
    }
    assert job.current_stage == "reorient"


def test_colmap_done_banner_marks_current_stage_done():
    # The "done. workspace=" banner closes out the current stage.
    job = _job()
    _parse_colmap(job, "=== [12:00:00] feature_extractor ===")
    _parse_colmap(job, "=== [12:05:00] done. workspace=/data/ws ===")
    assert job.stage_status["extract"] == "done"
    # current_stage is left pointing at the just-finished stage.
    assert job.current_stage == "extract"


def test_colmap_done_banner_without_current_stage_is_noop():
    job = _job()
    _parse_colmap(job, "=== [12:05:00] done. workspace=/data/ws ===")
    assert job.current_stage is None
    assert job.stage_status == {}


def test_colmap_reentering_skipped_stage_does_not_overwrite_a_done_previous():
    # If the previous stage was skipped (not "running"), entering a new stage
    # must not flip the skipped one to done.
    job = _job()
    _parse_colmap(job, "  skip extract")
    _parse_colmap(job, "=== [12:00:00] colmap mapper data ===")
    assert job.stage_status["extract"] == "skipped"
    assert job.stage_status["mapper"] == "running"


# --------------------------------------------------------------------------- #
# _parse_frames
# --------------------------------------------------------------------------- #
def test_frames_found_videos_sets_total():
    job = _job("frames")
    _parse_frames(job, "Found 3 videos in input/")
    assert job.meta["total"] == 3


def test_frames_progress_sets_cur_total_and_stage():
    job = _job("frames")
    _parse_frames(job, "######## [2/3] processing clip.mp4 ########")
    assert job.meta["cur"] == 2
    assert job.meta["total"] == 3
    assert job.current_stage == "video 2/3"


def test_frames_result_accumulates_across_calls():
    # kept/dropped sum over every video, not overwrite.
    job = _job("frames")
    _parse_frames(job, "-> 50 kept / 10 dropped")
    _parse_frames(job, "-> 30 kept / 5 dropped")
    assert job.meta["kept"] == 80
    assert job.meta["dropped"] == 15


def test_frames_unrelated_line_is_ignored():
    job = _job("frames")
    _parse_frames(job, "extracting frames at 2 fps")
    assert job.meta == {}
    assert job.current_stage is None


# --------------------------------------------------------------------------- #
# _parse_train
# --------------------------------------------------------------------------- #
def test_train_iter_marker():
    job = _job("train")
    _parse_train(job, "[ITER 100] saving snapshot")
    assert job.meta["iter"] == 100
    assert job.current_stage == "train"


def test_train_loss_case_insensitive():
    job = _job("train")
    _parse_train(job, "[ITER 200] Loss: 0.05")
    assert job.meta["loss"] == "0.05"
    assert job.meta["iter"] == 200


def test_train_loss_lowercase_and_equals_form():
    job = _job("train")
    _parse_train(job, "step done loss=0.123")
    assert job.meta["loss"] == "0.123"


def test_train_bar_requires_the_word_Training():
    # A "Training" bar line sets iter/total + phase "訓練中" + stage train.
    job = _job("train")
    _parse_train(job, "Training: 500/30000 [00:30<10:00, 50it/s]")
    assert job.meta["iter"] == 500
    assert job.meta["total"] == 30000
    assert job.meta["phase"] == "訓練中"
    assert job.current_stage == "train"


def test_train_loading_cameras_bar_is_not_the_training_iteration():
    # CRITICAL false-positive guard: the camera-loading bar also matches the
    # generic "n/total [" pattern, but must NOT be read as training progress.
    job = _job("train")
    _parse_train(job, "Loading train cameras: 100/245 [00:01<00:02]")
    # phase comes from the _TR_PHASES marker, not the bar handler.
    assert job.meta["phase"] == "載入訓練相機"
    assert "iter" not in job.meta
    assert "total" not in job.meta
    assert job.current_stage is None


def test_train_lichtfeld_bar_form():
    # LichtFeld postfix "<iter>/<total> | Loss: .." feeds both iter/total and loss.
    job = _job("train")
    _parse_train(job, "1000/30000 | Loss: 0.1 | Splats: 50000")
    assert job.meta["iter"] == 1000
    assert job.meta["total"] == 30000
    assert job.meta["loss"] == "0.1"
    assert job.meta["phase"] == "訓練中"
    assert job.current_stage == "train"


def test_train_phase_marker_only():
    job = _job("train")
    _parse_train(job, "Loading test cameras now")
    assert job.meta["phase"] == "載入測試相機"
    assert job.current_stage is None


def test_train_complete_sets_done():
    job = _job("train")
    _parse_train(job, "Training complete!")
    assert job.meta["complete"] is True
    assert job.meta["phase"] == "完成"
    assert job.current_stage == "done"


def test_train_lichtfeld_completed_phrasing_also_completes():
    # _TR_DONE matches "Training complete" inside "Training completed in ..".
    job = _job("train")
    _parse_train(job, "Training completed in 12m")
    assert job.meta["complete"] is True


# --------------------------------------------------------------------------- #
# _parse_mesh
# --------------------------------------------------------------------------- #
def test_mesh_extracting_sets_stage():
    job = _job("mesh")
    _parse_mesh(job, "Extracting mesh from TSDF volume")
    assert job.current_stage == "extract"
    # "Extracting mesh" also matches a phase marker.
    assert job.meta["phase"] == "抽取 mesh"


def test_mesh_vertices_count():
    job = _job("mesh")
    _parse_mesh(job, "Num vertices post: 12345")
    assert job.meta["vertices"] == 12345


def test_mesh_result_path_marks_done():
    job = _job("mesh")
    _parse_mesh(job, "[mesh] result: /a/b.ply")
    assert job.meta["mesh_path"] == "/a/b.ply"
    assert job.current_stage == "done"


def test_mesh_scale_factor_parsed_as_float():
    job = _job("mesh")
    _parse_mesh(job, "applying mm_per_unit=2.5 to mesh")
    assert job.meta["mm_per_unit"] == 2.5
    assert isinstance(job.meta["mm_per_unit"], float)


def test_mesh_scaled_result_path_and_phase():
    job = _job("mesh")
    _parse_mesh(job, "[mesh] scaled result: /a/c.ply")
    assert job.meta["mesh_scaled_path"] == "/a/c.ply"
    assert job.meta["phase"] == "已縮放 (mm)"
    # "[mesh] scaled result:" is NOT the plain result line, so mesh_path stays unset.
    assert "mesh_path" not in job.meta


def test_mesh_full_sequence():
    job = _job("mesh")
    _feed(_parse_mesh, job, [
        "Found 12 cameras",
        "Extracting mesh from TSDF",
        "Num vertices post: 999",
        "[mesh] result: /out/raw.ply",
        "mm_per_unit=10.0",
        "[mesh] scaled result: /out/scaled.ply",
    ])
    assert job.meta["vertices"] == 999
    assert job.meta["mesh_path"] == "/out/raw.ply"
    assert job.meta["mm_per_unit"] == 10.0
    assert job.meta["mesh_scaled_path"] == "/out/scaled.ply"
    assert job.meta["phase"] == "已縮放 (mm)"
    assert job.current_stage == "done"


# --------------------------------------------------------------------------- #
# _parse_gcs
# --------------------------------------------------------------------------- #
def test_gcs_sync_sets_phase():
    job = _job("gcs")
    _parse_gcs(job, "Building synchronization state...")
    assert job.meta["phase"] == "比對差異中"


def test_gcs_copy_increments_counter():
    job = _job("gcs")
    _parse_gcs(job, "Copying gs://bucket/a.jpg...")
    _parse_gcs(job, "Copying gs://bucket/b.jpg...")
    assert job.meta["copied"] == 2
    assert job.meta["phase"] == "下載中"


def test_gcs_done_parses_objects_and_size():
    job = _job("gcs")
    _parse_gcs(job, "Operation completed over 42 objects/1.2 GiB.")
    assert job.meta["objects"] == 42
    assert job.meta["size"] == "1.2 GiB"
    assert job.meta["phase"] == "完成"
    assert job.current_stage == "done"


def test_gcs_sync_phase_does_not_overwrite_existing_phase():
    # _GCS_SYNC uses setdefault, so a later sync line won't clobber "下載中".
    job = _job("gcs")
    _parse_gcs(job, "Copying gs://bucket/a.jpg...")
    _parse_gcs(job, "Building synchronization state...")
    assert job.meta["phase"] == "下載中"


def test_gcs_full_flow():
    job = _job("gcs")
    _feed(_parse_gcs, job, [
        "Building synchronization state...",
        "Copying gs://bucket/a.jpg...",
        "Copying gs://bucket/b.jpg...",
        "Operation completed over 2 objects/512 KiB.",
    ])
    assert job.meta["copied"] == 2
    assert job.meta["objects"] == 2
    assert job.meta["size"] == "512 KiB"
    assert job.current_stage == "done"


# --------------------------------------------------------------------------- #
# _parse_depth
# --------------------------------------------------------------------------- #
def test_depth_header_sets_total_and_out():
    job = _job("depth")
    _parse_depth(job, "Images: 42 under /data/scene/images")
    assert job.meta["total"] == 42
    assert job.meta["out_dir"] == "/data/scene/images"
    assert job.meta["phase"] == "推論中"
    assert job.current_stage == "depth"


def test_depth_progress_updates_cur_total():
    job = _job("depth")
    _feed(_parse_depth, job, [
        "[1/3] a.jpg",
        "[2/3] sub/b.jpg",
    ])
    assert job.meta["cur"] == 2
    assert job.meta["total"] == 3


def test_depth_done_parses_counts_and_marks_done():
    job = _job("depth")
    _parse_depth(job, "Done. processed=40 skipped=1")
    assert job.meta["written"] == 40
    assert job.meta["skipped"] == 1
    assert job.meta["phase"] == "完成"
    assert job.current_stage == "done"


def test_depth_unrelated_line_is_ignored():
    job = _job("depth")
    _parse_depth(job, "Some unrelated chatter from torch")
    assert job.meta == {}


# --------------------------------------------------------------------------- #
# PARSERS registry
# --------------------------------------------------------------------------- #
def test_parsers_registry_maps_to_correct_callables():
    assert PARSERS["colmap"] is _parse_colmap
    assert PARSERS["frames"] is _parse_frames
    assert PARSERS["train"] is _parse_train
    assert PARSERS["mesh"] is _parse_mesh
    assert PARSERS["gcs"] is _parse_gcs
    assert PARSERS["depth"] is _parse_depth


def test_parsers_registry_has_exactly_the_known_kinds():
    assert set(PARSERS) == {"colmap", "frames", "train", "mesh", "gcs", "blocksplit", "depth"}


def test_parsers_registry_dispatch_round_trips():
    # Using the registry the way the worker does drives the same mutation.
    job = _job("frames")
    PARSERS[job.kind](job, "Found 7 videos")
    assert job.meta["total"] == 7
