"""Multi-camera rig grouping / staging / config generation.

The value of these is the contract with COLMAP's `rig_configurator`, which is
unforgiving in two specific ways (both verified against the real binary):
  * images belong to the same frame only when the name left after stripping
    `image_prefix` is byte-identical, and
  * the reference sensor must be the FIRST entry of the config array, not
    merely flagged, or `Rig::AddSensor` aborts.
"""
from __future__ import annotations

import json

import pytest

from pipeline.colmap._rig import (
    build_staging,
    compile_rig_regex,
    group_auto,
    group_images,
    summarize,
    write_rig_config,
)

# Per-body serials differ, so nothing matches by name — the case that makes
# rig_configurator abort when it is left to group images itself.
OBLIQUE = [
    f"{cam}/{cam}-1_0-{serial}_{i:05d}.jpg"
    for cam, serial in (("N", "61214"), ("F", "61294"), ("B", "70924"))
    for i in range(3)
]
OBLIQUE_RX = r"^(?P<cam>[NFB])-\d+_0-\d+_(?P<frame>\d+)\.jpg$"


def test_folder_mode_groups_by_first_path_component():
    names = [f"{c}/{i:04d}.jpg" for c in ("N", "F", "B") for i in range(3)]
    g = group_images(names, "folder")
    assert g.cameras == ["B", "F", "N"]
    assert len(g.complete_frames()) == 3
    assert not g.unmatched


def test_folder_mode_reports_flat_names_as_unmatched():
    g = group_images(["a.jpg", "b.jpg"], "folder")
    assert g.unmatched == ["a.jpg", "b.jpg"]
    assert g.cameras == []


def test_regex_mode_groups_despite_per_body_serials():
    g = group_images(OBLIQUE, "regex", regex=OBLIQUE_RX)
    assert g.cameras == ["B", "F", "N"]
    # the shared trailing index is what binds a frame together
    assert g.complete_frames() == ["00000", "00001", "00002"]


def test_regex_must_define_both_named_groups():
    with pytest.raises(ValueError, match="frame"):
        compile_rig_regex(r"^(?P<cam>[NFB])-.*$")


def test_gps_mode_clusters_simultaneous_exposures():
    names, gps = [], {}
    for station, (lat, lon) in enumerate([(24.1, 120.6), (24.2, 120.6)]):
        for cam in ("N", "F", "B"):
            # deliberately unrelated names: only position ties them together
            n = f"{cam}/{cam}_{station * 7919:06d}.jpg"
            names.append(n)
            gps[n] = (lat, lon)
    g = group_images(names, "gps", gps=gps, gps_tol=0.5)
    assert g.cameras == ["B", "F", "N"]
    assert len(g.complete_frames()) == 2


def test_summarize_flags_a_grouping_that_constrains_nothing():
    # same camera prefix but no shared frame key -> every frame is partial
    g = group_images([f"{c}/{c}_{i}.jpg" for c in ("N", "F") for i in range(3)], "folder")
    assert g.complete_frames() == []
    assert any("WARNING" in line for line in summarize(g))


def test_staging_normalises_names_so_colmap_can_group(tmp_path):
    root = tmp_path / "img"
    for n in OBLIQUE:
        p = root / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    g = group_images(OBLIQUE, "regex", regex=OBLIQUE_RX)

    assert build_staging(g, root, tmp_path / "stage") == len(OBLIQUE)
    # the stem is identical across cameras; the extension is kept
    for cam in ("N", "F", "B"):
        assert (tmp_path / "stage" / cam / "00000.jpg").is_symlink()
    # and it points at the real original, which is untouched
    assert (tmp_path / "stage" / "N" / "00000.jpg").resolve() == (
        root / "N/N-1_0-61214_00000.jpg").resolve()


def test_rig_config_puts_the_ref_sensor_first(tmp_path):
    g = group_images(OBLIQUE, "regex", regex=OBLIQUE_RX)
    path = tmp_path / "rig_config.json"
    assert write_rig_config(g, path, ref_camera="N") == "N"

    cfg = json.loads(path.read_text(encoding="utf-8"))
    cams = cfg[0]["cameras"]
    # ApplyRigConfig walks this array in order and Rig::AddSensor aborts unless
    # the ref sensor has already been added, so position matters, not just the flag.
    assert cams[0] == {"image_prefix": "N/", "ref_sensor": True}
    assert sum("ref_sensor" in c for c in cams) == 1
    assert [c["image_prefix"] for c in cams[1:]] == ["B/", "F/"]


def test_rig_config_rejects_an_unknown_ref_camera(tmp_path):
    g = group_images(OBLIQUE, "regex", regex=OBLIQUE_RX)
    with pytest.raises(ValueError, match="ref camera"):
        write_rig_config(g, tmp_path / "c.json", ref_camera="ZZZ")


# --- auto mode -------------------------------------------------------------
# Shape of a real oblique block: <cam>-<strip>_<index>-<serial>.jpg, where the
# serial differs per body AND per image, so only <strip>_<index> is shared.
def _oblique_block():
    names = []
    for cam, base in (("nadir", 61214), ("forward", 61294), ("backward", 70924)):
        for strip in (1, 2):
            for idx in range(3):
                names.append(f"{cam}/{cam[0].upper()}-{strip}_{idx}-{base + strip * 10 + idx}.jpg")
    return names


def test_auto_discovers_the_shared_exposure_key_without_config():
    g, notes = group_auto(_oblique_block())
    assert g.cameras == ["backward", "forward", "nadir"]
    # strip+index, i.e. the two leading digit fields — found, not supplied
    assert g.complete_frames() == ["1_0", "1_1", "1_2", "2_0", "2_1", "2_2"]
    assert any("auto picked digit field(s) [0, 1]" in n for n in notes)


def test_auto_drops_keys_that_repeat_within_a_camera():
    # a duplicated <strip>_<index> cannot identify an exposure; binding it would
    # pair images from different shots, so both copies are dropped everywhere.
    names = _oblique_block() + [f"{c}/{c[0].upper()}-1_0-{9000 + i}.jpg"
                                for i, c in enumerate(("nadir", "forward", "backward"))]
    g, notes = group_auto(names)
    assert "1_0" not in g.complete_frames()
    assert any("ambiguous" in n for n in notes)
    assert len(g.unmatched) == 6          # the original 1_0 trio plus the clashing trio


def test_auto_reports_failure_when_no_key_is_shared():
    # disjoint numbering per camera: no digit field can ever line up
    names = ([f"N/N_{100 + i}.jpg" for i in range(3)]
             + [f"F/F_{200 + i}.jpg" for i in range(3)])
    g, notes = group_auto(names)
    assert g.complete_frames() == []
    assert any("could not find a shared exposure key" in n for n in notes)


def test_auto_falls_back_to_a_filename_prefix_when_there_are_no_folders():
    # flat dataset, camera only distinguishable by the leading prefix
    names = [f"{cam}_{i}-{base + i}.jpg"
             for cam, base in (("alpha", 500), ("bravo", 900)) for i in range(3)]
    g, notes = group_auto(names)
    assert g.cameras == ["alpha", "bravo"]
    assert g.complete_frames() == ["0", "1", "2"]
    assert any("by filename prefix" in n for n in notes)


def test_auto_reports_when_only_one_camera_can_be_identified():
    # one folder AND one filename prefix: nothing to constrain against
    g, notes = group_auto([f"N/img_{i}.jpg" for i in range(3)])
    assert g.complete_frames() == []
    assert any("only one camera" in n for n in notes)
