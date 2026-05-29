"""Layout detection decides how COLMAP groups input images into cameras.

The pipeline supports exactly three input shapes (single / multi / nested), and
getting the detection wrong means images land in the wrong camera group or the
run aborts. These tests lock down the listing helpers (which feed the matcher)
and resolve_layout's auto-detection, the explicit overrides, the force_nested
flag, the workspace-exclusion rule, and the empty-root error path.
"""
from pathlib import Path

import pytest

from pipeline.colmap._layout import (
    IMAGE_EXTS,
    _has_images,
    list_image_names,
    list_images,
    resolve_layout,
)


def _touch(p: Path) -> Path:
    """Create an empty file (and parents). We never need real image bytes:
    layout detection keys off the file extension, not the contents."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


# --------------------------------------------------------------------------
# IMAGE_EXTS
# --------------------------------------------------------------------------

def test_image_exts_are_lowercase_dotted():
    # detection lower-cases suffixes before comparing, so the set must be lower.
    assert {".jpg", ".jpeg", ".png", ".tif", ".tiff"} == IMAGE_EXTS
    assert all(e == e.lower() and e.startswith(".") for e in IMAGE_EXTS)


# --------------------------------------------------------------------------
# list_images
# --------------------------------------------------------------------------

def test_list_images_with_prefix_returns_sorted_prefixed_names(tmp_path):
    _touch(tmp_path / "b.jpg")
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "c.tiff")
    assert list_images(tmp_path, "cam0") == [
        "cam0/a.png",
        "cam0/b.jpg",
        "cam0/c.tiff",
    ]


def test_list_images_empty_prefix_returns_bare_names(tmp_path):
    _touch(tmp_path / "z.jpg")
    _touch(tmp_path / "a.jpeg")
    # empty prefix -> bare names, no leading slash.
    assert list_images(tmp_path, "") == ["a.jpeg", "z.jpg"]


def test_list_images_ignores_non_image_files(tmp_path):
    _touch(tmp_path / "keep.jpg")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "data.json")
    _touch(tmp_path / "movie.mp4")
    assert list_images(tmp_path, "g") == ["g/keep.jpg"]


def test_list_images_ignores_subdirectories(tmp_path):
    _touch(tmp_path / "top.jpg")
    # a subdir that happens to look like an image name must not be listed.
    (tmp_path / "nested.jpg").mkdir()
    _touch(tmp_path / "nested.jpg" / "inside.png")
    assert list_images(tmp_path, "") == ["top.jpg"]


def test_list_images_extension_matching_is_case_insensitive(tmp_path):
    _touch(tmp_path / "UPPER.JPG")
    _touch(tmp_path / "Mixed.PnG")
    assert list_images(tmp_path, "") == ["Mixed.PnG", "UPPER.JPG"]


def test_list_images_missing_folder_returns_empty(tmp_path):
    assert list_images(tmp_path / "nope", "x") == []


def test_list_images_folder_with_only_junk_returns_empty(tmp_path):
    _touch(tmp_path / "a.txt")
    assert list_images(tmp_path, "g") == []


# --------------------------------------------------------------------------
# list_image_names
# --------------------------------------------------------------------------

def test_list_image_names_returns_bare_sorted_names(tmp_path):
    _touch(tmp_path / "frame_002.jpg")
    _touch(tmp_path / "frame_001.jpg")
    _touch(tmp_path / "frame_010.png")
    # no prefix is ever attached here, even though list_images can add one.
    assert list_image_names(tmp_path) == [
        "frame_001.jpg",
        "frame_002.jpg",
        "frame_010.png",
    ]


def test_list_image_names_filters_to_image_exts_only(tmp_path):
    _touch(tmp_path / "good.tiff")
    _touch(tmp_path / "bad.raw")
    (tmp_path / "subdir").mkdir()
    assert list_image_names(tmp_path) == ["good.tiff"]


def test_list_image_names_missing_folder_returns_empty(tmp_path):
    assert list_image_names(tmp_path / "ghost") == []


# --------------------------------------------------------------------------
# _has_images
# --------------------------------------------------------------------------

def test_has_images_true_when_image_directly_present(tmp_path):
    _touch(tmp_path / "x.jpg")
    assert _has_images(tmp_path) is True


def test_has_images_false_when_only_non_images(tmp_path):
    _touch(tmp_path / "readme.md")
    assert _has_images(tmp_path) is False


def test_has_images_false_when_images_only_in_subdir(tmp_path):
    # images one level down must NOT count: _has_images checks the folder directly.
    _touch(tmp_path / "group" / "x.jpg")
    assert _has_images(tmp_path) is False


def test_has_images_false_when_not_a_directory(tmp_path):
    f = _touch(tmp_path / "file.jpg")
    assert _has_images(f) is False
    assert _has_images(tmp_path / "missing") is False


# --------------------------------------------------------------------------
# resolve_layout: single
# --------------------------------------------------------------------------

def test_resolve_single_auto_when_root_holds_images(tmp_path):
    _touch(tmp_path / "0001.jpg")
    _touch(tmp_path / "0002.jpg")
    folders, nested, name = resolve_layout(str(tmp_path), [], "auto", False)
    assert folders == [""]
    assert nested is False
    assert name == "single"


def test_resolve_single_explicit_overrides_subfolders(tmp_path):
    # explicit single wins even when the root looks like a multi tree.
    _touch(tmp_path / "camA" / "a.jpg")
    _touch(tmp_path / "camB" / "b.jpg")
    folders, nested, name = resolve_layout(str(tmp_path), [], "single", False)
    assert folders == [""]
    assert nested is False
    assert name == "single"


def test_resolve_single_explicit_keeps_provided_folders(tmp_path):
    # when folders are passed, explicit single returns them as-is (not [""]).
    _touch(tmp_path / "0001.jpg")
    folders, nested, name = resolve_layout(str(tmp_path), ["sub"], "single", False)
    assert folders == ["sub"]
    assert nested is False
    assert name == "single"


# --------------------------------------------------------------------------
# resolve_layout: multi
# --------------------------------------------------------------------------

def test_resolve_multi_auto_from_groups_with_direct_images(tmp_path):
    _touch(tmp_path / "cam0" / "a.jpg")
    _touch(tmp_path / "cam1" / "b.jpg")
    _touch(tmp_path / "cam2" / "c.png")
    folders, nested, name = resolve_layout(str(tmp_path), [], "auto", False)
    assert folders == ["cam0", "cam1", "cam2"]
    assert nested is False
    assert name == "multi"


def test_resolve_multi_explicit(tmp_path):
    # explicit multi keeps groups flat even if a group has nested subdirs.
    _touch(tmp_path / "cam0" / "vid0" / "a.jpg")
    _touch(tmp_path / "cam1" / "vid0" / "b.jpg")
    folders, nested, name = resolve_layout(str(tmp_path), [], "multi", False)
    assert folders == ["cam0", "cam1"]
    assert nested is False
    assert name == "multi"


def test_resolve_multi_ignores_hidden_subfolders(tmp_path):
    _touch(tmp_path / "cam0" / "a.jpg")
    _touch(tmp_path / ".hidden" / "x.jpg")
    folders, nested, name = resolve_layout(str(tmp_path), [], "auto", False)
    assert folders == ["cam0"]
    assert name == "multi"


# --------------------------------------------------------------------------
# resolve_layout: nested
# --------------------------------------------------------------------------

def test_resolve_nested_auto_detected_from_first_group(tmp_path):
    # first group has no direct images but does contain subdirs -> nested.
    _touch(tmp_path / "groupA" / "vid0" / "a.jpg")
    _touch(tmp_path / "groupA" / "vid1" / "b.jpg")
    _touch(tmp_path / "groupB" / "vid0" / "c.jpg")
    folders, nested, name = resolve_layout(str(tmp_path), [], "auto", False)
    assert folders == ["groupA", "groupB"]
    assert nested is True
    assert name == "nested"


def test_resolve_nested_explicit(tmp_path):
    _touch(tmp_path / "g0" / "v0" / "a.jpg")
    folders, nested, name = resolve_layout(str(tmp_path), [], "nested", False)
    assert folders == ["g0"]
    assert nested is True
    assert name == "nested"


def test_resolve_force_nested_overrides_multi_shape(tmp_path):
    # groups have direct images (would be multi) but force_nested wins.
    _touch(tmp_path / "cam0" / "a.jpg")
    _touch(tmp_path / "cam1" / "b.jpg")
    folders, nested, name = resolve_layout(str(tmp_path), [], "auto", True)
    assert folders == ["cam0", "cam1"]
    assert nested is True
    assert name == "nested"


# --------------------------------------------------------------------------
# resolve_layout: workspace exclusion
# --------------------------------------------------------------------------

def test_resolve_excludes_workspace_dir_nested_under_root(tmp_path):
    _touch(tmp_path / "cam0" / "a.jpg")
    _touch(tmp_path / "cam1" / "b.jpg")
    ws = tmp_path / "workspace"
    ws.mkdir()
    folders, nested, name = resolve_layout(
        str(tmp_path), [], "auto", False, workspace=str(ws))
    assert "workspace" not in folders
    assert folders == ["cam0", "cam1"]
    assert name == "multi"


def test_resolve_workspace_not_under_root_does_not_skip_anything(tmp_path):
    # a workspace elsewhere on disk must not remove a same-named group.
    _touch(tmp_path / "root" / "ws" / "a.jpg")
    _touch(tmp_path / "root" / "cam1" / "b.jpg")
    outside_ws = tmp_path / "elsewhere" / "ws"
    outside_ws.mkdir(parents=True)
    folders, nested, name = resolve_layout(
        str(tmp_path / "root"), [], "auto", False, workspace=str(outside_ws))
    # "ws" under root is kept because the real workspace lives elsewhere.
    assert folders == ["cam1", "ws"]


# --------------------------------------------------------------------------
# resolve_layout: error / empty
# --------------------------------------------------------------------------

def test_resolve_raises_when_root_empty(tmp_path):
    # no images and no subfolders -> nothing to reconstruct.
    with pytest.raises(FileNotFoundError):
        resolve_layout(str(tmp_path), [], "auto", False)


def test_resolve_raises_when_only_non_image_files(tmp_path):
    _touch(tmp_path / "readme.txt")
    with pytest.raises(FileNotFoundError):
        resolve_layout(str(tmp_path), [], "auto", False)


def test_resolve_multi_explicit_empty_root_raises(tmp_path):
    # explicit multi still needs subfolders; an empty root has no chosen groups.
    with pytest.raises(FileNotFoundError):
        resolve_layout(str(tmp_path), [], "multi", False)
