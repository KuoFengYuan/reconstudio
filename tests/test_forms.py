"""Form → validated pipeline params (web/services/forms.py).

These three helpers are the trust boundary between raw HTML form input and the
COLMAP pipeline: they decide what label a dataset shows under, whether marker
scaling kicks in (and with what board geometry), and they reject malformed
numeric/enum fields BEFORE they reach the trainer. Locking down the defaults,
the override-wins behaviour, and every ValueError path is what keeps a typo in
a text box from silently producing a broken reconstruction.
"""
import pytest

from web.services.forms import build_colmap_params, parse_marker, scene_label

# --- scene_label ---------------------------------------------------------

def test_scene_label_uses_parent_and_leaf():
    # The COLMAP workspace (parent) is what distinguishes otherwise-generic leaves.
    assert scene_label("/x/0520_colmap/gs2m") == "0520_colmap/gs2m"


def test_scene_label_bare_name_has_no_parent():
    # Path("gs2m").parent.name == "" so the leaf alone is returned, no slash.
    assert scene_label("gs2m") == "gs2m"


def test_scene_label_root_anchored_name_has_no_parent():
    # Path("/gs2m").parent is "/" whose .name is "" -> just the leaf.
    assert scene_label("/gs2m") == "gs2m"


# --- parse_marker --------------------------------------------------------

def _spec():
    return {"marker_defaults": {
        "squares_x": 5, "squares_y": 7,
        "square_mm": 30.0, "marker_mm": 22.0,
        "dict": "DICT_4X4_50",
    }}


def test_parse_marker_returns_none_when_not_enabled():
    assert parse_marker({}, _spec()) is None
    # falsy explicit value also disables
    assert parse_marker({"marker_enable": ""}, _spec()) is None


def test_parse_marker_raises_when_spec_has_no_defaults():
    with pytest.raises(ValueError):
        parse_marker({"marker_enable": "1"}, {})
    with pytest.raises(ValueError):
        parse_marker({"marker_enable": "1"}, {"marker_defaults": {}})


def test_parse_marker_uses_configured_defaults_when_blank():
    # Blank/absent override fields fall back to the backend's marker_defaults.
    got = parse_marker({"marker_enable": "1", "marker_squares_x": "   "}, _spec())
    assert got == {
        "enable": True, "squares_x": 5, "squares_y": 7,
        "square_mm": 30.0, "marker_mm": 22.0, "dict": "DICT_4X4_50",
    }


def test_parse_marker_per_job_override_wins():
    form = {
        "marker_enable": "1",
        "marker_squares_x": "8", "marker_squares_y": "11",
        "marker_square_mm": "40", "marker_mm_value_unused": "x",
        "marker_marker_mm": "30",
        "marker_dict": "DICT_6X6_250",
    }
    got = parse_marker(form, _spec())
    assert got["squares_x"] == 8
    assert got["squares_y"] == 11
    assert got["square_mm"] == 40.0
    assert got["marker_mm"] == 30.0
    assert got["dict"] == "DICT_6X6_250"
    assert got["enable"] is True


def test_parse_marker_dict_defaults_when_default_missing():
    # marker_defaults has no "dict" and form supplies none -> hard-coded fallback.
    spec = {"marker_defaults": {
        "squares_x": 5, "squares_y": 7, "square_mm": 30.0, "marker_mm": 22.0,
    }}
    got = parse_marker({"marker_enable": "1"}, spec)
    assert got["dict"] == "DICT_5X5_100"


def test_parse_marker_non_numeric_override_raises():
    form = {"marker_enable": "1", "marker_squares_x": "five"}
    with pytest.raises(ValueError):
        parse_marker(form, _spec())


def test_parse_marker_squares_below_two_raises():
    form = {"marker_enable": "1", "marker_squares_x": "1"}
    with pytest.raises(ValueError):
        parse_marker(form, _spec())


def test_parse_marker_non_positive_square_mm_raises():
    form = {"marker_enable": "1", "marker_square_mm": "0"}
    with pytest.raises(ValueError):
        parse_marker(form, _spec())


def test_parse_marker_non_positive_marker_mm_raises():
    # marker_mm must be > 0; -1 trips the min(...) <= 0 check before the >= check.
    form = {"marker_enable": "1", "marker_marker_mm": "-1"}
    with pytest.raises(ValueError):
        parse_marker(form, _spec())


def test_parse_marker_marker_mm_not_smaller_than_square_raises():
    # marker must be embedded inside the square: marker_mm < square_mm.
    form = {"marker_enable": "1", "marker_square_mm": "30", "marker_marker_mm": "30"}
    with pytest.raises(ValueError):
        parse_marker(form, _spec())


# --- build_colmap_params -------------------------------------------------

def _build(form):
    return build_colmap_params(
        "/imgs", "/ws", ["folderA", "folderB"], ["sfm", "mvs"], form,
    )


def test_build_colmap_params_minimal_form_applies_defaults():
    p = _build({})
    assert p["image_root"] == "/imgs"
    assert p["workspace"] == "/ws"
    assert p["folders"] == ["folderA", "folderB"]
    assert p["stages"] == ["sfm", "mvs"]
    assert p["matcher"] == "vocab"
    assert p["mapper"] == "global"
    assert p["camera_mode"] == "per_folder"
    assert p["max_features"] == "4096"
    assert p["camera_model"] == "OPENCV"
    assert p["layout"] == "auto"
    assert p["resize"] == "fullhd"
    assert p["gps_align_type"] == "enu"


def test_build_colmap_params_boolean_flags_map_correctly():
    # GUIDED_MATCHING -> "1"/"0" string; FORCE -> python bool.
    p = _build({})
    assert p["guided_matching"] == "0"
    assert p["force"] is False
    p2 = _build({"GUIDED_MATCHING": "on", "FORCE": "yes"})
    assert p2["guided_matching"] == "1"
    assert p2["force"] is True


def test_build_colmap_params_other_booleans():
    p = _build({"SPATIAL_IGNORE_Z": "1", "GPS_ALIGN": "1",
                "MAPPER_BA_GPU": "1", "SIMPLIFY": "1", "REORIENT": "1"})
    assert p["spatial_ignore_z"] == "1"
    assert p["gps_align"] is True
    assert p["ba_gpu"] is True
    assert p["simplify"] is True
    assert p["reorient"] is True


def test_build_colmap_params_bad_matcher_raises():
    with pytest.raises(ValueError):
        _build({"MATCHER": "nope"})


def test_build_colmap_params_bad_mapper_raises():
    with pytest.raises(ValueError):
        _build({"MAPPER": "nope"})


def test_build_colmap_params_bad_camera_mode_raises():
    with pytest.raises(ValueError):
        _build({"CAMERA_MODE": "nope"})


def test_build_colmap_params_bad_layout_raises():
    with pytest.raises(ValueError):
        _build({"layout": "nope"})


def test_build_colmap_params_bad_resize_raises():
    with pytest.raises(ValueError):
        _build({"resize": "nope"})


def test_build_colmap_params_bad_gps_align_type_raises():
    with pytest.raises(ValueError):
        _build({"GPS_ALIGN_TYPE": "nope"})


def test_build_colmap_params_integer_field_rejects_non_digit():
    with pytest.raises(ValueError):
        _build({"MAX_FEATURES": "12x"})


def test_build_colmap_params_integer_field_rejects_negative():
    # .isdigit() is False for "-5", so a negative is rejected as non-integer.
    with pytest.raises(ValueError):
        _build({"SEQ_OVERLAP": "-5"})


def test_build_colmap_params_float_field_rejects_non_number():
    with pytest.raises(ValueError):
        _build({"PRIOR_STD_X": "abc"})


def test_build_colmap_params_float_field_accepts_decimal():
    p = _build({"SPATIAL_MAX_DISTANCE": "12.5"})
    assert p["spatial_max_distance"] == "12.5"


def test_build_colmap_params_loop_matches_odd_count_raises():
    with pytest.raises(ValueError):
        _build({"CM_LOOP_MATCHES": "1 2 3"})


def test_build_colmap_params_loop_matches_even_count_accepted():
    p = _build({"CM_LOOP_MATCHES": "1 2 3 4"})
    assert p["cm_loop_matches"] == "1 2 3 4"
    # commas are normalised to spaces and still parse as integer pairs
    p2 = _build({"CM_LOOP_MATCHES": "1,2,3,4"})
    assert p2["cm_loop_matches"] == "1,2,3,4"


def test_build_colmap_params_loop_matches_non_integer_raises():
    with pytest.raises(ValueError):
        _build({"CM_LOOP_MATCHES": "1 two"})


def test_build_colmap_params_optional_numeric_blank_ok_but_bad_rejected():
    # blank optional numerics are allowed (use COLMAP default)...
    p = _build({"MAX_IMAGE_SIZE": "", "FOCAL_FACTOR": "", "HM_NUM_WORKERS": ""})
    assert p["max_image_size"] == ""
    assert p["focal_factor"] == ""
    # ...but a provided non-digit MAX_IMAGE_SIZE is rejected
    with pytest.raises(ValueError):
        _build({"MAX_IMAGE_SIZE": "1.5"})


def test_build_colmap_params_hm_num_workers_allows_negative_one():
    # -1 = auto: int() accepts the leading minus where .isdigit() would not.
    p = _build({"HM_NUM_WORKERS": "-1"})
    assert p["hm_num_workers"] == "-1"
    with pytest.raises(ValueError):
        _build({"HM_NUM_WORKERS": "auto"})


def test_build_colmap_params_focal_factor_accepts_float():
    p = _build({"FOCAL_FACTOR": "1.25"})
    assert p["focal_factor"] == "1.25"
    with pytest.raises(ValueError):
        _build({"FOCAL_FACTOR": "x"})


def test_sift_max_image_size_accepts_auto_and_the_colmap_sentinel():
    """The panel pre-fills "auto"; a digits-only validator would reject it here
    and the feature would be unreachable from the UI."""
    assert _build({"SIFT_MAX_IMAGE_SIZE": "auto"})["sift_max_image_size"] == "auto"
    assert _build({"SIFT_MAX_IMAGE_SIZE": "-1"})["sift_max_image_size"] == "-1"
    assert _build({"SIFT_MAX_IMAGE_SIZE": "8192"})["sift_max_image_size"] == "8192"
    assert _build({})["sift_max_image_size"] == "auto"          # default


def test_sift_max_image_size_still_rejects_nonsense():
    import pytest
    with pytest.raises(ValueError, match="SIFT_MAX_IMAGE_SIZE"):
        _build({"SIFT_MAX_IMAGE_SIZE": "big"})


def test_prior_std_accepts_auto_and_numbers():
    p = _build({})
    assert (p["prior_std_x"], p["prior_std_y"], p["prior_std_z"]) == ("auto",) * 3
    p2 = _build({"PRIOR_STD_X": "0.02", "PRIOR_STD_Z": "auto"})
    assert p2["prior_std_x"] == "0.02"
    assert p2["prior_std_z"] == "auto"


def test_prior_std_still_rejects_nonsense():
    with pytest.raises(ValueError, match="prior_std_x"):
        _build({"PRIOR_STD_X": "loose"})
