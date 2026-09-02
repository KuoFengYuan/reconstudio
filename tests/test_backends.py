"""build_cli turns a param schema + submitted values into CLI tokens.

This is the heart of "backends are data": the form fields a backend declares
become the trainer's command line here, so its edge cases (empty skip, bool flags,
inverted bools, shell-quoting) are worth locking down.
"""
import pytest

from pipeline.backends import build_cli


def test_value_param_emits_flag_and_value():
    params = [{"key": "iter", "flag": "--iterations", "type": "int"}]
    assert build_cli(params, {"iter": "30000"}) == "--iterations 30000"


def test_empty_or_missing_value_is_skipped():
    params = [{"key": "iter", "flag": "--iterations", "type": "int"}]
    assert build_cli(params, {"iter": ""}) == ""
    assert build_cli(params, {}) == ""


def test_bool_emits_bare_flag_only_when_true():
    params = [{"key": "mat", "flag": "--material", "type": "bool"}]
    assert build_cli(params, {"mat": True}) == "--material"
    assert build_cli(params, {"mat": False}) == ""


def test_inverted_bool_emits_disable_flag_only_when_unchecked():
    # LichtFeld --no-cpu-cache: checkbox reads as the feature ON (default checked);
    # the disable flag is emitted only when UNchecked.
    params = [{"key": "cpu", "flag": "--no-cpu-cache", "type": "bool", "invert": True}]
    assert build_cli(params, {"cpu": True}) == ""
    assert build_cli(params, {"cpu": False}) == "--no-cpu-cache"


def test_values_are_shell_quoted():
    params = [{"key": "masks", "flag": "--masks", "type": "str"}]
    assert build_cli(params, {"masks": "/a b/masks"}) == "--masks '/a b/masks'"


def test_multiple_params_join_in_order():
    params = [
        {"key": "iter", "flag": "--iterations", "type": "int"},
        {"key": "mat", "flag": "--material", "type": "bool"},
    ]
    assert build_cli(params, {"iter": "100", "mat": True}) == "--iterations 100 --material"


# --- csv_options (comma-separated enum, e.g. LichtFeld --export) ----------

_EXPORT = [{"key": "fmt", "flag": "--export", "type": "text",
            "csv_options": ["ply", "sog", "spz"], "label": "匯出格式"}]


def test_csv_options_accepts_a_single_value_and_a_list():
    assert build_cli(_EXPORT, {"fmt": "sog"}) == "--export sog"
    assert build_cli(_EXPORT, {"fmt": "sog,ply"}) == "--export sog,ply"


def test_csv_options_normalises_whitespace_and_empty_items():
    # a hand-typed "sog, ply" must not reach argv with a space inside the value
    assert build_cli(_EXPORT, {"fmt": " sog , ply , "}) == "--export sog,ply"


def test_csv_options_blank_emits_nothing():
    # blank = don't pass the flag at all (LichtFeld then writes only project.licht)
    assert build_cli(_EXPORT, {"fmt": "   "}) == ""


def test_csv_options_rejects_an_unknown_format():
    # a typo must fail at submit time, not 20 minutes into training
    with pytest.raises(ValueError, match="glb"):
        build_cli(_EXPORT, {"fmt": "sog,glb"})


def test_mrnf_backend_exports_sog_by_default():
    """The panel default: a trained job must leave behind a usable file, not just
    the internal project.licht that nothing else can read."""
    from pipeline.backends import BUILTIN_BACKENDS
    prs = BUILTIN_BACKENDS["lichtfeld-mrnf"]["params"]
    pr = next(p for p in prs if p["key"] == "export_formats")
    assert pr["default"] == "sog"
    assert build_cli(prs, {p["key"]: p.get("default") for p in prs}).count("--export sog") == 1


def test_mrnf_export_dropdown_offers_sog_both_and_ply():
    """A dropdown, not a text box — the only choices that matter are "the small one",
    "the one other tools read", and "both". Each option must be a valid csv value."""
    from pipeline.backends import BUILTIN_BACKENDS
    prs = BUILTIN_BACKENDS["lichtfeld-mrnf"]["params"]
    pr = next(p for p in prs if p["key"] == "export_formats")
    assert pr["type"] == "select"
    assert pr["options"] == ["sog", "sog,ply", "ply"]
    assert pr["default"] in pr["options"]
    for opt in pr["options"]:
        assert build_cli([pr], {"export_formats": opt}) == f"--export {opt}"
