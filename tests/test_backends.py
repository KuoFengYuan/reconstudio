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


# --- option_flags / conflicts_with (LichtFeld mutually exclusive options) ---

_APPEARANCE = [{"key": "app", "type": "select", "options": ["a", "b"], "default": "a",
                "option_flags": {"a": [], "b": ["--no-x", "--y"]}, "label": "外觀"}]


def test_option_flags_emit_the_tokens_of_the_chosen_option():
    assert build_cli(_APPEARANCE, {"app": "a"}) == ""
    assert build_cli(_APPEARANCE, {"app": "b"}) == "--no-x --y"


def test_option_flags_fall_back_to_the_default_when_missing():
    assert build_cli(_APPEARANCE, {}) == ""


def test_option_flags_reject_an_unknown_option():
    with pytest.raises(ValueError, match="zzz"):
        build_cli(_APPEARANCE, {"app": "zzz"})


_ITERS = [{"key": "iterations", "flag": "--iter", "type": "int", "label": "迭代次數"},
          {"key": "steps_scaler", "flag": "--steps-scaler", "type": "float",
           "conflicts_with": "iterations", "label": "步數縮放"}]


def test_conflicting_params_alone_are_fine():
    assert build_cli(_ITERS, {"iterations": "30000"}) == "--iter 30000"
    assert build_cli(_ITERS, {"steps_scaler": "2"}) == "--steps-scaler 2"
    assert build_cli(_ITERS, {"iterations": "30000", "steps_scaler": " "}) == "--iter 30000"


def test_conflicting_params_together_are_rejected():
    with pytest.raises(ValueError, match="擇一"):
        build_cli(_ITERS, {"iterations": "30000", "steps_scaler": "2"})


def _mrnf_tokens(**overrides):
    from pipeline.backends import BUILTIN_BACKENDS
    prs = BUILTIN_BACKENDS["lichtfeld-mrnf"]["params"]
    values = {p["key"]: p.get("default") for p in prs}
    values.update(overrides)
    return build_cli(prs, values).split()


def test_mrnf_defaults_never_pass_iter_and_steps_scaler_together():
    """LichtFeld exits on argv when both are given."""
    toks = _mrnf_tokens()
    assert "--iter" in toks
    assert "--steps-scaler" not in toks


def test_mrnf_appearance_options_never_combine_exposure_correction():
    """MRNF enables exposure correction by default and rejects it alongside
    --ppisp / --bilateral-grid, so every option that uses either must also
    pass --no-exposure-correction."""
    from pipeline.backends import BUILTIN_BACKENDS
    prs = BUILTIN_BACKENDS["lichtfeld-mrnf"]["params"]
    pr = next(p for p in prs if p["key"] == "appearance")
    assert pr["default"] == "exposure_correction"
    assert set(pr["options"]) == set(pr["option_flags"])
    for opt in pr["options"]:
        toks = _mrnf_tokens(appearance=opt)
        if "--ppisp" in toks or "--bilateral-grid" in toks:
            assert "--no-exposure-correction" in toks, opt
    assert not {"--ppisp", "--bilateral-grid", "--no-exposure-correction"} & set(_mrnf_tokens())


def test_mrnf_no_longer_offers_removed_flags():
    toks = " ".join(_mrnf_tokens(**{"background_improvements": True}))
    assert "--background-improvements" not in toks
