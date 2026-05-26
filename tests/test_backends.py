"""build_cli turns a param schema + submitted values into CLI tokens.

This is the heart of "backends are data": the form fields a backend declares
become the trainer's command line here, so its edge cases (empty skip, bool flags,
inverted bools, shell-quoting) are worth locking down.
"""
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
