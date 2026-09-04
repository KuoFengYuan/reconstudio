"""Guards on the 去背 box picker's browser code, which no other test can reach.

The picker is plain JS inside templates/index.html driving an htmx fragment, so
a stale copy of a helper is not a syntax error anywhere — it just wins hoisting
and silently replaces the live one. That is exactly what happened once: an
earlier single-frame `mpInit` survived a rewrite, kept writing boxes into
`mpState.boxes` while the new renderer read `mpState.frames`, and the UI drew
nothing at all while reporting "0 框". Both checks below are for that class of
bug, not for cosmetics.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
INDEX = BASE / "templates" / "index.html"
FRAGMENT = BASE / "templates" / "_matte_pick.html"


@pytest.fixture(scope="module")
def index_html() -> str:
    return INDEX.read_text()


def test_no_picker_helper_is_defined_twice(index_html):
    names = re.findall(r"function\s+(mp[A-Za-z]*|syncMatte)\s*\(", index_html)
    dupes = {n: c for n, c in collections.Counter(names).items() if c > 1}
    assert not dupes, f"defined more than once (the later one silently wins): {dupes}"


def test_no_reference_to_the_retired_single_frame_state(index_html):
    """Boxes live in `mpState.frames[rel]` now. A leftover `mpState.boxes` or
    `mpState.ref` reads as undefined and drops every box on the floor."""
    assert "mpState.boxes" not in index_html
    assert not re.search(r"mpState\.ref\b", index_html)


def test_every_element_the_fragment_scripts_exists_in_the_fragment():
    """The fragment's inline `mpInit(...)` binds by id; a renamed div in the
    template would leave the overlay unhooked with no console error."""
    fragment = FRAGMENT.read_text()
    for element_id in ("mp_stage", "mp_ov", "mp_img", "mp_status", "mp_count", "mp_frames"):
        assert f'id="{element_id}"' in fragment, element_id


def test_picker_targets_the_run_panel_not_the_form_column(index_html):
    """The picker deliberately renders into #main: a 260px form column is not
    enough photo to place a box on."""
    fragment = FRAGMENT.read_text()
    assert fragment.count('hx-target="#main"') >= 3      # prev, next, jump select
    assert 'hx-target="#main"' in index_html             # the ⬚ 匡選物體 button


def _options_of(html: str, select_id: str) -> set[str]:
    block = html.split(f'id="{select_id}"', 1)[1].split("</select>", 1)[0]
    return set(re.findall(r'<option value="([\w,]+)"', block))


def test_form_and_pipeline_agree_on_the_prompt_modes(index_html):
    """A mode the <select> offers but run_matte rejects is a validation error on
    submit — and the two lists live in different files, so nothing else notices."""
    from pipeline.matte import BOX_SOURCES
    offered = _options_of(index_html, "matte_boxes_mode")
    assert offered, "the prompt-mode select has no options"
    assert offered <= set(BOX_SOURCES), offered - set(BOX_SOURCES)


def test_form_and_pipeline_agree_on_the_engines(index_html):
    from pipeline.matte import ENGINES
    offered = _options_of(index_html, "matte_engine")
    assert offered and offered <= set(ENGINES), offered - set(ENGINES)


def test_the_engine_options_take_their_default_from_the_pipeline(index_html):
    """Every engine option must derive `selected` from matte_defaults rather than
    hard-coding it, or changing MATTE_DEFAULTS moves the pipeline's default while
    the form keeps pre-selecting the old model. (Asserted on the template source:
    rendering index.html would need the backends registry and nvidia-smi, neither
    of which exists in the offline suite.)"""
    from pipeline.matte import ENGINES, MATTE_DEFAULTS
    block = index_html.split('id="matte_engine"', 1)[1].split("</select>", 1)[0]
    for engine in ENGINES:
        line = next(ln for ln in block.splitlines() if f'value="{engine}"' in ln)
        assert "matte_defaults.matte_engine" in line, engine
    assert MATTE_DEFAULTS["matte_engine"] in ENGINES
