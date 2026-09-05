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


# --------------------------------------------------------------------------- #
# The step gate. A first run fails the same way every time — 開始去背 pressed
# with no box drawn — and the server's (correct) error arrives only after a
# submit that swaps #main, taking the user away from everything they filled in.
# --------------------------------------------------------------------------- #
CREATE = BASE / "web" / "routers" / "create.py"


def _gate_modes(index_html: str) -> set[str]:
    m = re.search(r"var needs = \[([^\]]+)\]\.indexOf\(mode\.value\)", index_html)
    assert m, "matteGate no longer decides which modes need a box"
    return set(re.findall(r"'([a-z]+)'", m.group(1)))


def test_the_gate_and_the_server_agree_on_which_modes_need_a_box(index_html):
    """Two lists in two files. Drift one way lets the form submit into a server
    error; drift the other way blocks a mode that never needed a box."""
    served = re.search(r'if boxes in \(([^)]+)\):', CREATE.read_text())
    assert served, "create_matte no longer gates on the prompt mode"
    assert _gate_modes(index_html) == set(re.findall(r'"([a-z]+)"', served.group(1)))


@pytest.mark.parametrize("caller", ["syncMatte", "mpRender"])
def test_the_gate_reruns_whenever_its_inputs_change(index_html, caller):
    """Called from too few places and the button just stays wrong — still
    disabled after you draw a box, or still enabled after you clear one."""
    body = index_html.split(f"function {caller}(")[1].split("\n  }")[0]
    assert "matteGate()" in body, f"{caller} does not refresh the gate"


def test_the_path_field_refreshes_the_gate(index_html):
    assert re.search(r'id="matte_images"[^>]*oninput="matteGate\(\)"', index_html, re.S)


def test_picking_a_folder_notifies_the_field(index_html):
    """`el.value = path` fires no event, so without this the gate would still be
    asking for a folder that is already filled in."""
    body = index_html.split("function pickDir(")[1].split("\n  }")[0]
    assert "new Event('input'" in body


def test_boxes_remember_which_folder_they_were_drawn_in(index_html):
    """They are keyed by filename inside one folder; re-pointing the form would
    otherwise send SAM a prompt naming files that are not there."""
    assert "h.dataset.dir" in index_html
    body = index_html.split("function matteGate(")[1].split("\n  }")[0]
    assert "h.dataset.dir !== dir" in body


def test_the_form_leads_with_the_steps_not_the_reference(index_html):
    form = index_html.split('id="form-matte"')[1]
    steps, first_label = form.index('class="steps"'), form.index("<label>")
    assert steps < first_label, "the numbered steps must come before the fields"
