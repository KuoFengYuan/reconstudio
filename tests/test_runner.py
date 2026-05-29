"""Runner is the spine of every pipeline: it both *records* what happened (the
console log the browser tails, plus an optional mirror) and *executes* the child
processes that do the actual work, streaming their merged output into that same
log. Getting these wrong breaks observability or correctness, so we lock down:

* the exact bytes `log`/`banner` write (the UI parses these),
* that `run` reports child exit status faithfully (check=True must abort like
  `set -e`; check=False must surface the code),
* that the EXIF noise filter actually drops the spammy lines but keeps real ones,
* that `on_line` sees every streamed line and that a buggy callback can't crash
  the run,
* that cancellation flips the flag and that `check_cancel` raises, and
* that `stderr_to` diverts stderr to its own file and keeps it out of the console.

All children are `sys.executable -c ...` so the suite needs no external binaries,
no network, and no GPU.
"""
import re
import sys

import pytest

from pipeline.runner import _LOG_NOISE, Cancelled, PipelineError, Runner


def test_log_writes_message_then_newline(tmp_path):
    console = tmp_path / "console.log"
    r = Runner(console)
    try:
        r.log("hello world")
    finally:
        r.close()
    assert console.read_text() == "hello world\n"


def test_log_with_no_arg_writes_blank_line(tmp_path):
    console = tmp_path / "console.log"
    r = Runner(console)
    try:
        r.log()
    finally:
        r.close()
    assert console.read_text() == "\n"


def test_log_coerces_non_string(tmp_path):
    console = tmp_path / "console.log"
    r = Runner(console)
    try:
        r.log(42)
    finally:
        r.close()
    assert console.read_text() == "42\n"


def test_banner_matches_timestamped_format(tmp_path):
    console = tmp_path / "console.log"
    r = Runner(console)
    try:
        r.banner("BUILD")
    finally:
        r.close()
    text = console.read_text()
    # banner emits a leading newline, then the === line, then a trailing newline.
    assert text.startswith("\n")
    line = text.strip("\n")
    assert re.fullmatch(r"=== \[\d\d:\d\d:\d\d\] BUILD ===", line), repr(line)


def test_mirror_receives_same_bytes_as_console(tmp_path):
    console = tmp_path / "console.log"
    mirror = tmp_path / "mirror.log"
    r = Runner(console, mirror=mirror)
    try:
        r.log("twice")
    finally:
        r.close()
    assert console.read_text() == "twice\n"
    assert mirror.read_text() == "twice\n"


def test_run_returns_zero_and_logs_stdout(tmp_path):
    console = tmp_path / "console.log"
    r = Runner(console)
    try:
        rc = r.run([sys.executable, "-c", "print('hello')"])
    finally:
        r.close()
    assert rc == 0
    assert "hello" in console.read_text()


def test_run_nonzero_with_check_raises_pipeline_error(tmp_path):
    console = tmp_path / "console.log"
    r = Runner(console)
    try:
        with pytest.raises(PipelineError):
            r.run([sys.executable, "-c", "import sys; sys.exit(3)"], check=True)
    finally:
        r.close()


def test_run_nonzero_without_check_returns_code(tmp_path):
    console = tmp_path / "console.log"
    r = Runner(console)
    try:
        rc = r.run([sys.executable, "-c", "import sys; sys.exit(3)"], check=False)
    finally:
        r.close()
    assert rc == 3


def test_log_noise_regex_matches_exif_spam_only():
    assert _LOG_NOISE.search("add_exif_item_to_spec: didn't know how to process foo")
    assert not _LOG_NOISE.search("normal informative line")


def test_run_filters_exif_noise_but_keeps_real_lines(tmp_path):
    console = tmp_path / "console.log"
    r = Runner(console)
    code = (
        "print(\"add_exif_item_to_spec: didn't know how to process tag 42\")\n"
        "print('keep me: feature extraction done')\n"
    )
    try:
        rc = r.run([sys.executable, "-c", code])
    finally:
        r.close()
    assert rc == 0
    text = console.read_text()
    assert "add_exif_item_to_spec" not in text
    assert "keep me: feature extraction done" in text


def test_on_line_receives_each_streamed_line(tmp_path):
    console = tmp_path / "console.log"
    seen = []
    r = Runner(console, on_line=seen.append)
    code = "print('line-a')\nprint('line-b')\n"
    try:
        r.run([sys.executable, "-c", code])
    finally:
        r.close()
    assert "line-a" in seen
    assert "line-b" in seen


def test_on_line_exception_is_swallowed_and_run_completes(tmp_path):
    console = tmp_path / "console.log"

    def boom(_line):
        raise ValueError("callback blew up")

    r = Runner(console, on_line=boom)
    try:
        rc = r.run([sys.executable, "-c", "print('still logged')"])
    finally:
        r.close()
    # The buggy callback must not take down the run...
    assert rc == 0
    # ...and the line still reaches the console log.
    assert "still logged" in console.read_text()


def test_cancel_sets_flag_and_check_cancel_raises(tmp_path):
    console = tmp_path / "console.log"
    r = Runner(console)
    try:
        assert r.cancelled is False
        r.check_cancel()  # no-op before cancel
        r.cancel()
        assert r.cancelled is True
        with pytest.raises(Cancelled):
            r.check_cancel()
    finally:
        r.close()


def test_run_after_cancel_raises_cancelled_before_spawning(tmp_path):
    console = tmp_path / "console.log"
    r = Runner(console)
    try:
        r.cancel()
        with pytest.raises(Cancelled):
            r.run([sys.executable, "-c", "print('should not run')"])
    finally:
        r.close()
    # Cancelled before any child output was written.
    assert console.read_text() == ""


def test_stderr_to_diverts_stderr_and_keeps_console_clean(tmp_path):
    console = tmp_path / "console.log"
    errfile = tmp_path / "err.log"
    code = (
        "import sys\n"
        "sys.stdout.write('to stdout\\n')\n"
        "sys.stderr.write('to stderr\\n')\n"
    )
    r = Runner(console)
    try:
        rc = r.run([sys.executable, "-c", code], stderr_to=errfile)
    finally:
        r.close()
    assert rc == 0
    # stderr lands in the dedicated file...
    assert "to stderr" in errfile.read_text()
    # ...and neither stderr nor (DEVNULL'd) stdout reaches the console log.
    console_text = console.read_text()
    assert "to stderr" not in console_text
    assert "to stdout" not in console_text
