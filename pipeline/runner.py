"""Runner: emits log lines (to the job console + optional mirror file) and runs
child processes, streaming their output through the same log. Cancellation kills
the current child's process group and is checked between steps."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

# Pure-noise subprocess lines that bloat the log without informing anyone:
# COLMAP's EXIF reader prints ~3 of these per image, so a 4–5k-photo dataset
# adds ~14k useless lines that help freeze the browser log pane. Dropped before
# they ever reach the console log / parser.
_LOG_NOISE = re.compile(r"^add_exif_item_to_spec: didn't know how to process")


class Cancelled(Exception):
    """Raised inside a pipeline when the job has been cancelled."""


class PipelineError(RuntimeError):
    """A child process exited non-zero (mirrors `set -e` aborting the script)."""


class Runner:
    def __init__(self, console: Path | str, on_line: Callable[[str], None] | None = None,
                 mirror: Path | str | None = None):
        self._fh = open(console, "w", buffering=1)
        self._mirror = open(mirror, "w", buffering=1) if mirror else None
        self._on_line = on_line
        self._cancelled = False
        self._procs: set[subprocess.Popen] = set()   # concurrent children
        self._lock = threading.Lock()                # serialize log writes

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if self._mirror:
                self._mirror.close()

    def cancel(self) -> None:
        """Called from another thread: stop the pipeline and kill all children."""
        self._cancelled = True
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check_cancel(self) -> None:
        if self._cancelled:
            raise Cancelled()

    # -- logging ------------------------------------------------------------ #
    def _write(self, text: str) -> None:
        with self._lock:                 # atomic line writes across worker threads
            self._fh.write(text)
            if self._mirror:
                self._mirror.write(text)
            if self._on_line:
                for ln in text.splitlines():
                    try:
                        self._on_line(ln)
                    except Exception:
                        pass

    def log(self, msg: str = "") -> None:
        self._write(str(msg) + "\n")

    def banner(self, msg: str) -> None:
        """Mirror the shell `log()` helper: `=== [HH:MM:SS] msg ===`."""
        self._write(f"\n=== [{time.strftime('%H:%M:%S')}] {msg} ===\n")

    # -- subprocess --------------------------------------------------------- #
    def run(self, argv: Sequence[str], *, cwd: str | None = None,
            env: dict | None = None, check: bool = True,
            stderr_to: Path | str | None = None) -> int:
        """Run a child process.

        Default: merge stdout+stderr and stream every line into the log.
        stderr_to: send stderr to that file instead (stdout discarded) — used for
        ffmpeg's verbose output, which the shell script kept out of the console.
        """
        self.check_cancel()
        run_env = {**os.environ, **(env or {})}
        if stderr_to:
            errfh = open(stderr_to, "w")
            try:
                proc = subprocess.Popen(list(argv), stdout=subprocess.DEVNULL,
                                        stderr=errfh, cwd=cwd, env=run_env,
                                        start_new_session=True)
                with self._lock:
                    self._procs.add(proc)
                proc.wait()
            finally:
                errfh.close()
                with self._lock:
                    self._procs.discard(proc)
        else:
            proc = subprocess.Popen(list(argv), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, cwd=cwd, env=run_env,
                                    text=True, bufsize=1, start_new_session=True)
            with self._lock:
                self._procs.add(proc)
            assert proc.stdout is not None
            for line in proc.stdout:
                if _LOG_NOISE.search(line):
                    continue
                self._write(line)
            proc.wait()
            with self._lock:
                self._procs.discard(proc)

        if self._cancelled:
            raise Cancelled()
        if check and proc.returncode != 0:
            raise PipelineError(f"{argv[0]} {argv[1] if len(argv) > 1 else ''} "
                                f"exited with code {proc.returncode}")
        return proc.returncode
