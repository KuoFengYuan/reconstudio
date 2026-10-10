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
    # COLMAP >= 4.3 (#4662) traps the first SIGINT/SIGTERM and tries to stop
    # cleanly. Its feature extractor and exhaustive matcher can then wait on
    # their own worker queues forever (measured on 4ae1aafb: >10 min, every
    # thread in futex), leaving a cancelled job "running" with the GPU and the
    # worker slot held. A second signal is COLMAP's forced exit, so cancel()
    # escalates on a timer: SIGTERM, SIGTERM again, then SIGKILL.
    CANCEL_GRACE_S = 10.0     # first SIGTERM -> second SIGTERM
    CANCEL_FORCE_S = 5.0      # second SIGTERM -> SIGKILL

    def __init__(self, console: Path | str, on_line: Callable[[str], None] | None = None,
                 mirror: Path | str | None = None):
        self._fh = open(console, "w", buffering=1)
        self._mirror = open(mirror, "w", buffering=1) if mirror else None
        self._on_line = on_line
        self._cancelled = False
        self._procs: set[subprocess.Popen] = set()   # concurrent children
        self._escalating = False
        self._lock = threading.Lock()                # serialize log writes
        # extra env applied to every child process (e.g. CUDA_VISIBLE_DEVICES for a
        # whole stage); per-call run(env=...) still takes precedence over this.
        self.default_env: dict[str, str] = {}

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if self._mirror:
                self._mirror.close()

    def cancel(self) -> None:
        """Called from another thread: stop the pipeline and kill all children.

        Signals at once, then escalates in the background (see CANCEL_GRACE_S).
        A repeated call (a second click on 取消) signals again immediately."""
        self._cancelled = True
        with self._lock:
            procs = list(self._procs)
            escalate = bool(procs) and not self._escalating
            self._escalating = self._escalating or escalate
        self._signal(procs, signal.SIGTERM)
        if escalate:
            threading.Thread(target=self._escalate, args=(procs,), daemon=True).start()

    @staticmethod
    def _signal(procs: list[subprocess.Popen], sig: int) -> None:
        for proc in procs:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except (ProcessLookupError, PermissionError):
                    pass

    def _escalate(self, procs: list[subprocess.Popen]) -> None:
        for sig, wait in ((signal.SIGTERM, self.CANCEL_GRACE_S),
                          (signal.SIGKILL, self.CANCEL_FORCE_S)):
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if all(p.poll() is not None for p in procs):
                    return
                time.sleep(0.1)
            alive = [p for p in procs if p.poll() is None]
            if alive:
                try:
                    self.log(f"[cancel] {len(alive)} 個子程序未結束,送出 {signal.Signals(sig).name}")
                except ValueError:    # the job already closed its console
                    pass
            self._signal(alive, sig)

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
    def _spawn(self, argv, **kwargs):
        # Registration and the cancellation snapshot share a lock. Otherwise
        # cancel() can miss a child created between check_cancel() and add().
        with self._lock:
            self.check_cancel()
            proc = subprocess.Popen(argv, **kwargs)
            self._procs.add(proc)
            return proc

    def run(self, argv: Sequence[str], *, cwd: str | None = None,
            env: dict | None = None, check: bool = True,
            stderr_to: Path | str | None = None) -> int:
        """Run a child process.

        Default: merge stdout+stderr and stream every line into the log.
        stderr_to: send stderr to that file instead (stdout discarded) — used for
        ffmpeg's verbose output, which the shell script kept out of the console.
        """
        self.check_cancel()
        run_env = {**os.environ, **self.default_env, **(env or {})}
        errfh = open(stderr_to, "w") if stderr_to else None
        proc = None
        try:
            proc = self._spawn(
                list(argv), stdout=subprocess.DEVNULL if errfh else subprocess.PIPE,
                stderr=errfh if errfh else subprocess.STDOUT, cwd=cwd, env=run_env,
                text=True, bufsize=1, start_new_session=True,
            )
            if proc.stdout is not None:
                for line in proc.stdout:
                    if not _LOG_NOISE.search(line):
                        self._write(line)
            proc.wait()
        finally:
            if errfh:
                errfh.close()
            if proc is not None:
                if proc.stdout is not None:
                    proc.stdout.close()
                with self._lock:
                    self._procs.discard(proc)

        if self._cancelled:
            raise Cancelled()
        if check and proc.returncode != 0:
            raise PipelineError(f"{argv[0]} {argv[1] if len(argv) > 1 else ''} "
                                f"exited with code {proc.returncode}")
        return proc.returncode
