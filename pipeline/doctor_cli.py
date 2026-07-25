"""`python -m pipeline.doctor_cli` — the /doctor page, in the terminal.

Standing up a new machine used to mean installing everything, starting the
server, and port-forwarding a browser just to find out what was still missing.
This prints the same report over SSH, from the same `pipeline.doctor()` call, so
the two can't drift.

    ./run.sh --doctor            # normal use: run.sh loads local.env first
    ./run.sh --doctor --fast     # skip the per-backend torch/CUDA probe
    ./run.sh --doctor --json     # machine-readable (for scripts / CI)

Exit status is the point of the whole thing: 0 = every REQUIRED piece is ready,
1 = at least one hard failure, so setup.sh (or a deploy script) can branch on it.
Warnings — optional features, low disk, stale local.env vars — don't fail the run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .backends import doctor

# Colour only when writing to a real terminal, and never under NO_COLOR.
_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


# status -> (badge, colour). Padded to the same width so the labels line up.
_BADGE = {
    "ok":   ("  OK  ", "1;32"),
    "warn": (" WARN ", "1;33"),
    "err":  (" FAIL ", "1;31"),
    "skip": (" SKIP ", "1;90"),
}


def _line(status: str, label: str, value: str = "", detail: str = "", hint: str = "") -> None:
    badge, colour = _BADGE.get(status, ("  ??  ", "0"))
    print(f"{_c(badge, colour)} {label}" + (f"  {_c(value, '36')}" if value else ""))
    if detail:
        print(_c(f"       {detail}", "90"))
    if hint:
        print(_c(f"       → {hint}", "33"))       # hints are actionable: keep them bright


def render(report: dict) -> int:
    """Print the report; return the number of hard failures.

    Every section is now just a loop over check rows — the labels, statuses and
    remedies all live in the report data (pipeline/preflight.py and
    backends._backend_checks), so this renderer and doctor.html can't drift apart
    the way they did when each decided for itself what a missing env python means.
    """
    fails = warns = 0
    print(_c("\n=== Recon Studio 環境檢查 ===", "1"))

    print(_c("\n[系統 / 外部工具]", "1;34"))
    for chk in report["system"]:
        fails += chk["status"] == "err"
        warns += chk["status"] == "warn"
        _line(chk["status"], chk["label"], chk["value"], chk["detail"], chk["hint"])

    print(_c("\n[GPU]", "1;34"))
    if report["gpus"]:
        for g in report["gpus"]:
            _line("ok", f"#{g['index']} {g['name']}", f"{g['mem_mb'] / 1024:.1f} GB")
    else:
        _line("warn", "nvidia-smi / GPU", "未偵測到",
              "訓練和 GPU 解碼都需要 NVIDIA GPU", "確認驅動已安裝且 nvidia-smi 在 PATH")

    print(_c("\n[訓練 backends]", "1;34"))
    print(_c("       (未就緒只是該後端不能用,不算部署失敗)", "90"))
    for name, b in report["backends"].items():
        _line("ok" if b["ready"] else "warn", f"{b['label']} ({name})")
        for chk in b["checks"]:
            _line(chk["status"], f"  {chk['label']}", chk["value"], chk["detail"], chk["hint"])

    depth = report["depth"]
    print(_c("\n[深度 / 法向量 (LichtFeld preprocess)]", "1;34"))
    _line("ok" if depth["ready"] else "warn", "preprocess binary", depth["exec"],
          "" if depth["ready"] else "🌊 深度 工具不能用(訓練用其他後端不受影響)",
          "" if depth["ready"] else "在 ../LichtFeld-Studio 建置(與 lichtfeld-* 後端共用同一份 binary)")

    print()
    if fails:
        print(_c(f"✗ {fails} 項必要條件未滿足 — 上面 FAIL 的部分修好再開始用。", "1;31"))
    elif warns:
        print(_c(f"✓ 必要條件都通過({warns} 項選用功能有警告,見上面 WARN)。", "1;32"))
    else:
        print(_c("✓ 全部通過。", "1;32"))
    return fails


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="run.sh --doctor",
        description="Recon Studio 環境檢查(等同 /doctor 頁面,但輸出到終端機)")
    ap.add_argument("--fast", action="store_true",
                    help="跳過每個 backend 的 torch/CUDA 探測(快,但不會抓到顯卡 arch 不合)")
    ap.add_argument("--json", action="store_true", help="輸出原始 JSON")
    args = ap.parse_args(argv)

    report = doctor(deep=not args.fast)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    return 1 if render(report) else 0


if __name__ == "__main__":
    sys.exit(main())
