"""Known interior orientation -> COLMAP camera parameters.

Aerial vendors ship a calibration certificate per camera head, in the
photogrammetric (Australis / Brown) convention: principal distance C and
principal-point offsets XP/YP in millimetres, radial K1..K3 and decentring
P1/P2 in per-millimetre powers, plus affinity terms B1/B2. COLMAP instead wants
pixels and normalised image coordinates.

What this module converts, and what it deliberately does not:

  focal length      C / pixel_size. Unambiguous, and the highest-value part:
                    without it COLMAP guesses from EXIF, which on the block this
                    was written for is ~2% off, differently per head.
  principal point   W/2 + XP/pixel_size, H/2 - YP/pixel_size. The y sign is a
                    convention (the certificate's y axis points up, COLMAP's
                    image y points down); `flip_pp_y=False` is there for
                    certificates that already use image coordinates.
  sensor size       straight from the certificate, so the model is not tied to
                    whatever resolution the images happen to be at.
  distortion        NOT converted. The K/P/B terms are per-mm and follow the
                    "correction to add" sign convention, while COLMAP's models
                    apply distortion in normalised coordinates; COLMAP's OPENCV
                    cannot represent K3/B1/B2 at all. A sign or scale slip there
                    silently deforms the geometry, and COLMAP self-calibrates
                    distortion well from a few hundred images, so the parameters
                    are emitted as zeros and left to the bundle. Seeding them is
                    a separate exercise that needs its own validation.

The output is per-camera `camera_params`, which is exactly what a rig config
accepts (scene/rig.cc copies them into the database), so a rig gets one correct
calibration per head rather than one shared guess.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# COLMAP parameter orders we emit. Distortion terms are always zero here.
_MODEL_PARAMS = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k1"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "FULL_OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2",
                    "k3", "k4", "k5", "k6"),
}


@dataclass(frozen=True)
class Calibration:
    """One camera head's interior orientation, in certificate units."""
    name: str                 # as printed in the certificate, e.g. "Nadir"
    width: int                # sensor columns, px
    height: int               # sensor rows, px
    pixel_size_mm: float      # e.g. 0.00376 for 3.76 um
    c_mm: float               # principal distance
    xp_mm: float = 0.0        # principal point offset, certificate convention
    yp_mm: float = 0.0
    detail: str = ""          # free text (model / serial), for the log

    @property
    def focal_px(self) -> float:
        return self.c_mm / self.pixel_size_mm

    def principal_point_px(self, flip_pp_y: bool = True) -> tuple[float, float]:
        cx = self.width / 2.0 + self.xp_mm / self.pixel_size_mm
        dy = self.yp_mm / self.pixel_size_mm
        cy = self.height / 2.0 + (-dy if flip_pp_y else dy)
        return cx, cy

    def colmap_params(self, model: str = "OPENCV", *, use_pp: bool = True,
                      flip_pp_y: bool = True) -> list[float]:
        """Parameter vector in COLMAP's order for `model`, distortion zeroed."""
        model = model.upper()
        if model not in _MODEL_PARAMS:
            raise ValueError(f"unsupported camera model {model!r}; "
                             f"expected one of {sorted(_MODEL_PARAMS)}")
        f = self.focal_px
        cx, cy = (self.principal_point_px(flip_pp_y) if use_pp
                  else (self.width / 2.0, self.height / 2.0))
        named = {"f": f, "fx": f, "fy": f, "cx": cx, "cy": cy}
        return [float(named.get(p, 0.0)) for p in _MODEL_PARAMS[model]]


# "C   89.8040   0.00000   89.8040 ..." — take the FINAL value (3rd column) when
# the row has adjustment columns, else the first number on the row.
_ROW = r"^\s*{key}\s+(-?\s?[0-9.]+(?:[eE][-+]?\d+)?)"


def _num(body: str, key: str) -> float | None:
    m = re.search(_ROW.format(key=re.escape(key)), body, re.M)
    return float(m.group(1).replace(" ", "")) if m else None


def parse_australis_report(text: str) -> list[Calibration]:
    """Pull one Calibration per '<name> Camera Calibration Report' section.

    Tolerant by design: certificates are PDFs converted with pdftotext, so
    spacing is unreliable and unrelated sections come and go. A section is only
    accepted when it carries the three values that matter (resolution, pixel
    size, C) — anything else is skipped rather than half-parsed.
    """
    out: list[Calibration] = []
    parts = re.split(r"\n\s*\d+\.\s+(\w[\w /-]*?)\s+Camera Calibration Report", text)
    for i in range(1, len(parts) - 1, 2):
        name, body = parts[i].strip(), parts[i + 1]
        res = re.search(r"Sensor Resolution:\s*x\s*(\d+)\s*\*\s*y\s*(\d+)", body)
        px = re.search(r"Pixel Size:\s*([0-9.]+)", body)
        c = _num(body, "C")
        if not (res and px and c):
            continue
        cam = re.search(r"Camera:\s*(.+)", body)
        out.append(Calibration(
            name=name,
            width=int(res.group(1)),
            height=int(res.group(2)),
            pixel_size_mm=float(px.group(1)) / 1000.0,   # um -> mm
            c_mm=c,
            xp_mm=_num(body, "XP") or 0.0,
            yp_mm=_num(body, "YP") or 0.0,
            detail=(cam.group(1).strip() if cam else ""),
        ))
    return out


def match_to_cameras(cals: list[Calibration], cameras: list[str],
                     ) -> tuple[dict[str, Calibration], list[str]]:
    """Map rig camera ids -> Calibration by case-insensitive name.

    The rig's camera ids come from the data (folder names), and the certificate's
    head names come from the vendor, so nothing here assumes a naming scheme
    beyond "they refer to the same head by the same word". Returns the mapping
    plus the camera ids that found no calibration.
    """
    by_name = {c.name.strip().lower(): c for c in cals}
    hit: dict[str, Calibration] = {}
    missed: list[str] = []
    for cam in cameras:
        key = cam.strip().lower()
        cal = by_name.get(key)
        if cal is None:                       # allow "nadir_1" / "cam-nadir"
            cal = next((v for k, v in by_name.items() if k in key or key in k), None)
        if cal is None:
            missed.append(cam)
        else:
            hit[cam] = cal
    return hit, missed


# --------------------------------------------------------------------------- #
# discovery: find a certificate in the dataset instead of asking for a path
# --------------------------------------------------------------------------- #
# Vendors drop the certificate in with the imagery, under whatever name and in
# whatever language. Rather than matching filenames, every candidate document is
# read and offered to the parsers — a file either yields calibrations or it does
# not, which is a far more robust test than guessing at names.
_DOC_SUFFIXES = (".pdf", ".txt", ".json")
_MAX_DOC_BYTES = 64 * 1024 * 1024        # skip anything implausibly large
_SCAN_DEPTH = 2                          # dataset root, plus one level of subdirs


def _pdf_to_text(path: Path) -> str:
    """pdftotext -layout, or "" when the tool or the file is unusable.

    -layout matters: the certificates are tables, and without it the columns
    interleave and the parameter rows stop being parseable.
    """
    exe = shutil.which("pdftotext")
    if not exe:
        return ""
    try:
        p = subprocess.run([exe, "-layout", str(path), "-"],
                           capture_output=True, text=True, timeout=120, check=False)
        return p.stdout if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _read_json_calibrations(text: str) -> list[Calibration]:
    """A hand-written escape hatch, for certificates this cannot parse.

    [{"name": "nadir", "width": 14204, "height": 10652,
      "pixel_size_um": 3.76, "c_mm": 89.804, "xp_mm": 0.0005, "yp_mm": -0.4069}]
    """
    try:
        rows = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(rows, list):
        return []
    out: list[Calibration] = []
    for row in rows:
        try:
            out.append(Calibration(
                name=str(row["name"]),
                width=int(row["width"]),
                height=int(row["height"]),
                pixel_size_mm=float(row["pixel_size_um"]) / 1000.0,
                c_mm=float(row["c_mm"]),
                xp_mm=float(row.get("xp_mm", 0.0)),
                yp_mm=float(row.get("yp_mm", 0.0)),
                detail=str(row.get("detail", "hand-written JSON")),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def calibrations_from_file(path: Path) -> list[Calibration]:
    """Parse one document, trying each known format. [] if it is not one."""
    try:
        if path.stat().st_size > _MAX_DOC_BYTES:
            return []
    except OSError:
        return []
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_australis_report(_pdf_to_text(path))
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if suffix == ".json":
        return _read_json_calibrations(text)
    return parse_australis_report(text)


def discover_calibrations(roots: Iterable[Path],
                          ) -> tuple[list[Calibration], Path | None]:
    """Scan `roots` for a calibration document and return the first that parses.

    Deterministic: candidates are visited in sorted order, so the same dataset
    always resolves to the same certificate. Returns ([], None) when the dataset
    ships no calibration, which is the normal case and not an error.
    """
    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for depth in range(_SCAN_DEPTH):
            pattern = "/".join(["*"] * (depth + 1))
            for p in sorted(root.glob(pattern)):
                rp = p.resolve()
                if (p.is_file() and p.suffix.lower() in _DOC_SUFFIXES
                        and rp not in seen):
                    seen.add(rp)
                    candidates.append(p)
    for path in candidates:
        cals = calibrations_from_file(path)
        if cals:
            return cals, path
    return [], None
