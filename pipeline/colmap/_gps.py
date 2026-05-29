"""EXIF / GPS helpers used by the COLMAP pipeline.

Pure stdlib (no Pillow/piexif): JPEG segments are walked to find the Exif APP1,
which we parse for the GPS IFD or graft back onto a re-encoded copy.
"""
from __future__ import annotations

import concurrent.futures as futures
import struct
from pathlib import Path

from ..runner import Cancelled, Runner


def _jpeg_gps_present(path: Path) -> bool:
    """True iff the JPEG at `path` carries a non-empty EXIF GPS IFD (lat + lon).

    Walks JPEG segments to the Exif APP1, parses the TIFF/IFD0 for the GPS IFD
    pointer (tag 0x8825), then confirms that GPS IFD holds GPSLatitude (0x0002)
    and GPSLongitude (0x0004). Best-effort — any parse hiccup or non-JPEG
    returns False."""
    try:
        with path.open("rb") as fh:
            if fh.read(2) != b"\xff\xd8":                 # not a JPEG (no SOI)
                return False
            app1 = b""
            while True:                                   # find the Exif APP1 segment
                hdr = fh.read(2)
                if len(hdr) < 2 or hdr[0] != 0xFF:
                    return False
                marker = hdr[1]
                if marker == 0xD8 or 0xD0 <= marker <= 0xD7:
                    continue                              # standalone markers, no length
                if marker == 0xDA or marker == 0xD9:      # SOS / EOI: pixel data, give up
                    return False
                seg_len = int.from_bytes(fh.read(2), "big")
                if seg_len < 2:
                    return False
                body = fh.read(seg_len - 2)
                if marker == 0xE1 and body[:6] == b"Exif\x00\x00":
                    app1 = body[6:]                       # the TIFF block
                    break
        if len(app1) < 8 or app1[:2] not in (b"II", b"MM"):
            return False
        bo = "<" if app1[:2] == b"II" else ">"
        ifd0 = struct.unpack(bo + "I", app1[4:8])[0]

        def tags(off: int) -> dict[int, int]:
            n = struct.unpack(bo + "H", app1[off:off + 2])[0]
            out: dict[int, int] = {}
            for k in range(n):
                e = off + 2 + k * 12
                tag = struct.unpack(bo + "H", app1[e:e + 2])[0]
                out[tag] = struct.unpack(bo + "I", app1[e + 8:e + 12])[0]
            return out

        gps_off = tags(ifd0).get(0x8825)
        if not gps_off:
            return False
        g = tags(gps_off)
        return 0x0002 in g and 0x0004 in g                # GPSLatitude + GPSLongitude
    except Exception:                                     # noqa: BLE001 — detection is best-effort
        return False


def read_app1_exif(path: Path) -> bytes | None:
    """Return the raw Exif APP1 segment (FF E1 + length + body) from a JPEG, or None.
    Walks the JPEG segments rather than scanning, so it grabs the real Exif block."""
    try:
        with path.open("rb") as fh:
            if fh.read(2) != b"\xff\xd8":
                return None
            while True:
                hdr = fh.read(2)
                if len(hdr) < 2 or hdr[0] != 0xFF:
                    return None
                marker = hdr[1]
                if marker == 0xD8 or 0xD0 <= marker <= 0xD7:
                    continue
                if marker == 0xDA or marker == 0xD9:      # SOS / EOI: header is over
                    return None
                ln = fh.read(2)
                body = fh.read(int.from_bytes(ln, "big") - 2)
                if marker == 0xE1 and body[:6] == b"Exif\x00\x00":
                    return b"\xff\xe1" + ln + body
    except Exception:                                     # noqa: BLE001
        return None


def graft_app1(dst: Path, app1: bytes) -> bool:
    """Splice an Exif APP1 segment in right after the SOI of the JPEG at `dst`.
    ffmpeg's mjpeg output has no EXIF, so re-inserting the original's restores GPS
    (and focal). The TIFF block is copied verbatim — its internal offsets stay valid.
    Returns True on success."""
    try:
        data = dst.read_bytes()
        if data[:2] != b"\xff\xd8" or not app1:
            return False
        dst.write_bytes(data[:2] + app1 + data[2:])
        return True
    except OSError:
        return False


def gps_coverage(img_root: str, lines: list[str], r: Runner, workers: int
                 ) -> tuple[int, int]:
    """Return (n_with_gps, n_total) for the inputs. The GPS pipeline needs a prior on
    EVERY image (a frame without one can't be spatially matched or anchored), so this
    can't sample — it checks each image (in parallel, the set can be large). Only JPEGs
    can carry EXIF GPS, so non-JPEG inputs count toward n_total but never toward
    n_with_gps — any PNG/TIFF therefore makes coverage incomplete."""
    n_total = len(lines)
    cand = [ln for ln in lines if Path(ln).suffix.lower() in (".jpg", ".jpeg")]
    if not cand:
        return 0, n_total
    n_gps = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        pending = [ex.submit(_jpeg_gps_present, Path(img_root) / rel) for rel in cand]
        for fut in futures.as_completed(pending):
            if r.cancelled:
                ex.shutdown(wait=False, cancel_futures=True)
                raise Cancelled()
            if fut.result():
                n_gps += 1
    return n_gps, n_total
