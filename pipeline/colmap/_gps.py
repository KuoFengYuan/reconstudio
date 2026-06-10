"""EXIF / GPS helpers used by the COLMAP pipeline.

Pure stdlib (no Pillow/piexif): JPEG segments are walked to find the Exif APP1,
which we parse for the GPS IFD or graft back onto a re-encoded copy. A *raw TIFF*
is itself a TIFF/IFD block (the very thing the JPEG APP1 wraps), so the same IFD
walk reads its GPS straight from byte 0 — no extra dependency.

Why TIFF needs special handling: COLMAP's own EXIF reader only parses JPEG GPS
(verified on 4.0.4 — a GPS-tagged TIFF yields zero `pose_priors` rows while the
identical JPEG yields one per image). So for TIFF inputs we read the GPS here and
write the priors into the DB ourselves via `inject_pose_priors`, matching the exact
row layout COLMAP writes for JPEG (coordinate_system=0 WGS84, position=<3d lat/lon/alt).
"""
from __future__ import annotations

import concurrent.futures as futures
import sqlite3
import struct
from pathlib import Path

from ..runner import Cancelled, Runner

TIFF_EXTS = (".tif", ".tiff")
JPEG_EXTS = (".jpg", ".jpeg")


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


def _jpeg_exif_tiff_block(path: Path) -> bytes | None:
    """For a JPEG, return just the Exif APP1's TIFF block (small — the EXIF segment, not
    the pixels). Walks segments via the file handle, so it never reads the whole image."""
    app1 = read_app1_exif(path)                           # b'\xff\xe1' + len(2) + body
    if not app1 or len(app1) < 12:
        return None
    tiff = app1[10:]                                      # skip FFE1 + len(2) + 'Exif\0\0'(6)
    return tiff if tiff[:2] in (b"II", b"MM") else None


def _parse_gps_ifd(read_at, ifd0_off: int, bo: str
                   ) -> tuple[float, float, float | None] | None:
    """Walk IFD0 -> GPS IFD and resolve (lat, lon, alt). `read_at(off, n)` returns n bytes
    at byte-offset `off` relative to the TIFF header (a seek+read for TIFF, a slice for the
    in-memory JPEG APP1). Reads only the few IFD/rational bytes it needs — never the pixels,
    so a 300 MB aerial TIFF costs a handful of small reads, not a full-file load."""
    def read_ifd(off: int) -> dict[int, tuple[int, int, int, int]]:
        # entry = (type, count, value_field_off, value_as_long). The entry's last 4 bytes
        # are the value inline when it fits (count*size <= 4) else a long offset to the
        # data; we keep both so scalars/refs read inline while arrays (rationals) and the
        # GPS IFD *pointer* read from value_as_long.
        cnt_b = read_at(off, 2)
        if len(cnt_b) < 2:
            return {}
        n = struct.unpack(bo + "H", cnt_b)[0]
        raw = read_at(off + 2, n * 12)
        out: dict[int, tuple[int, int, int, int]] = {}
        for k in range(n):
            e = k * 12
            tag, typ, cnt = struct.unpack(bo + "HHI", raw[e:e + 8])
            val_long = struct.unpack(bo + "I", raw[e + 8:e + 12])[0]
            out[tag] = (typ, cnt, off + 2 + e + 8, val_long)
        return out

    def data_off(entry: tuple[int, int, int, int]) -> int:
        typ, cnt, vfield, vlong = entry
        return vfield if cnt * _TYPE_SZ.get(typ, 0) <= 4 else vlong

    gps_ptr = read_ifd(ifd0_off).get(0x8825)
    if not gps_ptr:
        return None
    g = read_ifd(gps_ptr[3])                              # pointer's LONG = GPS IFD offset

    def rationals(tag: int) -> list[float] | None:
        if tag not in g:
            return None
        cnt = g[tag][1]
        raw = read_at(data_off(g[tag]), cnt * 8)
        out = []
        for k in range(cnt):
            num, den = struct.unpack(bo + "II", raw[k * 8:k * 8 + 8])
            out.append(num / den if den else 0.0)
        return out

    def ascii1(tag: int) -> str:
        if tag not in g:
            return ""
        cnt = g[tag][1]
        return read_at(data_off(g[tag]), cnt).decode("ascii", "ignore").strip("\x00").strip()

    lat, lon = rationals(0x0002), rationals(0x0004)
    if not lat or not lon or len(lat) < 3 or len(lon) < 3:
        return None
    dlat = lat[0] + lat[1] / 60 + lat[2] / 3600
    dlon = lon[0] + lon[1] / 60 + lon[2] / 3600
    if ascii1(0x0001).upper().startswith("S"):
        dlat = -dlat
    if ascii1(0x0003).upper().startswith("W"):
        dlon = -dlon
    alt_r = rationals(0x0006)
    alt = alt_r[0] if alt_r else None
    if (alt is not None and 0x0005 in g                  # AltitudeRef 1 = below sea level
            and read_at(data_off(g[0x0005]), 1) == b"\x01"):
        alt = -alt
    return dlat, dlon, alt


def image_gps_latlonalt(path: Path) -> tuple[float, float, float | None] | None:
    """Read (lat, lon, alt) in WGS84 (decimal degrees, metres) from a JPEG or TIFF's EXIF
    GPS IFD, applying N/S, E/W and altitude-ref signs. alt is None when absent. Returns None
    if there's no usable GPS. Best-effort: any parse hiccup returns None.

    Crucially seek-based for TIFF — these can be 100s of MB (aerial). The GPS pre-flight
    checks every image and the inject reads them again, so a full-file load per image would
    mean tens of GB of pointless I/O; here each image costs only a few small reads."""
    ext = path.suffix.lower()
    try:
        if ext in TIFF_EXTS:
            with path.open("rb") as fh:
                hdr = fh.read(8)
                if hdr[:2] not in (b"II", b"MM"):
                    return None
                bo = "<" if hdr[:2] == b"II" else ">"
                ifd0 = struct.unpack(bo + "I", hdr[4:8])[0]

                def read_at(off: int, n: int) -> bytes:
                    fh.seek(off)
                    return fh.read(n)

                return _parse_gps_ifd(read_at, ifd0, bo)
        else:                                            # JPEG (or anything with Exif APP1)
            tiff = _jpeg_exif_tiff_block(path)
            if not tiff:
                return None
            bo = "<" if tiff[:2] == b"II" else ">"
            ifd0 = struct.unpack(bo + "I", tiff[4:8])[0]
            return _parse_gps_ifd(lambda off, n: tiff[off:off + n], ifd0, bo)
    except Exception:                                    # noqa: BLE001 — best-effort
        return None


# EXIF type -> element byte size (for the inline-vs-offset 4-byte test)
_TYPE_SZ = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}


def inject_pose_priors(db_path: Path, img_root: str, names: list[str], r: Runner,
                       std_xyz: tuple[float, float, float] = (3.0, 3.0, 5.0)
                       ) -> tuple[int, int]:
    """Fill the COLMAP DB `pose_priors` table from EXIF GPS read off the ORIGINAL images.

    Only inserts for images that don't already have a prior — COLMAP populates JPEG GPS
    itself at feature extraction, so this targets the TIFF/PNG (and any JPEG COLMAP
    missed) inputs, and is a no-op when every image already has one. Rows are written in
    COLMAP's own layout (pose_prior_id = image_id, corr_data_id = image_id,
    corr_sensor_id = camera_id, corr_sensor_type = 0, position = <3d WGS84 lat/lon/alt,
    coordinate_system = 0) so pose_prior_mapper consumes them identically.

    `std_xyz` is the GPS position uncertainty in metres (x, y, z); we write a real diagonal
    covariance = diag(sx², sy², sz²), NOT NaN. COLMAP writes NaN for JPEG and relies on
    `--overwrite_priors_covariance` at mapping, but a plain incremental/global BA aligns to
    priors *opportunistically* and skips any prior whose covariance is NaN ("No pose priors
    with valid covariance found"). Baking a valid covariance in makes the in-BA GPS
    alignment work for every mapper; pose_prior_mapper's overwrite flag still takes
    precedence when set. Reads from `img_root` (the originals), NOT the EXIF-stripped FullHD
    copies. Returns (n_injected, n_existing)."""
    sx, sy, sz = std_xyz
    cov = struct.pack("<9d", sx * sx, 0.0, 0.0, 0.0, sy * sy, 0.0, 0.0, 0.0, sz * sz)
    grav = struct.pack("<3d", *([float("nan")] * 3))
    try:
        db = sqlite3.connect(str(db_path))
    except sqlite3.Error:
        return 0, 0
    try:
        tbls = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "pose_priors" not in tbls or "images" not in tbls:
            # empty/old DB or a COLMAP build without pose-prior support: nothing we can
            # safely write. pose_prior_mapper / spatial_matcher would also fall back.
            r.log("gps_inject: DB has no pose_priors/images table yet — skipping")
            return 0, 0
        have = {row[0] for row in db.execute("SELECT corr_data_id FROM pose_priors")}
        n_inj = 0
        for image_id, name, camera_id in db.execute(
                "SELECT image_id, name, camera_id FROM images").fetchall():
            if image_id in have:
                continue
            if r.cancelled:
                raise Cancelled()
            gps = image_gps_latlonalt(Path(img_root) / name)
            if not gps:
                continue
            lat, lon, alt = gps
            pos = struct.pack("<3d", lat, lon, 0.0 if alt is None else alt)
            db.execute(
                "INSERT INTO pose_priors(pose_prior_id, corr_data_id, corr_sensor_id, "
                "corr_sensor_type, position, position_covariance, gravity, "
                "coordinate_system) VALUES(?,?,?,?,?,?,?,?)",
                (image_id, image_id, camera_id, 0, pos, cov, grav, 0))
            n_inj += 1
        db.commit()
        return n_inj, len(have)
    finally:
        db.close()


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


def _has_gps(path: Path) -> bool:
    """True iff the image carries a usable EXIF GPS fix. JPEG uses the fast presence
    walk; TIFF reuses the full reader (a raw TIFF is the same IFD block, read from 0)."""
    ext = path.suffix.lower()
    if ext in JPEG_EXTS:
        return _jpeg_gps_present(path)
    if ext in TIFF_EXTS:
        return image_gps_latlonalt(path) is not None
    return False                                          # PNG etc. carry no EXIF GPS


def gps_coverage(img_root: str, lines: list[str], r: Runner, workers: int
                 ) -> tuple[int, int]:
    """Return (n_with_gps, n_total) for the inputs. The GPS pipeline needs a prior on
    EVERY image (a frame without one can't be spatially matched or anchored), so this
    can't sample — it checks each image (in parallel, the set can be large). JPEG and
    TIFF can both carry an EXIF GPS IFD; other formats (PNG) never do, so they count
    toward n_total but never toward n_with_gps and thus make coverage incomplete."""
    n_total = len(lines)
    cand = [ln for ln in lines
            if Path(ln).suffix.lower() in (JPEG_EXTS + TIFF_EXTS)]
    if not cand:
        return 0, n_total
    n_gps = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        pending = [ex.submit(_has_gps, Path(img_root) / rel) for rel in cand]
        for fut in futures.as_completed(pending):
            if r.cancelled:
                ex.shutdown(wait=False, cancel_futures=True)
                raise Cancelled()
            if fut.result():
                n_gps += 1
    return n_gps, n_total
