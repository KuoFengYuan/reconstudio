#!/usr/bin/env python
# ruff: noqa: UP031  # printf-style tables keep their column widths readable
"""COLMAP 重建成果驗收 —— 多鏡頭 rig / 航測專用。

    verify_recon.py <model_dir> [--manifest case.yaml] [--db DB] [--undistorted]

沒有 manifest 時只跑「與 case 無關」的三項(1/1b/5),任何 COLMAP 模型都能直接吃。
給了 manifest 才會加跑需要外部真值的三項(2/3/4)。

為什麼需要這個工具:
  model_aligner 報的 EO RMS 只驗證參考鏡頭的「位置」。實測遇過 EO RMS 1.83 m 看似正常,
  實際上有一整台相機 0 個觀測、rig 偏心量錯 100 倍、重建其實裂成三個互不相連的塊。
  下面每一項都是那次沒被抓到的東西。

回傳 0 = 全過, 1 = 有問題。
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import re
import sqlite3
import sys

import numpy as np
import pycolmap

# ─────────────────────────────────────────────────────────── manifest

DEFAULT_THRESH = {
    "epipolar_px": 5.0,        # 跨感測器對極 Sampson 中位數上限
    "roundtrip_px": 0.1,       # 畸變往返誤差上限
    "lever_arm_m": 0.5,        # rig 偏心量 vs 證書的容許差
    "colocation_m": 0.5,       # 同幀各感測器中心的容許差(僅在證書缺席時當粗略門檻)
}

def load_manifest(path):
    if path is None:
        return None
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        m = yaml.safe_load(text)
    else:
        import json
        m = json.loads(text)
    t = dict(DEFAULT_THRESH)
    t.update(m.get("thresholds") or {})
    m["thresholds"] = t
    return m

def cert_lever_arms(man):
    """證書 nodal-point 表 -> {sensor: C 在參考感測器相機座標系 (m)}。

    表格是在機體座標系(通常 X=前, Y=右, Z=下)量的,要轉到 COLMAP 的參考相機座標系
    (+X=影像右, +Y=影像下, +Z=光軸)。manifest 用 axes 宣告對應,例如
        axes: {front: "-Y", right: "+X", down: "+Z"}
    代表「機體 +X(前) = 相機 -Y」。
    """
    rig = (man or {}).get("rig")
    if not rig or not rig.get("nodal_points"):
        return None
    scale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[rig.get("units", "mm")]
    ax = rig.get("axes") or {"front": "-Y", "right": "+X", "down": "+Z"}
    basis = {}
    for body_axis, spec in ax.items():
        sign = -1.0 if spec.strip()[0] == "-" else 1.0
        idx = "XYZ".index(spec.strip()[-1].upper())
        v = np.zeros(3)
        v[idx] = sign
        basis[body_axis] = v
    ref = rig["reference"]
    tbl = rig["nodal_points"]
    if ref not in tbl:
        raise SystemExit(f"manifest: rig.reference '{ref}' 不在 nodal_points 裡")
    out = {}
    for name, xyz in tbl.items():
        d = (np.asarray(xyz, float) - np.asarray(tbl[ref], float)) * scale
        out[name] = d[0] * basis["front"] + d[1] * basis["right"] + d[2] * basis["down"]
    return out

# ─────────────────────────────────────────────────────────── 模型讀取

def sensor_names(rec, man):
    """camera_id -> 感測器名稱。優先 manifest,其次影像名的子目錄,最後 cam<id>。"""
    declared = {}
    for c in ((man or {}).get("cameras") or []):
        if "id" in c:
            declared[int(c["id"])] = c["name"]
    out = {}
    for im in rec.images.values():
        cid = im.camera_id
        if cid in out:
            continue
        if cid in declared:
            out[cid] = declared[cid]
        elif "/" in im.name:
            out[cid] = im.name.rsplit("/", 1)[0]
        else:
            out[cid] = f"cam{cid}"
    return out

def frame_of(rec, man):
    """image_id -> 幀鍵。優先用模型自帶的 frame_id(COLMAP 4.x),
    沒有 rig/frames 時才退回 manifest 提供的檔名 regex。"""
    if len(rec.frames) > 0:
        return {im.image_id: im.frame_id for im in rec.images.values()}
    pat = ((man or {}).get("frame_key") or {}).get("regex")
    if not pat:
        return None
    rx = re.compile(pat)
    out = {}
    for im in rec.images.values():
        mo = rx.search(im.name)
        if not mo:
            return None
        out[im.image_id] = mo.group(1)
    return out

def ref_sensor(rec, s2n, man):
    if len(rec.rigs):
        rig = next(iter(rec.rigs.values()))
        for sid in rig.sensor_ids():
            if rig.is_ref_sensor(sid):
                return s2n.get(sid.id)
    return ((man or {}).get("rig") or {}).get("reference")

def sec(t):
    print("\n" + "─" * 74 + "\n" + t)

# ─────────────────────────────────────────────────────────── 檢核

def check_observations(rec, s2n, order, bad):
    sec("1. 每感測器觀測數 / 重投影誤差")
    st = collections.defaultdict(lambda: dict(n=0, obs=0, err=[]))
    for im in rec.images.values():
        s = st[s2n[im.camera_id]]
        s["n"] += 1
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world()
        for p2 in im.points2D:
            if not p2.has_point3D():
                continue
            s["obs"] += 1
            uv = cam.img_from_cam(T * rec.points3D[p2.point3D_id].xyz)
            if uv is not None:
                s["err"].append(np.linalg.norm(uv - p2.xy))
    print("  %-12s %6s %10s %9s %11s" % ("感測器", "張數", "觀測數", "觀測/張", "重投影px"))
    for L in order:
        s = st[L]
        e = np.array(s["err"]) if s["err"] else np.array([np.nan])
        print("  %-12s %6d %10d %9.0f %11.3f"
              % (L, s["n"], s["obs"], s["obs"] / max(s["n"], 1), np.nanmedian(e)))
        if s["obs"] == 0:
            bad.append(f"{L} 觀測數為 0 —— 整台相機沒有任何三維點")
    if rec.num_points3D():
        tl = np.array([len(p.track.elements) for p in rec.points3D.values()])
        print("  track 長度 平均 %.2f 中位 %d  |  ≥3 視 %.1f%%  ≥5 視 %.1f%%"
              % (tl.mean(), np.median(tl), 100 * (tl >= 3).mean(), 100 * (tl >= 5).mean()))
        ref = order[0]
        seen = collections.Counter()
        for p in rec.points3D.values():
            seen[tuple(sorted({s2n[rec.images[e.image_id].camera_id] for e in p.track.elements}))] += 1
        nm = sum(v for k, v in seen.items() if ref in k and len(k) > 1)
        print("  含 %s 且跨感測器的三維點: %d (%.1f%%)"
              % (ref, nm, 100 * nm / rec.num_points3D()))

def check_intrinsics(rec, s2n, thr, bad):
    sec("1b. 內方位 / 畸變可逆性")
    for cid in sorted(rec.cameras):
        c = rec.cameras[cid]
        u, v = np.meshgrid(np.linspace(0, c.width - 1, 40), np.linspace(0, c.height - 1, 40))
        xy = np.c_[u.ravel(), v.ravel()]
        nn = c.cam_from_img(xy)
        ok = np.isfinite(nn).all(1)
        emax = np.inf
        if ok.any():
            bk = c.img_from_cam(np.c_[nn[ok], np.ones(ok.sum())])
            emax = np.linalg.norm(bk - xy[ok], axis=1).max()
        flag = " ⚠" if (ok.sum() < len(xy) or emax > thr["roundtrip_px"]) else ""
        print("  %-12s %-12s f=%9.1f pp=(%8.1f,%8.1f) 往返最大 %.4f px%s"
              % (s2n[cid], c.model.name, c.focal_length_x,
                 c.principal_point_x, c.principal_point_y, emax, flag))
        if flag:
            bad.append(f"{s2n[cid]} 畸變參數不可逆 (往返 {emax:.1f} px)")

def check_rig(rec, s2n, cert, thr, bad):
    if not len(rec.rigs):
        sec("2. rig 外參 —— 略過 (模型沒有 rig)")
        return
    sec("2. rig 外參" + (" vs 率定證書" if cert else " (無證書, 僅列出)"))
    rig = next(iter(rec.rigs.values()))
    print("  %-12s %30s %10s %10s %10s" % ("感測器", "C 在參考相機座標系(m)", "|C|", "證書|C|", "光軸夾角"))
    for sid in rig.sensor_ids():
        L = s2n.get(sid.id, f"cam{sid.id}")
        if rig.is_ref_sensor(sid):
            print("  %-12s  (參考感測器)" % L)
            continue
        s = rig.sensor_from_rig(sid)
        R = s.rotation.matrix()
        C = -R.T @ s.translation
        ang = np.degrees(np.arccos(np.clip((R.T @ np.array([0, 0, 1.0]))[2], -1, 1)))
        if cert and L in cert:
            d = float(np.linalg.norm(C - cert[L]))
            flag = " ⚠" if d > thr["lever_arm_m"] else ""
            print("  %-12s [%+8.3f %+8.3f %+8.3f] %10.3f %10.3f %9.2f°  Δ=%.3fm%s"
                  % (L, *C, np.linalg.norm(C), np.linalg.norm(cert[L]), ang, d, flag))
            if flag:
                bad.append(f"{L} rig 偏心量偏離證書 {d:.2f} m")
        else:
            print("  %-12s [%+8.3f %+8.3f %+8.3f] %10.3f %10s %9.2f°"
                  % (L, *C, np.linalg.norm(C), "-", ang))

def check_eo(rec, man, bad):
    eo_cfg = (man or {}).get("eo")
    if not eo_cfg:
        sec("3. 相機中心 vs 廠商 EO —— 略過 (manifest 未提供)")
        return
    sec("3. 相機中心 vs 廠商 EO")
    sch = eo_cfg.get("schema") or {}
    org = eo_cfg.get("local_origin") or {"E": 0, "N": 0, "H": 0}
    head = eo_cfg["head"]
    # ID -> 模型影像名。用「檔名主幹」比對,比不到就是比不到,不靜默略過。
    stem2name = {}
    for im in rec.images.values():
        base = im.name.rsplit("/", 1)[-1]
        stem2name.setdefault(os.path.splitext(base)[0], im.name)
    eo = {}
    unmatched = 0
    with open(eo_cfg["file"], newline="", encoding=eo_cfg.get("encoding", "utf-8-sig")) as fh:
        for row in csv.DictReader(fh):
            sid = row[sch.get("id", "ID")].strip()
            nm = stem2name.get(sid)
            if nm is None:
                unmatched += 1
                continue
            if head and not nm.startswith(head):
                continue
            eo[nm] = np.array([float(row[sch.get("e", "EASTING")]) - org["E"],
                               float(row[sch.get("n", "NORTHING")]) - org["N"],
                               float(row[sch.get("h", "ELLIPSOID HEIGHT")]) - org["H"]])
    d = np.array([im.projection_center() - eo[im.name]
                  for im in rec.images.values() if im.name in eo])
    if not len(d):
        bad.append("EO 與模型影像完全對不上 (檢查 schema.id 與檔名主幹)")
        print("  ✗ 0 筆對上")
        return
    rms = float(np.sqrt((d ** 2).sum(1).mean()))
    print("  n=%d  (EO 檔中 %d 筆在模型裡找不到對應影像)" % (len(d), unmatched))
    print("  bias E/N/U = {:+.3f} {:+.3f} {:+.3f} m".format(*tuple(d.mean(0))))
    print("  sd   E/N/U = {:6.3f} {:6.3f} {:6.3f} m   RMS3D {:.3f} m  max {:.3f} m".format(*d.std(0), rms, np.linalg.norm(d, axis=1).max()))
    lim = eo_cfg.get("max_rms_m")
    if lim and rms > lim:
        bad.append(f"EO RMS {rms:.2f} m 超過門檻 {lim:.2f} m")

def check_colocation(rec, s2n, fmap, ref, cert, thr, bad):
    if fmap is None or ref is None:
        sec("4. 同幀各感測器中心 —— 略過 (無 frame 資訊)")
        return
    sec("4. 同幀各感測器中心相對位移 (轉進參考相機座標系, 應等於證書偏心量)")
    # 必須先轉進參考相機座標系才能跟證書比:世界座標系下的位移會隨航向轉,
    # 而航帶是往返飛的(κ 差 180°),直接平均會互相抵銷。
    ref_cid = next(c for c in rec.cameras if s2n[c] == ref)
    fr = collections.defaultdict(dict)
    Rref = {}
    for im in rec.images.values():
        f = fmap[im.image_id]
        fr[f][s2n[im.camera_id]] = im.projection_center()
        if im.camera_id == ref_cid:
            Rref[f] = im.cam_from_world().rotation.matrix()
    for L in sorted({s2n[c] for c in rec.cameras}):
        if ref == L:
            continue
        dd = np.array([Rref[f] @ (v[L] - v[ref])
                       for f, v in fr.items() if L in v and ref in v and f in Rref])
        if not len(dd):
            continue
        mean = dd.mean(0)
        spread = float(np.linalg.norm(dd - mean, axis=1).max())
        nrm = float(np.linalg.norm(mean))
        if cert and L in cert:
            err = float(np.linalg.norm(mean - cert[L]))
            flag = " \u26a0" if err > thr["lever_arm_m"] else ""
            print("  %-12s n=%3d  [%+8.3f %+8.3f %+8.3f] |C|=%.3f  vs\u8b49\u66f8 \u0394=%.3f m  \u6563\u5e03\u6700\u5927 %.3f m%s"
                  % (L, len(dd), *mean, nrm, err, spread, flag))
            if flag:
                bad.append(f"{L} \u5be6\u969b\u4f4d\u79fb\u504f\u96e2\u8b49\u66f8 {err:.2f} m")
        else:
            flag = " \u26a0" if nrm > thr["colocation_m"] else ""
            print("  %-12s n=%3d  [%+8.3f %+8.3f %+8.3f] |C|=%.3f m  \u6563\u5e03\u6700\u5927 %.3f m%s"
                  % (L, len(dd), *mean, nrm, spread, flag))

def check_epipolar(rec, s2n, fmap, db_path, maxper, thr, bad):
    sec("5. 對極一致性 (DB 內點比對 + 模型姿態, 每組取 %d 對影像的中位 Sampson)" % maxper)
    db = sqlite3.connect(db_path)
    cur = db.cursor()
    kcur = db.cursor()          # 分開的 cursor: gk() 會重入,共用會把外層迭代重置
    ids = {i: n for i, n in cur.execute("select image_id,name from images")}
    kp = {}
    def gk(i):
        if i not in kp:
            q = kcur.execute("select rows,cols,data from keypoints where image_id=?", (i,)).fetchone()
            kp[i] = np.frombuffer(q[2], dtype=np.float32).reshape(q[0], q[1])[:, :2].astype(np.float64)
        return kp[i]
    pairs = cur.execute(
        "select pair_id,rows,cols,data from two_view_geometries where rows>50").fetchall()
    random.seed(0)
    random.shuffle(pairs)
    n2i = {im.name: im for im in rec.images.values()}
    res = collections.defaultdict(list)
    cnt = collections.Counter()
    for pid, rows, cols, data in pairs:
        i1 = pid // 2147483647
        i2 = pid % 2147483647
        if i2 == 0:
            i1 -= 1
            i2 = 2147483647
        n1, n2 = ids.get(i1), ids.get(i2)
        if n1 not in n2i or n2 not in n2i:
            continue
        im1, im2 = n2i[n1], n2i[n2]
        # 同幀 rig 基線 ~0.1 m,對極幾何退化,排除
        if fmap is not None and fmap[im1.image_id] == fmap[im2.image_id]:
            continue
        k = tuple(sorted((s2n[im1.camera_id], s2n[im2.camera_id])))
        if cnt[k] >= maxper:
            continue
        cnt[k] += 1
        m = np.frombuffer(data, dtype=np.uint32).reshape(rows, cols)
        c1, c2 = rec.cameras[im1.camera_id], rec.cameras[im2.camera_id]
        A = np.c_[c1.cam_from_img(gk(i1)[m[:, 0]]), np.ones(rows)]
        B = np.c_[c2.cam_from_img(gk(i2)[m[:, 1]]), np.ones(rows)]
        T = im2.cam_from_world() * im1.cam_from_world().inverse()
        R = T.rotation.matrix()
        t = T.translation
        E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
        Ea = A @ E.T
        Etb = B @ E
        den = np.sqrt(Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2)
        res[k].append(float(np.median(np.abs(np.einsum("ij,ij->i", B, Ea))
                                      / np.maximum(den, 1e-12)) * c1.focal_length_x))
    if not res:
        print("  (沒有可用的比對對) —— DB 與模型是否對得上?")
        return
    worst, worst_k = 0.0, None
    for k in sorted(res, key=lambda k: np.median(res[k])):
        v = float(np.median(res[k]))
        if v > worst:
            worst, worst_k = v, k
        print("  %-26s n=%2d  %9.2f px%s"
              % ("+".join(k), len(res[k]), v, " ⚠" if v > thr["epipolar_px"] else ""))
    if worst > thr["epipolar_px"]:
        bad.append("對極誤差最差 {:.1f} px ({}) 超過門檻 {:.1f}".format(worst, "+".join(worst_k), thr["epipolar_px"]))

# ─────────────────────────────────────────────────────────── main

def main():
    ap = argparse.ArgumentParser(description="COLMAP 多鏡頭重建驗收")
    ap.add_argument("model", help="COLMAP sparse 模型目錄")
    ap.add_argument("--manifest", help="case 描述檔 (.yaml/.json);沒給就只跑 1/1b/5")
    ap.add_argument("--db", help="database.db (對極測試用);未給則取 manifest 的 database")
    ap.add_argument("--undistorted", action="store_true",
                    help="模型已去畸變 —— 座標系與 DB keypoints 不同,跳過對極測試")
    ap.add_argument("--maxper", type=int, default=12)
    a = ap.parse_args()

    man = load_manifest(a.manifest)
    thr = (man or {}).get("thresholds", DEFAULT_THRESH)
    rec = pycolmap.Reconstruction(a.model)
    s2n = sensor_names(rec, man)
    fmap = frame_of(rec, man)
    ref = ref_sensor(rec, s2n, man) or s2n[min(rec.cameras)]
    cert = cert_lever_arms(man)
    order = [ref] + sorted({v for v in s2n.values()} - {ref})

    print("模型:", a.model, "" if man else "  (無 manifest —— 只跑與 case 無關的 1/1b/5)")
    print("  cameras %d  images %d  frames %d  points3D %d  參考感測器 %s"
          % (rec.num_cameras(), rec.num_images(), len(rec.frames), rec.num_points3D(), ref))

    bad = []
    check_observations(rec, s2n, order, bad)
    check_intrinsics(rec, s2n, thr, bad)
    check_rig(rec, s2n, cert, thr, bad)
    check_eo(rec, man, bad)
    check_colocation(rec, s2n, fmap, ref, cert, thr, bad)

    db = a.db or (man or {}).get("database")
    if a.undistorted:
        sec("5. 對極一致性 —— 略過 (--undistorted)")
    elif not db:
        sec("5. 對極一致性 —— 略過 (未提供 --db)")
    else:
        check_epipolar(rec, s2n, fmap, db, a.maxper, thr, bad)

    sec("驗收結果")
    if bad:
        for b in bad:
            print("  ✗", b)
    else:
        print("  ✓ 全部通過")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
