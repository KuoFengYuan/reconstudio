"""User-facing job categories and lightweight history search (no pipeline payloads)."""
import unicodedata
from collections import Counter

# Keep search vocabulary and displayed names together.
CATEGORIES = [
    {"key": "frames", "label": "影片抽幀", "description": "影片 → 清晰照片", "aliases": "抽幀 抽帧 frames video 照片擷取"},
    {"key": "colmap", "label": "照片重建", "description": "COLMAP · 相機定位與點雲", "aliases": "colmap sfm 相機位姿 稀疏重建"},
    {"key": "train", "label": "模型訓練", "description": "3DGS · 高斯模型", "aliases": "train training 3dgs 訓練 训练 gaussian splat"},
    {"key": "mesh", "label": "網格模型", "description": "Mesh · 產生三角網格", "aliases": "mesh 網格 网格 三角網格"},
    {"key": "gcs", "label": "雲端傳輸", "description": "GCS · 上傳與下載", "aliases": "gcs 雲端資料 云端 傳輸 传输 upload download"},
    {"key": "depth", "label": "深度與法線", "description": "產生深度圖、法向量", "aliases": "depth normal 深度 法線 法线"},
    {"key": "matte", "label": "影像去背", "description": "移除照片背景", "aliases": "matte mask 去背 遮罩 背景移除"},
    {"key": "blocksplit", "label": "大場景分塊", "description": "切成可獨立訓練的子場景", "aliases": "blocksplit 分塊 分块 切塊"},
]
STATUS_LABELS = {"running": "執行中", "queued": "排隊中", "done": "已完成", "failed": "失敗", "cancelled": "已取消"}


def normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def history_context(all_jobs: list[dict], q: str, kind: str, status: str, limit: int) -> dict:
    categories = [dict(category) for category in CATEGORIES]
    known = {category["key"] for category in categories}
    for key in sorted({job["kind"] for job in all_jobs} - known):
        categories.append({"key": key, "label": key, "description": "其他任務", "aliases": key})
    labels = {category["key"]: category["label"] for category in categories}
    vocabulary = {category["key"]: " ".join(category[field] for field in ("key", "label", "aliases")) for category in categories}
    kind = kind if kind in labels else "all"
    status = status if status in STATUS_LABELS else "all"
    q = q.strip()
    terms = normalize(q).split()

    def matches(job):
        # Subtitles contain source/output paths, allowing one project to span job kinds.
        haystack = normalize(" ".join(str(job.get(field) or "") for field in
                                      ("title", "subtitle", "id", "kind", "status")) + " " +
                             vocabulary.get(job["kind"], "") + " " + STATUS_LABELS.get(job["status"], ""))
        return all(term in haystack for term in terms)

    searched = [job for job in all_jobs if matches(job)] if terms else all_jobs
    # Each facet includes search + the OTHER facet, so counts predict switching it.
    kind_scope = [job for job in searched if status == "all" or job["status"] == status]
    status_scope = [job for job in searched if kind == "all" or job["kind"] == kind]
    filtered = [job for job in kind_scope if kind == "all" or job["kind"] == kind]
    limit = max(10, min(int(limit or 50), 1000))
    return {
        "jobs": filtered[:limit], "total": len(filtered), "all_total": len(all_jobs),
        "categories": categories, "kinds_map": labels, "status_labels": STATUS_LABELS,
        "summary_counts": dict(Counter(job["status"] for job in all_jobs)),
        "kind_counts": dict(Counter(job["kind"] for job in kind_scope)), "kind_total": len(kind_scope),
        "status_counts": dict(Counter(job["status"] for job in status_scope)), "status_total": len(status_scope),
        "q": q, "sel_kind": kind, "sel_status": status, "limit": limit,
    }
