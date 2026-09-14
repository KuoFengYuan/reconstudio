"""History search combines project paths, category vocabulary and contextual facets."""
import asyncio

import pytest
from starlette.requests import Request

from web.routers import jobs as routes
from web.services.job_history import history_context


def job(id, kind="train", status="done", title="corrected", subtitle="/project/花瓶/corrected"):
    return dict(id=id, kind=kind, status=status, title=title, subtitle=subtitle,
                created_at=1, started_at=None, finished_at=None, current_stage="done")


@pytest.fixture
def records():
    return [job("train-a"), job("colmap-b", "colmap", "failed"),
            job("train-c", status="failed", subtitle="/project/木/model"),
            job("split-d", "blocksplit"), job("train-e", title="other", subtitle="/project/花瓶/model")]


@pytest.mark.parametrize(("query", "ids"), [
    ("花瓶 訓練", ["train-a", "train-e"]),
    ("ＣＯＬＭＡＰ　失敗", ["colmap-b"]),
    ("花瓶 /corrected", ["train-a", "colmap-b", "split-d"]),
    ("TRAIN-A", ["train-a"]),
    ("  3dgs  花瓶  ", ["train-a", "train-e"]),
    ("不存在", []),
])
def test_search_project_paths_aliases_and_multiple_terms(records, query, ids):
    context = history_context(records, query, "all", "all", 50)
    assert [row["id"] for row in context["jobs"]] == ids


def test_facet_counts_apply_other_axis_but_global_summary_is_stable(records):
    context = history_context(records, "花瓶", "train", "failed", 50)
    assert context["total"] == 0
    assert context["kind_counts"] == {"colmap": 1}
    assert context["kind_total"] == 1
    assert context["status_counts"] == {"done": 2}
    assert context["status_total"] == 2
    assert context["summary_counts"] == {"done": 3, "failed": 2}
    assert context["all_total"] == 5


def test_search_happens_before_pagination_and_does_not_mutate_summaries():
    records = [job(str(i), subtitle="/other") for i in range(60)] + [job("last")]
    context = history_context(records, "花瓶", "train", "all", 10)
    assert [row["id"] for row in context["jobs"]] == ["last"]
    assert context["kind_total"] == context["total"] == 1
    assert "params" not in records[-1]
    assert len(history_context(records, "", "all", "all", 10)["jobs"]) == 10


def test_unknown_kind_is_discoverable_and_invalid_filters_reset():
    records = [job("future", kind="new-kind")]
    context = history_context(records, "", "new-kind", "bogus", 50)
    assert context["categories"][-1]["label"] == "new-kind"
    assert context["sel_kind"] == "new-kind" and context["sel_status"] == "all"
    assert history_context(records, "", "missing", "all", 50)["sel_kind"] == "all"


def test_route_uses_lightweight_summaries_and_escapes_search_and_paths(monkeypatch):
    class Manager:
        def list(self, *, summaries):
            assert summaries is True
            return [job("test", subtitle='<script>alert("x")</script>')]

    monkeypatch.setattr(routes, "manager", Manager())
    request = Request({"type": "http", "method": "GET", "path": "/ui/joblist", "headers": []})
    response = asyncio.run(routes.joblist(request, q="<script>"))
    html = response.body.decode()
    assert "找到 1 筆任務" in html
    assert '<script>alert(' not in html
    assert "&lt;script&gt;" in html
    assert "依工作類型尋找" in html and "照片重建" in html
    assert 'aria-pressed="true"' in html


def test_empty_history_renders_without_missing_counts(monkeypatch):
    class Manager:
        def list(self, **kwargs):
            return []

    monkeypatch.setattr(routes, "manager", Manager())
    request = Request({"type": "http", "method": "GET", "path": "/ui/joblist", "headers": []})
    html = asyncio.run(routes.joblist(request)).body.decode()
    assert "還沒有任務紀錄" in html
