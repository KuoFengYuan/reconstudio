"""The two routes that hand a trained splat to the viewer / to a download.

The bundled SuperSplat chooses its reader from the `filename` it is given — not
from the URL, not from the MIME type — so the *name* is part of the contract, not
cosmetic: a .sog announced as point_cloud.ply is parsed as PLY and fails. These
call the route coroutines directly (the repo has no HTTP test client) to lock the
filename round-trip and the 404 paths.
"""
from __future__ import annotations

import asyncio
import types

import pytest
from fastapi import HTTPException

from web.routers import viz


class _Manager:
    def __init__(self, job=None):
        self._job = job

    def get(self, _job_id):
        return self._job


def _job(model_path):
    return types.SimpleNamespace(meta={"model_path": str(model_path)}, params={})


def _use(monkeypatch, job):
    monkeypatch.setattr(viz, "manager", _Manager(job))


def test_splat_info_reports_the_real_filename(tmp_path, monkeypatch):
    d = tmp_path / "20260902_131140_2240" / "model"
    d.mkdir(parents=True)
    (d / "20260902_131140_2240.sog").write_bytes(b"sogbytes")
    _use(monkeypatch, _job(d))
    got = asyncio.run(viz.splat_info("j1"))
    body = got.body.decode()
    assert '"filename":"20260902_131140_2240.sog"' in body.replace(" ", "")
    assert '"size":8' in body.replace(" ", "")


def test_splat_download_serves_the_file_under_its_own_name(tmp_path, monkeypatch):
    d = tmp_path / "model"
    d.mkdir()
    f = d / "proj.sog"
    f.write_bytes(b"sogbytes")
    _use(monkeypatch, _job(d))
    resp = asyncio.run(viz.gaussians_splat("j1"))
    assert str(resp.path) == str(f)
    assert resp.filename == "proj.sog"     # NOT "point_cloud.ply"


def test_splat_404_when_the_job_is_unknown(monkeypatch):
    _use(monkeypatch, None)
    with pytest.raises(HTTPException) as e:
        asyncio.run(viz.splat_info("nope"))
    assert e.value.status_code == 404


def test_splat_404_explains_a_licht_only_output(tmp_path, monkeypatch):
    """The common case: training ran with 匯出格式 blank, so only project.licht
    exists. The message has to name the fix, or the user just sees "404"."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "project.licht").write_bytes(b"x")
    _use(monkeypatch, _job(d))
    with pytest.raises(HTTPException) as e:
        asyncio.run(viz.splat_info("j1"))
    assert e.value.status_code == 404
    assert "sog" in e.value.detail
