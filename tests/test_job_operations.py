"""Job operations must stay responsive and report their actual outcome."""
import asyncio
import json
import threading

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import jobs as jobs_mod
from jobs import Job, JobManager
from web.routers import jobs as routes


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod, "JOBS_DIR", tmp_path)
    return JobManager()


def add_job(manager, status="done", **kw):
    job = Job(id="example", kind="frames", title="test", subtitle="", status=status, **kw)
    manager.register(job)
    return job


def request_with(data):
    async def receive():
        return {"type": "http.request", "body": json.dumps(data).encode()}
    return Request({"type": "http", "method": "POST", "path": "/api/jobs/delete"}, receive)


def test_summary_does_not_copy_pipeline_payload(manager):
    class Expensive:
        def __deepcopy__(self, memo):
            raise AssertionError("list UI must not copy reconstruction payloads")

    job = add_job(manager)
    job.params["points"] = Expensive()
    summary = manager.list(summaries=True)[0]
    assert summary["id"] == job.id
    assert "params" not in summary and "meta" not in summary
    assert summary["status"] == job.status


def test_cancel_queued_has_finish_time_and_persists(manager):
    job = add_job(manager, status="queued")
    assert asyncio.run(manager.cancel(job.id))
    assert job.status == "cancelled" and job.finished_at is not None
    assert json.loads((job.dir / "job.json").read_text())["finished_at"] == job.finished_at


def test_delete_cancelled_retains_record_until_next_action(manager):
    job = add_job(manager, status="queued")
    assert asyncio.run(manager.delete(job.id)) == "cancelled"
    assert manager.get(job.id) is job and job.dir.exists()
    assert asyncio.run(manager.delete(job.id)) == "deleted"
    assert manager.get(job.id) is None and not job.dir.exists()


def test_delete_failed_on_disk_keeps_record(manager, monkeypatch):
    job = add_job(manager)
    main_thread = threading.get_ident()

    def fail(path):
        assert threading.get_ident() != main_thread
        raise PermissionError("read-only")

    monkeypatch.setattr(jobs_mod.shutil, "rmtree", fail)
    assert asyncio.run(manager.delete(job.id)) == "failed"
    assert manager.get(job.id) is job


def test_delete_running_without_runner_reports_failure(manager):
    job = add_job(manager, status="running")
    assert asyncio.run(manager.delete(job.id)) == "failed"
    assert manager.get(job.id) is job


def test_cancel_runner_is_off_event_loop(manager):
    job = add_job(manager, status="running")
    calls = []

    class FakeRunner:
        def cancel(self):
            calls.append(threading.get_ident())

    manager.runners[job.id] = FakeRunner()
    assert asyncio.run(manager.cancel(job.id))
    assert calls and calls[0] != threading.get_ident()


def test_duplicate_batch_ids_cannot_cancel_then_delete(manager, monkeypatch):
    monkeypatch.setattr(routes, "manager", manager)
    job = add_job(manager, status="queued")
    response = asyncio.run(routes.delete_jobs(request_with({"ids": [job.id, job.id]})))
    assert response == {"results": {job.id: "cancelled"}}
    assert manager.get(job.id) is job


@pytest.mark.parametrize("data", [{}, {"ids": "example"}, {"ids": [None]}, {"ids": []}, {"ids": [[]]}, {"ids": ["x"] * 1001}])
def test_invalid_batch_does_not_mutate_jobs(manager, monkeypatch, data):
    monkeypatch.setattr(routes, "manager", manager)
    job = add_job(manager)
    with pytest.raises(HTTPException) as err:
        asyncio.run(routes.delete_jobs(request_with(data)))
    assert err.value.status_code == 422
    assert manager.get(job.id) is job


def test_full_log_download_uses_registered_job_path(manager, monkeypatch):
    monkeypatch.setattr(routes, "manager", manager)
    job = add_job(manager)
    job.log_path.write_text("完整紀錄\n")
    response = asyncio.run(routes.download_log(job.id))
    assert "example-console.log" in response.headers["content-disposition"]

    async def content():
        return b"".join([chunk async for chunk in response.body_iterator])

    assert asyncio.run(content()) == "完整紀錄\n".encode()
    with pytest.raises(HTTPException) as err:
        asyncio.run(routes.download_log("missing"))
    assert err.value.status_code == 404


def test_finished_pipeline_still_honors_cancellation(manager, monkeypatch):
    job = add_job(manager, status="queued")
    monkeypatch.setitem(jobs_mod.RUN_FUNCS, "frames", lambda params, runner: runner.cancel())
    asyncio.run(manager._run_job(job))
    assert job.status == "cancelled"
    assert job.finished_at is not None


def test_switch_log_during_slow_read_keeps_new_cursor(manager, monkeypatch):
    """A's pending disk read must not advance B's cursor or send A as B."""
    monkeypatch.setattr(routes, "manager", manager)
    a = add_job(manager, status="running")
    b = Job(id="second", kind="frames", title="B", subtitle="", status="running")
    manager.register(b)

    async def scenario():
        inbox = asyncio.Queue()
        output = []
        reads = []
        switched = asyncio.Event()
        real_sleep = asyncio.sleep

        class Socket:
            async def accept(self):
                for invalid in ([], 4, {"action": "watch_log", "job_id": []}):
                    await inbox.put(invalid)
                await inbox.put({"action": "watch_log", "job_id": a.id})

            async def receive_json(self):
                return await inbox.get()

            async def send_json(self, message):
                output.append(message)
                if message.get("type") == "log" and message.get("job") == b.id:
                    raise RuntimeError("test complete")

        async def read_in_thread(fn, path, pos, *args):
            reads.append((path, pos))
            if path == a.log_path:
                await inbox.put({"action": "watch_log", "job_id": b.id})
                await real_sleep(0)  # let the reader replace its subscription
                switched.set()
                return "A\n", 800, False, False
            assert switched.is_set()
            return "B\n", 2, False, False

        async def tick(_):
            await real_sleep(0)

        monkeypatch.setattr(routes.asyncio, "to_thread", read_in_thread)
        monkeypatch.setattr(routes.asyncio, "sleep", tick)
        await asyncio.wait_for(routes.ws_bus(Socket()), timeout=2)
        assert reads == [(a.log_path, None), (b.log_path, None)]
        logs = [m for m in output if m["type"] in ("log", "log_reset")]
        assert logs == [
            {"type": "log_reset", "job": b.id},
            {"type": "log", "job": b.id, "lines": ["B"]},
        ]

    asyncio.run(scenario())


def test_log_download_stops_at_snapshot_size_when_file_grows(tmp_path):
    path = tmp_path / "console.log"
    path.write_bytes(b"a" * 70000)
    chunks = routes._log_chunks(path)
    first = next(chunks)
    with path.open("ab") as stream:
        stream.write(b"new live output")
    assert first + b"".join(chunks) == b"a" * 70000
