"""Job.save() must be crash-safe and thread-safe: job.json is replaced
atomically (a crash mid-write must never leave a truncated file that
_load_existing would silently drop), and the Runner thread's parser
mutations may not tear the snapshot that save()/to_dict() serialize."""
import json
import threading

import jobs as jobs_mod
from jobs import PARSERS, Job


def _job(jid="j1", **kw):
    return Job(id=jid, kind="frames", title="t", subtitle="s", **kw)


def test_save_writes_valid_json_and_no_tmp_leftover(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod, "JOBS_DIR", tmp_path)
    j = _job(meta={"total": 3})
    j.save()
    data = json.loads((tmp_path / "j1" / "job.json").read_text())
    assert data["id"] == "j1" and data["meta"] == {"total": 3}
    assert not (tmp_path / "j1" / "job.json.tmp").exists()


def test_save_overwrites_previous_record(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod, "JOBS_DIR", tmp_path)
    j = _job()
    j.save()
    j.status = "done"
    j.save()
    assert json.loads((tmp_path / "j1" / "job.json").read_text())["status"] == "done"


def test_roundtrip_through_dataclass_fields(tmp_path, monkeypatch):
    # _load_existing reconstructs via Job(**fields) — the _lock must not leak
    # into asdict() nor be expected as a constructor argument.
    monkeypatch.setattr(jobs_mod, "JOBS_DIR", tmp_path)
    j = _job(params={"fps": 1}, status="done")
    j.save()
    data = json.loads((tmp_path / "j1" / "job.json").read_text())
    assert "_lock" not in data
    j2 = Job(**{k: data[k] for k in data if k in Job.__dataclass_fields__})
    assert j2.params == {"fps": 1} and j2.status == "done"


def test_to_dict_is_a_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod, "JOBS_DIR", tmp_path)
    j = _job(meta={"kept": 1})
    d = j.to_dict()
    j.meta["kept"] = 2
    assert d["meta"]["kept"] == 1


def test_concurrent_parser_and_save(tmp_path, monkeypatch):
    """Hammer save() from one thread while the frames parser mutates meta from
    another (the real Runner-thread shape); every written job.json must parse."""
    monkeypatch.setattr(jobs_mod, "JOBS_DIR", tmp_path)
    j = _job()
    parser = PARSERS["frames"]
    stop = threading.Event()

    def feed():
        i = 0
        while not stop.is_set():
            i += 1
            j.parse_line(parser, f"######## [{i % 9 + 1}/9] clip.mp4")
            j.parse_line(parser, "-> 12 kept / 3 dropped")

    t = threading.Thread(target=feed)
    t.start()
    try:
        for _ in range(200):
            j.save()
            json.loads((tmp_path / "j1" / "job.json").read_text())
    finally:
        stop.set()
        t.join()
    assert j.meta["kept"] > 0
