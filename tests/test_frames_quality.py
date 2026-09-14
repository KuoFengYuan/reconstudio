"""Quality gates must reject bad scores and never mix reruns with stale images."""
import asyncio
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
from starlette.requests import Request

from pipeline import frames
from pipeline.runner import Cancelled, PipelineError, Runner
from web.routers import create


@pytest.mark.parametrize('params', [
    {'fps': '0'}, {'fps': 0}, {'fps': '-1'}, {'fps': 'nan'}, {'fps': 'inf'},
    {'mode': 'other'}, {'mode': 'hybrid', 'keep_pct': '0'},
    {'mode': 'hybrid', 'keep_pct': '101'}, {'mode': 'hybrid', 'keep_pct': '1.5'},
    {'mode': 'hybrid', 'threshold': '-1'}, {'mode': 'hybrid', 'threshold': 'nan'},
    {'mode': 'threshold', 'threshold': 'inf'}, {'mode': 'hybrid', 'threshold': 0},
])
def test_invalid_options_rejected(params):
    with pytest.raises(ValueError):
        frames.validate_frame_options(params)


def test_defaults_and_legacy_saved_jobs():
    assert frames.validate_frame_options({}) == dict(mode='hybrid', fps='1', threshold='8', keep_pct='70')
    assert frames.validate_frame_options({'keep_pct': '90'})['mode'] == 'percentile'
    assert frames.validate_frame_options({'threshold': '5', 'keep_pct': '90'})['keep_pct'] is None
    assert frames.validate_frame_options({'mode': 'threshold', 'keep_pct': 'invalid'})['threshold'] == '8'


def test_hybrid_never_fills_quota_with_blurry_frames():
    rows = [('a', 3), ('b', 5), ('c', 12), ('d', 20), ('e', math.nan)]
    decisions, _ = frames._select_frames(rows, '8', '90')
    assert decisions == dict(a='kept', b='kept', c='above_threshold', d='above_threshold', e='invalid_score')


def test_tied_scores_do_not_exceed_quota_and_threshold_only_keeps_all_good():
    rows = [('c', 3), ('a', 3), ('b', 3), ('d', math.inf)]
    decisions, _ = frames._select_frames(rows, '8', '70')
    assert {k for k, v in decisions.items() if v == 'kept'} == {'a', 'b'}
    decisions, _ = frames._select_frames(rows, '8', None)
    assert sum(v == 'kept' for v in decisions.values()) == 3


class FakeRunner:
    def __init__(self, scores, image_count=None):
        self.scores = scores
        self.image_count = len(scores) if image_count is None else image_count
        self.logs = []
        self.commands = []

    def log(self, line):
        self.logs.append(line)

    def check_cancel(self):
        pass

    def run(self, cmd, *, stderr_to, check):
        self.commands.append(cmd)
        folder = Path(cmd[-1]).parent
        for i in range(self.image_count):
            (folder / f'frame_{i + 1:06}.jpg').write_bytes(f'new {i + 1}'.encode())
        Path(stderr_to).write_text(''.join(f'[Parsed_blurdetect_1 @ 0x1] blur: {s}\n' for s in self.scores) +
                                  '[Parsed_blurdetect_1 @ 0x1] blur mean: 99.0\n')


def process(out, runner, **kw):
    options = dict(fps='1', threshold='8', keep_pct='70', flatten=True, hwaccel=False, ffmpeg='ffmpeg', r=runner)
    options.update(kw)
    return frames._process_video('/fake/video.mp4', str(out), **options)


def test_rerun_replaces_old_generated_frames_and_reports_why(tmp_path):
    (tmp_path / 'frame_000004.jpg').write_bytes(b'old blurry frame')
    (tmp_path / 'frame_000099.jpg').write_bytes(b'stale tail')
    (tmp_path / 'notes.txt').write_text('keep me')
    runner = FakeRunner([3, '-nan', 4, 12, 20])
    assert process(tmp_path, runner)
    assert sorted(p.name for p in tmp_path.glob('frame_*.jpg')) == ['frame_000001.jpg', 'frame_000003.jpg']
    assert (tmp_path / 'notes.txt').read_text() == 'keep me'
    with (tmp_path / 'blur_scores.csv').open() as f:
        report = list(csv.DictReader(f))
    assert [r['decision'] for r in report] == ['kept', 'invalid_score', 'kept', 'above_threshold', 'above_threshold']
    assert json.loads((tmp_path / 'frame_quality.json').read_text())['kept'] == 2
    assert not list(tmp_path.glob('.frames-work-*'))
    assert any('-> 2 kept / 3 dropped' in line for line in runner.logs)


@pytest.mark.parametrize('scores,count', [([3, 4], 3), ([3, 4, 5], 2)])
def test_score_count_mismatch_preserves_previous_output(tmp_path, scores, count):
    previous = tmp_path / 'frame_000099.jpg'
    previous.write_bytes(b'previous result')
    with pytest.raises(PipelineError, match='不一致'):
        process(tmp_path, FakeRunner(scores, count))
    assert previous.read_bytes() == b'previous result'
    assert not (tmp_path / 'frame_quality.json').exists()
    assert not list(tmp_path.glob('.frames-work-*'))


@pytest.mark.parametrize('scores', [[12, 15], ['-nan', 'inf'], [-1, '-inf']])
def test_all_bad_does_not_publish_a_success_or_delete_previous_result(tmp_path, scores):
    old = tmp_path / 'frame_000001.jpg'
    old.write_bytes(b'previous result')
    with pytest.raises(PipelineError, match='沒有符合'):
        process(tmp_path, FakeRunner(scores))
    assert old.read_bytes() == b'previous result'
    assert json.loads((tmp_path / 'rejected_quality.json').read_text())['kept'] == 0


def test_publish_cancellation_rolls_back_old_frames(tmp_path):
    old = tmp_path / 'frame_000001.jpg'
    old.write_bytes(b'old')
    stage = tmp_path / 'stage'
    stage.mkdir()
    fresh = stage / 'frame_000002.jpg'
    fresh.write_bytes(b'new')

    class CancelAfterFirstMove:
        calls = 0

        def check_cancel(self):
            self.calls += 1
            if self.calls == 3:
                raise Cancelled()

    with pytest.raises(Cancelled):
        frames._publish_frames(stage, tmp_path, [(fresh, tmp_path / fresh.name)], CancelAfterFirstMove())
    assert old.read_bytes() == b'old'
    assert not (tmp_path / fresh.name).exists()
    assert not (stage / 'previous').exists()


def test_gpu_retry_clears_partial_frames(tmp_path):
    class RetryRunner(FakeRunner):
        def run(self, cmd, **kw):
            if '-hwaccel' in cmd:
                (Path(cmd[-1]).parent / 'frame_999999.jpg').write_bytes(b'partial')
                raise PipelineError('NVDEC failed')
            super().run(cmd, **kw)

    assert process(tmp_path, RetryRunner([3, 4]), hwaccel=True, keep_pct=None)
    assert not (tmp_path / 'frame_999999.jpg').exists()
    assert len(list(tmp_path.glob('frame_*.jpg'))) == 2


def test_colliding_video_outputs_are_rejected_before_extraction(tmp_path, monkeypatch):
    for suffix in ('.mp4', '.mov'):
        (tmp_path / ('clip' + suffix)).touch()
    monkeypatch.setattr(frames, '_supports_cuda', lambda _: False)
    with pytest.raises(ValueError, match='同一輸出'):
        frames.run_frames(dict(inputs=[str(tmp_path)], out_dir=str(tmp_path / 'out')), FakeRunner([]))
    assert not (tmp_path / 'out').exists()


def test_partial_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    for name in ('a.mp4', 'b.mp4'):
        (tmp_path / name).touch()
    monkeypatch.setattr(frames, '_supports_cuda', lambda _: False)
    monkeypatch.setattr(frames, '_process_video', lambda video, *a, **kw: video.endswith('a.mp4'))
    runner = FakeRunner([])
    runner.cancelled = False
    with pytest.raises(RuntimeError, match='1 video'):
        frames.run_frames(dict(inputs=[str(tmp_path)], workers=1), runner)


def test_create_route_stores_both_quality_constraints(tmp_path, monkeypatch):
    video = tmp_path / 'clip.mp4'
    video.touch()
    submitted = []
    monkeypatch.setattr(create.manager, 'submit', submitted.append)
    from urllib.parse import urlencode
    body = urlencode(dict(input=str(video), out_dir=str(tmp_path / 'out'), mode='hybrid',
                          fps='2', threshold='6', keep_pct='80')).encode()

    async def receive():
        return {'type': 'http.request', 'body': body}

    request = Request({'type': 'http', 'method': 'POST', 'path': '/ui/frames',
                       'headers': [(b'content-type', b'application/x-www-form-urlencoded')]}, receive)
    response = asyncio.run(create.create_frames(request))
    assert response.status_code == 200
    assert len(submitted) == 1
    assert submitted[0].params['threshold'] == '6' and submitted[0].params['keep_pct'] == '80'
    assert submitted[0].params['mode'] == 'hybrid'


def test_real_ffmpeg_rejects_blurred_and_flat_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(frames, "_supports_cuda", lambda _: False)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('ffmpeg not installed')
    filters = subprocess.run([ffmpeg, '-hide_banner', '-filters'], capture_output=True, text=True, check=True)
    if 'blurdetect' not in filters.stdout:
        pytest.skip('ffmpeg lacks blurdetect')
    video = tmp_path / 'mixed.mkv'
    # Three clear, three defocused, then three featureless frames. Lossless codec
    # avoids compression noise masquerading as image detail.
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc2=size=640x360:rate=1:duration=3', '-f', 'lavfi', '-i',
                    'testsrc2=size=640x360:rate=1:duration=3,gblur=sigma=8', '-f', 'lavfi', '-i',
                    'color=gray:size=640x360:rate=1:duration=3', '-filter_complex',
                    '[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]', '-map', '[v]', '-c:v', 'ffv1', str(video)],
                   capture_output=True, text=True, check=True, timeout=30)
    runner = Runner(tmp_path / 'console.log')
    try:
        frames.run_frames(dict(inputs=[str(video)], out_dir=str(tmp_path / 'out'), mode='hybrid',
                               fps='1', threshold='8', keep_pct='70', workers=1, ffmpeg_bin=ffmpeg), runner)
    finally:
        runner.close()
    output = tmp_path / 'out' / 'frames_mixed'
    report = json.loads((output / 'frame_quality.json').read_text())
    assert report['extracted'] == 9 and report['kept'] == 3
    assert report['reasons']['above_threshold'] == 3
    assert report['reasons']['invalid_score'] == 3
    assert sorted(p.name for p in output.glob('frame_*.jpg')) == [f'frame_{i:06}.jpg' for i in range(1, 4)]
