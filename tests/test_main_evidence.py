from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from survng.app.main_evidence import MainEvidenceBatch, MainEvidenceFrame, MainEvidenceProvider
from survng.app.main_evidence_ring import AccessUnit, EncodedRing, receive_control, send_control
from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector
from survng.app.motion_pipeline.object_detection import _DecodedRecordedFrame
from survng.app.motion_pipeline.recorded_decode_budget import RecordedDecodeBudget


def au(ordinal, *, pts=None, dts=None, size=10, idr=False, epoch=None, received=None):
    return AccessUnit(ordinal, ordinal * 100 if pts is None else pts, dts, 100,
                      time.monotonic() if received is None else received,
                      100 + ordinal if epoch is None else epoch, b"x" * size, idr)


def test_ring_requires_random_access_and_evicts_whole_dependencies():
    ring = EncodedRing(max_bytes=40, history_seconds=20)
    assert not ring.add(au(1), config="a")
    assert ring.add(au(2, idr=True), config="a")
    ring.add(au(3), config="a")
    ring.add(au(4, idr=True), config="a")
    ring.add(au(5), config="a")
    ring.add(au(6), config="a")
    assert ring.bytes == 30
    assert [item.ordinal for group in ring.gops for item in group] == [4, 5, 6]
    assert ring.window([103], max_bytes=40)[0] == "miss"


def test_b_frame_pts_reordering_preserves_generation_but_bad_dts_does_not():
    ring = EncodedRing(max_bytes=100, history_seconds=20)
    ring.add(au(1, idr=True, pts=300, dts=100), config="a")
    generation = ring.generation
    ring.add(au(2, pts=100, dts=200), config="a")
    assert ring.generation == generation
    assert ring.units == 2
    ring.add(au(3, pts=400, dts=50), config="a")
    assert ring.generation > generation
    assert ring.units == 0


@pytest.mark.parametrize("change", ["ordinal", "configuration", "discontinuity"])
def test_gaps_invalidate_dependent_frames_until_next_idr(change):
    ring = EncodedRing(max_bytes=100, history_seconds=20)
    ring.add(au(1, idr=True), config="a")
    ring.add(au(3 if change == "ordinal" else 2),
             config="b" if change == "configuration" else "a",
             discontinuity=change == "discontinuity")
    assert ring.units == 0
    ring.add(au(4, idr=True), config="b" if change == "configuration" else "a")
    assert ring.units == 1


def test_active_gop_can_be_exported_before_next_idr_and_window_is_immutable():
    ring = EncodedRing(max_bytes=100, history_seconds=20)
    ring.add(au(1, idr=True), config="a")
    ring.add(au(2), config="a")
    state, window = ring.window([101.5], max_bytes=100)
    assert state == "ready"
    ring.add(au(3), config="a")
    assert len(window) == 2
    assert ring.window([104], max_bytes=100)[0] == "pending"
    assert ring.window([101.5], max_bytes=10)[0] == "capacity_denied"


def test_oversized_gop_and_idle_expiry_do_not_leave_decodable_claim():
    ring = EncodedRing(max_bytes=20, history_seconds=1)
    ring.add(au(1, idr=True), config="a")
    ring.add(au(2), config="a")
    ring.add(au(3), config="a")
    assert ring.units == 0
    ring.add(au(4, idr=True, received=time.monotonic() - 2), config="a")
    assert ring.status()["state"] == "warming"
    assert ring.bytes == 0


def test_private_control_transfers_readonly_descriptor():
    import fcntl
    left, right = socket.socketpair()
    original = os.memfd_create("test", os.MFD_ALLOW_SEALING)
    received = None
    try:
        os.write(original, b"encoded")
        fcntl.fcntl(original, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE)
        send_control(left, {"version": 1, "size": 7}, original)
        value, received = receive_control(right)
        assert value == {"version": 1, "size": 7}
        assert os.pread(received, 7, 0) == b"encoded"
        with pytest.raises(OSError):
            os.write(received, b"x")
    finally:
        left.close()
        right.close()
        os.close(original)
        if received is not None:
            os.close(received)


def test_provider_disabled_until_explicit_start_and_status_never_does_io():
    provider = MainEvidenceProvider("gate", "rtsp://example.invalid/test", RecordedDecodeBudget())
    with patch.object(provider, "_exchange", side_effect=AssertionError("observer did IPC")):
        assert provider.status()["state"] == "stopped"
    assert not provider.running
    assert provider.frames_at([time.time()], deadline=time.monotonic() + 1).status == "miss"
    provider.stop()


def backend_for_batch(batch):
    backend = RecordedMotionObjectDetector(SimpleNamespace(id="gate"),
        SimpleNamespace(config=SimpleNamespace(event_confirmation_frames=2)),
        Mock(), lambda: None, main_evidence=SimpleNamespace(frames_at=lambda *a, **kw: batch))
    backend._detect_objects = Mock(return_value=[{"label": "person", "confidence": .9,
        "incident_eligible": True, "box": {"x1": 20, "y1": 20, "x2": 60, "y2": 70}}])
    backend._enrich_selected_faces = lambda frame, objects, timing: objects
    backend._enrich_depth = lambda frame, objects, *args, **kw: objects
    return backend


def test_buffered_result_uses_own_source_identity_and_releases_memory():
    frame = np.random.default_rng(4).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    lease = Mock()
    frames = {epoch: MainEvidenceFrame(frame, epoch,
        {"session": "s", "generation": 1, "ordinal": n, "source": "buffered_main"}, .5)
        for n, epoch in enumerate((99.5, 100., 100.5))}
    batch = MainEvidenceBatch("ready", frames, memory_lease=lease)
    backend = backend_for_batch(batch)
    result = backend._detect_buffered(event_epoch=100, stages=((-1., -.5, 0., .5, 1.),),
        deadline=time.monotonic()+1, timing={}, workflow_started=time.monotonic(),
        minimum_last_offset_seconds=None)
    assert result is not None
    assert result.frame_source == "buffered_main"
    assert not result.frame_timestamp_exact
    assert result.frame_reference["session"] == "s"
    assert result.recording_path == ""
    lease.release.assert_called_once()


@pytest.mark.parametrize("same_frame,mixed_session", [(True, False), (False, True)])
def test_buffer_cohort_cannot_confirm_duplicate_frame_or_cross_generation(same_frame, mixed_session):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = {epoch: MainEvidenceFrame(frame, epoch,
        {"session": "other" if mixed_session and n == 1 else "s", "generation": 1,
         "ordinal": 1 if same_frame else n}, .5)
        for n, epoch in enumerate((99.5, 100., 100.5))}
    backend = backend_for_batch(MainEvidenceBatch("ready", frames))
    assert backend._detect_buffered(event_epoch=100, stages=((-1., -.5, 0., .5, 1.),),
        deadline=time.monotonic()+1, timing={}, workflow_started=time.monotonic(),
        minimum_last_offset_seconds=None) is None


def test_attribution_continuation_observes_later_stage_despite_core_confirmation():
    from datetime import datetime, timezone
    frame = np.random.default_rng(5).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    recorder = SimpleNamespace(hardware_acceleration="none",
        recording_rows_between=lambda *args, **kwargs: [],
        recording_at=lambda *args: {"path": "/synthetic/main.mp4", "start_epoch": 90., "end_epoch": 120.})
    config = SimpleNamespace(event_confirmation_frames=2,
        event_refinement_stages=[[-.5, 0., .5], [4., 4.5]])
    backend = RecordedMotionObjectDetector(SimpleNamespace(id="gate"),
        SimpleNamespace(config=config), recorder, lambda: None)
    backend._detect_objects = lambda *args, **kwargs: [{"label": "person", "confidence": .9,
        "incident_eligible": True, "box": {"x1": 20, "y1": 20, "x2": 60, "y2": 70}}]
    backend._enrich_selected_faces = lambda frame, objects, timing: objects
    backend._enrich_depth = lambda frame, objects, *args, **kw: objects
    requested = []
    def read(_path, offsets, **kwargs):
        requested.extend(offsets)
        return {offset: _DecodedRecordedFrame(frame, offset, True) for offset in offsets}, 1
    backend._read_recorded_frames = read
    event = datetime.fromtimestamp(100., timezone.utc)
    backend.detect(event)
    assert not any(offset >= 14. for offset in requested)
    requested.clear()
    backend.detect(event, {"evidence_sampling": {"minimum_last_offset_seconds": 4., "timeout_seconds": 1.}})
    assert any(offset >= 14. for offset in requested)


def test_settle_wait_cannot_extend_callers_remaining_evidence_deadline():
    from datetime import datetime, timezone
    clock = [100.]
    recorder = Mock()
    backend = RecordedMotionObjectDetector(SimpleNamespace(id="gate"),
        SimpleNamespace(config=SimpleNamespace(event_confirmation_frames=2)), recorder, lambda: None)
    with patch("survng.app.motion_pipeline.object_detection.time.monotonic", side_effect=lambda: clock[0]), patch(
        "survng.app.motion_pipeline.object_detection.time.time", return_value=200.), patch(
        "survng.app.motion_pipeline.object_detection.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, round(clock[0] + seconds, 9))):
        result = backend.detect(datetime.fromtimestamp(200., timezone.utc),
                                {"evidence_sampling": {"timeout_seconds": .2}})
    assert result.frame is None
    assert clock[0] <= 100.200001
    recorder.recording_rows_between.assert_not_called()


NATIVE_FIXTURE = r'''
import json,os,subprocess,tempfile,sys
from survng.main_evidence_worker import _gst,decode
Gst=_gst()
codec=os.environ.get('SURVNG_TEST_CODEC','h264')
for name in ['tsdemux',codec+'parse','avdec_'+codec,'appsrc','appsink','videoconvert']:
 if not Gst.ElementFactory.find(name):raise SystemExit(77)
with tempfile.TemporaryDirectory() as directory:
 path=directory+'/fixture.ts'
 encoded=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i',
   'testsrc2=size=160x96:rate=10','-frames:v','24','-c:v','libx264' if codec=='h264' else 'libx265',
   '-bf','2','-g','10','-preset','ultrafast',
   *([] if codec=='h264' else ['-x265-params','pools=1:frame-threads=1:log-level=error']),
   '-f','mpegts',path],capture_output=True)
 if encoded.returncode:raise SystemExit(77)
 pipeline=Gst.parse_launch('filesrc name=file ! tsdemux ! '+codec+'parse config-interval=-1 ! video/x-'+codec+',stream-format=byte-stream,alignment=au ! appsink name=sink sync=false')
 pipeline.get_by_name('file').set_property('location',path)
 sink=pipeline.get_by_name('sink');pipeline.set_state(Gst.State.PLAYING)
 units=[];fd=os.memfd_create('fixture');offset=0;caps=''
 while True:
  sample=sink.emit('try-pull-sample',Gst.SECOND)
  if sample is None:break
  buffer=sample.get_buffer();data=buffer.extract_dup(0,buffer.get_size());os.write(fd,data)
  caps=sample.get_caps().to_string()
  units.append(dict(offset=offset,size=len(data),ordinal=len(units)+1,pts_ns=int(buffer.pts),
    dts_ns=None if buffer.dts==Gst.CLOCK_TIME_NONE else int(buffer.dts),
    duration_ns=None if buffer.duration==Gst.CLOCK_TIME_NONE else int(buffer.duration),
    epoch=100+int(buffer.pts)/Gst.SECOND))
  offset+=len(data)
 pipeline.set_state(Gst.State.NULL)
 assert len(units)==24,len(units)
 assert any(b['pts_ns']<a['pts_ns'] for a,b in zip(units,units[1:])), 'fixture lacks B-frame reorder'
 ordered=sorted(units,key=lambda x:x['pts_ns']);targets=[ordered[3]['epoch'],ordered[13]['epoch']]
 output=os.memfd_create('output')
 config=dict(codec=codec,caps=caps,units=units,targets=targets,decoder='cpu',
   timeout_seconds=5,maximum_frame_offset_seconds=.01,maximum_output_bytes=160*96*3*2)
 worker=subprocess.run([sys.executable,'-m','survng.main_evidence_worker','decode',
   '--input-fd',str(fd),'--output-fd',str(output)],pass_fds=(fd,output),
   input=json.dumps(config)+'\n',capture_output=True,text=True,timeout=10)
 assert worker.returncode==0,worker.stderr[-1000:]
 result=json.loads(worker.stdout)
 assert result['status']=='ready',result
 assert [f['unit']['ordinal'] for f in result['frames']]==[ordered[3]['ordinal'],ordered[13]['ordinal']],result
 assert os.fstat(output).st_size==160*96*3*2
 print(json.dumps({'status':result['status'],'frames':len(result['frames']),'b_frame_reordering':True}))
 os.close(fd);os.close(output)
'''


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_native_gstreamer_b_frame_decode_selects_original_access_units(codec):
    if not Path("/usr/bin/python3").exists():
        pytest.skip("system GI runtime unavailable")
    probe = subprocess.run(["/usr/bin/python3", "-c", "import gi"], capture_output=True)
    if probe.returncode:
        pytest.skip("system GI runtime unavailable")
    result = subprocess.run(["/usr/bin/python3", "-c", NATIVE_FIXTURE],
                            capture_output=True, text=True, timeout=20,
                            env={**os.environ, "SURVNG_TEST_CODEC": codec})
    if result.returncode == 77:
        pytest.skip("local encoded fixture codec/plugins unavailable")
    assert result.returncode == 0, result.stderr[-2000:] + result.stdout[-2000:]
    assert json.loads(result.stdout)["b_frame_reordering"] is True


def test_auto_decoder_retries_cpu_with_clean_output_and_remaining_deadline(monkeypatch):
    from survng import main_evidence_worker as worker
    monkeypatch.setattr(worker, '_gst', lambda: SimpleNamespace(
        ElementFactory=SimpleNamespace(find=lambda name: True)))
    now = [100.0]
    monkeypatch.setattr(worker.time, 'monotonic', lambda: now[0])
    attempts = []
    def decode_attempt(gst, config, input_fd, output_fd, chosen):
        attempts.append((chosen, config['timeout_seconds']))
        if chosen == 'vah264dec':
            os.write(output_fd, b'failed partial output')
            now[0] += 2
            raise RuntimeError('VA device could not initialize')
        assert os.fstat(output_fd).st_size == 0
        assert os.lseek(output_fd, 0, os.SEEK_CUR) == 0
        os.write(output_fd, b'cpu')
        return {'status': 'ready', 'frames': []}
    monkeypatch.setattr(worker, '_decode_with_decoder', decode_attempt)
    fd = os.memfd_create('decoder-fallback-test')
    try:
        result = worker.decode({'codec': 'h264', 'decoder': 'auto', 'timeout_seconds': 5}, -1, fd)
        assert result['status'] == 'ready'
        assert attempts == [('vah264dec', 5), ('avdec_h264', 3)]
        assert os.pread(fd, 3, 0) == b'cpu'
    finally:
        os.close(fd)


@pytest.mark.parametrize('policy,reason,elapsed', [
    ('va', 'decode_error', 0),
    ('auto', 'dependency_window_incomplete', 0),
    ('auto', 'decode_error', 5),
])
def test_decoder_fallback_respects_policy_source_failures_and_deadline(monkeypatch, policy, reason, elapsed):
    from survng import main_evidence_worker as worker
    monkeypatch.setattr(worker, '_gst', lambda: SimpleNamespace(
        ElementFactory=SimpleNamespace(find=lambda name: True)))
    now = [100.0]
    monkeypatch.setattr(worker.time, 'monotonic', lambda: now[0])
    attempted = []
    def attempt(gst, config, input_fd, output_fd, chosen):
        attempted.append(chosen)
        now[0] += elapsed
        return {'status': 'miss', 'reason': reason}
    monkeypatch.setattr(worker, '_decode_with_decoder', attempt)
    fd = os.memfd_create('decoder-deadline-test')
    try:
        result = worker.decode({'codec': 'h265', 'decoder': policy, 'timeout_seconds': 5}, -1, fd)
        assert result['reason'] == reason
        assert attempted == ['vah265dec']
    finally:
        os.close(fd)


def test_decoder_initialization_failure_tears_down_pipeline_before_fallback():
    from survng.main_evidence_worker import _decode_with_decoder
    pipeline = Mock()
    pipeline.get_by_name.return_value.set_property.side_effect = RuntimeError('caps rejected')
    gst = SimpleNamespace(parse_launch=lambda command: pipeline,
                          Caps=SimpleNamespace(from_string=lambda caps: caps),
                          StateChangeReturn=SimpleNamespace(FAILURE='failure'),
                          State=SimpleNamespace(NULL='null'))
    with pytest.raises(RuntimeError, match='caps rejected'):
        _decode_with_decoder(gst, {'codec': 'h264', 'caps': 'caps', 'timeout_seconds': 1}, -1, -1, 'vah264dec')
    pipeline.set_state.assert_called_once_with('null')


def test_auto_decoder_does_not_replace_a_decoder_with_incomplete_cleanup(monkeypatch):
    from survng import main_evidence_worker as worker
    monkeypatch.setattr(worker, '_gst', lambda: SimpleNamespace(
        ElementFactory=SimpleNamespace(find=lambda name: True)))
    attempt = Mock(side_effect=worker.DecoderCleanupIncomplete())
    monkeypatch.setattr(worker, '_decode_with_decoder', attempt)
    fd = os.memfd_create('decoder-cleanup-test')
    try:
        result = worker.decode({'codec': 'h264', 'decoder': 'auto', 'timeout_seconds': 5}, -1, fd)
        assert result['reason'] == 'decoder_cleanup_incomplete'
        assert attempt.call_count == 1
    finally:
        os.close(fd)


def test_failed_native_teardown_fences_decoder_fallback():
    from survng.main_evidence_worker import DecoderCleanupIncomplete, _decode_with_decoder
    pipeline = Mock()
    pipeline.get_by_name.return_value.set_property.side_effect = RuntimeError('initialization failed')
    pipeline.set_state.return_value = 'failure'
    gst = SimpleNamespace(parse_launch=lambda command: pipeline,
                          Caps=SimpleNamespace(from_string=lambda caps: caps),
                          State=SimpleNamespace(NULL='null'),
                          StateChangeReturn=SimpleNamespace(FAILURE='failure'))
    with pytest.raises(DecoderCleanupIncomplete):
        _decode_with_decoder(gst, {'codec': 'h264', 'caps': 'caps', 'timeout_seconds': 1}, -1, -1, 'vah264dec')
