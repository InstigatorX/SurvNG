import json
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from survng.app.config import DetectorConfig
from survng.app.face_capture import service as service_module
from survng.app.face_capture.pipeline import CapturedFace, CaptureResult
from survng.app.face_capture.service import MAX_ATTEMPTS, FaceCaptureService
from survng.app.face_capture.store import FaceCaptureStore
from survng.app.inference_runtime.types import InferenceUnavailable

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _person_event(consensus=True):
    return json.dumps([{"label": "person", "temporal_consensus": consensus, "box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}])


def _database(tmp_path: Path, rows):
    path = tmp_path / "survng.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        "create table events (id integer primary key, camera_id text, objects_json text, created_at text);"
        "create table scene_observations (event_id integer, payload_json text, captured_epoch real);"
    )
    connection.executemany("insert into events values (?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()
    return path


def _service(tmp_path, rows, enabled=True):
    config = SimpleNamespace(
        detector=DetectorConfig(face_capture_enabled=enabled, face_capture_backfill_hours=24.0),
        ffmpeg_path="ffmpeg", hardware_acceleration="",
    )
    return FaceCaptureService(
        config=lambda: config, events_db_path=_database(tmp_path, rows),
        inference=lambda: SimpleNamespace(capture_faces=lambda crop: [], detect_enrichment=lambda *a, **k: []),
        recording_rows=lambda *a, **k: [],
    )


def _at(minutes_ago):
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def test_the_newest_settled_confirmed_person_event_comes_first(tmp_path):
    service = _service(tmp_path, [
        (1, "gate", _person_event(), _at(120)),
        (2, "gate", _person_event(), _at(30)),
        (3, "gate", _person_event(consensus=False), _at(20)),  # no confirmed person
        (4, "gate", _person_event(), _at(0.5)),  # recordings not settled yet
        (5, "gate", _person_event(), _at(60 * 30)),  # older than the backfill
    ])
    assert service.next_event(NOW)[0] == 2


def test_done_events_are_skipped_and_failures_retry_a_few_times(tmp_path):
    service = _service(tmp_path, [(2, "gate", _person_event(), _at(60)), (1, "gate", _person_event(), _at(90))])
    service.store.record(event_id=2, camera_id="gate", event_epoch=0.0, result=CaptureResult(),
                         embedding_model="m", seconds=1.0)
    assert service.next_event(NOW)[0] == 1
    for _ in range(MAX_ATTEMPTS):
        assert service.next_event(NOW)[0] == 1
        service.store.record_failure(event_id=1, camera_id="gate", event_epoch=0.0, error="boom")
        _age_failures(service.store)
    assert service.next_event(NOW) is None


def _age_failures(store, seconds=service_module.FAILURE_RETRY_SECONDS + 1):
    old = datetime.fromtimestamp(time.time() - seconds, timezone.utc).isoformat()
    with sqlite3.connect(store.path) as connection:
        connection.execute("update face_capture_events set updated_at = ? where state = 'failed'", (old,))
    connection.close()


def test_a_new_service_spaces_the_retry_of_a_recent_failure(tmp_path):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))])
    service.store.record_failure(event_id=1, camera_id="gate", event_epoch=0.0, error="boom")
    # A fresh service (a manager reload) has no hold in memory for event 1.
    assert service.next_event(NOW) is None
    service._deferred.clear()
    _age_failures(service.store)
    assert service.next_event(NOW)[0] == 1


def test_disabled_capture_does_nothing_and_leaves_no_store(tmp_path):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))], enabled=False)
    assert service.run_once(NOW) is False
    assert not (tmp_path / "face-captures.sqlite3").exists()
    assert service.status()["faces"] == 0


def test_shed_inference_defers_without_using_an_attempt(tmp_path, monkeypatch):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))])

    def shed(**kwargs):
        raise InferenceUnavailable("shed")

    monkeypatch.setattr(service_module, "capture_event", shed)
    assert service.run_once(NOW) is False
    assert service.store.attempts([1]) == {}


def test_a_capture_is_stored_and_a_failure_is_recorded(tmp_path, monkeypatch):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(60)), (2, "gate", _person_event(), _at(30))])
    face = CapturedFace(0, 0.5, (0, 0, 60, 60), [[1.0, 1.0]] * 5, 0.9, 17.0, np.ones(512, np.float32),
                        np.zeros((20, 20, 3), np.uint8))
    outcomes = iter([CaptureResult(faces=[face], moment_source="sub-stream"), RuntimeError("decoder failed")])

    def fake_capture(**kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(service_module, "capture_event", fake_capture)
    assert service.run_once(NOW) is True
    assert service.run_once(NOW) is False  # a failure waits the idle interval
    assert service.store.attempts([1, 2]) == {2: ("done", 1, 1), 1: ("failed", 1, 1)}
    stored = service.store.faces(2)
    assert len(stored) == 1 and Path(stored[0]["crop_path"]).is_file()
    assert np.frombuffer(stored[0]["embedding"], np.float32).shape == (512,)


def test_faces_of_events_removed_by_retention_are_forgotten(tmp_path):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))])
    face = CapturedFace(0, 0.5, (0, 0, 60, 60), [], 0.9, 17.0, np.ones(512, np.float32), np.zeros((20, 20, 3), np.uint8))
    for event_id in (1, 99):
        service.store.record(event_id=event_id, camera_id="gate", event_epoch=0.0,
                             result=CaptureResult(faces=[face]), embedding_model="m", seconds=1.0)
    crop = Path(service.store.faces(99)[0]["crop_path"])
    service._prune_if_due()
    assert service.store.captured_event_ids() == [1]
    assert not crop.exists()


def test_a_rerun_replaces_the_event_faces_and_their_crops(tmp_path):
    store = FaceCaptureStore(tmp_path)
    face = CapturedFace(0, 0.5, (0, 0, 60, 60), [], 0.9, 17.0, np.ones(512, np.float32), np.zeros((20, 20, 3), np.uint8))
    store.record(event_id=7, camera_id="gate", event_epoch=0.0, result=CaptureResult(faces=[face, face]),
                 embedding_model="m", seconds=1.0)
    first = {row["crop_path"] for row in store.faces(7)}
    store.record(event_id=7, camera_id="gate", event_epoch=0.0, result=CaptureResult(faces=[face]),
                 embedding_model="m", seconds=1.0)
    second = [row["crop_path"] for row in store.faces(7)]
    assert len(second) == 1 and Path(second[0]).is_file()
    assert not first & set(second)  # a rerun never overwrites crops the old rows point at
    assert not any(Path(p).exists() for p in first)
    assert store.attempts([7]) == {7: ("done", 2, 1)}


def test_the_face_worker_runs_for_recognition_or_capture():
    assert not DetectorConfig().face_worker_enabled()
    assert DetectorConfig(face_capture_enabled=True).face_worker_enabled()
    assert DetectorConfig(face_recognition_enabled=True).face_worker_enabled()


def test_events_older_than_a_captured_batch_are_still_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(service_module, "CANDIDATE_BATCH", 2)
    service = _service(tmp_path, [(i, "gate", _person_event(), _at(100 - i)) for i in range(1, 6)])
    for event_id in (5, 4, 3):
        service.store.record(event_id=event_id, camera_id="gate", event_epoch=0.0, result=CaptureResult(),
                             embedding_model="m", seconds=1.0)
    assert service.next_event(NOW)[0] == 2


def test_unanswered_person_detection_defers_instead_of_finishing_the_event():
    from survng.app.face_capture.service import _people

    inference = SimpleNamespace(detect_enrichment=lambda *a, **k: [{"status": "detector_unavailable"}])
    with pytest.raises(InferenceUnavailable):
        _people(inference, DetectorConfig(), np.zeros((8, 8, 3), np.uint8))


def test_main_stream_errors_without_faces_count_as_a_failed_attempt(tmp_path, monkeypatch):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))])
    monkeypatch.setattr(service_module, "capture_event",
                        lambda **kwargs: CaptureResult(errors=["main stream -1.0..+1.0: face inference failed"]))
    assert service.run_once(NOW) is False
    assert service.store.attempts([1]) == {1: ("failed", 1, 1)}


def test_a_failed_lookup_before_the_capture_uses_an_attempt(tmp_path):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))])

    def broken_rows(*args, **kwargs):
        raise OSError("recording index locked")

    service._recording_rows = broken_rows
    assert service.run_once(NOW) is False
    assert service.store.attempts([1]) == {1: ("failed", 1, 1)}


def test_a_window_without_main_stream_frames_is_retried_not_done(tmp_path, monkeypatch):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))])
    monkeypatch.setattr(service_module, "capture_event", lambda **kwargs: CaptureResult(main_frames=0))
    assert service.run_once(NOW) is False
    assert service.store.attempts([1]) == {1: ("failed", 1, 1)}


def test_a_failure_to_store_faces_uses_an_attempt(tmp_path, monkeypatch):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))])
    monkeypatch.setattr(service_module, "capture_event", lambda **kwargs: CaptureResult(main_frames=5))

    def disk_full(**kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(service.store, "record", disk_full)
    assert service.run_once(NOW) is False
    assert service.store.attempts([1]) == {1: ("failed", 1, 1)}


def test_a_deferred_event_backs_off_so_older_events_are_reached(tmp_path, monkeypatch):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(60)), (2, "gate", _person_event(), _at(30))])

    def shed(**kwargs):
        raise InferenceUnavailable("worker timed out")

    monkeypatch.setattr(service_module, "capture_event", shed)
    assert service.run_once(NOW) is False
    assert service.next_event(NOW)[0] == 1
    assert service.store.attempts([1, 2]) == {}


def test_unloaded_capture_models_defer_instead_of_using_attempts(tmp_path, monkeypatch):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))])
    service._inference = lambda: SimpleNamespace(face_capture_unavailable=lambda: "models missing")
    monkeypatch.setattr(service_module, "capture_event", lambda **kwargs: pytest.fail("captured"))
    assert service.run_once(NOW) is False
    assert service.store.attempts([1]) == {}


def test_a_malformed_stored_observation_is_skipped():
    from survng.app.face_capture.events import stored_people

    connection = sqlite3.connect(":memory:")
    connection.execute("create table scene_observations (event_id integer, payload_json text, captured_epoch real)")
    connection.executemany("insert into scene_observations values (1, ?, ?)", [
        ("not json", 10.0),
        ("[1, 2]", 10.0),
        (json.dumps({"label": "person", "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 20}}), None),
        (json.dumps({"label": "person", "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 20}}), 11.0),
    ])
    assert stored_people(connection, 1, 10.0, None) == [(1.0, [(0.0, 0.0, 10.0, 20.0)])]


def test_a_failed_event_waits_before_its_next_attempt(tmp_path, monkeypatch):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(60)), (2, "gate", _person_event(), _at(30))])
    monkeypatch.setattr(service_module, "capture_event", lambda **kwargs: CaptureResult(main_frames=0))
    assert service.run_once(NOW) is False
    assert service.store.attempts([2]) == {2: ("failed", 1, 1)}
    assert service.next_event(NOW)[0] == 1


def test_an_event_whose_epoch_has_not_settled_waits(tmp_path):
    qualification = {"status": "motion_qualification", "motion_qualification": {
        "evidence_frame_at_epoch": NOW.timestamp() - 30.0}}
    objects = json.dumps([json.loads(_person_event())[0], qualification])
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(60)), (2, "gate", objects, _at(30))])
    assert service.next_event(NOW)[0] == 1


def test_a_new_capture_version_retries_events_that_used_their_attempts(tmp_path, monkeypatch):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))])
    for _ in range(MAX_ATTEMPTS):
        service.store.record_failure(event_id=1, camera_id="gate", event_epoch=0.0, error="boom")
    assert service.next_event(NOW) is None
    monkeypatch.setattr(service_module, "CAPTURE_VERSION", 2)
    service._full_scan_due = 0.0  # a new version arrives with a restart: a full scan
    assert service.next_event(NOW)[0] == 1
    from survng.app.face_capture import store as store_module

    monkeypatch.setattr(store_module, "CAPTURE_VERSION", 2)
    service.store.record_failure(event_id=1, camera_id="gate", event_epoch=0.0, error="boom")
    assert service.store.attempts([1]) == {1: ("failed", 1, 2)}


def test_after_a_full_scan_only_newer_events_are_read_but_waiting_ones_are_kept(tmp_path, monkeypatch):
    rows = [(1, "gate", _person_event(), _at(90)), (2, "gate", _person_event(), _at(60))]
    service = _service(tmp_path, rows)
    for event_id in (1, 2):
        service.store.record(event_id=event_id, camera_id="gate", event_epoch=0.0, result=CaptureResult(),
                             embedding_model="m", seconds=1.0)
    assert service.next_event(NOW) is None
    assert service._scan_floor == 2
    reads = []
    real = service_module.person_events
    monkeypatch.setattr(service_module, "person_events",
                        lambda connection, **kw: reads.append(kw) or real(connection, **kw))
    connection = sqlite3.connect(tmp_path / "survng.sqlite3")
    connection.execute("insert into events values (3, 'gate', ?, ?)", (_person_event(), _at(30)))
    connection.commit()
    connection.close()
    assert service.next_event(NOW)[0] == 3
    assert len(reads) == 1  # one page, stopped at the floor
    # A deferred event keeps the floor below it, so it is not skipped later.
    service._deferred[3] = (1, time.monotonic() + 60)
    assert service.next_event(NOW) is None
    assert service._scan_floor == 2


def test_a_crop_that_cannot_be_written_leaves_no_files_behind(tmp_path):
    store = FaceCaptureStore(tmp_path)
    good = CapturedFace(track=0, offset=0.0, box=(0, 0, 8, 8), landmarks=[], confidence=0.9, norm=20.0,
                        embedding=np.zeros(512, np.float32), crop=np.zeros((8, 8, 3), np.uint8))
    bad = CapturedFace(track=0, offset=1.0, box=(0, 0, 8, 8), landmarks=[], confidence=0.9, norm=20.0,
                       embedding=np.zeros(512, np.float32), crop=np.empty((0, 0, 3), np.uint8))
    with pytest.raises(Exception):
        store.record(event_id=1, camera_id="gate", event_epoch=0.0, result=CaptureResult(faces=[good, bad]),
                     embedding_model="m", seconds=1.0)
    assert not [path for path in store.crop_dir.rglob("*") if path.is_file()]


def test_an_event_deferred_again_and_again_becomes_a_failed_attempt(tmp_path, monkeypatch):
    service = _service(tmp_path, [(1, "gate", _person_event(), _at(30))])

    def shed(**kwargs):
        raise InferenceUnavailable("face inference request timed out")

    monkeypatch.setattr(service_module, "capture_event", shed)
    for _ in range(service_module.MAX_CONSECUTIVE_DEFERRALS - 1):
        assert service.capture(*service.next_event(NOW)) is None
        service._deferred[1] = (service._deferred[1][0], 0.0)  # the hold has passed
    assert service.store.attempts([1]) == {}
    service.capture(*service.next_event(NOW))
    assert service.store.attempts([1])[1][:2] == ("failed", 1)
