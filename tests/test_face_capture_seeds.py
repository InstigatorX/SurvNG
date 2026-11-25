import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from survng.app.config import DetectorConfig
from survng.app.face_capture import service as service_module
from survng.app.face_capture.pipeline import CapturedFace, CaptureResult
from survng.app.face_capture.routes import FaceCaptureRouteDependencies, SeedRequest, create_face_capture_router
from survng.app.face_capture.service import FaceCaptureService
from survng.app.face_capture.store import FaceCaptureStore, SeedConflict
from survng.app.security import required_api_scope

NOW = datetime.now(timezone.utc)
PERSON = json.dumps([{"label": "person", "temporal_consensus": True}])


def _at(hours_ago):
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _service(tmp_path: Path, rows, *, backfill_hours=24.0, enabled=True):
    path = tmp_path / "survng.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        "create table events (id integer primary key, camera_id text, objects_json text, created_at text);"
        "create table scene_observations (event_id integer, payload_json text, captured_epoch real);"
    )
    connection.executemany("insert into events values (?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()
    config = SimpleNamespace(
        detector=DetectorConfig(face_capture_enabled=enabled, face_capture_backfill_hours=backfill_hours),
        ffmpeg_path="ffmpeg", hardware_acceleration="",
        cameras=[SimpleNamespace(id="gate", name="Gate"), SimpleNamespace(id="foyer", name="")],
    )
    return FaceCaptureService(
        config=lambda: config, events_db_path=path,
        inference=lambda: SimpleNamespace(capture_faces=lambda crop: [], detect_enrichment=lambda *a, **k: []),
        recording_rows=lambda *a, **k: [],
    ), config


def _face():
    return CapturedFace(0, 0.5, (0, 0, 60, 60), [], 0.9, 17.0, np.ones(512, np.float32), np.zeros((20, 20, 3), np.uint8))


def test_one_seed_at_a_time_and_only_allowed_state_changes(tmp_path):
    store = FaceCaptureStore(tmp_path)
    seed = store.create_seed(start_epoch=0, end_epoch=10, cameras=["gate"], total=5)
    with pytest.raises(SeedConflict):
        store.create_seed(start_epoch=0, end_epoch=10, cameras=[], total=1)
    assert store.set_seed_state(seed, "paused")["state"] == "paused"
    with pytest.raises(SeedConflict):
        store.set_seed_state(seed, "done")  # only a running seed finishes
    assert store.set_seed_state(seed, "running")["state"] == "running"
    cancelled = store.set_seed_state(seed, "cancelled")
    assert cancelled["state"] == "cancelled" and cancelled["finished_at"]
    assert store.active_seed() is None
    assert store.create_seed(start_epoch=0, end_epoch=10, cameras=[], total=1) > seed
    with pytest.raises(KeyError):
        store.set_seed_state(999, "paused")


def test_a_seed_runs_only_when_nothing_new_waits_then_finishes(tmp_path, monkeypatch):
    # Events 1-3 are three to five days old (outside the 24 h backfill);
    # event 4 is new.
    rows = [(1, "gate", PERSON, _at(120)), (2, "foyer", PERSON, _at(96)), (3, "gate", PERSON, _at(72)),
            (4, "gate", PERSON, _at(1))]
    service, _ = _service(tmp_path, rows)
    captured = []

    def fake_capture(**kwargs):
        captured.append(kwargs["plan_offsets"] is not None)
        return CaptureResult(faces=[_face()], main_frames=10)

    monkeypatch.setattr(service_module, "capture_event", fake_capture)
    start, end = (NOW - timedelta(days=6)).timestamp(), (NOW - timedelta(days=2)).timestamp()
    assert service.count_seed_events(start, end, ["gate"]) == 2
    seed_id = service.store.create_seed(start_epoch=start, end_epoch=end, cameras=["gate"], total=2)
    order = []
    real = service.capture
    monkeypatch.setattr(service, "capture", lambda *c: order.append(c[0]) or real(*c))
    while service.run_once(NOW):
        pass
    service.run_once(NOW)  # nothing left: the seed is marked done
    # The new event first, then the seed's events newest first; foyer is not in the seed.
    assert order == [4, 3, 1]
    seed = service.store.seeds(1)[0]
    assert (seed["id"], seed["state"], seed["processed"], seed["faces"]) == (seed_id, "done", 2, 2)


def test_a_paused_seed_does_nothing_until_resumed(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, [(1, "gate", PERSON, _at(72))])
    monkeypatch.setattr(service_module, "capture_event", lambda **k: CaptureResult(faces=[_face()], main_frames=5))
    seed_id = service.store.create_seed(
        start_epoch=(NOW - timedelta(days=5)).timestamp(), end_epoch=NOW.timestamp(), cameras=[], total=1)
    service.store.set_seed_state(seed_id, "paused")
    assert service.run_once(NOW) is False
    assert service.store.attempts([1]) == {}
    service.store.set_seed_state(seed_id, "running")
    assert service.run_once(NOW) is True
    assert service.store.attempts([1])[1][0] == "done"


def test_routes_validate_the_range_and_cameras_and_need_capture_on(tmp_path):
    service, config = _service(tmp_path, [(1, "gate", PERSON, _at(72))])
    manager = SimpleNamespace(face_capture=service, config=config)
    router = create_face_capture_router(FaceCaptureRouteDependencies(get_manager=lambda: manager))
    endpoint = {route.path + ":" + ",".join(sorted(route.methods)): route.endpoint for route in router.routes}
    create = endpoint["/api/face-capture/seeds:POST"]
    change = endpoint["/api/face-capture/seeds/{seed_id}/{action}:POST"]
    status = endpoint["/api/face-capture/status:GET"]

    def request(start, end, cameras=None):
        return SeedRequest(start=start, end=end, cameras=cameras)

    with pytest.raises(HTTPException) as error:
        create(request("2026-11-01T00:00:00", "2026-11-02T00:00:00+00:00"))
    assert error.value.status_code == 400  # no timezone
    with pytest.raises(HTTPException) as error:
        create(request(_at(10), _at(20)))
    assert error.value.status_code == 400  # start after end
    with pytest.raises(HTTPException) as error:
        create(request(_at(100), _at(1), ["gate", "attic"]))
    assert error.value.status_code == 400 and "attic" in error.value.detail
    seed = create(request(_at(100), (NOW + timedelta(days=1)).isoformat(), ["gate"]))
    assert seed["state"] == "running" and seed["total"] == 1 and seed["cameras"] == ["gate"]
    assert seed["end_epoch"] <= datetime.now(timezone.utc).timestamp()  # the end is capped at now
    with pytest.raises(HTTPException) as error:
        create(request(_at(100), _at(1)))
    assert error.value.status_code == 409
    assert change(seed["id"], "pause")["state"] == "paused"
    with pytest.raises(HTTPException) as error:
        change(seed["id"], "pause")
    assert error.value.status_code == 409
    body = status()
    assert body["cameras"] == [{"id": "gate", "name": "Gate"}, {"id": "foyer", "name": "foyer"}]
    assert body["seeds"][0]["id"] == seed["id"]
    config.detector = DetectorConfig(face_capture_enabled=False)
    with pytest.raises(HTTPException) as error:
        create(request(_at(100), _at(1)))
    assert error.value.status_code == 409


def test_every_face_capture_route_needs_the_admin_scope():
    for method, path in (("GET", "/api/face-capture/status"), ("POST", "/api/face-capture/seeds"),
                         ("POST", "/api/face-capture/seeds/1/pause")):
        assert required_api_scope(method, path) == "admin"


def test_a_seed_pass_resumes_below_the_last_event_and_rescans_before_finishing(tmp_path, monkeypatch):
    rows = [(i, "gate", PERSON, _at(72 + i)) for i in range(1, 6)]
    service, _ = _service(tmp_path, rows)
    monkeypatch.setattr(service_module, "capture_event", lambda **k: CaptureResult(faces=[_face()], main_frames=5))
    service.store.create_seed(
        start_epoch=(NOW - timedelta(days=5)).timestamp(), end_epoch=(NOW - timedelta(days=1)).timestamp(),
        cameras=[], total=5)
    scans = []
    real = service.seed_events
    monkeypatch.setattr(service, "seed_events", lambda *a, **k: scans.append(k.get("before_id")) or real(*a, **k))
    while service.run_once(NOW):
        pass
    # After the first full scan each pass resumes at the event last taken;
    # only the final pass, which finds nothing below, rescans the whole range.
    assert scans == [None, 6, 5, 4, 3, 2, None]
    assert service.store.seeds(1)[0]["state"] == "done"


def test_a_seed_paused_while_it_is_being_finished_is_left_paused(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, [])
    seed_id = service.store.create_seed(
        start_epoch=(NOW - timedelta(days=5)).timestamp(), end_epoch=(NOW - timedelta(days=4)).timestamp(),
        cameras=[], total=0)
    real = service.next_seed_event

    def pause_meanwhile(seed, now=None):
        service.store.set_seed_state(seed_id, "paused")
        return real(seed, now)

    monkeypatch.setattr(service, "next_seed_event", pause_meanwhile)
    assert service.run_once(NOW) is False  # no SeedConflict out of the pass
    assert service.store.seeds(1)[0]["state"] == "paused"


def test_a_finished_seed_counts_every_event_in_its_range(tmp_path):
    store = FaceCaptureStore(tmp_path)
    seed = store.create_seed(start_epoch=0, end_epoch=10, cameras=[], total=5)
    store.seed_progress(seed, faces=2)
    assert store.set_seed_state(seed, "done")["processed"] == 5


def test_events_the_regular_catch_up_captures_count_toward_the_seed(tmp_path, monkeypatch):
    # Both events are inside the 24 h catch-up window and the seed's range.
    service, _ = _service(tmp_path, [(1, "gate", PERSON, _at(5)), (2, "gate", PERSON, _at(3))])
    monkeypatch.setattr(service_module, "capture_event", lambda **k: CaptureResult(faces=[_face()], main_frames=5))
    start, end = (NOW - timedelta(hours=10)).timestamp(), (NOW - timedelta(hours=1)).timestamp()
    seed_id = service.store.create_seed(start_epoch=start, end_epoch=end, cameras=[], total=2)
    while service.run_once(NOW):
        pass
    service.run_once(NOW)
    seed = service.store.seeds(1)[0]
    assert (seed["id"], seed["processed"], seed["faces"], seed["state"]) == (seed_id, 2, 2, "done")


def test_a_running_seed_gets_every_fourth_capture_while_new_events_wait(tmp_path, monkeypatch):
    new = [(10 + i, "gate", PERSON, _at(1 + i * 0.1)) for i in range(8)]   # inside the 24 h catch-up
    old = [(1, "foyer", PERSON, _at(72)), (2, "foyer", PERSON, _at(73))]   # only the seed reaches these
    service, _ = _service(tmp_path, old + new)
    monkeypatch.setattr(service_module, "capture_event", lambda **k: CaptureResult(faces=[_face()], main_frames=5))
    service.store.create_seed(start_epoch=(NOW - timedelta(days=5)).timestamp(),
                              end_epoch=(NOW - timedelta(days=2)).timestamp(), cameras=["foyer"], total=2)
    order = []
    real = service.capture
    monkeypatch.setattr(service, "capture", lambda *c: order.append(c[0]) or real(*c))
    for _ in range(8):
        service.run_once(NOW)
    assert [i for i, event in enumerate(order) if event in (1, 2)] == [3, 7]
