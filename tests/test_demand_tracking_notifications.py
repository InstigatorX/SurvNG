"""Viewer-requested analysis adds evidence, not new live security activity."""
import json
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np

from survng.app.event_store import EventStore
from survng.app.evidence_projection import EvidenceProjection
from survng.app.state_events import StateEventBroker
from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.object_tracking import ObjectTrackingSession
from survng.app.video_frames import VideoFrameReference


def person(epoch=1000):
    return {"label": "person", "confidence": .8, "incident_eligible": True,
            "alert_eligible": True, "captured_at_epoch": epoch,
            "frame_source": "recorded_main",
            "track_id": 7,
            "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 70}}


def prepared(tmp_path, *, demand=True):
    store = EventStore(tmp_path)
    event = store.add_event("gate", "motion",
                            created_at=datetime.fromtimestamp(1000, timezone.utc).isoformat(),
                            objects_json=json.dumps([person()]))
    store.enqueue_scene_tracking(event["id"], 1000, 1045)
    job = store.claim_scene_tracking("gate", "worker")
    store.settle_scene_incidents()
    with store._connect() as conn:
        if demand:
            conn.execute("update scene_analysis_jobs set admission='demand',request_end_epoch=1045 "
                         "where episode_id=?", (job["episode_id"],))
        conn.execute("delete from scene_notification_outbox")
        conn.execute("delete from event_evidence_outbox")
    identity = {"episode_id": job["episode_id"], "lease_owner": "worker"}
    return store, event, identity


def snapshot(store):
    with store._connect() as conn:
        episode = conn.execute("select last_activity_epoch from scene_episodes").fetchone()[0]
        incident = dict(conn.execute("select state,revision from scene_incidents").fetchone())
        establishment = conn.execute("select decision_id from scene_event_establishment").fetchone()[0]
    return episode, incident, establishment


@pytest.mark.parametrize("state", ["active", "complete"])
def test_demand_tracking_retains_evidence_without_live_activity_or_notifications(tmp_path, state):
    store, event, identity = prepared(tmp_path)
    before = snapshot(store)
    store.update_object_tracking(event["id"], {
        "scene_analysis_job": identity, "state": state,
        "window_end_epoch": 1045, "scene_cursor_epoch": 1045,
        "scene_observations": [person(1020)],
    })
    after = snapshot(store)
    assert after[0] == before[0]
    assert after[1]["state"] == before[1]["state"] == "complete"
    assert after[1]["revision"] > before[1]["revision"]
    assert after[2] == before[2]
    assert store.scene_pending_notifications() == []
    with store._connect() as conn:
        assert conn.execute("select 1 from scene_observations where captured_epoch=1020").fetchone()


def test_payload_cannot_make_automatic_job_non_notifying(tmp_path):
    store, event, identity = prepared(tmp_path, demand=False)
    store.update_object_tracking(event["id"], {
        "scene_analysis_job": {**identity, "admission": "demand"}, "state": "active",
        "scene_observations": [person(1020)],
    })
    assert store.scene_pending_notifications()


def test_demand_checkpoint_second_projection_keeps_suppression_after_lease_release(tmp_path):
    store, event, identity = prepared(tmp_path)
    checkpoint = store._checkpoint_scene_tracking

    def checkpoint_and_extend(*args):
        checkpoint(*args)
        return True

    store._checkpoint_scene_tracking = checkpoint_and_extend
    store._scene_ingest = Mock(wraps=store._scene_ingest)
    store.update_object_tracking(event["id"], {
        "scene_analysis_job": identity, "state": "complete",
        "window_end_epoch": 1045, "scene_cursor_epoch": 1045,
        "scene_observations": [person(1020)],
    })
    assert store._scene_ingest.call_count == 2
    assert all(call.kwargs["notify"] is False and call.kwargs["activity"] is False
               for call in store._scene_ingest.call_args_list)
    assert store.scene_pending_notifications() == []


def test_demand_context_continuation_preserves_non_notifying_provenance(tmp_path):
    store, event, identity = prepared(tmp_path)
    for index in range(205):
        epoch = 1001 + index / 10
        store.acquire_scene_sample(sample_id=f"context-{index:03d}", camera_id="gate",
                                   captured_epoch=epoch, source="recorded_main", status="complete",
                                   observations=[person(epoch)])
    before = snapshot(store)
    store.update_object_tracking(event["id"], {
        "scene_analysis_job": identity, "state": "complete",
        "window_end_epoch": 1045, "scene_cursor_epoch": 1045,
    })
    with store._connect() as conn:
        job = conn.execute("select * from scene_context_projection_jobs").fetchone()
        assert job is not None and job["notify"] == 0
    assert store.scene_pending_notifications() == []
    # The obligation must remain safe after a service restart, too.
    resumed = EventStore(tmp_path)
    assert resumed.project_pending_scene_context() > 0
    assert resumed.scene_pending_notifications() == []
    after = snapshot(resumed)
    assert after[0] == before[0]
    assert after[1]["state"] == "complete"
    with resumed._connect() as conn:
        assert conn.execute("select count(*) from scene_context_projection_jobs").fetchone()[0] == 0
        assert conn.execute("select count(*) from scene_observations where captured_epoch>1000").fetchone()[0] == 205


def project_evidence(store):
    notification = Mock()
    broker = StateEventBroker()
    subscriber = broker.subscribe()
    projection = EvidenceProjection(store, lambda: SimpleNamespace(config=SimpleNamespace(enabled=False)),
                                    broker, notification)
    projection.run_once()
    return notification, subscriber


def test_demand_cover_and_projection_do_not_reopen_notifications(tmp_path):
    store, event, identity = prepared(tmp_path)
    cover = tmp_path / "new.webp"
    cover.write_bytes(b"new cover")
    before = snapshot(store)
    promoted = store.promote_tracking_cover(event["id"], snapshot_path=str(cover),
        captured_at=1020, frame_width=100, frame_height=100, tracked_objects=[person(1020)],
        cover_metrics={}, scene_analysis_job=identity)
    assert promoted is not None and promoted["snapshot_path"] == "new.webp"
    notification, subscriber = project_evidence(store)
    notification.assert_not_called()
    assert subscriber.get_nowait().type == "incident"
    assert store.scene_pending_notifications() == []
    after = snapshot(store)
    assert after[0] == before[0] and after[2] == before[2]


@pytest.mark.parametrize("pending_eager", [False, True])
def test_evidence_coalescing_preserves_only_pending_eager_notification(tmp_path, pending_eager):
    store, event, identity = prepared(tmp_path)
    with store._connect() as conn:
        row = conn.execute("select * from events where id=?", (event["id"],)).fetchone()
        store._evidence_outbox(conn, row, "evidence_updated", notify=True)
        if not pending_eager:
            conn.execute("update event_evidence_outbox set publication_done=1")
    store.update_object_tracking(event["id"], {
        "scene_analysis_job": identity, "state": "active",
        "scene_observations": [{**person(1020), "snapshot_path": "new.webp"}],
    })
    notification, subscriber = project_evidence(store)
    assert bool(notification.call_count) is pending_eager
    assert subscriber.get_nowait().type == "incident"


def test_stale_demand_lease_cannot_promote_cover(tmp_path):
    store, event, identity = prepared(tmp_path)
    cover = tmp_path / "snapshots" / "new.webp"
    cover.parent.mkdir()
    cover.write_bytes(b"new cover")
    promoted = store.promote_tracking_cover(event["id"], snapshot_path=str(cover),
        captured_at=1020, frame_width=100, frame_height=100, tracked_objects=[person(1020)],
        cover_metrics={}, scene_analysis_job={**identity, "lease_owner": "obsolete"})
    assert promoted is None
    assert not cover.exists()
    assert store.get(event["id"])["snapshot_path"] != "new.webp"
    assert store.scene_pending_notifications() == []


def test_eager_obligation_can_follow_demand_at_same_evidence_revision(tmp_path):
    store, event, _identity = prepared(tmp_path)
    with store._connect() as conn:
        row = conn.execute("select * from events where id=?", (event["id"],)).fetchone()
        store._evidence_outbox(conn, row, "evidence_updated", notify=False)
    notification, _subscriber = project_evidence(store)
    notification.assert_not_called()
    # Also exercise a coalesced obligation whose demand publication was not
    # acknowledged yet (for example, its semantic encoder is still working).
    with store._connect() as conn:
        row = conn.execute("select * from events where id=?", (event["id"],)).fetchone()
        store._evidence_outbox(conn, row, "evidence_updated", notify=False)
        store._evidence_outbox(conn, row, "evidence_updated", notify=True)
    notification, _subscriber = project_evidence(store)
    notification.assert_called_once_with("gate", event["id"])


def test_session_cover_promotion_and_satisfied_requirement_remain_non_alerting(tmp_path):
    store, event, identity = prepared(tmp_path)
    now = time.time()
    with store._connect() as conn:
        conn.execute("insert into event_cover_requirements "
                     "(event_id,state,deadline_epoch,available_at_epoch,payload_json,created_at,updated_at) "
                     "values(?,'pending',?,?,'{}',?,?)", (event["id"], now + 300, now, now, now))
    cover = tmp_path / "new.webp"

    def snapshot_writer(_frame, _at):
        cover.write_bytes(b"new cover")
        return str(cover)

    verified = {**person(1004), "box": {"x1": 60, "y1": 35, "x2": 210, "y2": 145}}
    session = ObjectTrackingSession(
        camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        config=ObjectTrackingConfig(analysis_mode="on_demand"),
        detector=SimpleNamespace(config=SimpleNamespace(confidence_threshold=.7),
                                 detect=lambda *_args, **_kwargs: [verified]),
        frame_provider=lambda: None, update_event=store.update_object_tracking,
        publisher=None, limiter=threading.BoundedSemaphore(1),
        cover_frame_provider=lambda *_args: np.full((200, 300, 3), 127, dtype=np.uint8),
        snapshot_writer=snapshot_writer, cover_promoter=store.promote_tracking_cover,
    )
    session._scene_analysis_job = {**identity, "admission": "demand"}
    session._frame_width = session._frame_height = 100
    image = np.full((100, 100, 3), 127, dtype=np.uint8)
    initial = {**person(), "snapshot_primary_subject": True,
               "box": {"x1": 45, "y1": 45, "x2": 55, "y2": 55}}
    closer = {**person(1004), "box": {"x1": 25, "y1": 20, "x2": 75, "y2": 70}}
    session._consider_cover_candidate(image, 1000, [initial], {7})
    session._consider_cover_candidate(image, 1004, [closer], {7}, VideoFrameReference(
        source_path=tmp_path / "segment.mp4", seek_offset_seconds=4, pts=123,
        pts_seconds=0, time_base_num=1, time_base_den=90000, captured_at=1004))
    session._promote_cover_candidate(event["id"])
    assert session._cover_promotion["cover_promoted"] is True
    assert store.get(event["id"])["snapshot_path"] == "new.webp"
    assert store.cover_requirement(event["id"])["state"] == "satisfied"
    notification, subscriber = project_evidence(store)
    notification.assert_not_called()
    assert subscriber.get_nowait().type == "incident"
    assert store.scene_pending_notifications() == []
