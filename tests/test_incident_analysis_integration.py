"""Exercise the real durable store and tracking worker through viewer admission."""
from datetime import datetime, timezone
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.event_store import EventStore
from survng.app.incident_queries import IncidentQueryService
from survng.app.object_tracking import ObjectTrackingSession
from survng.app.object_tracking_lifecycle import ObjectTrackingLifecycle


@pytest.mark.parametrize("missing_media", [False, True])
def test_bookmark_open_analyze_and_reopen_without_extra_inference(tmp_path, missing_media):
    store = EventStore(tmp_path)
    at = datetime.fromtimestamp(1000, timezone.utc)
    person = {"label": "person", "confidence": .9, "incident_eligible": True,
              "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 70},
              "detection_frame_width": 100, "detection_frame_height": 100}
    event = store.add_event("gate", "motion", created_at=at.isoformat(), objects_json=json.dumps([person]))
    incident_id = store.scene_incident(event_id=event["id"])["id"]
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
    config = ObjectTrackingConfig(analysis_mode="on_demand", capacity_wait_seconds=0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detect = Mock(side_effect=lambda *_args, **_kwargs: [dict(person)])
    decode = Mock(side_effect=lambda start, end, fps, width, **kwargs:
                  [] if missing_media else [(epoch, frame) for epoch in np.arange(1000, 1045.01, .5) if start <= epoch <= end])
    publisher = Mock()
    session = ObjectTrackingSession(
        camera=camera, config=config,
        detector=SimpleNamespace(config=SimpleNamespace(confidence_threshold=.45, require_incident_zone=False), detect=detect),
        frame_provider=Mock(), catchup_frame_provider=decode,
        update_event=store.update_object_tracking, publisher=publisher,
        limiter=threading.BoundedSemaphore(1), window_provider=lambda *_: (1000, 1045),
    )
    session.set_accepting(True)
    factory = Mock()
    factory.create.return_value = session
    lifecycle = ObjectTrackingLifecycle(
        camera=camera, factory=factory, frame_provider=Mock(), catchup_frame_provider=decode,
        prewarm_frame_provider=Mock(), history=Mock(), accepting=lambda: True,
        lifecycle_lock=threading.RLock(), scene_job_store=store,
    )
    # Old footage must not contaminate the live stationary-subject memory.
    session.activity_attributor = Mock()
    manager = SimpleNamespace(events=store, workers={"gate": SimpleNamespace(tracking_lifecycle=lifecycle)},
                              config=SimpleNamespace(detector=SimpleNamespace(tracking=config)))
    assert lifecycle.start_incident(event["id"], at, [person])
    assert IncidentQueryService.analysis(manager, incident_id)["status"] == "deferred"
    assert lifecycle.resume_pending_scene() is False
    detect.assert_not_called()
    decode.assert_not_called()
    assert store.scene_incident(incident_id)["labels"] == ["person"]
    store.settle_scene_incidents()
    with store._connect() as conn:
        conn.execute("delete from scene_notification_outbox")
    assert IncidentQueryService.analysis(manager, incident_id, request=True)["status"] == "queued"
    try:
        assert lifecycle.resume_pending_scene()
        assert session.wait_stopped(10)
    finally:
        session.stop()
    expected = "unavailable" if missing_media else "complete"
    assert IncidentQueryService.analysis(manager, incident_id)["status"] == expected
    assert (detect.call_count == 0) if missing_media else (detect.call_count > 0)
    assert decode.call_count > 0
    assert store.scene_pending_notifications() == []
    session.activity_attributor.stamp_recorded.assert_not_called()
    assert publisher.call_args.args[1]["scene_analysis_job"]["admission"] == "demand"
    previous_calls = detect.call_count
    assert IncidentQueryService.analysis(manager, incident_id, request=True)["status"] == expected
    assert lifecycle.resume_pending_scene() is False
    assert detect.call_count == previous_calls
