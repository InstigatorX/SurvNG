from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.config import CameraTransitionRoute, MotionQualificationConfig
from survng.app.detection_watch import RouteDetectionWatch
from survng.app.events import EventStore
from survng.app.motion import MotionQualificationResult
from survng.app.motion_decisions import MotionDecisionOrchestrator
from survng.app.motion_events import MotionEventCoordinator, MotionTrigger, RetryDisposition
from survng.app.motion_incidents import MotionIncidentService
from survng.app.motion_pipeline.decision_handler import MotionDecisionHandler
from survng.app.motion_pipeline.object_detection import RecordedDetectionResult


@pytest.mark.parametrize("fail_first", [False, True])
def test_route_occurrences_have_independent_delivery_and_incident_outcomes(
    tmp_path: Path, fail_first: bool,
) -> None:
    store = EventStore(tmp_path)
    events = MotionEventCoordinator(
        queue_size=4, retry_limit=0, camera_id="gate", durable_store=store,
    )
    watches = RouteDetectionWatch([
        CameraTransitionRoute(from_camera="source", to_camera="gate"),
    ])
    now = datetime.now(timezone.utc)
    expected_ids = [f"route:gate:source:{index}" for index in (101, 102)]
    provider_calls = []
    objects = [{
        "label": "person", "confidence": 0.9,
        "box": {"x1": 1, "y1": 1, "x2": 10, "y2": 10},
        "detection_frame_width": 20, "detection_frame_height": 20,
        "incident_eligible": True,
    }]

    def provider(at, qualification):
        intent_id = qualification["detection_intent_id"]
        provider_calls.append(intent_id)
        # The other occurrence must not even be leased during the first decision.
        if intent_id == expected_ids[0]:
            assert store.motion_trigger_status("gate") == {"queued": 1, "running": 1}
            if fail_first:
                raise RuntimeError("first occurrence inference failed")
        return RecordedDetectionResult(
            np.zeros((20, 20, 3), dtype=np.uint8), objects, "", {},
            frame_captured_at_epoch=at.timestamp(), frame_source="recorded_main",
        )

    publish = Mock()
    admission = Mock(side_effect=watches.consume_origin)
    tracking = Mock(return_value=True)
    handler = MotionDecisionHandler(
        "gate", store, provider, lambda *_args: "", json.dumps,
        event_callback=publish, route_admission_callback=admission,
    )
    incidents = MotionIncidentService(
        camera_id="gate", decision_processor=handler,
        tracking_enabled=lambda: True, has_trackable_objects=bool,
        start_tracking=tracking, prewarm_tracking=lambda: None,
        image_reader=lambda _path: None, refinement_store=store,
    )
    qualification = Mock()
    qualification.settings.return_value = ("camera_rescue", "balanced", 320)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    qualification.with_pipeline_telemetry.side_effect = lambda result: result
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate", events=events, audit_recorder=handler,
        config=MotionQualificationConfig(burst_quiet_seconds=0.1),
        qualification=qualification, incidents=incidents,
        media=Mock(), analysis=Mock(), state=Mock(),
    )
    for index, intent_id in zip((101, 102), expected_ids):
        at = now + timedelta(seconds=index - 101)
        watch = watches.observe_incident(
            camera_id="source", event_id=index, event_at=at.timestamp(), objects=objects,
        )[0]
        trigger = MotionTrigger(
            topic="adaptive/visual_backup", message="route watch", event_at=at,
            received_at=at.timestamp(), episode_id=intent_id,
            detection_intent_id=intent_id,
            prequalified=MotionQualificationResult(
                True, 0.9, 0.4, "ema_v2_qualified", 4,
                {"ema_v2": True, "motion_regions": [[0, 0, 1, 1]],
                 "route_detection_watch": watch.as_dict()},
            ),
        )
        assert events.enqueue(trigger)
        # Repeated notices of this durable identity need no additional decision.
        duplicate = MotionTrigger.from_durable_payload(trigger.durable_payload(), "")
        assert events.enqueue(duplicate)
    stop = threading.Event()
    for position, intent_id in enumerate(expected_ids):
        first = events.next_trigger(0.1)
        assert first is not None
        batch = events.coalesce(first, quiet_seconds=0.1, stop_event=stop)
        assert batch is not None
        assert [trigger.detection_intent_id for trigger in batch] == [intent_id]
        events.set_active(batch)
        if fail_first and position == 0:
            with pytest.raises(RuntimeError, match="first occurrence"):
                orchestrator._process_batch(batch, stop)
            terminal = Mock(wraps=orchestrator._fail_episode_intents)
            orchestrator._fail_episode_intents = terminal
            failed = events.take_failed_active()
            assert failed is not None
            assert orchestrator.retry_batch(failed, stop) == RetryDisposition.DROPPED
            assert [
                trigger.detection_intent_id for trigger in terminal.call_args.args[0]
            ] == [intent_id]
            assert store.motion_trigger_status("gate") == {"failed": 1, "queued": 1}
        else:
            orchestrator._process_batch(batch, stop)
        if position == 0:
            assert watches.status(now.timestamp())["active"] == (2 if fail_first else 1)

    assert provider_calls == expected_ids
    with store._connect() as connection:
        rows = list(connection.execute(
            "select id, detection_intent_id, objects_json from events order by id"
        ))
        admissions = list(connection.execute(
            "select origin_event_id, event_id from route_incident_admissions order by event_id"
        ))
    successful_ids = expected_ids[1:] if fail_first else expected_ids
    assert [row["detection_intent_id"] for row in rows] == successful_ids
    assert [call.args for call in admission.call_args_list] == [
        ("gate", "source", int(identity.rsplit(":", 1)[1])) for identity in successful_ids
    ]
    assert [row["origin_event_id"] for row in admissions] == [
        int(identity.rsplit(":", 1)[1]) for identity in successful_ids
    ]
    event_ids = [row["id"] for row in rows]
    assert [call.args[0] for call in tracking.call_args_list] == event_ids
    for kind in ("incident", "object"):
        assert [
            call.args[1]["event_id"]
            for call in publish.call_args_list if call.args[0] == kind
        ] == event_ids
    for row in rows:
        persisted = next(
            item["motion_qualification"] for item in json.loads(row["objects_json"])
            if "motion_qualification" in item
        )
        assert persisted["detection_intent_id"] == row["detection_intent_id"]
        watch = persisted["features"]["route_detection_watch"]
        assert watch["source_event_id"] == int(row["detection_intent_id"].rsplit(":", 1)[1])
    assert store.motion_trigger_status("gate") == ({"failed": 1} if fail_first else {})
