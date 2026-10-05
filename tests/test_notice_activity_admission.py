"""A notice and an unrelated detected object must not manufacture an incident."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import numpy as np
import pytest

from survng.app.events import EventStore
from survng.app.motion_pipeline import MotionDecisionHandler
from survng.app.motion_pipeline.object_detection import RecordedDetectionResult
from survng.app.scene_zone_admission import evaluate_scene_establishment
from tests.test_scene_zone_admission import box, physical, policy, zone


def evidence(*, movement=0.00383, entry=False, arrival=False, y=80):
    samples = physical(["car"] * 3, [box(20, y)] * 3, confidence=0.93)
    for sample in samples:
        sample["metadata"]["activity_witnesses"] = []
        for item in sample["observations"]:
            item.update(
                incident_eligible=True, activity_role="indeterminate",
                temporal_track_observations=2 if arrival else 3,
                temporal_robust_new_appearance=arrival,
                temporal_zone_entry=entry,
                temporal_motion={"version": 2, "excursion_ratio": movement},
                temporal_center_displacement_ratio=movement,
                temporal_center_path_ratio=movement,
            )
    return samples


def notice(*, source="motion"):
    return {"source": source, "epoch": 1001,
            "activity_policy": {"version": 1, "require_motion_correlation": True,
                                "alignment": {"reliable": True}},
            "features": {"motion_regions": [[0.1, 0.5, 0.4, 0.9]]}}


@pytest.mark.parametrize("source", ["motion", "camera"])
@pytest.mark.parametrize("restricted", [True, False])
def test_uncorrelated_notice_does_not_establish_even_with_high_confidence(source, restricted):
    samples = evidence()
    original = deepcopy(samples)
    result = evaluate_scene_establishment(
        samples, policy=policy(zone("Entry", "incident", 0.5, 1)) if restricted else policy(require=False),
        notice=notice(source=source),
    )
    assert result["status"] == "pending"
    assert result["reason"] == "object_not_motion_correlated"
    assert result["supporting_observation_ids"] == []
    assert samples == original  # assessment must not rewrite raw acquisitions


@pytest.mark.parametrize("changes", [{"movement": 0.06}, {"entry": True}, {"arrival": True}])
def test_motion_entry_and_arrival_can_establish(changes):
    result = evaluate_scene_establishment(
        evidence(**changes), policy=policy(zone("Entry", "incident", 0.5, 1)), notice=notice(),
    )
    assert result["status"] == "supported"
    assert result["reason"] == "verified_object_activity"
    assert result["evidence_kind"] == "motion_correlated_object"


def test_correlated_object_outside_zone_cannot_use_unrelated_in_zone_box():
    outside = evidence(movement=0.06, y=40)
    inside = evidence()
    for sample in outside:
        for item in sample["observations"]:
            item["id"] += "-outside"
    result = evaluate_scene_establishment(
        outside + inside, policy=policy(zone("Entry", "incident", 0.5, 1)), notice=notice(),
    )
    assert result["status"] != "supported"
    assert result["supporting_observation_ids"] == []


def test_independent_localized_evidence_still_establishes():
    samples = physical(["person"] * 2, [box(20, 80), box(40, 80)], confidence=0.9)
    result = evaluate_scene_establishment(
        samples, policy=policy(zone("Entry", "incident", 0.5, 1)), notice=notice(),
    )
    assert result["status"] == "supported"
    assert result["reason"] == "eligible_zone_activity"


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("notifications", [False, True])
def test_handler_and_durable_admission_agree_and_survive_restart(tmp_path, moving, notifications):
    store = EventStore(tmp_path)
    samples = evidence(movement=0.06 if moving else 0.00383)
    observations = [item for sample in samples for item in sample["observations"]]
    cover = deepcopy(observations[1])
    # A notification rejection is not a rejection of physical activity.
    cover["incident_eligible"] = notifications
    if not notifications:
        cover["incident_ineligible_reasons"] = ["notifications_disabled"]
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    provider = RecordedDetectionResult(
        frame, [cover, {"status": "scene_observations", "observations": observations, "samples": samples}],
        "", {}, frame_captured_at_epoch=1001, frame_source="recorded_main",
    )
    handler = MotionDecisionHandler(
        camera_id="gate", events=store, detection_provider=lambda _at: provider,
        snapshot_writer=lambda *_: "", object_serializer=json.dumps,
        establishment_zone_policy=lambda: policy(zone("Entry", "incident", 0.5, 1), confidence_threshold=0.7),
    )
    outcome = handler.handle(
        "adaptive/visual_backup", "visual backup", datetime.fromtimestamp(1001, timezone.utc),
        {"trigger_source": "visual_backup", "features": notice()["features"]},
        require_motion_correlation=True,
    )
    assert outcome.event_id is not None  # raw evidence remains accessible
    assert any(item.get("alert_eligible") for item in outcome.detected_objects) is (moving and notifications)
    assert (store.scene_incident(event_id=outcome.event_id) is not None) is moving
    with store._connect() as conn:
        decision = conn.execute(
            "select d.* from scene_activity_decisions d join scene_event_establishment x on x.decision_id=d.id where x.event_id=?",
            (outcome.event_id,),
        ).fetchone()
        assert decision["reason"] == ("verified_object_activity" if moving else "object_not_motion_correlated")
        assert conn.execute("select count(*) from acquired_observations").fetchone()[0] >= 3
    restarted = EventStore(tmp_path)
    assert (restarted.scene_incident(event_id=outcome.event_id) is not None) is moving


def test_unavailable_evidence_remains_unresolved():
    samples = [{"id": "failed", "camera_id": "gate", "captured_epoch": 1001,
                "status": "failed", "observations": [], "metadata": {}}]
    result = evaluate_scene_establishment(samples, policy=policy(require=False), notice=notice())
    assert result["status"] == "incomplete"
    assert result["reason"] == "source_evidence_unavailable"


def test_full_frame_measured_activity_remains_sufficient():
    samples = [{"id": "motion", "camera_id": "gate", "captured_epoch": 1001,
                "status": "complete", "observations": [],
                "metadata": {"validated_motion": True, "motion_source": "measured_pixels"}}]
    result = evaluate_scene_establishment(samples, policy=policy(require=False), notice=notice())
    assert result["status"] == "supported"
    assert result["reason"] == "measured_motion"


@pytest.mark.parametrize("entry", [False, True])
def test_recorded_scene_retains_arrival_and_entry_evidence(entry, monkeypatch):
    import time
    from types import SimpleNamespace
    from survng.app.config import CameraConfig
    from survng.app.motion_pipeline.object_detection import (
        RecordedMotionObjectDetector, _RecordedDetectionSample, _temporal_consensus,
    )
    samples = []
    for index, offset in enumerate((-2.0, -1.0, 0.0, 0.5)):
        objects = []
        if entry or offset >= 0:
            objects = [{"label": "car", "confidence": 0.95, "box": box(20, 80),
                        "incident_eligible": True, "detection_frame_width": 160,
                        "detection_frame_height": 100,
                        "spatial_zones": ["Entry"] if index >= 2 else []}]
        samples.append(_RecordedDetectionSample(offset, np.zeros((100, 160, 3), dtype=np.uint8), objects, ""))
    selected, objects = _temporal_consensus(samples, 2)
    expected = "temporal_zone_entry" if entry else "temporal_robust_new_appearance"
    assert objects[0][expected] is True
    backend = RecordedMotionObjectDetector(
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        SimpleNamespace(config=SimpleNamespace()), SimpleNamespace(), lambda: None,
    )
    monkeypatch.setattr(backend, "_enrich_selected_faces", lambda frame, objects, timing: objects)
    monkeypatch.setattr(backend, "_enrich_depth", lambda frame, objects, offset, timing: objects)
    result = backend._recorded_result(selected, objects, samples, {}, time.monotonic(),
                                      refinement_pending=False, event_epoch=1001)
    batch = next(item for item in result.objects if item.get("status") == "scene_observations")
    assert all(item[expected] is True for item in batch["observations"])
    assessment = evaluate_scene_establishment(batch["samples"], policy=policy(require=False), notice=notice())
    assert assessment["status"] == "supported"


def test_pending_evidence_can_establish_later_without_noise_prolonging_it(tmp_path):
    store = EventStore(tmp_path)

    def objects_at(epoch, movement):
        samples = evidence(movement=movement)
        for sample in samples:
            sample["id"] += f"-{epoch}"
            sample["captured_epoch"] += epoch - 1001
            for item in sample["observations"]:
                item["id"] += f"-{epoch}"
                item["captured_at_epoch"] += epoch - 1001
        return json.dumps([
            {"status": "scene_observations", "samples": samples,
             "observations": [item for sample in samples for item in sample["observations"]]},
            {"status": "motion_qualification", "motion_qualification": {
                "trigger_source": "visual_backup", "features": notice()["features"],
                "notice_activity_policy": notice()["activity_policy"],
                "establishment_zone_policy": policy(zone("Entry", "incident", 0.5, 1)),
            }},
        ])

    event = store.add_event("gate", "motion", created_at=datetime.fromtimestamp(1001, timezone.utc).isoformat(),
                            objects_json=objects_at(1001, 0.00383))
    assert store.scene_incident(event_id=event["id"]) is None
    store.refine_event_evidence(event["id"], snapshot_path="", recording_path="", objects_json=objects_at(1004, 0.06))
    incident = store.scene_incident(event_id=event["id"])
    assert incident is not None
    with store._connect() as conn:
        before = conn.execute("select last_activity_epoch from scene_episodes where incident_id=?", (incident["id"],)).fetchone()[0]
    noise = store.add_event("gate", "motion", created_at=datetime.fromtimestamp(1007, timezone.utc).isoformat(),
                            objects_json=objects_at(1007, 0.00383))
    with store._connect() as conn:
        after = conn.execute("select last_activity_epoch from scene_episodes where incident_id=?", (incident["id"],)).fetchone()[0]
        assert after == before
        verdict = conn.execute("select d.verdict from scene_activity_decisions d join scene_event_establishment x on x.decision_id=d.id where x.event_id=?", (noise["id"],)).fetchone()[0]
        assert verdict != "supported"
