"""Zones gate establishment and prolongation, not scene contents or alerts."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from survng.app.config import CameraConfig, DetectionZone
from survng.app.events import EventStore
from survng.app.incident_utils import DEFAULT_INCIDENT_GAP_SECONDS
from survng.app.scene_activity_evidence import scene_sample_records
from survng.app.scene_activity import evaluate_scene_activity
from survng.app.motion_pipeline.scene_acquisition import acquire_detection_result
from survng.app.scene_zone_admission import (
    discovery_requires_confirmation,
    establishment_zone_policy,
    evaluate_scene_establishment,
    interpret_establishment_observation,
)
from tests.test_manager_lifecycle import manager_with_mocks


def zone(name, behavior, y0, y1, *, classes=None, x0=0.0, x1=1.0, minimum=None, maximum=None, notifications=True):
    return {
        "name": name, "enabled": True, "behavior": behavior, "object_classes": classes or [],
        "points": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}],
        "min_depth_m": minimum, "max_depth_m": maximum, "notifications_enabled": notifications,
    }


def policy(*zones, require=True, confidence_threshold=None, class_confidence_thresholds=None):
    payload = {"version": 1, "require_incident_zone": require, "zones": list(zones)}
    if confidence_threshold is not None:
        payload["confidence_threshold"] = confidence_threshold
    if class_confidence_thresholds:
        payload["class_confidence_thresholds"] = class_confidence_thresholds
    return payload


def box(x, y2, y1=None):
    top = y2 - 20 if y1 is None else y1
    return {"x1": x, "y1": top, "x2": x + 20, "y2": y2}


def physical(labels, boxes, *, confidence=0.3, track="subject", alert=False, zone_names=None):
    samples = []
    for index, (label, item) in enumerate(zip(labels, boxes)):
        observation = {
            "id": f"observation-{index}", "label": label, "confidence": confidence, "box": item,
            "detection_frame_width": 160, "detection_frame_height": 100, "alert_eligible": alert,
            "incident_eligible": alert, "captured_at_epoch": 1000 + index, "scene_track_key": track,
        }
        if zone_names:
            observation["zones"] = zone_names
        samples.append({
            "id": f"sample-{index}", "camera_id": "gate", "captured_epoch": 1000 + index, "status": "complete",
            "observations": [observation], "metadata": {"frame_width": 160, "frame_height": 100, "activity_witnesses": []},
        })
    if len(samples) > 1:
        samples[-1]["metadata"]["activity_witnesses"] = [{
            "kind": "localized_movement", "validated": True, "from_sample_id": samples[-2]["id"],
            "to_sample_id": samples[-1]["id"], "camera_stable": True, "local_change_fraction": 0.2,
            "background_change_fraction": 0.01, "normalized_displacement": 0.05,
            "observation_ids": [samples[-1]["observations"][0]["id"]],
        }]
    return samples


def restricting():
    return policy(zone("NoMotion", "ignore", 0, 0.5), zone("Yard", "incident", 0.5, 1))


def test_ignore_only_movement_is_not_establishment():
    decision = evaluate_scene_establishment(physical(["car", "car"], [box(20, 40), box(40, 40)]), policy=restricting())
    assert decision["status"] == "unsupported"
    assert decision["reason"] == "ignored_zone"
    assert decision["activity_epoch"] is None
    assert decision["physical_evidence"]["status"] == "supported"
    assert decision["zone_interpretation"]["establishment_eligible"] is False


def test_overlap_with_an_eligible_zone_keeps_ignore_precedence():
    overlapped = policy(zone("NoMotion", "ignore", 0, 1), zone("Yard", "incident", 0, 1))
    decision = evaluate_scene_establishment(physical(["car", "car"], [box(20, 80), box(40, 80)]), policy=overlapped)
    assert decision["reason"] == "ignored_zone"
    assert decision["zone_interpretation"]["observations"][-1]["zones"] == ["NoMotion", "Yard"]


def test_notifications_and_confidence_are_not_establishment_filters():
    zones = policy(zone("Road", "incident", 0.5, 1, notifications=False), zone("NoMotion", "ignore", 0, 0.5))
    decision = evaluate_scene_establishment(
        physical(["car", "car"], [box(20, 80), box(40, 80)], confidence=0.2, alert=False), policy=zones)
    assert decision["status"] == "supported"
    assert decision["reason"] == "eligible_zone_activity"
    assert "notifications_enabled" not in json.dumps(decision["zone_interpretation"])


def test_class_rule_and_full_frame_and_required_zone():
    person_ignore = policy(zone("People", "ignore", 0, 1, classes=["person"]), require=False)
    assert evaluate_scene_establishment(physical(["car", "car"], [box(20, 80), box(40, 80)]), policy=person_ignore)["status"] == "supported"
    assert evaluate_scene_establishment(physical(["person", "person"], [box(20, 80), box(40, 80)]), policy=person_ignore)["reason"] == "ignored_zone"
    full = policy(zone("Corner", "incident", 0, 1, x0=0.9), require=False)
    assert evaluate_scene_establishment(physical(["car", "car"], [box(20, 80), box(40, 80)]), policy=full)["zone_interpretation"]["reason"] == "full_frame"
    required = policy(zone("Corner", "incident", 0, 1, x0=0.9), require=True)
    outside = evaluate_scene_establishment(physical(["car", "car"], [box(20, 80), box(40, 80)]), policy=required)
    assert outside["reason"] == "outside_incident_zone"


def test_depth_ignore_does_not_make_depth_an_incident_zone_filter():
    ignored = interpret_establishment_observation({
        "id": "car", "label": "car", "box": box(20, 80), "depth_stats": {"median_m": 4},
    }, policy(zone("Far", "ignore", 0, 1, minimum=2, maximum=8)), frame_width=160, frame_height=100)
    assert ignored["reason"] == "depth_ignore_zone"
    assert ignored["establishment_eligible"] is False
    admitted = interpret_establishment_observation({
        "id": "car", "label": "car", "box": box(20, 80), "depth_stats": {"median_m": 20},
    }, policy(zone("Near", "incident", 0, 1, minimum=2, maximum=8)), frame_width=160, frame_height=100)
    assert admitted["establishment_eligible"] is True


def test_mixed_ignore_and_outside_witnesses_are_not_described_as_ignore_only():
    zones = policy(zone("NoMotion", "ignore", 0, 0.4), zone("Yard", "incident", 0.7, 1))
    ignored = physical(["car", "car"], [box(20, 30), box(40, 30)])
    outside = physical(["truck", "truck"], [box(20, 50), box(40, 55)], track="truck")
    outside[0]["id"] = "sample-out-0"
    outside[0]["captured_epoch"] = 1010
    outside[0]["observations"][0]["id"] = "observation-out-0"
    outside[0]["observations"][0]["captured_at_epoch"] = 1010
    outside[1]["id"] = "sample-out-1"
    outside[1]["captured_epoch"] = 1011
    outside[1]["observations"][0]["id"] = "observation-out-1"
    outside[1]["observations"][0]["captured_at_epoch"] = 1011
    outside[1]["metadata"]["activity_witnesses"][0]["from_sample_id"] = "sample-out-0"
    outside[1]["metadata"]["activity_witnesses"][0]["to_sample_id"] = "sample-out-1"
    outside[1]["metadata"]["activity_witnesses"][0]["observation_ids"] = ["observation-out-1"]
    decision = evaluate_scene_establishment([*ignored, *outside], policy=zones)
    assert decision["physical_evidence"]["status"] == "supported"
    assert decision["status"] == "unsupported"
    assert decision["reason"] == "ineligible_zone"
    assert "only in an ignored zone" not in decision["summary"]


def test_discovery_confirmation_follows_establishment_eligibility():
    zones = restricting()
    ignored = {"label": "car", "box": box(20, 40)}
    eligible = {"label": "person", "box": box(20, 80)}
    assert discovery_requires_confirmation([ignored], zones, frame_width=160, frame_height=100) is False
    assert discovery_requires_confirmation([ignored, eligible], zones, frame_width=160, frame_height=100) is True
    assert discovery_requires_confirmation([ignored], None, frame_width=160, frame_height=100) is True
    assert discovery_requires_confirmation([ignored], zones, frame_width=0, frame_height=0) is True


def test_ignore_only_discovery_stays_in_the_ledger_without_confirmation(tmp_path):
    store = EventStore(tmp_path)
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    ignored = {
        "label": "car", "confidence": 0.8, "box": box(20, 40),
        "detection_frame_width": 160, "detection_frame_height": 100,
    }
    eligible = {**ignored, "label": "person", "box": box(20, 80)}
    result = SimpleNamespace(frame_captured_at_epoch=1000, frame_source="live_discovery")
    acquire_detection_result(
        store, "gate", datetime.fromtimestamp(1000, timezone.utc),
        {"scene_discovery": True, "establishment_zone_policy": restricting()},
        frame, [ignored, {"status": "scene_observations", "observations": [ignored]}],
        result, lambda *_args: "",
    )
    with store._connect() as conn:
        assert conn.execute("select count(*) from acquired_observations").fetchone()[0] == 1
        assert conn.execute("select count(*) from scene_candidate_jobs").fetchone()[0] == 0
    acquire_detection_result(
        store, "gate", datetime.fromtimestamp(1010, timezone.utc),
        {"scene_discovery": True, "establishment_zone_policy": restricting()},
        frame, [eligible, {"status": "scene_observations", "observations": [eligible]}],
        SimpleNamespace(frame_captured_at_epoch=1010, frame_source="live_discovery"),
        lambda *_args: "",
    )
    with store._connect() as conn:
        assert conn.execute("select count(*) from acquired_observations").fetchone()[0] == 2
        assert conn.execute("select count(*) from scene_candidate_jobs").fetchone()[0] == 1


def test_unlocalized_notice_and_measured_motion_cannot_bypass_ignore():
    zones = restricting()
    notice = evaluate_scene_establishment([
        {"id": "bare", "camera_id": "gate", "captured_epoch": 1000, "status": "complete", "observations": [], "metadata": {}},
    ], policy=zones, notice={"source": "camera", "epoch": 1000})
    assert notice["status"] == "unsupported"
    assert notice["reason"] == "insufficient_spatial_evidence"
    measured = evaluate_scene_establishment([
        {"id": "pixels", "camera_id": "gate", "captured_epoch": 1000, "status": "complete", "observations": [],
         "metadata": {"validated_motion": True, "motion_source": "measured_pixels"}},
    ], policy=zones)
    assert measured["status"] == "unsupported"
    assert measured["reason"] == "insufficient_spatial_evidence"
    assert measured["physical_evidence"]["status"] == "supported"
    verified = evaluate_scene_establishment(physical(["car"], [box(20, 80)], alert=False), policy=zones, notice={"source": "motion", "epoch": 1000})
    assert verified["status"] == "supported"
    assert verified["reason"] == "verified_camera_notice"


def _notice_samples(confidences, *, label="car", y2=80, median=None, track="subject"):
    observations = []
    for index, confidence in enumerate(confidences):
        observation = {
            "id": f"observation-{index}", "label": label, "confidence": confidence, "box": box(20 + index * 30, y2),
            "detection_frame_width": 160, "detection_frame_height": 100, "captured_at_epoch": 1000 + index,
            "scene_track_key": track,
        }
        if median is not None:
            observation["semantic_median_confidence"] = median
        observations.append(observation)
    return [{
        "id": "sample-0", "camera_id": "gate", "captured_epoch": 1000, "status": "complete",
        "observations": observations, "metadata": {},
    }]


def test_detector_confidence_gates_establishment_without_hiding_the_zone():
    road = policy(zone("Road", "incident", 0, 1), confidence_threshold=0.7)
    notice = {"source": "camera", "epoch": 1000}
    weak = evaluate_scene_establishment(_notice_samples([0.26, 0.43]), policy=road, notice=notice)
    assert weak["status"] == "unsupported"
    assert weak["reason"] == "below_confidence"
    assert weak["supporting_observation_ids"] == []
    assert all(item["establishment_eligible"] for item in weak["zone_interpretation"]["observations"])
    assert all(item["reason"] == "incident_zone" for item in weak["zone_interpretation"]["observations"])
    assert discovery_requires_confirmation(
        [{"label": "car", "confidence": 0.26, "box": box(20, 80)}], road, frame_width=160, frame_height=100,
    ) is True
    same_track = evaluate_scene_establishment(_notice_samples([0.26, 0.43], median=0.3465), policy=road, notice=notice)
    assert same_track["reason"] == "below_confidence"
    admitted = evaluate_scene_establishment(_notice_samples([0.8]), policy=road, notice=notice)
    assert admitted["status"] == "supported"
    assert admitted["reason"] == "verified_camera_notice"
    ignored = evaluate_scene_establishment(
        _notice_samples([0.8], y2=40),
        policy=policy(zone("NoMotion", "ignore", 0, 0.5), zone("Road", "incident", 0.5, 1), confidence_threshold=0.7),
        notice=notice,
    )
    assert ignored["status"] == "unsupported"
    assert ignored["reason"] == "ignored_zone"
    median = evaluate_scene_establishment(_notice_samples([0.62, 0.75, 0.71], median=0.71), policy=road, notice=notice)
    assert median["status"] == "supported"
    assert median["reason"] == "verified_camera_notice"
    movement = evaluate_scene_establishment(
        physical(["car", "car"], [box(20, 80), box(40, 80)], confidence=0.26), policy=road,
    )
    assert movement["status"] == "unsupported"
    assert movement["reason"] == "below_confidence"
    mower = evaluate_scene_establishment(
        _notice_samples([0.75], label="robot_lawnmower"),
        policy=policy(zone("Road", "incident", 0, 1), confidence_threshold=0.7, class_confidence_thresholds={"robot_lawnmower": 0.8}),
        notice=notice,
    )
    assert mower["reason"] == "below_confidence"


def test_policy_snapshot_records_the_detector_threshold_without_alert_settings():
    camera = CameraConfig(
        id="gate", name="Gate", stream_url="rtsp://example.invalid/main",
        zones=[DetectionZone(
            name="Road", behavior="incident", confidence_threshold=0.2, notifications_enabled=False,
            points=[{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}],
        )],
    )
    stamped = establishment_zone_policy(
        camera, confidence_threshold=0.7, class_confidence_thresholds={"Robot_Lawnmower": 0.8},
    )
    assert stamped["confidence_threshold"] == 0.7
    assert stamped["class_confidence_thresholds"] == {"robot_lawnmower": 0.8}
    assert "notifications_enabled" not in json.dumps(stamped)
    assert "confidence_threshold" not in stamped["zones"][0]
    assert "confidence_threshold" not in establishment_zone_policy(camera)


def commit(store, camera, at, frames, zones, *, label="person", track="subject", alert=False, named=None, confirmation=True, confidence=0.3, include_policy=True):
    observed = []
    recorded = []
    for index, boxes in enumerate(frames):
        items = []
        for item_index, (item_label, item_box) in enumerate(boxes):
            item = {
                "label": item_label, "confidence": confidence, "alert_eligible": alert, "incident_eligible": alert,
                "box": item_box, "detection_frame_width": 160, "detection_frame_height": 100,
                "scene_track_key": track if item_index == 0 else f"{track}-{item_index}",
                "captured_at_epoch": at + index, "frame_source": "recorded_main",
            }
            if named and item_index == 0:
                item["zones"] = named
            items.append(item)
            observed.append(item)
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        for item in items:
            bounds = item["box"]
            frame[int(bounds["y1"]):int(bounds["y2"]), int(bounds["x1"]):int(bounds["x2"])] = 220
        recorded.append(SimpleNamespace(offset=at + index, frame=frame, objects=items, recording_path="main.mp4", exact_timestamp=True))
    samples = scene_sample_records(recorded, observed, 0, camera, source="recorded_main")
    qualification = {"trigger_source": "camera"}
    if include_policy:
        qualification["establishment_zone_policy"] = zones
    if confirmation:
        qualification["scene_confirmation"] = True
    event = store.add_event(
        camera, "motion", created_at=datetime.fromtimestamp(at, timezone.utc).isoformat(),
        objects_json=json.dumps([observed[-1], {"status": "scene_observations", "observations": observed, "samples": samples},
                                 {"status": "motion_qualification", "motion_qualification": qualification}]),
    )
    return event, samples


def test_ignore_only_physical_movement_is_retained_without_an_incident(tmp_path):
    store = EventStore(tmp_path)
    zones = restricting()
    event, _ = commit(store, "gate", 1000, [[("car", box(20, 40))], [("car", box(40, 40))]], zones, label="car", track="car")
    assert store.scene_incident(event_id=event["id"]) is None
    with store._connect() as conn:
        assert conn.execute("select count(*) from acquired_observations").fetchone()[0] >= 2
        assert conn.execute("select count(*) from scene_incidents").fetchone()[0] == 0
        decision = conn.execute("select reason,verdict from scene_activity_decisions").fetchone()
    assert decision["verdict"] == "unsupported" and decision["reason"] == "ignored_zone"
    restarted = EventStore(tmp_path)
    with restarted._connect() as conn:
        assert conn.execute("select count(*) from scene_incidents").fetchone()[0] == 0
        assert conn.execute("select count(*) from acquired_observations").fetchone()[0] >= 2


def test_eligible_movement_contains_ignored_uncertain_and_stationary_objects(tmp_path):
    store = EventStore(tmp_path)
    zones = restricting()
    frames = [
        [("person", box(20, 80)), ("car", box(20, 40))],
        [("person", box(50, 80)), ("car", box(20, 40))],
    ]
    event, samples = commit(store, "gate", 1000, frames, zones, confidence=0.2, alert=False)
    incident = store.scene_incident(event_id=event["id"])
    assert incident["establishment"]["status"] == "established"
    assert incident["establishment"]["reason"] == "eligible_zone_activity"
    assert {item["label"] for item in incident["scene_objects"]} == {"person", "car"}
    assert not any(decision["eligible"] for item in incident["alert_decisions"] for decision in item["objects"])
    assert incident["establishment"]["zone_interpretation"]["ignored_observation_ids"]
    before = [tuple(row) for row in store._connect().execute("select id,captured_epoch,payload_json from acquired_observations order by id")]
    store.update_object_tracking(event["id"], {"scene_observations": [], "scene_samples": samples, "state": "complete"})
    with store._connect() as conn:
        assert [tuple(row) for row in conn.execute("select id,captured_epoch,payload_json from acquired_observations order by id")] == before
        assert conn.execute("select count(*) from scene_incidents").fetchone()[0] == 1
        assert conn.execute("select count(distinct object_id) from scene_observations").fetchone()[0] == 2


def test_ignored_movement_does_not_prolong_past_the_inactivity_grace(tmp_path):
    store = EventStore(tmp_path)
    zones = restricting()
    first, _ = commit(store, "gate", 1000, [[("person", box(20, 80))], [("person", box(50, 80))]], zones)
    incident = store.scene_incident(event_id=first["id"])
    activity = datetime.fromisoformat(incident["episodes"][0]["last_activity_at"]).timestamp()
    ignored, _ = commit(store, "gate", activity + 20, [[("car", box(20, 40))], [("car", box(50, 40))]], zones, label="car", track="car")
    during = store.scene_incident(event_id=first["id"])
    assert datetime.fromisoformat(during["episodes"][0]["last_activity_at"]).timestamp() == pytest.approx(activity)
    assert "car" in {item["label"] for item in during["scene_objects"]}
    assert store.scene_incident(event_id=ignored["id"]) is None
    store.settle_scene_incidents(activity + DEFAULT_INCIDENT_GAP_SECONDS + 1)
    assert store.scene_incident(event_id=first["id"])["state"] == "complete"
    late, _ = commit(store, "gate", activity + DEFAULT_INCIDENT_GAP_SECONDS + 30, [[("truck", box(20, 40))], [("truck", box(50, 40))]], zones, label="truck", track="truck")
    settled = store.scene_incident(event_id=first["id"])
    assert settled["state"] == "complete"
    assert "truck" not in {item["label"] for item in settled["scene_objects"]}
    assert store.scene_incident(event_id=late["id"]) is None
    with store._connect() as conn:
        assert conn.execute("select count(*) from acquired_observations where payload_json like '%truck%'").fetchone()[0] >= 2


def test_track_crossing_from_ignore_to_eligible_keeps_one_identity(tmp_path):
    store = EventStore(tmp_path)
    event, _ = commit(
        store, "gate", 1000,
        [[("person", box(20, 40))], [("person", box(20, 80))]],
        restricting(), track="walker",
    )
    incident = store.scene_incident(event_id=event["id"])
    assert len(incident["scene_objects"]) == 1
    assert len(incident["scene_objects"][0]["observations"]) == 2
    corrected = store.correct_scene_incident(incident["id"], incident["revision"], {
        "operation": "label", "object_id": incident["scene_objects"][0]["id"], "label": "resident",
    })
    assert corrected["scene_objects"][0]["label"] == "resident"
    assert len(store.scene_incident(event_id=event["id"])["scene_objects"][0]["observations"]) == 2


def test_disabled_zone_notifications_still_establish_and_do_not_notify(tmp_path):
    store = EventStore(tmp_path)
    zones = policy(zone("Road", "incident", 0.5, 1, notifications=False))
    event, _ = commit(store, "gate", 1000, [[("car", box(20, 80))], [("car", box(50, 80))]], zones, label="car", track="car", alert=True, named=["Road"], confidence=0.8)
    incident = store.scene_incident(event_id=event["id"])
    assert incident["establishment"]["status"] == "established"
    manager = manager_with_mocks()
    manager.config.cameras[0].zones = [DetectionZone(name="Road", behavior="incident", notifications_enabled=False, points=[
        {"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}, {"x": 1, "y": 1}, {"x": 0, "y": 1},
    ])]
    manager.config.integration_notifications.enabled = True
    manager.config.mqtt.enabled = True
    manager.config.mqtt.incident_events_enabled = True
    payload = {
        "schema_version": 3, "historical": False, "camera_id": "gate", "incident_id": incident["id"],
        "events": [{"id": event["id"], "camera_id": "gate"}],
        "alert_decisions": incident["alert_decisions"],
    }
    assert manager.incident_notification_allowed(payload) is False
    manager._publish_incident_notification(payload)
    assert manager.mqtt.publish.called is False


def test_cross_camera_identity_requires_eligible_activity_on_the_admitted_camera(tmp_path):
    store = EventStore(tmp_path)
    eligible = policy(zone("Yard", "incident", 0.5, 1))
    ignored = policy(zone("NoMotion", "ignore", 0, 1))
    first, _ = commit(store, "gate", 1000, [[("person", box(20, 80))], [("person", box(50, 80))]], eligible)
    second, _ = commit(store, "yard", 1010, [[("person", box(20, 40))], [("person", box(50, 40))]], ignored)
    identity = [{"identity_id": 7, "status": "confirmed", "name": "Resident"}]
    store.update_scene_identities(first["id"], identity)
    store.update_scene_identities(second["id"], identity)
    assert store.scene_incident(event_id=second["id"]) is None
    assert store.scene_incident(event_id=first["id"])["camera_ids"] == ["gate"]
    third, _ = commit(store, "yard", 1020, [[("person", box(20, 80))], [("person", box(50, 80))]], eligible, track="other")
    store.update_scene_identities(third["id"], identity)
    assert set(store.scene_incident(event_id=first["id"])["camera_ids"]) == {"gate", "yard"}


def test_camera_notice_without_a_box_does_not_establish_when_zones_restrict(tmp_path):
    store = EventStore(tmp_path)
    event = store.add_event("gate", "motion", created_at=datetime.fromtimestamp(1000, timezone.utc).isoformat(), objects_json=json.dumps([
        {"status": "motion_qualification", "motion_qualification": {"trigger_source": "camera", "establishment_zone_policy": restricting()}},
    ]))
    assert store.scene_incident(event_id=event["id"]) is None
    with store._connect() as conn:
        decision = conn.execute("select reason from scene_activity_decisions").fetchone()
        assert decision["reason"] == "insufficient_spatial_evidence"
        assert conn.execute("select count(*) from acquired_samples").fetchone()[0] == 1


def recorded_frames(camera, at, frames, *, track="subject"):
    observed = []
    recorded = []
    for index, boxes in enumerate(frames):
        items = []
        for item_index, (item_label, item_box) in enumerate(boxes):
            item = {
                "label": item_label, "confidence": 0.8, "alert_eligible": False, "incident_eligible": False,
                "box": item_box, "detection_frame_width": 160, "detection_frame_height": 100,
                "scene_track_key": track if item_index == 0 else f"{track}-{item_index}",
                "captured_at_epoch": at + index, "frame_source": "recorded_main",
            }
            items.append(item)
            observed.append(item)
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        for item in items:
            bounds = item["box"]
            frame[int(bounds["y1"]):int(bounds["y2"]), int(bounds["x1"]):int(bounds["x2"])] = 220
        recorded.append(SimpleNamespace(offset=at + index, frame=frame, objects=items, recording_path="main.mp4", exact_timestamp=True))
    return scene_sample_records(recorded, observed, 0, camera, source="recorded_main")


def test_tracking_fills_a_missing_zone_snapshot_without_prolonging_ignore_activity(tmp_path):
    store = EventStore(tmp_path)
    zones = restricting()
    event, _ = commit(
        store, "gate", 1000, [[("car", box(20, 40))], [("car", box(40, 40))]], zones,
        label="car", track="car", include_policy=False,
    )
    incident = store.scene_incident(event_id=event["id"])
    assert incident["establishment"]["reason"] == "video_verified_activity"
    activity = datetime.fromisoformat(incident["episodes"][0]["last_activity_at"]).timestamp()
    samples = recorded_frames("gate", activity + 20, [[("car", box(20, 40))], [("car", box(50, 40))]], track="later")
    assert evaluate_scene_activity(samples)["status"] == "supported"
    assert evaluate_scene_establishment(samples, policy=zones)["status"] == "unsupported"
    observations = [item for sample in samples for item in sample["observations"]]
    payload = {
        "scene_observations": observations, "scene_samples": samples, "state": "complete",
        "frame_width": 160, "frame_height": 100, "establishment_zone_policy": zones,
    }
    store.update_object_tracking(event["id"], payload)
    during = store.scene_incident(event_id=event["id"])
    assert datetime.fromisoformat(during["episodes"][0]["last_activity_at"]).timestamp() == pytest.approx(activity)
    assert during["establishment"]["reason"] == "video_verified_activity"
    with store._connect() as conn:
        stored = json.loads(conn.execute("select objects_json from events where id=?", (event["id"],)).fetchone()[0])
        tracking = next(item["object_tracking"] for item in stored if item.get("status") == "object_tracking")
        qualification = next(item["motion_qualification"] for item in stored if item.get("status") == "motion_qualification")
        retained = conn.execute("select count(*) from acquired_observations").fetchone()[0]
    assert "establishment_zone_policy" not in tracking
    assert qualification["establishment_zone_policy"]["zones"][0]["name"] == "NoMotion"
    assert retained >= 4
    store.update_object_tracking(event["id"], payload)
    with store._connect() as conn:
        assert conn.execute("select count(*) from acquired_observations").fetchone()[0] == retained
        assert conn.execute("select count(*) from scene_incidents").fetchone()[0] == 1
    store.settle_scene_incidents(activity + DEFAULT_INCIDENT_GAP_SECONDS + 1)
    assert store.scene_incident(event_id=event["id"])["state"] == "complete"


def test_tracking_does_not_replace_an_existing_zone_snapshot(tmp_path):
    store = EventStore(tmp_path)
    zones = restricting()
    event, _ = commit(store, "gate", 1000, [[("person", box(20, 80))], [("person", box(50, 80))]], zones)
    incident = store.scene_incident(event_id=event["id"])
    activity = datetime.fromisoformat(incident["episodes"][0]["last_activity_at"]).timestamp()
    samples = recorded_frames("gate", activity + 20, [[("car", box(20, 40))], [("car", box(50, 40))]], track="ignored")
    assert evaluate_scene_activity(samples)["status"] == "supported"
    assert evaluate_scene_establishment(samples, policy=policy(require=False))["status"] == "supported"
    store.update_object_tracking(event["id"], {
        "scene_observations": [item for sample in samples for item in sample["observations"]],
        "scene_samples": samples, "state": "complete",
        "frame_width": 160, "frame_height": 100,
        "establishment_zone_policy": policy(require=False),
    })
    during = store.scene_incident(event_id=event["id"])
    assert datetime.fromisoformat(during["episodes"][0]["last_activity_at"]).timestamp() == pytest.approx(activity)
    with store._connect() as conn:
        stored = json.loads(conn.execute("select objects_json from events where id=?", (event["id"],)).fetchone()[0])
    qualification = next(item["motion_qualification"] for item in stored if item.get("status") == "motion_qualification")
    assert qualification["establishment_zone_policy"]["zones"][0]["name"] == "NoMotion"
    assert "establishment_zone_policy" not in next(item["object_tracking"] for item in stored if item.get("status") == "object_tracking")
