"""Stationary subjects are one durable fact shared by admission, establishment, and alerts."""
from survng.app.config import CameraConfig, DetectorConfig
from survng.app.events import EventStore
from survng.app.object_activity import (
    ObjectActivityAttributor,
    apply_stationary_alert,
    stationary_subject_effect,
)
from survng.app.scene_activity import evaluate_scene_activity
from survng.app.scene_zone_admission import (
    establishment_zone_policy,
    evaluate_scene_establishment,
)
from tests.test_object_activity import observation
from tests.test_scene_zone_admission import box, physical, restricting


def _stable(attributor, event_key, epoch):
    return attributor.admit(
        [observation()],
        {},
        event_key=event_key,
        observed_at_epoch=epoch,
    )[0]


def test_scene_context_memory_survives_a_new_attributor(tmp_path):
    store = EventStore(tmp_path)
    memory = store.scene_context_memory("gate")
    first = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    _stable(first, "one", 1000.0)
    _stable(first, "two", 1060.0)
    store._scene_context_memories.clear()
    restarted = ObjectActivityAttributor(
        "enforce",
        memory=store.scene_context_memory("gate"),
        camera_id="gate",
    )

    third = _stable(restarted, "three", 1120.0)

    assert third.attribution.role.value == "scene_context"
    assert restarted.status()["scene_context_memory_entries"] == 1
    assert store.scene_context_status() == {"gate": 1}


def test_ledger_writes_only_when_the_sighting_changes(tmp_path):
    store = EventStore(tmp_path)
    memory = store.scene_context_memory("gate")
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    _stable(attributor, "one", 1000.0)
    _stable(attributor, "two", 1060.0)
    written = memory.writes
    _stable(attributor, "two", 1061.0)

    assert memory.writes == written
    assert len(store.load_scene_context("gate")[0]["sightings"]) == 2


def test_expired_subjects_are_pruned_and_movement_invalidates_the_location(tmp_path):
    store = EventStore(tmp_path)
    memory = store.scene_context_memory("gate")
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    _stable(attributor, "one", 1000.0)
    _stable(attributor, "two", 1000.0 + 3 * 60 * 60)

    assert store.load_scene_context("gate")[0]["sightings"] == ["two"]

    moved = attributor.admit(
        [observation(temporal_center_displacement_ratio=0.05, temporal_center_path_ratio=0.08)],
        {},
        event_key="moved",
        observed_at_epoch=1000.0 + 3 * 60 * 60 + 1,
    )[0]
    assert moved.attribution.role.value == "active"
    assert store.load_scene_context("gate") == []


def test_turning_attribution_off_clears_the_camera_ledger(tmp_path):
    store = EventStore(tmp_path)
    attributor = ObjectActivityAttributor(
        "enforce",
        memory=store.scene_context_memory("gate"),
        camera_id="gate",
    )
    _stable(attributor, "one", 1000.0)
    attributor.reconfigure("off")
    assert store.load_scene_context("gate") == []


def _parked_notice(role="scene_context"):
    samples = physical(["car"], [box(20, 80)])
    samples[0]["observations"][0]["activity_role"] = role
    return samples


def _policy(**values):
    payload = restricting()
    payload.update(values)
    return payload


def test_known_stationary_subject_does_not_verify_a_notice():
    decision = evaluate_scene_establishment(
        _parked_notice(),
        policy=_policy(stationary_subject_presence="ignore", object_activity_attribution="enforce"),
        notice={"source": "motion", "epoch": 1000},
    )
    assert decision["status"] == "unsupported"
    assert decision["reason"] == "stationary_scene_context"
    assert "already known" in decision["summary"]
    assert decision["policy_version"] == "establishment_zones_v2"


def test_presence_activity_shadow_and_off_still_verify_the_notice():
    for attribution, presence, counterfactual in (
        ("enforce", "activity", False),
        ("shadow", "ignore", True),
        ("off", "ignore", False),
    ):
        decision = evaluate_scene_establishment(
            _parked_notice(),
            policy=_policy(stationary_subject_presence=presence, object_activity_attribution=attribution),
            notice={"source": "motion", "epoch": 1000},
        )
        assert decision["status"] == "supported", (attribution, presence)
        assert decision["reason"] == "verified_camera_notice"
        assert bool(decision["diagnostics"].get("stationary_scene_context_counterfactual")) is counterfactual


def test_snapshot_without_a_presence_policy_keeps_the_recorded_meaning():
    decision = evaluate_scene_establishment(
        _parked_notice(),
        policy=restricting(),
        notice={"source": "motion", "epoch": 1000},
    )
    assert decision["status"] == "supported"
    assert decision["reason"] == "verified_camera_notice"
    assert decision["policy_version"] == "establishment_zones_v1"


def test_inplace_motion_on_a_stationary_subject_is_not_activity():
    samples = physical(["car", "car"], [box(20, 80), box(20, 80)])
    witness = samples[-1]["metadata"]["activity_witnesses"][0]
    witness["kind"] = "localized_motion"
    witness["normalized_displacement"] = 0.0004
    for sample in samples:
        sample["observations"][0]["activity_role"] = "scene_context"
        sample["metadata"]["confirmation_complete"] = True
    decision = evaluate_scene_activity(samples, ignore_stationary_scene_context=True)
    assert decision["status"] == "unsupported"
    established = evaluate_scene_establishment(
        samples,
        policy=_policy(stationary_subject_presence="ignore", object_activity_attribution="enforce"),
    )
    assert established["status"] != "supported"


def test_measured_movement_still_establishes_and_forgets_the_parked_subject(tmp_path):
    store = EventStore(tmp_path)
    attributor = ObjectActivityAttributor(
        "enforce",
        memory=store.scene_context_memory("gate"),
        camera_id="gate",
    )
    _stable(attributor, "one", 1000.0)
    _stable(attributor, "two", 1060.0)
    moving = observation(
        temporal_center_displacement_ratio=0.05,
        temporal_center_path_ratio=0.08,
    )
    attributor.stamp_recorded([moving], event_key="moved", observed_at_epoch=1120.0)
    assert moving["activity_role"] == "active"
    assert store.load_scene_context("gate") == []

    samples = physical(["car", "car"], [box(20, 80), box(22, 80)])
    samples[-1]["observations"][0]["activity_role"] = "active"
    decision = evaluate_scene_establishment(
        samples,
        policy=_policy(stationary_subject_presence="ignore", object_activity_attribution="enforce"),
    )
    assert decision["status"] == "supported"
    assert decision["reason"] == "eligible_zone_activity"


def test_later_frames_of_one_event_stay_scene_context():
    attributor = ObjectActivityAttributor("enforce")
    _stable(attributor, "one", 1000.0)
    _stable(attributor, "two", 1060.0)
    frames = [observation(), observation()]

    attributor.stamp_recorded(frames, event_key="three", observed_at_epoch=1120.0)

    assert [frame["activity_role"] for frame in frames] == ["scene_context", "scene_context"]


def test_overlapping_ledger_rows_collapse_to_one_subject(tmp_path):
    store = EventStore(tmp_path)
    store.save_scene_context("gate", {
        "label": "car",
        "box": (0.1, 0.1, 0.3, 0.3),
        "first_seen_epoch": 1000.0,
        "last_seen_epoch": 1000.0,
        "sightings": ["one", "two"],
    })
    store.save_scene_context("gate", {
        "label": "car",
        "box": (0.11, 0.1, 0.3, 0.3),
        "first_seen_epoch": 1060.0,
        "last_seen_epoch": 1060.0,
        "sightings": ["three", "four"],
    })
    attributor = ObjectActivityAttributor(
        "enforce",
        memory=store.scene_context_memory("gate"),
        camera_id="gate",
    )
    stamped = observation()

    attributor.stamp_recorded([stamped], event_key="five", observed_at_epoch=1120.0)

    assert stamped["activity_role"] == "scene_context"
    rows = store.load_scene_context("gate")
    assert len(rows) == 1
    assert rows[0]["sightings"] == ["one", "two", "three", "four", "five"]


def _parked_box():
    return observation(
        box={"x1": 20, "y1": 60, "x2": 40, "y2": 80},
        detection_frame_width=160,
        detection_frame_height=100,
    )


def test_unstamped_box_on_a_known_subject_does_not_verify_a_notice(tmp_path):
    store = EventStore(tmp_path)
    attributor = ObjectActivityAttributor(
        "enforce",
        memory=store.scene_context_memory("gate"),
        camera_id="gate",
    )
    for key, epoch in (("one", 1000.0), ("two", 1060.0)):
        attributor.admit([_parked_box()], {}, event_key=key, observed_at_epoch=epoch)
    samples = physical(["car"], [box(20, 80)])

    decision = evaluate_scene_establishment(
        samples,
        policy=_policy(stationary_subject_presence="ignore", object_activity_attribution="enforce"),
        notice={"source": "motion", "epoch": 1120},
        scene_context_memory=store.scene_context_memory("gate"),
        event_key="three",
        observed_at_epoch=1120.0,
    )

    assert decision["status"] == "unsupported"
    assert decision["reason"] == "stationary_scene_context"
    assert samples[0]["observations"][0]["activity_role"] == "scene_context"


def test_an_explicit_indeterminate_role_still_verifies_the_notice(tmp_path):
    store = EventStore(tmp_path)
    attributor = ObjectActivityAttributor(
        "enforce",
        memory=store.scene_context_memory("gate"),
        camera_id="gate",
    )
    for key, epoch in (("one", 1000.0), ("two", 1060.0)):
        attributor.admit([_parked_box()], {}, event_key=key, observed_at_epoch=epoch)
    samples = physical(["car"], [box(20, 80)])
    samples[0]["observations"][0]["activity_role"] = "indeterminate"

    decision = evaluate_scene_establishment(
        samples,
        policy=_policy(stationary_subject_presence="ignore", object_activity_attribution="enforce"),
        notice={"source": "motion", "epoch": 1120},
        scene_context_memory=store.scene_context_memory("gate"),
        event_key="three",
        observed_at_epoch=1120.0,
    )

    assert decision["status"] == "supported"
    assert samples[0]["observations"][0]["activity_role"] == "indeterminate"


def test_a_new_arrival_fails_open():
    admission = ObjectActivityAttributor("enforce").admit([observation()], {}, event_key="new", observed_at_epoch=1000.0)[0]
    assert admission.attribution.role.value == "indeterminate"
    assert admission.admitted is True
    samples = physical(["car"], [box(20, 80)])
    decision = evaluate_scene_establishment(
        samples,
        policy=_policy(stationary_subject_presence="ignore", object_activity_attribution="enforce"),
        notice={"source": "camera", "epoch": 1000},
    )
    assert decision["status"] == "supported"


def test_enforced_stationary_subject_does_not_alert_and_presence_activity_does():
    parked = {"label": "car", "activity_role": "scene_context", "alert_eligible": True, "alert_reasons": []}
    apply_stationary_alert(parked, presence="ignore", mode="enforce")
    assert parked["alert_eligible"] is False
    assert parked["alert_reasons"] == ["stationary_scene_context"]

    allowed = {"label": "car", "activity_role": "scene_context", "alert_eligible": True, "alert_reasons": []}
    apply_stationary_alert(allowed, presence="activity", mode="enforce")
    assert allowed["alert_eligible"] is True

    shadow = {"label": "car", "activity_role": "scene_context", "alert_eligible": True, "alert_reasons": []}
    apply_stationary_alert(shadow, presence="ignore", mode="shadow")
    assert shadow["alert_eligible"] is True
    assert shadow["alert_shadow_reasons"] == ["stationary_scene_context"]
    assert stationary_subject_effect("indeterminate", "ignore", "enforce") == ""


def test_establishment_snapshot_carries_the_effective_presence_policy():
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main", stationary_subject_presence="inherit")
    detector = DetectorConfig(stationary_subject_presence="activity", object_activity_attribution="shadow")
    policy = establishment_zone_policy(camera, detector_config=detector)
    assert policy["stationary_subject_presence"] == "activity"
    assert policy["object_activity_attribution"] == "shadow"
    camera.stationary_subject_presence = "ignore"
    assert establishment_zone_policy(camera, detector_config=detector)["stationary_subject_presence"] == "ignore"
