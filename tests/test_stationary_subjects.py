"""Stationary subjects are one durable fact shared by admission, establishment, and alerts."""
import pytest

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


def _memory(kind, tmp_path):
    if kind == "durable":
        return EventStore(tmp_path).scene_context_memory("gate")
    from survng.app.scene_context_memory import InMemorySceneContextMemory

    return InMemorySceneContextMemory()


def _sighting(attributor, event_key, epoch, **values):
    return attributor.admit(
        [observation(**values)], {}, event_key=event_key, observed_at_epoch=epoch,
    )[0]


def _scaled(scale):
    """The default 200 px box at (100, 100)-(300, 300), scaled about its centre."""
    half = 100.0 * scale
    return {"x1": 200.0 - half, "y1": 200.0 - half, "x2": 200.0 + half, "y2": 200.0 + half}


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_opposite_box_errors_still_match_one_subject(kind, tmp_path):
    memory = _memory(kind, tmp_path)
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    _sighting(attributor, "one", 1000.0)
    # Each later box is within the match IoU of the true box, but the two
    # disagree with each other: one detection is 15% large, the next 12.5%
    # small.
    _sighting(attributor, "two", 1060.0, box=_scaled(1.15))
    third = _sighting(attributor, "three", 1120.0, box=_scaled(0.875))
    fourth = _sighting(attributor, "four", 1180.0)

    assert third.attribution.role.value == "scene_context"
    assert fourth.attribution.role.value == "scene_context"
    assert memory.count("gate") == 1


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_a_vehicle_flipping_between_car_and_truck_is_one_subject(kind, tmp_path):
    memory = _memory(kind, tmp_path)
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    labels = ["car", "truck", "car", "bus"]
    admissions = [
        _sighting(attributor, f"event-{index}", 1000.0 + index * 60.0, label=label)
        for index, label in enumerate(labels)
    ]

    assert [admission.admitted for admission in admissions] == [True, True, False, False]
    assert memory.count("gate") == 1
    if kind == "durable":
        assert memory._store.load_scene_context("gate")[0]["label"] == "car"


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_non_vehicle_labels_still_match_exactly(kind, tmp_path):
    memory = _memory(kind, tmp_path)
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    _sighting(attributor, "one", 1000.0, label="car")
    _sighting(attributor, "two", 1060.0, label="car")
    person = _sighting(attributor, "three", 1120.0, label="person")
    motorcycle = _sighting(attributor, "four", 1180.0, label="motorcycle")

    assert person.attribution.role.value == "indeterminate"
    assert motorcycle.attribution.role.value == "indeterminate"
    assert memory.count("gate") == 3


def test_movement_of_a_relabelled_vehicle_invalidates_its_location(tmp_path):
    store = EventStore(tmp_path)
    attributor = ObjectActivityAttributor(
        "enforce", memory=store.scene_context_memory("gate"), camera_id="gate",
    )
    _sighting(attributor, "one", 1000.0, label="car")
    _sighting(attributor, "two", 1060.0, label="car")

    moved = _sighting(
        attributor,
        "moved",
        1120.0,
        label="truck",
        temporal_center_displacement_ratio=0.05,
        temporal_center_path_ratio=0.08,
    )

    assert moved.attribution.role.value == "active"
    assert store.load_scene_context("gate") == []


def test_reference_box_moves_only_with_a_new_sighting(tmp_path):
    store = EventStore(tmp_path)
    memory = store.scene_context_memory("gate")
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    _sighting(attributor, "one", 1000.0)
    _sighting(attributor, "two", 1060.0, box=_scaled(1.1))
    written = memory.writes
    persisted = store.load_scene_context("gate")[0]

    # Later frames of the same event, with jittered boxes, change nothing.
    _sighting(attributor, "two", 1061.0, box=_scaled(0.9))
    _sighting(attributor, "two", 1062.0, box=_scaled(1.12))

    assert memory.writes == written
    assert store.load_scene_context("gate") == [persisted]
    # The reference is a damped blend, not the newest box.
    assert 0.09 < persisted["box_x1"] < 0.1


def test_establishment_snapshot_matches_a_relabelled_vehicle(tmp_path):
    from survng.app.scene_context_memory import SceneContextSnapshot, SceneContextSubject

    snapshot = SceneContextSnapshot([SceneContextSubject(
        label="car",
        box=(0.1, 0.1, 0.3, 0.3),
        last_seen_epoch=1000.0,
        first_seen_epoch=900.0,
        stable_event_keys=["one", "two"],
    )])

    def match(label):
        return snapshot.match(
            camera_id="gate", label=label, box=(0.1, 0.1, 0.3, 0.3), event_key="three",
            observed_at_epoch=1060.0, min_iou=0.72, ttl_seconds=7200.0,
        )[0]

    assert match("truck") is True
    assert match("bus") is True
    assert match("person") is False


def _parked_for_an_hour(attributor):
    """Four still sightings at 1000-1180 s, then a live one at 2005 s."""
    for index in range(4):
        _sighting(attributor, f"parked-{index}", 1000.0 + index * 60.0)
    return _sighting(attributor, "live", 2005.0)


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_recorded_analysis_that_finishes_after_a_later_event_still_knows_the_subject(kind, tmp_path):
    memory = _memory(kind, tmp_path)
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    live = _parked_for_an_hour(attributor)
    assert live.attribution.role.value == "scene_context"

    # The recording of an event at 2000 s is analysed after the live event at
    # 2005 s has refreshed the subject. The car was already parked at 2000 s.
    late = _sighting(attributor, "recorded", 2000.0)

    assert late.attribution.role.value == "scene_context"
    assert late.admitted is False
    assert late.attribution.evidence.scene_context_memory_sightings == 5
    assert memory.count("gate") == 1


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_a_late_sighting_never_moves_the_subject_back_in_time(kind, tmp_path):
    memory = _memory(kind, tmp_path)
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    _parked_for_an_hour(attributor)
    _sighting(attributor, "recorded", 2000.0)

    ttl = attributor.stationary_policy.scene_memory_ttl_seconds
    # Still inside the TTL of the 2005 s sighting, past that of 2000 s.
    after = _sighting(attributor, "after", 2005.0 + ttl - 1.0)

    assert after.attribution.role.value == "scene_context"
    assert after.attribution.evidence.scene_context_memory_sightings == 6
    assert memory.count("gate") == 1


def test_a_late_sighting_is_written_once_and_reloads_as_it_is_held(tmp_path):
    store = EventStore(tmp_path)
    memory = store.scene_context_memory("gate")
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    _parked_for_an_hour(attributor)
    written = memory.writes

    _sighting(attributor, "recorded", 2000.0)
    assert memory.writes == written + 1
    # Later frames of the late event change nothing.
    _sighting(attributor, "recorded", 2001.0)
    _sighting(attributor, "live", 2007.0)
    assert memory.writes == written + 1

    store._scene_context_memories.clear()
    reloaded = store.scene_context_memory("gate")
    assert [
        (s.label, s.box, s.first_seen_epoch, s.last_seen_epoch, s.stable_event_keys)
        for s in reloaded._subjects
    ] == [
        (s.label, s.box, s.first_seen_epoch, s.last_seen_epoch, s.stable_event_keys)
        for s in memory._subjects
    ]
    row = store.load_scene_context("gate")[0]
    assert (row["first_seen_epoch"], row["last_seen_epoch"]) == (1000.0, 2005.0)


def test_the_establishment_snapshot_also_knows_the_subject_for_a_late_event(tmp_path):
    store = EventStore(tmp_path)
    memory = store.scene_context_memory("gate")
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    _parked_for_an_hour(attributor)
    with store._connect() as conn:
        snapshot = store.scene_context_snapshot(conn, "gate")
    query = dict(
        camera_id="gate", label="car", box=(0.1, 0.1, 0.3, 0.3), event_key="recorded",
        observed_at_epoch=2000.0, min_iou=0.72, ttl_seconds=7200.0,
    )

    assert snapshot.match(**query) == memory.match(**query)
    assert snapshot.match(**query)[:2] == (True, 5)


@pytest.mark.parametrize("kind", ["memory", "durable", "snapshot"])
def test_a_car_that_parked_after_an_event_cannot_suppress_its_arrival(kind, tmp_path):
    memory = _memory("durable" if kind == "snapshot" else kind, tmp_path)
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    for index in range(4):
        _sighting(attributor, f"parked-{index}", 1060.0 + index * 60.0)

    if kind == "snapshot":
        store = memory._store
        with store._connect() as conn:
            snapshot = store.scene_context_snapshot(conn, "gate")
        matched = snapshot.match(
            camera_id="gate", label="car", box=(0.1, 0.1, 0.3, 0.3), event_key="arrival",
            observed_at_epoch=1000.0, min_iou=0.72, ttl_seconds=7200.0,
        )
        assert matched == (False, 0, None)
        return
    # The arrival at 1000 s is analysed last. Sightings that began after it
    # are not evidence that the car was already there.
    arrival = _sighting(attributor, "arrival", 1000.0)

    assert arrival.attribution.role.value == "indeterminate"
    assert arrival.admitted is True
    assert arrival.attribution.evidence.scene_context_memory_match is False


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_a_revoked_first_sighting_no_longer_dates_the_subject(kind, tmp_path):
    memory = _memory(kind, tmp_path)
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    for key, epoch in (("one", 1000.0), ("two", 1060.0), ("three", 1120.0)):
        _sighting(attributor, key, epoch)
    # Refinement shows the first event moved, below the bound that forgets
    # the location. Its sighting is revoked; the later two remain.
    refined = _sighting(
        attributor, "one", 1000.0,
        temporal_center_displacement_ratio=0.015, temporal_center_path_ratio=0.02,
    )
    assert refined.attribution.role.value == "active"
    assert memory.count("gate") == 1

    # Nothing that remains was seen before 1030 s, so the late event is new.
    late = _sighting(attributor, "late", 1030.0)

    assert late.attribution.role.value == "indeterminate"
    assert late.admitted is True
