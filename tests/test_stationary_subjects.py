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
    # The four sightings before it; the live one at 2005 s came after.
    assert late.attribution.evidence.scene_context_memory_sightings == 4
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
        (s.label, s.box, s.first_seen_epoch, s.last_seen_epoch, s.stable_event_keys, s.sighting_epochs)
        for s in reloaded._subjects
    ] == [
        (s.label, s.box, s.first_seen_epoch, s.last_seen_epoch, s.stable_event_keys, s.sighting_epochs)
        for s in memory._subjects
    ]
    row = store.load_scene_context("gate")[0]
    assert (row["first_seen_epoch"], row["last_seen_epoch"]) == (1000.0, 2005.0)
    assert row["sighting_epochs"]["recorded"] == 2000.0


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
    assert snapshot.match(**query)[:2] == (True, 4)


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


@pytest.mark.parametrize("kind", ["memory", "durable", "snapshot"])
def test_a_late_event_counts_only_sightings_from_before_it(kind, tmp_path):
    memory = _memory("durable" if kind == "snapshot" else kind, tmp_path)
    attributor = ObjectActivityAttributor("enforce", memory=memory, camera_id="gate")
    # One car was seen once at 1000 s and left unseen. Another parks in the
    # same bay at 2000 s, and a live event sees it at 2005 s.
    _sighting(attributor, "first-car", 1000.0)
    _sighting(attributor, "live", 2005.0)
    query = dict(
        camera_id="gate", label="car", box=(0.1, 0.1, 0.3, 0.3), event_key="arrival",
        observed_at_epoch=2000.0, min_iou=0.72, ttl_seconds=7200.0,
    )
    if kind == "snapshot":
        store = memory._store
        with store._connect() as conn:
            snapshot = store.scene_context_snapshot(conn, "gate")
        assert snapshot.match(**query)[:2] == (True, 1)
        return

    # The recording of 2000 s is analysed last. In order, it would have had
    # one prior sighting; the sighting at 2005 s is not evidence before it.
    arrival = _sighting(attributor, "arrival", 2000.0)

    assert arrival.attribution.evidence.scene_context_memory_sightings == 1
    assert arrival.attribution.role.value == "indeterminate"
    assert arrival.admitted is True
    assert memory.count("gate") == 1


def test_a_ledger_from_before_sighting_times_still_loads_and_matches(tmp_path):
    store = EventStore(tmp_path)
    with store._connect() as conn:
        conn.execute("drop table scene_context_subjects")
        conn.execute(
            """create table scene_context_subjects (
                id integer primary key, camera_id text not null, label text not null,
                box_x1 real not null, box_y1 real not null, box_x2 real not null,
                box_y2 real not null, first_seen_epoch real not null,
                last_seen_epoch real not null, sightings_json text not null,
                last_moved_epoch real)"""
        )
        conn.execute(
            "insert into scene_context_subjects(camera_id, label, box_x1, box_y1, box_x2, "
            "box_y2, first_seen_epoch, last_seen_epoch, sightings_json) "
            "values('gate','car',0.1,0.1,0.3,0.3,1000.0,1060.0,'[\"one\",\"two\"]')"
        )
    upgraded = EventStore(tmp_path)
    attributor = ObjectActivityAttributor(
        "enforce", memory=upgraded.scene_context_memory("gate"), camera_id="gate",
    )

    # Sightings without times date from the last one, so in-order events
    # count them as before, and a late event before it does not.
    assert _sighting(attributor, "three", 1120.0).attribution.role.value == "scene_context"
    late = _sighting(attributor, "late", 1030.0)
    assert late.attribution.role.value == "indeterminate"
    assert upgraded.load_scene_context("gate")[0]["sighting_epochs"] == {
        "three": 1120.0, "late": 1030.0,
    }


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_a_late_sighting_older_than_every_kept_one_changes_nothing(kind, tmp_path):
    memory = _memory(kind, tmp_path)
    common = dict(camera_id="gate", label="car", min_iou=0.72, ttl_seconds=7200.0,
                  max_entries=64, max_sightings=16)
    memory.remember(box=(0.1, 0.1, 0.3, 0.3), event_key="first", observed_at_epoch=1000.0, **common)
    for index in range(16):
        memory.remember(box=(0.1, 0.1, 0.3, 0.3), event_key=f"later-{index}",
                        observed_at_epoch=2000.0 + index, **common)
    subjects = memory._subjects
    subject = subjects[0][1] if kind == "memory" else subjects[0]
    held = (list(subject.stable_event_keys), subject.box, dict(subject.sighting_epochs))
    written = memory.writes

    # Every frame of a late event at 1500 s: it is older than all sixteen
    # kept sightings, so it would be trimmed at once.
    for _frame in range(4):
        memory.remember(box=(0.115, 0.1, 0.315, 0.3), event_key="late",
                        observed_at_epoch=1500.0, **common)

    assert (list(subject.stable_event_keys), subject.box, dict(subject.sighting_epochs)) == held
    assert memory.writes == written


def test_a_corrupt_sighting_time_is_ignored(tmp_path):
    store = EventStore(tmp_path)
    store.save_scene_context("gate", {
        "label": "car", "box": (0.1, 0.1, 0.3, 0.3), "first_seen_epoch": 1000.0,
        "last_seen_epoch": 1060.0, "sightings": ["one", "two"],
        "sighting_epochs": {"one": True, "two": 1060.0},
    })
    assert store.load_scene_context("gate")[0]["sighting_epochs"] == {"two": 1060.0}


def _full_set(memory):
    """Sixteen sightings at 2000-2015 s, the cap."""
    common = dict(camera_id="gate", label="car", min_iou=0.72, ttl_seconds=7200.0,
                  max_entries=64, max_sightings=16)
    for index in range(16):
        memory.remember(box=(0.1, 0.1, 0.3, 0.3), event_key=f"kept-{index}",
                        observed_at_epoch=2000.0 + index, **common)
    subjects = memory._subjects
    subject = subjects[0][1] if isinstance(subjects[0], tuple) else subjects[0]
    return subject, common


def _held(subject):
    return (list(subject.stable_event_keys), subject.box, dict(subject.sighting_epochs))


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_an_event_as_old_as_the_oldest_kept_sighting_does_not_evict_it(kind, tmp_path):
    memory = _memory(kind, tmp_path)
    subject, common = _full_set(memory)
    held = _held(subject)
    written = memory.writes

    # Frames of two events at the same time as the oldest kept sighting,
    # alternating. Neither may displace the other.
    for _frame in range(3):
        memory.remember(box=(0.112, 0.1, 0.312, 0.3), event_key="tie",
                        observed_at_epoch=2000.0, **common)
        memory.remember(box=(0.1, 0.1, 0.3, 0.3), event_key="kept-0",
                        observed_at_epoch=2000.0, **common)

    assert _held(subject) == held
    assert memory.writes == written


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_an_evicted_sighting_that_returns_is_not_re_dated(kind, tmp_path):
    memory = _memory(kind, tmp_path)
    subject, common = _full_set(memory)

    # A newer event evicts the oldest sighting, once.
    memory.remember(box=(0.1, 0.1, 0.3, 0.3), event_key="newer",
                    observed_at_epoch=3000.0, **common)
    assert "kept-0" not in subject.stable_event_keys
    held = _held(subject)
    written = memory.writes

    # Later frames of the evicted event, and then of the next one it would
    # evict, would otherwise date them as the newest and swap on each frame.
    for offset in range(3):
        memory.remember(box=(0.112, 0.1, 0.312, 0.3), event_key="kept-0",
                        observed_at_epoch=3001.0 + offset, **common)

    assert _held(subject) == held
    assert subject.sighting_epochs["kept-1"] == 2001.0
    assert memory.writes == written


@pytest.mark.parametrize(
    "raw",
    [
        '{"one": 1' + "0" * 400 + ', "two": 1060.0}',  # float() overflows
        '{"one": 1' + "0" * 5000 + "}",  # beyond the integer digit limit
        '{"one":' * 100000 + "1" + "}" * 100000,  # nesting beyond the recursion limit
    ],
    ids=["huge_integer", "huge_integer_digits", "deep_nesting"],
)
def test_a_corrupt_sighting_time_column_loads_without_times(tmp_path, raw):
    store = EventStore(tmp_path)
    store.save_scene_context("gate", {
        "label": "car", "box": (0.1, 0.1, 0.3, 0.3), "first_seen_epoch": 1000.0,
        "last_seen_epoch": 1060.0, "sightings": ["one", "two"],
        "sighting_epochs": {"one": 1000.0, "two": 1060.0},
    })
    with store._connect() as conn:
        conn.execute("update scene_context_subjects set sighting_epochs_json=?", (raw,))

    rows = store.load_scene_context("gate")
    memory = store.scene_context_memory("gate")
    with store._connect() as conn:
        snapshot = store.scene_context_snapshot(conn, "gate")

    assert rows[0]["sightings"] == ["one", "two"]
    assert rows[0]["sighting_epochs"] in ({}, {"two": 1060.0})
    assert memory.count("gate") == 1
    assert snapshot.match(
        camera_id="gate", label="car", box=(0.1, 0.1, 0.3, 0.3), event_key="three",
        observed_at_epoch=1120.0, min_iou=0.72, ttl_seconds=7200.0,
    )[:2] == (True, 2)


def test_a_corrupt_sightings_column_loads_as_no_sightings(tmp_path):
    store = EventStore(tmp_path)
    store.save_scene_context("gate", {
        "label": "car", "box": (0.1, 0.1, 0.3, 0.3), "first_seen_epoch": 1000.0,
        "last_seen_epoch": 1060.0, "sightings": ["one"], "sighting_epochs": {"one": 1000.0},
    })
    with store._connect() as conn:
        conn.execute("update scene_context_subjects set sightings_json=?",
                     ("[" * 100000 + "]" * 100000,))

    assert store.load_scene_context("gate")[0]["sightings"] == []
    with store._connect() as conn:
        assert store.scene_context_snapshot(conn, "gate").match(
            camera_id="gate", label="car", box=(0.1, 0.1, 0.3, 0.3), event_key="two",
            observed_at_epoch=1120.0, min_iou=0.72, ttl_seconds=7200.0,
        )[:2] == (True, 0)


def _parked_then_banded(memory, *, tolerance="high"):
    from tests.test_object_activity import night_car

    attributor = ObjectActivityAttributor("enforce", tolerance, memory=memory, camera_id="gate")
    for index in range(8):
        attributor.admit([night_car(0.001)], {}, event_key=f"parked-{index}",
                         observed_at_epoch=1000.0 + index * 60.0)
    return attributor


def _banded(attributor, event_key, epoch):
    from tests.test_object_activity import banded_car

    return attributor.admit([banded_car()], {}, event_key=event_key, observed_at_epoch=epoch)[0]


def _subject(memory):
    subjects = memory._subjects
    return subjects[0][1] if isinstance(subjects[0], tuple) else subjects[0]


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_a_car_seen_only_in_the_band_stays_known_and_then_expires(kind, tmp_path):
    memory = _memory(kind, tmp_path)
    attributor = _parked_then_banded(memory)
    subject = _subject(memory)
    box, keys = subject.box, list(subject.stable_event_keys)
    ttl = attributor.stationary_policy.scene_memory_ttl_seconds

    # Three hours of night events, every one in the band, none still.
    start = 1420.0 + 600.0
    nights = [start + index * 600.0 for index in range(19)]
    assert nights[-1] - 1420.0 > ttl
    for index, epoch in enumerate(nights):
        admission = _banded(attributor, f"night-{index}", epoch)
        assert admission.attribution.role.value == "scene_context", index
        assert admission.admitted is False

    # Kept alive by its last sighting, never moved or extended.
    assert (subject.box, subject.stable_event_keys, subject.last_seen_epoch) == (box, keys, nights[-1])
    if kind == "durable":
        assert memory._store.load_scene_context("gate")[0]["last_seen_epoch"] == nights[-1]

    # Once nothing is seen there for the TTL, the subject expires.
    gone = _banded(attributor, "after", nights[-1] + ttl + 1.0)
    assert gone.attribution.role.value == "indeterminate"
    assert gone.admitted is True
    assert memory.count("gate") == 0


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_a_jitter_absorbed_sighting_also_keeps_the_subject_known(kind, tmp_path):
    from tests.test_object_activity import night_car

    memory = _memory(kind, tmp_path)
    attributor = _parked_then_banded(memory)
    ttl = attributor.stationary_policy.scene_memory_ttl_seconds
    epochs = [1420.0 + index * 1800.0 for index in range(1, 7)]
    for index, epoch in enumerate(epochs):
        admission = attributor.admit([night_car(0.00994)], {}, event_key=f"night-{index}",
                                     observed_at_epoch=epoch)[0]
        assert admission.attribution.evidence.scene_context_jitter_absorbed is True, index
    assert epochs[-1] - 1420.0 > ttl
    assert _subject(memory).last_seen_epoch == epochs[-1]


def test_a_refresh_writes_once_per_event_and_reloads_as_it_is_held(tmp_path):
    store = EventStore(tmp_path)
    memory = store.scene_context_memory("gate")
    attributor = _parked_then_banded(memory)
    written = memory.writes

    # Every frame of one event, then a late event older than the newest sighting.
    for offset in range(5):
        _banded(attributor, "night", 2000.0 + offset)
    assert memory.writes == written + 1
    _banded(attributor, "late", 1500.0)
    assert memory.writes == written + 1
    assert _subject(memory).last_seen_epoch == 2000.0

    store._scene_context_memories.clear()
    reloaded = store.scene_context_memory("gate")
    fields = lambda s: (s.label, s.box, s.first_seen_epoch, s.last_seen_epoch,  # noqa: E731
                        s.stable_event_keys, s.sighting_epochs)
    assert [fields(s) for s in reloaded._subjects] == [fields(s) for s in memory._subjects]


# Item: a still person is exempt from stationary suppression by default.

def _still_person(role="scene_context", label="person"):
    samples = physical([label], [box(20, 80)])
    samples[0]["observations"][0]["activity_role"] = role
    return samples


def _known(attributor, label, count=2):
    for index in range(count):
        attributor.admit([observation(label=label)], {}, event_key=f"seen-{label}-{index}",
                         observed_at_epoch=1000.0 + index * 60.0)


def test_person_suppression_defaults_to_exempt_and_cameras_inherit_it():
    from survng.app.scene_zone_admission import effective_stationary_person_suppression

    detector = DetectorConfig()
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
    assert detector.stationary_person_suppression == "exempt"
    assert camera.stationary_person_suppression == "inherit"
    assert effective_stationary_person_suppression(camera, detector) == "exempt"
    assert establishment_zone_policy(camera, detector_config=detector)["stationary_person_suppression"] == "exempt"
    camera.stationary_person_suppression = "suppress"
    assert effective_stationary_person_suppression(camera, detector) == "suppress"
    assert establishment_zone_policy(camera, detector_config=detector)["stationary_person_suppression"] == "suppress"
    camera.stationary_person_suppression = "inherit"
    detector.stationary_person_suppression = "suppress"
    assert effective_stationary_person_suppression(camera, detector) == "suppress"


@pytest.mark.parametrize(
    ("label", "setting", "admitted"),
    [("person", "exempt", True), ("person", "suppress", False),
     ("car", "exempt", False), ("dog", "exempt", False), ("face", "exempt", False)],
)
def test_live_admission_exempts_only_a_person(label, setting, admitted):
    attributor = ObjectActivityAttributor("enforce", person_suppression=setting)
    _known(attributor, label)

    admission = attributor.admit([observation(label=label)], {}, event_key="now",
                                 observed_at_epoch=1200.0)[0]

    assert admission.attribution.role.value == "scene_context"
    assert admission.admitted is admitted
    assert admission.counterfactual_suppressed is not admitted
    stored = admission.stored_observation()
    assert stored["activity_role"] == "scene_context"
    assert stored["activity_admission_reason"] == (
        "stationary_person_exempt" if admitted else "stationary_scene_context"
    )
    assert attributor.status()["enforced_suppressions"] == (0 if admitted else 1)


@pytest.mark.parametrize(("setting", "counterfactual"), [("exempt", False), ("suppress", True)])
def test_shadow_mode_records_the_counterfactual_only_where_enforcement_would_suppress(setting, counterfactual):
    attributor = ObjectActivityAttributor("shadow", person_suppression=setting)
    _known(attributor, "person")
    admission = attributor.admit([observation(label="person")], {}, event_key="now",
                                 observed_at_epoch=1200.0)[0]
    assert admission.admitted is True
    assert admission.counterfactual_suppressed is counterfactual


def test_the_exemption_hot_applies_to_a_running_attributor():
    attributor = ObjectActivityAttributor("enforce", person_suppression="suppress")
    _known(attributor, "person")
    assert attributor.admit([observation(label="person")], {}, event_key="a",
                            observed_at_epoch=1200.0)[0].admitted is False
    attributor.reconfigure_stationary_person_suppression("exempt")
    assert attributor.admit([observation(label="person")], {}, event_key="b",
                            observed_at_epoch=1260.0)[0].admitted is True
    assert attributor.status()["stationary_person_suppression"] == "exempt"


@pytest.mark.parametrize(
    ("label", "setting", "suppressed"),
    [("person", "exempt", False), ("person", "suppress", True), ("Person", "exempt", False),
     ("car", "exempt", True), ("dog", "exempt", True)],
)
def test_alerts_exempt_only_a_person(label, setting, suppressed):
    item = {"label": label, "activity_role": "scene_context", "alert_eligible": True, "alert_reasons": []}
    apply_stationary_alert(item, presence="ignore", mode="enforce", person_suppression=setting)
    assert item["alert_eligible"] is (not suppressed)
    shadow = {"label": label, "activity_role": "scene_context", "alert_eligible": True, "alert_reasons": []}
    apply_stationary_alert(shadow, presence="ignore", mode="shadow", person_suppression=setting)
    assert ("alert_shadow_reasons" in shadow) is suppressed
    assert stationary_subject_effect(
        "scene_context", "ignore", "enforce", label=label, person_suppression=setting,
    ) == ("enforce" if suppressed else "")


def test_an_alert_without_a_person_setting_keeps_suppressing():
    item = {"label": "person", "activity_role": "scene_context", "alert_eligible": True, "alert_reasons": []}
    apply_stationary_alert(item, presence="ignore", mode="enforce")
    assert item["alert_eligible"] is False


def _enforcing(**values):
    return _policy(stationary_subject_presence="ignore", object_activity_attribution="enforce", **values)


@pytest.mark.parametrize(
    ("label", "setting", "supported"),
    [("person", "exempt", True), ("person", "suppress", False), ("car", "exempt", False),
     ("dog", "exempt", False)],
)
def test_a_notice_is_verified_by_a_still_person_only_when_exempt(label, setting, supported):
    decision = evaluate_scene_establishment(
        _still_person(label=label),
        policy=_enforcing(stationary_person_suppression=setting),
        notice={"source": "motion", "epoch": 1000},
    )
    assert (decision["status"] == "supported") is supported
    interpreted = decision["zone_interpretation"]["observations"][0]
    assert interpreted["activity_role"] == "scene_context"
    assert (interpreted["admission_reason"] == "stationary_scene_context") is (not supported)


def test_a_snapshot_from_before_the_setting_keeps_suppressing_a_person():
    decision = evaluate_scene_establishment(
        _still_person(), policy=_enforcing(), notice={"source": "motion", "epoch": 1000},
    )
    assert decision["status"] == "unsupported"
    assert decision["reason"] == "stationary_scene_context"


@pytest.mark.parametrize(("setting", "counterfactual"), [("exempt", False), ("suppress", True)])
def test_establishment_shadow_counterfactual_follows_the_exemption(setting, counterfactual):
    decision = evaluate_scene_establishment(
        _still_person(),
        policy=_policy(stationary_subject_presence="ignore", object_activity_attribution="shadow",
                       stationary_person_suppression=setting),
        notice={"source": "motion", "epoch": 1000},
    )
    assert decision["status"] == "supported"
    assert bool(decision["diagnostics"].get("stationary_scene_context_counterfactual")) is counterfactual


@pytest.mark.parametrize(("label", "setting", "counted"),
                         [("person", "exempt", True), ("person", "suppress", False),
                          ("car", "exempt", False)])
def test_in_place_motion_of_a_still_person_is_activity_when_exempt(label, setting, counted):
    samples = physical([label, label], [box(20, 80), box(20, 80)])
    witness = samples[-1]["metadata"]["activity_witnesses"][0]
    witness["kind"] = "localized_motion"
    witness["normalized_displacement"] = 0.0004
    for sample in samples:
        sample["observations"][0]["activity_role"] = "scene_context"
        sample["metadata"]["confirmation_complete"] = True
    established = evaluate_scene_establishment(
        samples, policy=_enforcing(stationary_person_suppression=setting),
    )
    assert (established["status"] == "supported") is counted


@pytest.mark.parametrize(("label", "setting", "supported"),
                         [("person", "exempt", True), ("person", "suppress", False),
                          ("car", "exempt", False)])
def test_a_full_frame_camera_notice_follows_the_exemption(label, setting, supported):
    samples = _still_person(label=label)
    payload = samples[0]["observations"][0]
    payload.update(temporal_consensus=True, incident_eligible=True)
    policy = {"version": 1, "require_incident_zone": False, "zones": [],
              "stationary_subject_presence": "ignore", "object_activity_attribution": "enforce",
              "stationary_person_suppression": setting}
    decision = evaluate_scene_establishment(
        samples, policy=policy,
        notice={"source": "camera", "epoch": 1000,
                "camera_admission": {"version": 1, "trigger_class": "generic_camera"}},
    )
    assert (decision["status"] == "supported") is supported


@pytest.mark.parametrize(("label", "supported"), [("person", True), ("car", False)])
def test_a_role_assigned_from_memory_does_not_suppress_an_exempt_person(tmp_path, label, supported):
    store = EventStore(tmp_path)
    attributor = ObjectActivityAttributor("enforce", memory=store.scene_context_memory("gate"),
                                          camera_id="gate")
    for key, epoch in (("one", 1000.0), ("two", 1060.0)):
        parked = _parked_box()
        parked["label"] = label
        attributor.admit([parked], {}, event_key=key, observed_at_epoch=epoch)
    samples = physical([label], [box(20, 80)])

    decision = evaluate_scene_establishment(
        samples, policy=_enforcing(stationary_person_suppression="exempt"),
        notice={"source": "motion", "epoch": 1120},
        scene_context_memory=store.scene_context_memory("gate"),
        event_key="three", observed_at_epoch=1120.0,
    )

    # The role is still recorded for audit.
    assert samples[0]["observations"][0]["activity_role"] == "scene_context"
    assert (decision["status"] == "supported") is supported


@pytest.mark.parametrize(("label", "exempt", "rescued"),
                         [("person", True, True), ("person", False, False), ("car", True, False)])
def test_semantic_rescue_follows_the_exemption(label, exempt, rescued):
    from unittest.mock import patch

    from survng.app.motion_pipeline import decision_handler

    candidate = {"label": label, "activity_role": "scene_context", "spatial_zone_eligible": True,
                 "confidence_eligible": True, "temporal_incident_observations": 2,
                 "temporal_required_observations": 2, "incident_eligible": False,
                 "temporal_eligible": False, "semantic_tier": "rescue_candidate",
                 "incident_ineligible_reasons": ["pending_causal_confirmation"]}
    attributor = ObjectActivityAttributor(
        "enforce", person_suppression="exempt" if exempt else "suppress",
    )
    with patch.object(decision_handler, "motion_correlated_objects",
                      side_effect=lambda frame, items, *args, **kwargs: (list(items), {})):
        decision_handler._admit_semantic_rescues(
            object(), [candidate], {}, {}, stationary_suppressed=attributor.suppresses_stationary,
        )
    assert candidate["semantic_rescue_admitted"] is rescued


def test_a_sightings_column_with_repeated_events_counts_each_once(tmp_path):
    store = EventStore(tmp_path)
    store.save_scene_context("gate", {
        "label": "car", "box": (0.1, 0.1, 0.3, 0.3), "first_seen_epoch": 1000.0,
        "last_seen_epoch": 1060.0, "sightings": ["one"] * 6 + ["two"],
        "sighting_epochs": {"one": 1000.0, "two": 1060.0},
    })
    assert store.load_scene_context("gate")[0]["sightings"] == ["one", "two"]
    query = dict(camera_id="gate", label="car", box=(0.1, 0.1, 0.3, 0.3), event_key="three",
                 observed_at_epoch=1120.0, min_iou=0.72, ttl_seconds=7200.0)
    assert store.scene_context_memory("gate").match(**query)[:2] == (True, 2)
    with store._connect() as conn:
        assert store.scene_context_snapshot(conn, "gate").match(**query)[:2] == (True, 2)
