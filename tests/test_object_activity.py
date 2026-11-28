from __future__ import annotations

from copy import deepcopy

import pytest

from survng.app.object_activity import ObjectActivityAttributor, ObjectActivityRole


def observation(**values: object) -> dict[str, object]:
    result: dict[str, object] = {
        "label": "car",
        "confidence": 0.91,
        "incident_eligible": True,
        "box": {"x1": 100, "y1": 100, "x2": 300, "y2": 300},
        "detection_frame_width": 1000,
        "detection_frame_height": 1000,
        "temporal_consensus": True,
        "temporal_track_observations": 4,
        "temporal_pretrigger_observations": 2,
        "temporal_posttrigger_observations": 2,
        "temporal_center_displacement_ratio": 0.001,
        "temporal_center_path_ratio": 0.003,
        "temporal_robust_new_appearance": False,
        "temporal_zone_entry": False,
    }
    result.update(values)
    return result


def test_stable_object_without_cross_incident_evidence_fails_open() -> None:
    source = observation()
    original = deepcopy(source)
    admission = ObjectActivityAttributor("enforce").admit([source], {})[0]

    assert admission.attribution.role is ObjectActivityRole.INDETERMINATE
    assert admission.counterfactual_suppressed is False
    assert admission.admitted is True
    assert source == original
    stored = admission.stored_observation()
    assert stored["label"] == "car"
    assert stored["activity_role"] == "indeterminate"
    assert stored["incident_eligible"] is True
    assert stored["detector_incident_eligible"] is True


@pytest.mark.parametrize("label", ["person", "car", "dog", "robot_lawnmower"])
def test_activity_policy_is_class_agnostic(label: str) -> None:
    attributor = ObjectActivityAttributor("enforce")
    for index in range(2):
        attributor.admit(
            [observation(label=label)],
            {},
            event_key=f"prior-{index}",
            observed_at_epoch=1000.0 + index,
        )
    admission = attributor.admit(
        [observation(label=label)], {}, event_key="current", observed_at_epoch=1002.0
    )[0]

    assert admission.attribution.role is ObjectActivityRole.SCENE_CONTEXT
    assert admission.admitted is False


def test_credible_movement_is_active_and_admitted() -> None:
    admission = ObjectActivityAttributor("enforce").admit(
        [
            observation(
                temporal_center_displacement_ratio=0.025,
                temporal_center_path_ratio=0.04,
            )
        ],
        {},
    )[0]

    assert admission.attribution.role is ObjectActivityRole.ACTIVE
    assert admission.admitted is True


def test_robust_appearance_needs_causal_evidence_while_zone_entry_is_active() -> None:
    attributor = ObjectActivityAttributor("enforce")

    appearance, zone = attributor.admit(
        [
            observation(
                temporal_pretrigger_observations=0,
                temporal_robust_new_appearance=True,
            ),
            observation(temporal_zone_entry=True),
        ],
        {},
    )

    assert appearance.attribution.role is ObjectActivityRole.INDETERMINATE
    assert appearance.attribution.evidence.reasons == (
        "robust_new_appearance_without_causal_motion",
    )
    assert zone.attribution.role is ObjectActivityRole.ACTIVE


def test_repeated_stable_robust_appearance_builds_scene_context_memory() -> None:
    attributor = ObjectActivityAttributor("enforce")
    appeared = observation(
        temporal_pretrigger_observations=0,
        temporal_posttrigger_observations=4,
        temporal_robust_new_appearance=True,
    )

    first = attributor.admit(
        [appeared], {}, event_key="event-1", observed_at_epoch=1000.0
    )[0]
    second = attributor.admit(
        [appeared], {}, event_key="event-2", observed_at_epoch=1060.0
    )[0]
    third = attributor.admit(
        [appeared], {}, event_key="event-3", observed_at_epoch=1120.0
    )[0]

    assert first.attribution.role is ObjectActivityRole.INDETERMINATE
    assert second.attribution.role is ObjectActivityRole.INDETERMINATE
    assert third.attribution.role is ObjectActivityRole.SCENE_CONTEXT
    assert attributor.status()["scene_context_memory_entries"] == 1


def test_semantic_rescue_telemetry_is_grouped_by_trigger_source() -> None:
    attributor = ObjectActivityAttributor("enforce")

    attributor.record_semantic_rescue(
        source="visual_backup",
        candidates=3,
        admitted=1,
        rejected=2,
    )

    status = attributor.status()
    assert status["semantic_rescue_candidates"] == 3
    assert status["semantic_rescue_admissions"] == 1
    assert status["semantic_rescue_rejections"] == 2
    assert status["semantic_rescue_by_source"]["visual_backup"] == {
        "candidates": 3,
        "admitted": 1,
        "rejected": 2,
    }


def test_incomplete_evidence_fails_open() -> None:
    admission = ObjectActivityAttributor("enforce").admit(
        [
            observation(
                temporal_consensus=False,
                temporal_track_observations=1,
                temporal_pretrigger_observations=0,
                temporal_posttrigger_observations=1,
            )
        ],
        {},
    )[0]

    assert admission.attribution.role is ObjectActivityRole.INDETERMINATE
    assert admission.admitted is True


def test_shadow_mode_reports_counterfactual_without_changing_eligibility() -> None:
    attributor = ObjectActivityAttributor("shadow")
    attributor.admit([observation()], {}, event_key="one", observed_at_epoch=1000.0)
    attributor.admit([observation()], {}, event_key="two", observed_at_epoch=1001.0)
    admission = attributor.admit(
        [observation()], {}, event_key="three", observed_at_epoch=1002.0
    )[0]

    assert admission.counterfactual_suppressed is True
    assert admission.admitted is True
    assert admission.stored_observation()["incident_eligible"] is True
    assert attributor.status()["counterfactual_suppressions"] == 1
    assert attributor.status()["enforced_suppressions"] == 0


def test_ema_overlap_is_diagnostic_and_cannot_override_temporal_evidence() -> None:
    admission = ObjectActivityAttributor("enforce").admit(
        [observation()],
        {"features": {"motion_regions": [[0.0, 0.0, 0.5, 0.5]]}},
    )[0]

    assert admission.attribution.evidence.ema_region_overlap is True
    assert admission.attribution.evidence.ema_alignment_reliable is False
    assert admission.attribution.role is ObjectActivityRole.INDETERMINATE


def test_legacy_offsets_support_conservative_historical_replay() -> None:
    legacy = observation()
    legacy.pop("temporal_pretrigger_observations")
    legacy.pop("temporal_posttrigger_observations")
    legacy["temporal_first_observation_offset_seconds"] = -1.0
    legacy["temporal_last_observation_offset_seconds"] = 1.0

    admission = ObjectActivityAttributor("enforce").admit([legacy], {})[0]

    assert admission.attribution.role is ObjectActivityRole.INDETERMINATE
    assert admission.admitted is True


def test_repeated_stable_object_becomes_context_across_distinct_incidents() -> None:
    attributor = ObjectActivityAttributor("enforce")
    short_clip = observation(
        temporal_pretrigger_observations=0,
        temporal_posttrigger_observations=2,
    )

    first = attributor.admit(
        [short_clip], {}, event_key="event-1", observed_at_epoch=1000.0
    )[0]
    second = attributor.admit(
        [short_clip], {}, event_key="event-2", observed_at_epoch=1060.0
    )[0]
    third = attributor.admit(
        [short_clip], {}, event_key="event-3", observed_at_epoch=1120.0
    )[0]

    assert first.attribution.role is ObjectActivityRole.INDETERMINATE
    assert first.admitted is True
    assert second.attribution.role is ObjectActivityRole.INDETERMINATE
    assert second.admitted is True
    assert third.attribution.role is ObjectActivityRole.SCENE_CONTEXT
    assert third.attribution.evidence.scene_context_memory_match is True
    assert third.admitted is False


def test_delayed_refinement_cannot_learn_from_its_own_incident() -> None:
    attributor = ObjectActivityAttributor("enforce")
    short_clip = observation(
        temporal_pretrigger_observations=0,
        temporal_posttrigger_observations=2,
    )

    attributor.admit(
        [short_clip], {}, event_key="event-1", observed_at_epoch=1000.0
    )
    refinement = attributor.admit(
        [short_clip], {}, event_key="event-1", observed_at_epoch=1012.0
    )[0]

    assert refinement.attribution.role is ObjectActivityRole.INDETERMINATE
    assert refinement.admitted is True


def test_active_refinement_revokes_its_provisional_stable_memory() -> None:
    attributor = ObjectActivityAttributor("enforce")
    stable = observation(
        temporal_pretrigger_observations=0,
        temporal_posttrigger_observations=2,
    )
    active = observation(
        temporal_center_displacement_ratio=0.03,
        temporal_center_path_ratio=0.04,
    )
    attributor.admit(
        [stable], {}, event_key="event-1", observed_at_epoch=1000.0
    )

    attributor.admit(
        [active], {}, event_key="event-1", observed_at_epoch=1000.0
    )

    assert attributor.status()["scene_context_memory_entries"] == 0


def test_context_memory_expires_and_never_overrides_movement() -> None:
    attributor = ObjectActivityAttributor("enforce")
    short_clip = observation(
        temporal_pretrigger_observations=0,
        temporal_posttrigger_observations=2,
    )
    attributor.admit(
        [short_clip], {}, event_key="event-1", observed_at_epoch=1000.0
    )

    moved = attributor.admit(
        [
            observation(
                temporal_pretrigger_observations=0,
                temporal_posttrigger_observations=2,
                temporal_center_displacement_ratio=0.03,
                temporal_center_path_ratio=0.04,
            )
        ],
        {},
        event_key="event-2",
        observed_at_epoch=1060.0,
    )[0]
    expired = attributor.admit(
        [short_clip],
        {},
        event_key="event-3",
        observed_at_epoch=1000.0 + attributor.CONTEXT_MEMORY_TTL_SECONDS + 1,
    )[0]

    assert moved.attribution.role is ObjectActivityRole.ACTIVE
    assert moved.admitted is True
    assert expired.attribution.role is ObjectActivityRole.INDETERMINATE
    assert expired.admitted is True


def test_disabling_attribution_clears_scene_context_memory() -> None:
    attributor = ObjectActivityAttributor("shadow")
    attributor.admit(
        [observation()], {}, event_key="event-1", observed_at_epoch=1000.0
    )

    attributor.reconfigure("off")

    assert attributor.status()["mode"] == "off"
    assert attributor.status()["scene_context_memory_entries"] == 0


def test_out_of_order_event_cannot_learn_from_future_context() -> None:
    attributor = ObjectActivityAttributor("enforce")
    attributor.admit(
        [observation()], {}, event_key="future", observed_at_epoch=2000.0
    )

    older = attributor.admit(
        [observation()], {}, event_key="older", observed_at_epoch=1000.0
    )[0]

    assert older.attribution.evidence.scene_context_memory_match is False
    assert older.attribution.role is ObjectActivityRole.INDETERMINATE


def test_repeated_scene_context_history_is_bounded() -> None:
    attributor = ObjectActivityAttributor("shadow")

    for index in range(100):
        attributor.admit(
            [observation()],
            {},
            event_key=f"event-{index}",
            observed_at_epoch=1000.0 + index,
        )

    subjects = [subject for _, subject in attributor.memory._subjects]
    assert len(subjects) == 1
    assert len(subjects[0].stable_event_keys) == 16


# A parked car at night on a "high" tolerance camera: its box is 17.9% x 5.1%
# of the frame, so its movement threshold is 0.00745, and each event has only
# three track samples. These are the displacements and excursions measured
# for six of its events, all on the remembered spot.
PRODUCTION_NIGHT_JITTER = (0.00696, 0.00775, 0.00802, 0.00805, 0.00827, 0.00994)


def night_car(
    displacement: float,
    *,
    samples: int = 3,
    excursion: float | None = None,
    shift: float = 0.0,
    **values: object,
) -> dict[str, object]:
    excursion = displacement if excursion is None else excursion
    return observation(
        box={"x1": 400 + shift, "y1": 600, "x2": 579 + shift, "y2": 651},
        temporal_track_observations=samples,
        temporal_center_displacement_ratio=displacement,
        temporal_center_path_ratio=excursion,
        temporal_motion={
            "version": 2,
            "displacement_ratio": displacement,
            "excursion_ratio": excursion,
        },
        **values,
    )


def parked_overnight(tolerance: str = "high", sightings: int = 8) -> ObjectActivityAttributor:
    attributor = ObjectActivityAttributor("enforce", tolerance)
    for index in range(sightings):
        attributor.admit(
            [night_car(0.001)], {}, event_key=f"parked-{index}",
            observed_at_epoch=1000.0 + index * 60.0,
        )
    return attributor


def remembered(attributor: ObjectActivityAttributor) -> list[list[str]]:
    return [list(subject.stable_event_keys) for _, subject in attributor.memory._subjects]


@pytest.mark.parametrize("measured", PRODUCTION_NIGHT_JITTER)
def test_sparse_jitter_on_an_established_subject_is_scene_context(measured: float) -> None:
    attributor = parked_overnight()
    before = remembered(attributor)

    admission = attributor.admit(
        [night_car(measured)], {}, event_key="night", observed_at_epoch=2000.0
    )[0]

    evidence = admission.attribution.evidence
    assert evidence.movement_threshold == pytest.approx(0.00745, abs=1e-5)
    assert admission.attribution.role is ObjectActivityRole.SCENE_CONTEXT
    assert admission.admitted is False
    assert evidence.reasons == ("sparse_jitter_on_known_scene_context",)
    audit = evidence.as_dict()
    assert audit["scene_context_jitter_absorbed"] is True
    assert audit["scene_context_footprint_match"] is True
    assert audit["scene_context_footprint_sightings"] == 8
    assert audit["scene_context_jitter_allowance_ratio"] >= max(PRODUCTION_NIGHT_JITTER)
    # Neither erased nor extended: only still sightings build the memory.
    assert remembered(attributor) == before


@pytest.mark.parametrize("tolerance", ["low", "balanced", "high"])
def test_the_jitter_allowance_holds_at_every_tolerance(tolerance: str) -> None:
    attributor = parked_overnight(tolerance)
    admission = attributor.admit(
        [night_car(max(PRODUCTION_NIGHT_JITTER))], {}, event_key="night",
        observed_at_epoch=2000.0,
    )[0]
    assert admission.attribution.role is ObjectActivityRole.SCENE_CONTEXT


def test_a_car_pulling_away_from_its_remembered_spot_is_active_and_forgets_it() -> None:
    attributor = parked_overnight()

    admission = attributor.admit(
        [night_car(0.03, shift=15)], {}, event_key="leaving", observed_at_epoch=2000.0
    )[0]

    assert admission.attribution.role is ObjectActivityRole.ACTIVE
    assert admission.admitted is True
    assert attributor.memory.count() == 0


def test_creeping_off_the_remembered_footprint_is_active() -> None:
    attributor = parked_overnight()

    # Within the jitter allowance, but the box has slid a seventh of its
    # width: it still matches the memory, not the remembered footprint.
    admission = attributor.admit(
        [night_car(0.009, shift=25)], {}, event_key="creeping", observed_at_epoch=2000.0
    )[0]

    assert admission.attribution.role is ObjectActivityRole.ACTIVE
    assert admission.attribution.evidence.scene_context_memory_match is True
    assert admission.admitted is True


def test_zone_entry_at_the_remembered_spot_is_active_and_forgets_it() -> None:
    attributor = parked_overnight()

    admission = attributor.admit(
        [night_car(0.008, temporal_zone_entry=True)], {}, event_key="entry",
        observed_at_epoch=2000.0,
    )[0]

    assert admission.attribution.role is ObjectActivityRole.ACTIVE
    assert admission.admitted is True
    assert attributor.memory.count() == 0


@pytest.mark.parametrize(
    "sighting",
    [
        night_car(0.009, samples=12),
        night_car(0.0115),
        night_car(0.009, excursion=0.015),
        night_car(0.009, temporal_robust_new_appearance=True),
    ],
    ids=["many_samples", "beyond_allowance", "wide_excursion", "robust_appearance"],
)
def test_movement_the_jitter_allowance_does_not_cover_stays_active(sighting) -> None:
    attributor = parked_overnight()
    before = remembered(attributor)

    admission = attributor.admit([sighting], {}, event_key="moving", observed_at_epoch=2000.0)[0]

    assert admission.attribution.role is ObjectActivityRole.ACTIVE
    assert admission.admitted is True
    assert admission.attribution.evidence.as_dict()["scene_context_jitter_absorbed"] is False
    assert remembered(attributor) == before


def test_jitter_needs_more_prior_sightings_than_plain_scene_context() -> None:
    attributor = parked_overnight(sightings=5)

    jitter = attributor.admit(
        [night_car(0.008)], {}, event_key="night", observed_at_epoch=2000.0
    )[0]
    still = attributor.admit(
        [night_car(0.00696)], {}, event_key="night-2", observed_at_epoch=2060.0
    )[0]

    assert jitter.attribution.role is ObjectActivityRole.ACTIVE
    assert still.attribution.role is ObjectActivityRole.INDETERMINATE
    assert still.admitted is True


def test_jitter_without_a_remembered_subject_fails_open() -> None:
    attributor = ObjectActivityAttributor("enforce", "high")

    moving = attributor.admit([night_car(0.008)], {}, event_key="one", observed_at_epoch=1000.0)[0]
    unclear = attributor.admit([night_car(0.00696)], {}, event_key="two", observed_at_epoch=1060.0)[0]

    assert moving.attribution.role is ObjectActivityRole.ACTIVE
    assert unclear.attribution.role is ObjectActivityRole.INDETERMINATE
    assert unclear.admitted is True
    assert moving.attribution.evidence.as_dict()["scene_context_footprint_match"] is False
