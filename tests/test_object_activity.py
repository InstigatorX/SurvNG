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


def test_robust_appearance_and_zone_entry_are_active() -> None:
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

    assert appearance.attribution.role is ObjectActivityRole.ACTIVE
    assert zone.attribution.role is ObjectActivityRole.ACTIVE


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

    assert len(attributor._context_memory) == 1
    assert len(attributor._context_memory[0].stable_event_keys) == 16
