from __future__ import annotations

import math

from survng.app.object_motion import temporal_object_motion_evidence


def test_temporal_motion_uses_explicit_frame_shape_for_normalization() -> None:
    evidence = temporal_object_motion_evidence(
        {
            "box": {"x1": 20, "y1": 10, "x2": 60, "y2": 50},
            "detection_frame_width": 1000,
            "detection_frame_height": 1000,
        },
        frame_width=100,
        frame_height=100,
    )

    assert evidence.normalized_box == (0.2, 0.1, 0.6, 0.5)
    assert evidence.movement_threshold == 0.02


def test_temporal_motion_reuses_one_adaptive_threshold_and_path_policy() -> None:
    evidence = temporal_object_motion_evidence({
        "box": {"x1": 100, "y1": 100, "x2": 150, "y2": 200},
        "detection_frame_width": 1000,
        "detection_frame_height": 1000,
        "temporal_track_observations": 3,
        "temporal_center_displacement_ratio": 0.001,
        "temporal_center_path_ratio": 0.012,
    })

    expected = math.hypot(0.05, 0.1) * 0.04
    assert math.isclose(evidence.movement_threshold, expected)
    assert math.isclose(evidence.path_threshold, expected * 2.5)
    assert evidence.credible_movement is True


def test_temporal_motion_recovers_legacy_trigger_span_offsets() -> None:
    evidence = temporal_object_motion_evidence({
        "temporal_track_observations": 2,
        "temporal_first_observation_offset_seconds": -1.0,
        "temporal_last_observation_offset_seconds": 1.0,
    })

    assert evidence.pretrigger_observations == 1
    assert evidence.posttrigger_observations == 1
