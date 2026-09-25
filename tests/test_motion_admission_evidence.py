"""Trajectory regressions across detection, attribution, and motion admission."""

import json
from pathlib import Path

import numpy as np
import pytest

from survng.app.motion_pipeline.decision_handler import motion_correlated_objects
from survng.app.motion_pipeline.object_detection import (
    _RecordedDetectionSample,
    _temporal_consensus,
)
from survng.app.object_activity import ObjectActivityAttributor, ObjectActivityRole
from survng.app.object_motion import estimate_object_motion, temporal_object_motion_evidence
from survng.app.incident_presenter import _incident_event_payload


def observation(trajectory):
    estimate = estimate_object_motion(trajectory)
    return {
        "label": "dog",
        "confidence": 0.8,
        "box": {"x1": 1526, "y1": 524, "x2": 1642, "y2": 598},
        "detection_frame_width": 2560,
        "detection_frame_height": 1440,
        "incident_eligible": True,
        "temporal_consensus": True,
        "temporal_track_observations": len(trajectory),
        "temporal_pretrigger_observations": 2,
        "temporal_posttrigger_observations": len(trajectory) - 2,
        "temporal_center_displacement_ratio": estimate.raw_displacement_ratio,
        "temporal_center_path_ratio": estimate.raw_path_ratio,
        "temporal_motion": estimate.as_dict(),
    }


@pytest.mark.parametrize("count", [8, 80, 800])
def test_bounded_jitter_never_accumulates_into_activity(count):
    item = observation([(i * 0.5, 0.6, 0.4 + (i % 2) * 0.002) for i in range(count)])
    motion = temporal_object_motion_evidence(item)
    assert motion.path_ratio > motion.path_threshold
    assert not motion.credible_movement
    assert motion.stable(maximum_displacement_ratio=0.0025, maximum_path_ratio=0.006)


@pytest.mark.parametrize("index", [0, 1, 4, 6, 7])
def test_isolated_box_jump_does_not_establish_displacement_or_excursion(index):
    centers = [(i * 0.5, 0.6, 0.4) for i in range(8)]
    centers[index] = (index * 0.5, 0.7, 0.5)
    assert not temporal_object_motion_evidence(observation(centers)).credible_movement


@pytest.mark.parametrize("count", [2, 3, 4, 8, 24])
def test_slow_sustained_movement_survives_short_and_long_sequences(count):
    item = observation([(i * 0.5, 0.6 + i * 0.002, 0.4) for i in range(count)])
    motion = temporal_object_motion_evidence(item)
    if count == 2:
        assert not motion.credible_movement  # Insufficient travel, not a special exemption.
    else:
        assert motion.credible_movement


@pytest.mark.parametrize("step", [0.1, 0.25, 0.5, 1.0])
def test_out_and_back_survives_sampling_cadence_and_clock_offset(step):
    trajectory = [
        (i * step, 0.6 + (0.04 if 2 <= i * step < 4 else 0), 0.4)
        for i in range(int(6 / step) + 1)
    ]
    for origin in (0, 1_790_000_000):
        item = observation([(t + origin, x, y) for t, x, y in trajectory])
        motion = temporal_object_motion_evidence(item)
        assert motion.displacement_ratio == 0
        assert motion.excursion_ratio > motion.path_threshold
        assert motion.credible_movement


def test_two_observation_excursion_survives_temporal_filter():
    trajectory = [(i * 0.5, 0.64 if i in (3, 4) else 0.6, 0.4) for i in range(8)]
    assert temporal_object_motion_evidence(observation(trajectory)).credible_movement


@pytest.mark.parametrize("count", [8, 9, 38, 39])
def test_repeated_excursions_preserve_nonadjacent_support(count):
    trajectory = [(i * 0.4, 0.64 if i % 2 else 0.6, 0.4) for i in range(count)]
    assert_active_admission(trajectory)


def assert_active_admission(trajectory):
    item = observation(trajectory)
    attributor = ObjectActivityAttributor("enforce")
    for index in range(3):
        admission = attributor.admit(
            [item], {}, event_key=f"moving-{index}", observed_at_epoch=100 + index,
        )[0]
        assert admission.attribution.role is ObjectActivityRole.ACTIVE
        assert admission.admitted
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    correlated, _ = motion_correlated_objects(
        frame, [dict(item)], {"features": {"motion_regions": [[0.1, 0.1, 0.3, 0.3]]}},
    )
    assert len(correlated) == 1


@pytest.mark.parametrize("start", [2.0, 2.3, 2.4])
@pytest.mark.parametrize("support", [2, 4])
def test_brief_supported_excursion_in_long_track_remains_active(start, support):
    first = round(start * 10)
    trajectory = [
        (i / 10, 0.65 if first <= i < first + support else 0.6, 0.4)
        for i in range(41)
    ]
    assert_active_admission(trajectory)


@pytest.mark.parametrize("count", [5, 6, 7, 8])
def test_more_samples_do_not_reclassify_steady_drift_as_scene_context(count):
    trajectory = [(i * 0.5, 0.6 + i * 0.0008, 0.4) for i in range(count)]
    estimate = estimate_object_motion(trajectory)
    assert estimate.displacement_ratio == pytest.approx((count - 1) * 0.0008)
    assert_active_admission(trajectory)


def test_irregular_sampling_preserves_real_endpoint_travel():
    times = [0.0, 5.0, 5.1, 5.2, 5.3, 10.3]
    trajectory = [(t, 0.6 + t * 0.005, 0.4) for t in times]
    estimate = estimate_object_motion(trajectory)
    assert estimate.displacement_ratio == pytest.approx(0.0515)
    assert_active_admission(trajectory)


@pytest.mark.parametrize("count", [2, 3, 4])
def test_brief_transit_within_one_time_bin_is_not_erased(count):
    trajectory = [(i * 0.05, 0.6 + i * 0.02, 0.4) for i in range(count)]
    item = observation(trajectory)
    assert item["temporal_motion"]["samples"] == count
    assert temporal_object_motion_evidence(item).credible_movement


def test_duplicate_samples_do_not_manufacture_temporal_support():
    trajectory = [(0, 0.6, 0.4), (0.5, 0.6, 0.4), (1, 0.6, 0.4)]
    estimate = estimate_object_motion(trajectory * 10)
    assert estimate.samples == 3
    assert estimate.raw_path_ratio == 0


def test_duplicate_outlier_timestamps_do_not_count_as_support():
    trajectory = [(i * 0.5, 0.6, 0.4) for i in range(8)]
    trajectory[3] = (1.5, 0.7, 0.5)
    trajectory.extend([trajectory[3]] * 10)
    assert not temporal_object_motion_evidence(observation(trajectory)).credible_movement


def test_previous_version_estimates_remain_readable():
    item = observation([(i * 0.5, 0.6 + i * 0.005, 0.4) for i in range(8)])
    item["temporal_motion"] = {
        "version": 1, "method": "time_bin_median_excursion",
        "displacement_ratio": 0.02, "excursion_ratio": 0.025,
    }
    motion = temporal_object_motion_evidence(item)
    assert motion.displacement_ratio == 0.02
    assert motion.excursion_ratio == 0.025
    assert motion.credible_movement


def test_saved_79555_tracking_is_stationary_evidence():
    fixture = json.loads((Path(__file__).parent / "fixtures/motion/event_79555_tracking.json").read_text())
    trajectory = [(t, x / fixture["frame_width"], y / fixture["frame_height"])
                  for t, x, y in fixture["trajectory"]]
    item = observation(trajectory)
    motion = temporal_object_motion_evidence(item)
    assert motion.path_ratio > motion.path_threshold
    assert not motion.credible_movement
    assert motion.stable(maximum_displacement_ratio=0.0025, maximum_path_ratio=0.006)


def test_new_detection_evidence_drives_memory_and_correlation_consistently():
    # Eight sparse observations with a > .010 accumulated path, but a stationary
    # box oscillating only .002 in frame coordinates, like the admission failure.
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    samples = []
    for i in range(8):
        y = 400 + (i % 2) * 2
        candidate = {
            "label": "dog", "confidence": 0.8, "confidence_threshold": 0.65,
            "confidence_eligible": True, "incident_eligible": True,
            "spatial_zone_eligible": True,
            "box": {"x1": 600, "y1": y, "x2": 645, "y2": y + 50},
        }
        samples.append(_RecordedDetectionSample(i * 0.5 - 1, frame, [candidate], ""))
    _, detected = _temporal_consensus(samples, minimum_confirmations=2)
    item = next(o for o in detected if o.get("label") == "dog")
    item.update(detection_frame_width=1000, detection_frame_height=1000)
    assert item["temporal_motion"]["version"] == 2
    assert item["temporal_center_path_ratio"] > 0.01
    assert not temporal_object_motion_evidence(item).credible_movement
    payload = _incident_event_payload({"objects": [item]})
    assert payload["objects"][0]["temporal_motion"] == item["temporal_motion"]

    attribution = ObjectActivityAttributor("enforce")
    for i in range(3):
        admission = attribution.admit([item], {}, event_key=f"event-{i}", observed_at_epoch=100 + i)[0]
    assert admission.attribution.role is ObjectActivityRole.SCENE_CONTEXT
    assert not admission.admitted

    correlated, _ = motion_correlated_objects(
        frame, [dict(item)], {"features": {"motion_regions": [[0.1, 0.1, 0.3, 0.3]]}},
    )
    assert correlated == []

    # Real movement remains admissible even when the triggering region is elsewhere.
    moving = observation([(i * 0.5, 0.6 + i * 0.005, 0.4) for i in range(8)])
    moving.update(detection_frame_width=1000, detection_frame_height=1000,
                  box={"x1": 600, "y1": 400, "x2": 645, "y2": 450})
    correlated, _ = motion_correlated_objects(
        frame, [moving], {"features": {"motion_regions": [[0.1, 0.1, 0.3, 0.3]]}},
    )
    assert len(correlated) == 1
