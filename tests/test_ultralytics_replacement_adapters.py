"""Exercise the real optional pinned tracker algorithms, without model files."""
from __future__ import annotations

import importlib.util
from unittest.mock import patch

import numpy as np
import pytest

from survng.app.config import ObjectTrackingConfig
from survng.app.object_track.registry import (
    build_builtin_object_tracker_registry,
    ultralytics_botsort_dependency_status,
    ultralytics_tracktrack_dependency_status,
)

ENGINES = ("ultralytics_tracktrack", "ultralytics_botsort")
AVAILABLE = ultralytics_tracktrack_dependency_status()["available"] and ultralytics_botsort_dependency_status()["available"]


def det(label="person", embedding=None, x=10):
    item = {"label": label, "confidence": 0.9, "box": {"x1": x, "y1": 10, "x2": x + 30, "y2": 80}}
    if embedding is not None:
        item["_tracking_embedding"] = np.asarray(embedding, dtype=np.float32)
    return item


def make(engine, **kwargs):
    return build_builtin_object_tracker_registry().create(engine, ObjectTrackingConfig(
        min_confirmations=1, sample_fps=3, lost_timeout_seconds=1, **kwargs,
    ), 0.7)


def test_registry_is_available_without_importing_ultralytics():
    registry = build_builtin_object_tracker_registry()
    for engine in ENGINES:
        registry.require(engine)


def test_dependency_status_does_not_import_dotted_modules():
    original = importlib.util.find_spec
    def guarded(name):
        assert name in {"ultralytics", "lap"}
        return original(name)
    with patch("survng.app.object_track.registry.importlib.util.find_spec", side_effect=guarded):
        for status in (ultralytics_tracktrack_dependency_status(), ultralytics_botsort_dependency_status()):
            assert status["tested_version"] == "8.4.129"
            assert status["supported_version_range"] == "==8.4.129"


@pytest.mark.skipif(not AVAILABLE, reason="requires optional ultralytics==8.4.129 and LAP")
@pytest.mark.parametrize("engine", ENGINES)
class TestNativeReplacementAdapters:
    def test_supplied_embeddings_and_session_ids(self, engine):
        from ultralytics.trackers.utils.reid import ReID
        # Any accidental external encoder construction fails before a download.
        with patch.object(ReID, "__init__", side_effect=AssertionError("must use supplied embeddings")):
            one = make(engine, reid_enabled=True, reid_model_path="unused-person.xml")
            first = one.update([det(embedding=(3, 4))], 10, confirm_new=True)
            two = make(engine, reid_enabled=True, reid_model_path="unused-person.xml")
            assert two.update([det(embedding=(3, 4))], 10, confirm_new=True)[0]["track_id"] == 1
            assert first[0]["track_id"] == 1
            np.testing.assert_allclose(one._tracker.tracked_stracks[0].curr_feat, [0.6, 0.8])
            for timestamp in (10.333, 10.667):
                out = one.update([det(embedding=(3, 4)), det("car", x=150)], timestamp)
            assert {row["label"]: row["track_id"] for row in out} == {"person": 1, "car": 2}
            assert all("_tracking_embedding" not in row for row in out)

    def test_class_isolation_and_history(self, engine):
        tracker = make(engine, reid_enabled=True, reid_model_path="unused.xml",
                       vehicle_reid_enabled=True, vehicle_reid_model_path="unused-car.xml")
        first = tracker.update([det(embedding=(1, 0)), det("car", (1, 0))], 10, confirm_new=True)
        assert len(first) == 2
        second = tracker.update([det("car", (1, 0)), det(embedding=(1, 0, 0, 0))], 10.333)
        assert {row["label"]: row["track_id"] for row in first} == {row["label"]: row["track_id"] for row in second}
        for summary in tracker.summaries(10.333):
            assert summary["box_history"][-1] == [10.333, 10, 10, 40, 80]
        assert tracker._feature_dimension == 4

    def test_timestamp_expiry_without_intermediate_empty_frame(self, engine):
        tracker = make(engine)
        first = tracker.update([det()], 10, confirm_new=True)
        tracker.update([det()], 20)
        after = tracker.update([det()], 20.333)
        assert after[0]["track_id"] != first[0]["track_id"]
        assert tracker.diagnostics()["motion_timebase"] == "fixed_replay_cadence"
        assert tracker.diagnostics()["timestamp_retention"] is True

    def test_short_occlusion_and_reid_expiry(self, engine):
        tracker = make(engine, reid_enabled=True, reid_model_path="unused.xml", reid_max_age_seconds=2)
        first = tracker.update([det(embedding=(1, 0))], 10, confirm_new=True)
        tracker.update([], 10.333)
        assert tracker.update([det(embedding=(1, 0))], 10.667)[0]["track_id"] == first[0]["track_id"]
        tracker.update([], 11)
        tracker.update([det(embedding=(1, 0))], 13)
        after = tracker.update([det(embedding=(1, 0))], 13.333)
        assert after[0]["track_id"] != first[0]["track_id"]

    @pytest.mark.parametrize("label", ("dog", "truck"))
    def test_vehicle_reid_does_not_extend_retention_for_unconfigured_labels(self, engine, label):
        tracker = make(engine, vehicle_reid_enabled=True, vehicle_reid_model_path="unused-car.xml",
                       vehicle_reid_labels=["car"], reid_max_age_seconds=30)
        first = tracker.update([det(label)], 10, confirm_new=True)
        tracker.update([det(label)], 10.25)
        tracker.update([], 10.5)
        tracker.update([det(label)], 12)
        after = tracker.update([det(label)], 12.333)

        assert after[0]["track_id"] != first[0]["track_id"]

    def test_confirmation_filter_and_exact_threshold(self, engine):
        tracker = make(engine)
        ignored = det(); ignored["incident_eligible"] = False
        assert tracker.update([ignored], 10, confirm_new=True) == []
        value = det(); value["confidence"] = 0.7
        tracker.update([value], 10.333, confirm_new=True)
        assert tracker.update([value], 10.667)

    def test_ineligible_low_confidence_continues_but_cannot_start(self, engine):
        tracker = make(engine)
        first = tracker.update([det()], 10, confirm_new=True)
        low = det(); low.update(confidence=0.4, incident_eligible=False)
        continued = tracker.update([low], 10.333)
        assert continued[0]["track_id"] == first[0]["track_id"]
        high = det("car", x=150); high["incident_eligible"] = False
        tracker.update([high], 10.667)
        assert all(item["label"] != "car" for item in tracker.summaries(10.667))

    def test_seed_time_and_minimum_confirmations(self, engine):
        tracker = build_builtin_object_tracker_registry().create(engine, ObjectTrackingConfig(
            min_confirmations=3, sample_fps=3, lost_timeout_seconds=1,
        ), 0.7)
        seeded = det(); seeded["_tracking_first_seen_at"] = 8.0
        assert tracker.update([seeded], 10.0) == []
        assert tracker.update([det()], 10.333) == []
        confirmed = tracker.update([det()], 10.667)
        assert confirmed and confirmed[0]["track_observations"] >= 3
        assert tracker._records[confirmed[0]["track_id"]].first_seen == 8.0
