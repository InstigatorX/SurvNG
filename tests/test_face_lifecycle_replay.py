"""Deterministic replay of temporal face evidence through persistence."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from survng.app.face_candidates import FaceCandidateSample, collect_face_candidates
from survng.app.faces import FaceStore
from survng.app.motion_pipeline import MotionDecisionHandler
from survng.app.motion_pipeline.object_detection import RecordedDetectionResult


class _Events:
    def add_event(self, **payload: Any) -> dict[str, Any]:
        return {"id": 77, **payload}


def _face(x: int, quality: float) -> dict[str, Any]:
    return {
        "label": "face",
        "confidence": 0.9,
        "box": {"x1": x, "y1": 20, "x2": x + 40, "y2": 70},
        "face_quality_score": quality,
        "face_sharpness_score": quality,
        "face_exposure_score": 0.8,
        "detection_source": "dedicated_face",
    }


def _replay() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir)
        store = FaceStore(storage, start_recognition=False)
        frames = []
        samples = []
        for index, quality in enumerate((0.55, 0.91, 0.72)):
            frame = np.full((100, 160, 3), 30 + index * 20, dtype=np.uint8)
            frames.append(frame)
            samples.append(FaceCandidateSample(float(index), frame, (_face(30 + index, quality),)))
        candidates = collect_face_candidates(samples)
        result = RecordedDetectionResult(
            frame=frames[1],
            objects=[{"label": "person", "confidence": 0.9, "incident_eligible": True}],
            recording_path="recording.mp4",
            timings_ms={},
            face_candidates=candidates,
        )
        write_index = 0

        def write_snapshot(frame: np.ndarray, _at: datetime) -> str:
            nonlocal write_index
            write_index += 1
            path = storage / f"snapshot-{write_index}.png"
            assert cv2.imwrite(str(path), frame)
            return str(path)

        handler = MotionDecisionHandler(
            camera_id="front-door",
            events=_Events(),
            detection_provider=lambda _at: result,
            snapshot_writer=write_snapshot,
            object_serializer=lambda value: json.dumps(value, separators=(",", ":")),
            face_candidate_sink=store.ingest_candidates,
        )
        outcome = handler.handle(
            "onvif/motion",
            "motion",
            datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            {},
        )
        observation = store.observations()[0]
        return {
            "event_id": outcome.event_id,
            "candidate_count": observation["consensus"]["candidate_count"],
            "canonical_rank": observation["candidate_rank"],
            "quality_score": observation["quality_score"],
            "crop_shape": cv2.imread(str(store.snapshot_path(observation["id"])[0])).shape,
        }


def test_face_candidate_lifecycle_replay_is_deterministic() -> None:
    first = _replay()
    second = _replay()

    assert first == second
    assert first["event_id"] == 77
    assert first["candidate_count"] == 3
    assert first["canonical_rank"] == 1
    assert first["quality_score"] == 0.91
    assert first["crop_shape"] == (70, 56, 3)
