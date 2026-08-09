from __future__ import annotations

import numpy as np

from survng.app.face_candidates import FaceCandidateSample, collect_face_candidates


def _face(x: int, *, quality: float, confidence: float = 0.9) -> dict:
    return {
        "label": "face",
        "confidence": confidence,
        "box": {"x1": x, "y1": 20, "x2": x + 40, "y2": 70},
        "face_quality_score": quality,
        "face_sharpness_score": quality,
        "face_exposure_score": 0.8,
        "detection_source": "dedicated_face",
    }


def test_collect_face_candidates_ranks_diverse_frames_per_track() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    candidates = collect_face_candidates(
        (
            FaceCandidateSample(0.0, frame, (_face(20, quality=0.4),)),
            FaceCandidateSample(0.2, frame, (_face(22, quality=0.95),)),
            FaceCandidateSample(1.0, frame, (_face(25, quality=0.8),)),
            FaceCandidateSample(2.0, frame, (_face(30, quality=0.7),)),
        )
    )

    assert len(candidates) == 3
    assert {candidate.track_id for candidate in candidates} == {"face-1"}
    assert candidates[0].rank == 1
    assert candidates[0].offset_seconds == 0.2
    assert [candidate.rank for candidate in candidates] == [1, 2, 3]


def test_collect_face_candidates_keeps_people_separate_and_bounded() -> None:
    frame = np.zeros((120, 500, 3), dtype=np.uint8)
    samples = tuple(
        FaceCandidateSample(
            float(index),
            frame,
            (_face(20 + index, quality=0.8), _face(350 - index, quality=0.7)),
        )
        for index in range(6)
    )

    candidates = collect_face_candidates(samples, max_per_track=2)

    assert len(candidates) == 4
    assert {candidate.track_id for candidate in candidates} == {"face-1", "face-2"}
    assert all(
        sum(item.track_id == track_id for item in candidates) == 2
        for track_id in {"face-1", "face-2"}
    )


def test_collect_face_candidates_ignores_non_faces_and_invalid_boxes() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    candidates = collect_face_candidates((
        FaceCandidateSample(
            0.0,
            frame,
            (
                {"label": "person", "box": {"x1": 0, "y1": 0, "x2": 50, "y2": 100}},
                {"label": "face", "box": {"x1": 20, "y1": 20, "x2": 10, "y2": 30}},
            ),
        ),
    ))

    assert candidates == ()
