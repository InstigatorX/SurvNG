import cv2
import numpy as np
import pytest

from survng.app.face_capture.pipeline import (
    CaptureCancelled,
    CapturedFace,
    CaptureSettings,
    capture_event,
    event_window,
    keep_best,
)
from survng.app.inference_runtime.types import InferenceUnavailable

PERSON = (100.0, 50.0, 300.0, 650.0)
SETTINGS = CaptureSettings(min_norm=13.0, min_size=48.0, faces_per_track=2, margin_seconds=1.0, moment_radius_seconds=0.2)


def _frames(fps=10.0):
    def source(start, end):
        count = int(round((end - start) * fps))
        for index in range(count + 1):
            yield round(start + index / fps, 3), np.zeros((720, 1280, 3), np.uint8)
    return source


def _face(norm, side=60.0):
    return {"box": [10.0, 10.0, 10.0 + side, 10.0 + side], "landmarks": [[20.0, 20.0]] * 5,
            "confidence": 0.9, "norm": norm, "embedding": [0.0] * 512}


def test_window_covers_the_plan_reach_and_margin_and_is_capped():
    assert event_window([-0.5, 0.0, 8.5], 3.0) == (-4.4, 12.4)
    start, end = event_window([0.0, 40.0], 3.0)
    assert end - start == pytest.approx(30.0)


def test_moments_come_from_the_sub_stream_and_faces_pass_both_cuts():
    norms = iter([20.0, 9.0, 16.0, 30.0] * 100)
    result = capture_event(
        plan_offsets=[0.0], main_frames=_frames(), sub_frames=_frames(5.0),
        detect_people=lambda frame: [PERSON],
        capture_faces=lambda crop: [_face(next(norms)), _face(40.0, side=20.0)],
        settings=SETTINGS,
    )
    assert result.moment_source == "sub-stream"
    assert result.sub_frames > 0 and result.main_frames > 0
    # The 20 px face never passes; neither does norm 9; each track keeps two.
    assert all(f.norm >= 13.0 and f.box[2] - f.box[0] >= 48.0 for f in result.faces)
    assert len(result.faces) <= 2 * len({f.track for f in result.faces})
    assert result.candidates > len(result.faces)


def test_without_a_sub_stream_stored_boxes_then_plan_offsets_choose_the_moments():
    common = dict(main_frames=_frames(), detect_people=lambda frame: [], capture_faces=lambda crop: [],
                  settings=SETTINGS)
    stored = capture_event(plan_offsets=[0.0], stored_people=[(0.4, [PERSON]), (0.6, [PERSON])], **common)
    assert stored.moment_source == "stored observations"
    planned = capture_event(plan_offsets=[0.5, 0.0], **common)
    assert planned.moment_source == "plan offsets" and planned.moments == [0.0, 0.5]


def test_a_failed_decode_is_noted_and_capture_goes_on():
    def broken(start, end):
        raise RuntimeError("decoder failed")
        yield  # pragma: no cover

    result = capture_event(plan_offsets=[0.0], main_frames=broken, sub_frames=broken,
                           detect_people=lambda frame: [PERSON], capture_faces=lambda crop: [], settings=SETTINGS)
    assert result.moment_source == "plan offsets"
    assert any("sub-stream" in e for e in result.errors) and any("main stream" in e for e in result.errors)


def test_shed_inference_and_cancellation_stop_the_whole_capture():
    def shed(frame):
        raise InferenceUnavailable("shed for incident inference")

    with pytest.raises(InferenceUnavailable):
        capture_event(plan_offsets=[0.0], main_frames=_frames(), sub_frames=_frames(5.0), detect_people=shed,
                      capture_faces=lambda crop: [], settings=SETTINGS)
    with pytest.raises(CaptureCancelled):
        capture_event(plan_offsets=[0.0], main_frames=_frames(), detect_people=lambda f: [],
                      capture_faces=lambda crop: [], settings=SETTINGS, cancelled=lambda: True)


def test_keep_best_takes_the_highest_norms_spaced_apart_per_track():
    def face(track, offset, norm):
        return CapturedFace(track, offset, (0, 0, 60, 60), [], 0.9, norm, np.zeros(512, np.float32),
                            np.zeros((4, 4, 3), np.uint8))

    kept = keep_best([face(0, 1.0, 20), face(0, 1.1, 19), face(0, 2.0, 18), face(1, 1.0, 14)], per_track=2)
    assert [(f.track, f.offset) for f in kept] == [(0, 1.0), (0, 2.0), (1, 1.0)]


def test_one_person_searched_in_separate_windows_keeps_one_track():
    from survng.app.face_capture.moments import Track
    from survng.app.face_capture.pipeline import UNMATCHED_TRACK_BASE, person_keys

    first = [Track([(t / 5, PERSON) for t in range(0, 30)])]  # 0..5.8 s
    second = [Track([(0.5, PERSON), (0.6, PERSON)]), Track([(5.0, PERSON)]),
              Track([(2.0, (900.0, 50.0, 1100.0, 650.0))])]  # someone else
    assert person_keys(first, second) == [0, 0, UNMATCHED_TRACK_BASE + 2]


def test_two_people_the_first_pass_saw_as_one_keep_separate_tracks():
    from survng.app.face_capture.moments import Track
    from survng.app.face_capture.pipeline import UNMATCHED_TRACK_BASE, person_keys

    merged = (100.0, 50.0, 500.0, 650.0)
    first = [Track([(t / 5, merged) for t in range(0, 10)])]
    left, right = (100.0, 50.0, 300.0, 650.0), (300.0, 50.0, 500.0, 650.0)
    second = [Track([(0.5, left), (0.6, left), (0.7, left)]), Track([(0.5, right), (0.6, right)])]
    assert person_keys(first, second) == [0, UNMATCHED_TRACK_BASE + 1]


def test_skipped_main_stream_segments_are_reported_as_main_stream_errors():
    source = _frames()

    class Frames:
        errors = ["0001.mp4 frames 0-8: decoder timed out"]

        def __call__(self, start, end):
            return source(start, end)

    result = capture_event(plan_offsets=[0.0], main_frames=Frames(), detect_people=lambda frame: [],
                           capture_faces=lambda crop: [], settings=SETTINGS)
    assert result.errors == ["main stream 0001.mp4 frames 0-8: decoder timed out"]


def test_held_candidates_are_bounded_to_the_highest_norms_and_kept_crops_decode(monkeypatch):
    from survng.app.face_capture import pipeline

    crop_bytes = len(cv2.imencode(".jpg", np.zeros((84, 84, 3), np.uint8),
                                  [cv2.IMWRITE_JPEG_QUALITY, pipeline.CANDIDATE_JPEG_QUALITY])[1])
    monkeypatch.setattr(pipeline, "MAX_CANDIDATE_BYTES", 3 * crop_bytes)
    norms = iter(range(14, 200))
    result = capture_event(
        plan_offsets=[0.0], main_frames=_frames(), detect_people=lambda frame: [PERSON],
        capture_faces=lambda crop: [_face(float(next(norms)))],
        settings=CaptureSettings(faces_per_track=10, moment_radius_seconds=0.5),
    )
    assert result.candidates > 3
    assert len(result.faces) <= 3
    assert min(f.norm for f in result.faces) >= result.candidates + 14 - 3 - 1
    assert all(f.crop.ndim == 3 and f.crop.size for f in result.faces)


def test_covered_moments_are_not_decoded_again():
    decoded = []

    def frames(start, end):
        decoded.append((start, end))
        return iter(())

    result = capture_event(plan_offsets=[0.0, 2.0], main_frames=frames, detect_people=lambda frame: [PERSON],
                           capture_faces=lambda crop: [], settings=SETTINGS, covered=lambda moment: moment == 0.0)
    assert result.covered_moments == 1 and result.moments == [2.0]
    assert decoded and all(start > 0.5 for start, _ in decoded)


def test_only_moments_whose_window_was_read_count_as_decoded():
    frame = np.zeros((720, 1280, 3), np.uint8)

    def frames(start, end):
        # Recorded up to +0.1 s only: the window around 0.0 is read, and a
        # window that ends before its moment, or has no recording, is not.
        at = start
        while at <= min(end, 0.1):
            yield at, frame
            at += 0.2

    result = capture_event(plan_offsets=[0.0, 5.0], main_frames=frames, detect_people=lambda frame: [],
                           capture_faces=lambda crop: [], settings=SETTINGS)
    assert result.moments == [0.0, 5.0]
    assert result.decoded_moments == [0.0]
    wide = CaptureSettings(min_norm=13.0, min_size=48.0, faces_per_track=2, margin_seconds=1.0,
                           moment_radius_seconds=0.5)
    cut_short = capture_event(plan_offsets=[0.5], main_frames=frames, detect_people=lambda frame: [],
                              capture_faces=lambda crop: [], settings=wide)
    assert cut_short.main_frames > 0 and cut_short.decoded_moments == []
