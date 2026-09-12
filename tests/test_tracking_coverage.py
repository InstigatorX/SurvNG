"""Recording chronology regressions from delayed and interrupted incident tracking."""
from datetime import datetime, timezone
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from survng.app.camera_capture import CapturedFrame

from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.object_tracking import ObjectTrackingSession
from survng.app.object_track.types import TrackingFrameBatch, TrackingFrame


def detection():
    return {"label": "person", "confidence": 0.9,
            "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 80}}


def run_session(provider, *, fps=2.0, seconds=15.0, detector=None, on_update=None,
                cap=2, cursor_aware=False, live_provider=None):
    seed = datetime.fromtimestamp(time.time() - 50.0, timezone.utc).timestamp()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    updates, requests = [], []

    def read(start, end, sample_fps, width, *, after_epoch=None):
        requests.append((start - seed, end - seed))
        if cursor_aware:
            return provider(start - seed, end - seed, seed, frame, after=after_epoch - seed)
        return provider(start - seed, end - seed, seed, frame)

    def update(_id, payload, _objects):
        updates.append(payload)
        if on_update:
            on_update(session, payload)
        return {}

    session = ObjectTrackingSession(
        camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        config=ObjectTrackingConfig(sample_fps=fps, max_session_seconds=seconds,
                                    max_catchup_frames_per_tick=cap, lost_timeout_seconds=4,
                                    persist_interval_seconds=0),
        detector=detector or SimpleNamespace(
            config=SimpleNamespace(confidence_threshold=0.7, require_incident_zone=False),
            detect=lambda *_a, **_k: [detection()],
        ),
        frame_provider=(lambda: live_provider(seed, frame)) if live_provider else
                       (lambda: (frame, time.time(), time.monotonic())),
        catchup_frame_provider=read, update_event=update, publisher=None,
        limiter=threading.BoundedSemaphore(1),
    )
    session.set_accepting(True)
    with (
        patch("survng.app.object_track.session.TRACKING_CATCHUP_SETTLE_SECONDS", 0.03),
        patch("survng.app.object_track.session.TRACKING_CATCHUP_RETRY_SECONDS", 0.01),
    ):
        assert session.start(1, datetime.fromtimestamp(seed, timezone.utc), [detection()], frame)
        try:
            assert session.wait_stopped(2)
        finally:
            session.stop()
    assert session.limiter.acquire(blocking=False), "capacity must be released"
    session.limiter.release()
    return updates, requests, seed


def frames_at(offsets):
    def provider(start, end, seed, frame):
        return [(seed + offset, frame) for offset in offsets if start - 1e-6 <= offset <= end + 1e-6]
    return provider


def test_fifty_second_handoff_walks_all_batches_within_media_horizon():
    updates, requests, seed = run_session(frames_at([i / 2 for i in range(1, 31)]))
    result = updates[-1]
    assert result["state"] == "complete"
    assert result["completion_reason"] == "tracking_window_complete"
    assert result["frames_processed"] == 30
    assert len(requests) == 15
    assert all(end - start <= 1.0 + 1e-6 and end <= 15 + 1e-6 for start, end in requests)
    assert result["tracks"][0]["observations"] == 31
    assert result["tracks"][0]["state"] == "confirmed"  # wall time must not age it out
    assert abs(datetime.fromisoformat(result["analyzed_through"]).timestamp() - seed - 15) < 0.001


def test_front_side_irregular_timestamps_are_valid_observations():
    offsets = [0.529, 0.826, 1.16, 1.519, 1.882, 2.162, 2.37, 2.669, 3.158]
    updates, _, _ = run_session(frames_at(offsets), fps=3, seconds=4)
    assert updates[-1]["frames_processed"] >= 5
    assert len(updates[-1]["tracks"]) == 1


def test_readable_prefix_boundary_is_not_lost_on_next_batch():
    calls = []
    def provider(start, end, seed, frame):
        calls.append(start)
        return TrackingFrameBatch(((seed + 0.5, frame),), seed + 0.5, "recorder_epoch_changed")
    updates, _, _ = run_session(provider)
    assert len(calls) == 1
    assert updates[-1]["state"] == "interrupted"
    assert updates[-1]["completion_reason"] == "recorder_epoch_changed"
    assert updates[-1]["frames_processed"] == 1


def test_missing_interior_frames_never_jump_to_newer_recording_or_live():
    updates, _, _ = run_session(frames_at([0.5, 6, 6.5, 7]))
    assert updates[-1]["frames_processed"] == 1
    assert updates[-1]["state"] == "interrupted"
    assert updates[-1]["coverage_incomplete"]
    assert updates[-1]["tracks"][0]["state"] == "confirmed"
    assert updates[-1]["completion_reason"] == "missing_media_while_object_active"


def test_deferred_inference_retries_same_frame_before_later_frames():
    seen = []
    def detect(frame, **kwargs):
        index = int(frame[0, 0, 0])
        seen.append(index)
        return [{"status": "inference_deferred"}] if len(seen) <= 5 else [detection()]
    detector = SimpleNamespace(config=SimpleNamespace(confidence_threshold=0.7, require_incident_zone=False), detect=detect)
    def provider(start, end, seed, frame):
        return [(seed + i / 2, np.full_like(frame, i)) for i in range(1, 7) if start - 1e-6 <= i / 2 <= end + 1e-6]
    updates, _, _ = run_session(provider, seconds=3, detector=detector)
    assert seen == [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6]
    assert updates[-1]["state"] == "complete"


def test_processing_budget_exhaustion_preserves_cursor_and_reports_incomplete():
    def exhaust(session, payload):
        if payload["frames_processed"] == 1:
            session._deadline = time.monotonic() - 1
    updates, _, seed = run_session(frames_at([0.5, 1, 1.5]), on_update=exhaust)
    assert updates[-1]["state"] == "interrupted"
    assert updates[-1]["completion_reason"] == "processing_budget_exhausted"
    assert updates[-1]["frames_processed"] == 1
    assert updates[-1]["tracks"][0]["state"] == "confirmed"
    assert abs(datetime.fromisoformat(updates[-1]["analyzed_through"]).timestamp() - seed - 0.5) < 0.001


def test_explicit_stop_does_not_claim_completion():
    def stop(session, payload):
        if payload["frames_processed"] == 1:
            session.request_stop()
    updates, _, _ = run_session(frames_at([0.5, 1, 1.5]), on_update=stop)
    assert updates[-1]["state"] == "interrupted"
    assert updates[-1]["coverage_incomplete"]
    assert updates[-1]["completion_reason"] == "session_stopped"


def test_subsample_tail_completes_without_claiming_extra_analyzed_frames():
    updates, _, seed = run_session(frames_at([0.5, 1, 1.5, 2, 2.5]), fps=3, seconds=3)
    assert updates[-1]["state"] == "complete"
    assert updates[-1]["frames_processed"] == 5
    assert abs(datetime.fromisoformat(updates[-1]["analyzed_through"]).timestamp() - seed - 2.5) < 0.001
    assert updates[-1]["updated_at"] == updates[-1]["analyzed_through"]


def test_empty_boundary_batch_preserves_specific_continuity_reason():
    def provider(start, end, seed, frame):
        return TrackingFrameBatch((), seed + start, "capture_generation_changed")
    updates, requests, _ = run_session(provider)
    assert len(requests) == 1
    assert updates[-1]["frames_processed"] == 0
    assert updates[-1]["state"] == "interrupted"
    assert updates[-1]["completion_reason"] == "capture_generation_changed"
    assert updates[-1]["coverage_interruption"] == "capture_generation_changed"


def buffered_timeline_provider(offsets, *, boundary_at=None):
    from unittest.mock import Mock
    from survng.app.tracking_frames import CameraFrameTimeline
    timeline = None

    def provider(start, end, seed, frame, *, after):
        nonlocal timeline
        if timeline is None:
            recorder = Mock()
            recorder.recording_rows_between.return_value = []
            timeline = CameraFrameTimeline(
                camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
                capture=Mock(), recorder=recorder, stop_event=threading.Event(), sample_fps=lambda: 2,
            )
            timeline.frames.extend((seed + offset, frame) for offset in offsets)
            if boundary_at is not None:
                # A live restart records a continuity boundary without erasing
                # the separately retained main frames used by this fixture.
                timeline.clear("live", captured_at=seed + boundary_at)
        return timeline.read_recorded_frames(seed + start, seed + end, 2, 100,
                                             after_epoch=seed + after)
    return provider


def test_single_frame_batch_reads_positive_duration_through_actual_timeline():
    updates, requests, _ = run_session(
        buffered_timeline_provider([0.5, 1, 1.5, 2, 2.5, 3]),
        seconds=3, cap=1, cursor_aware=True,
    )
    assert updates[-1]["state"] == "complete"
    assert updates[-1]["frames_processed"] == 6
    assert all(0 < end - start <= 0.5 + 1e-6 for start, end in requests if start < 3 - 1e-6)
    assert requests[0][1] > requests[0][0]


def test_boundary_between_cursor_and_next_sample_is_not_skipped():
    updates, _, _ = run_session(
        buffered_timeline_provider([0.5, 1, 1.5, 2, 2.5, 3], boundary_at=1.25),
        seconds=3, cap=1, cursor_aware=True,
    )
    assert updates[-1]["state"] == "interrupted"
    assert updates[-1]["frames_processed"] == 2
    assert updates[-1]["completion_reason"] == "capture_generation_changed"


def test_available_tail_frame_is_analyzed_before_applying_horizon_tolerance():
    updates, _, seed = run_session(frames_at([0.5, 1, 1.5, 2, 2.5, 3]), fps=3, seconds=3)
    assert updates[-1]["state"] == "complete"
    assert updates[-1]["frames_processed"] == 6
    assert abs(datetime.fromisoformat(updates[-1]["analyzed_through"]).timestamp() - seed - 3) < 0.001


def test_boundary_in_tolerated_tail_is_not_hidden_by_completion():
    updates, _, _ = run_session(
        buffered_timeline_provider([0.5, 1, 1.5, 2, 2.5], boundary_at=2.8),
        seconds=3, cursor_aware=True,
    )
    assert updates[-1]["state"] == "interrupted"
    assert updates[-1]["completion_reason"] == "capture_generation_changed"


@pytest.mark.parametrize("qualified", [False, True])
def test_deferred_live_inference_retries_retained_sample_beyond_media_settle_timeout(qualified):
    calls, live_reads = [], []
    def live(seed, frame):
        live_reads.append(True)
        # The live capture slot may disappear or advance after the first read.
        if len(live_reads) != 1:
            return None
        if qualified:
            return TrackingFrame(CapturedFrame("live", frame, seed + 0.5, 1.0, "", 100, 100, 7,
                generation=1, source_pts=.5, source_session="fixture"), requires_inference=True)
        return frame, seed + 0.5, 1.0
    def detect(frame, **kwargs):
        calls.append(True)
        return [{"status": "inference_deferred"}] if len(calls) <= 5 else [detection()]
    def stop_after_success(session, payload):
        if payload["frames_processed"]:
            session.request_stop()
    detector = SimpleNamespace(config=SimpleNamespace(confidence_threshold=0.7, require_incident_zone=False), detect=detect)
    updates, _, seed = run_session(frames_at([]), seconds=3, detector=detector,
                                   live_provider=live, on_update=stop_after_success)
    assert len(calls) == 6
    assert len(live_reads) == 1
    assert updates[-1]["frames_processed"] == 1
    assert updates[-1]["completion_reason"] == "session_stopped"
    assert abs(datetime.fromisoformat(updates[-1]["analyzed_through"]).timestamp() - seed - 0.5) < 0.001


def test_live_fallback_checks_boundary_beyond_decoded_batch():
    def stop_after_success(session, payload):
        if payload["frames_processed"]:
            session.request_stop()
    updates, requests, _ = run_session(
        buffered_timeline_provider([], boundary_at=1.25), seconds=3, cap=1,
        cursor_aware=True, live_provider=lambda seed, frame: (frame, seed + 1.5, 1),
        on_update=stop_after_success,
    )
    assert updates[-1]["frames_processed"] == 0
    assert updates[-1]["completion_reason"] == "capture_generation_changed"
    assert updates[-1]["state"] == "interrupted"
    assert requests == [(0.5, 1.0), (1.5, 1.5)]  # Continuity check does not expand decoding.


def test_cancellation_during_tail_read_cannot_claim_completion():
    for cancellation in ("stop", "deadline"):
        session_ref = []
        def remember(session, payload):
            if not session_ref:
                session_ref.append(session)
        def provider(start, end, seed, frame):
            if start > 2.7:
                if cancellation == "stop":
                    session_ref[0].request_stop()
                else:
                    session_ref[0]._deadline = time.monotonic() - 1
            return [(seed + offset, frame) for offset in [0.5, 1, 1.5, 2, 2.5, 3]
                    if start - 1e-6 <= offset <= end + 1e-6]
        updates, _, _ = run_session(provider, fps=3, seconds=3, cap=1, on_update=remember)
        assert updates[-1]["frames_processed"] == 5
        assert updates[-1]["state"] == "interrupted"
        assert updates[-1]["completion_reason"] == (
            "session_stopped" if cancellation == "stop" else "processing_budget_exhausted"
        )


def test_cancellation_during_empty_live_acquisition_cannot_complete_tail():
    for cancellation in ("stop", "deadline"):
        session_ref = []
        def remember(session, payload):
            if not session_ref:
                session_ref.append(session)
        def live(seed, frame):
            if cancellation == "stop":
                session_ref[0].request_stop()
            else:
                session_ref[0]._deadline = time.monotonic() - 1
            return None
        updates, _, _ = run_session(frames_at([0.5, 1, 1.5, 2, 2.5]), fps=3,
                                     seconds=3, cap=1, on_update=remember, live_provider=live)
        assert updates[-1]["frames_processed"] == 5
        assert updates[-1]["state"] == "interrupted"
        assert updates[-1]["completion_reason"] == (
            "session_stopped" if cancellation == "stop" else "processing_budget_exhausted"
        )


def test_known_boundary_at_horizon_takes_priority_over_deadline_completion():
    def expire(session, payload):
        if payload["frames_processed"] == 6:
            session._deadline = time.monotonic() - 1
    updates, _, _ = run_session(
        buffered_timeline_provider([0.5, 1, 1.5, 2, 2.5, 3], boundary_at=3),
        seconds=3, cursor_aware=True, on_update=expire,
    )
    assert updates[-1]["state"] == "interrupted"
    assert updates[-1]["coverage_incomplete"]
    assert updates[-1]["completion_reason"] == "capture_generation_changed"
