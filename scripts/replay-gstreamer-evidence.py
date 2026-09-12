"""Replay exported native detections through live admission and tracking.

Inputs are a retained recording and --results-json from gstreamer-model-check.
No database, notifications, camera connection, or additional inference is used.
The clock is synthetic; this checks evidence contracts, not live latency.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survng.app.camera_capture import CapturedFrame
from survng.app.config import CameraConfig, DetectorConfig, ObjectTrackingConfig
from survng.app.live_detections import DetectionSnapshot
from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector, TimestampedLiveFrame
from survng.app.object_track.session import ObjectTrackingSession
from survng.app.object_track.types import TrackingFrame
from survng.app.tracking_frames import CameraFrameTimeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results_json.read_text())
    assert data["summary"]["eos"], "native replay did not complete"
    snapshots = [DetectionSnapshot.parse(item, session="replay") for item in data["results"]]
    assert snapshots and all(b.source_pts > a.source_pts for a, b in zip(snapshots, snapshots[1:]))
    camera = CameraConfig(id="replay", name="Replay", stream_url="rtsp://fixture.invalid/live")
    detector = SimpleNamespace(config=DetectorConfig(require_incident_zone=False), detect=Mock())
    reader = cv2.VideoCapture(str(args.video))
    assert reader.isOpened(), "recording unavailable"
    initial_results = []
    try:
        for snapshot in snapshots:
            # Pixels are used only to exercise grayscale provenance and shape.
            # Native inference PTS/boxes remain the authoritative evidence.
            reader.set(cv2.CAP_PROP_POS_MSEC, snapshot.source_pts * 1000)
            ok, bgr = reader.read()
            assert ok, f"cannot read snapshot near PTS {snapshot.source_pts}"
            gray = cv2.cvtColor(cv2.resize(bgr, (320, round(bgr.shape[0] * 320 / bgr.shape[1]))), cv2.COLOR_BGR2GRAY)
            now = time.time()
            sample = TimestampedLiveFrame(gray, now, time.monotonic(), snapshot.inference_sequence,
                1, 1, source_pts=snapshot.source_pts, source_session="replay",
                spatial_alignment={"reliable": True})
            backend = RecordedMotionObjectDetector(camera, detector, SimpleNamespace(), lambda: None,
                timestamped_live_frame_provider=lambda: sample,
                live_detections_provider=lambda _sample: snapshot)
            result = backend.detect_initial(datetime.fromtimestamp(now, timezone.utc))
            for obj in result.objects:
                if obj.get("label"):
                    assert obj["live_inference_sequence"] == snapshot.inference_sequence
                    assert obj["live_detection_session"] == "replay"
            initial_results.append((snapshot, gray, result))
    finally:
        reader.release()
    assert not detector.detect.called, "live admission performed duplicate inference"
    positives = [item for item in initial_results if any(obj.get("label") == "person"
                 and obj.get("incident_eligible") for obj in item[2].objects)]
    assert len(positives) >= 3, "expected person not admitted"
    reports = []
    for seed in (positives[0], positives[-3]):
        snap, gray, result = seed
        objects = [dict(obj) for obj in result.objects if obj.get("label") == "person" and obj.get("incident_eligible")]
        recorder = SimpleNamespace(ffmpeg_path="/usr/bin/ffmpeg", recording_rows_between=lambda *_a, **_kw: [])
        timeline = CameraFrameTimeline(camera=camera,
            capture=SimpleNamespace(matched_snapshot=lambda *_a, **_kw: None),
            recorder=recorder, stop_event=threading.Event(), sample_fps=lambda: 2)
        # The seed result arrives again on the following qualifier, as happens
        # with independent EMA/inference cadence. It must not count twice.
        selected = [(snap, gray, result), *[item for item in initial_results
                     if snap.source_pts < item[0].source_pts <= snap.source_pts + 10]]
        window_seconds = selected[-1][0].source_pts - snap.source_pts
        for index, (item, pixels, _result) in enumerate(selected):
            at = 1000 + item.source_pts - snap.source_pts + (0.5 if index == 0 else 0)
            frame = CapturedFrame("live", pixels, at, at, "", pixels.shape[1], pixels.shape[0],
                                  index + 1, 1, item.source_pts, "replay")
            timeline.live_frames.append(TrackingFrame(frame, item))
        updates = []
        session = ObjectTrackingSession(camera=camera,
            config=ObjectTrackingConfig(max_session_seconds=window_seconds, sample_fps=2, lost_timeout_seconds=2),
            detector=detector, frame_provider=lambda: None, catchup_frame_provider=timeline.read_recorded_frames,
            update_event=lambda _id, tracking, _objects: updates.append(tracking) or {},
            publisher=None, limiter=threading.BoundedSemaphore(1))
        appearances, covers = Mock(), Mock()
        session._annotate_appearances, session._consider_cover_candidate = appearances, covers
        session.set_accepting(True)
        try:
            assert session.start(1, datetime.fromtimestamp(1000, timezone.utc), objects,
                                 cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            assert session.wait_stopped(12)
        finally:
            session.stop()
        terminal = updates[-1]
        assert terminal["completion_reason"] in {"tracking_window_complete", "object_exited_recorded_window"}, terminal
        assert not appearances.called and not covers.called and not detector.detect.called
        reports.append({"seed_pts": snap.source_pts, "frames_processed": terminal["frames_processed"],
                        "reason": terminal["completion_reason"], "coverage_gap_count": terminal["coverage_gap_count"]})
    print(json.dumps({"native_frames_replayed": len(initial_results), "person_admissions": len(positives),
                      "duplicate_inferences": 0, "luma_color_enrichments": 0, "tracking": reports}))


if __name__ == "__main__":
    main()
