"""Capture provenance survives EMA, inference, persistence and deferred ReID."""
from datetime import datetime, timezone
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from survng.app.appearance_backfill import DeferredAppearanceBackfill
from survng.app.appearance_index import AppearanceIndex
from survng.app.camera import CameraWorker
from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.events import EventStore
from survng.app.motion_pipeline import MotionDecisionHandler
from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector
from tests.test_appearance_backfill import _Encoder
from tests.test_motion_analysis_service import _hooks, _service


@pytest.mark.parametrize("gray", [False, True])
@pytest.mark.parametrize("path", ["notice", "qualified", "fallback"])
def test_capture_to_saved_snapshot_preserves_color_authority(tmp_path, monkeypatch, gray, path):
    # Equal BGR channels are valid nighttime footage, so channel equality
    # cannot distinguish it from EMA-expanded luma.
    pixels = np.full((100, 200) if gray else (100, 200, 3), 77, np.uint8)
    analysis = _service(_hooks())
    analysis._preprocess_frame(pixels, 100, captured_at_monotonic=50,
                               capture_sequence=7, capture_generation=8,
                               lifecycle_generation=4, source_session="session")
    camera = CameraWorker.__new__(CameraWorker)
    camera.runtime_state = SimpleNamespace(lock=threading.Lock(), generation=4)
    camera._effective_spatial_alignment = {"reliable": True}
    camera.motion_analysis = analysis
    camera.tracking_frames = SimpleNamespace(captured=lambda _: SimpleNamespace(
        image=pixels, captured_at_epoch=100, captured_at_monotonic=50,
        sequence=7, generation=8, source="live", source_pts=1, source_session="session"))
    sample = camera._get_latest_detection_frame()
    assert sample.pixel_format == ("GRAY8" if gray else "BGR")
    detector = SimpleNamespace(
        config=SimpleNamespace(confidence_threshold=.5, require_incident_zone=False),
        detect=Mock(), detect_initial=Mock(return_value=[{
            "label": "car", "confidence": .9,
            "box": {"x1": 20, "y1": 10, "x2": 180, "y2": 90},
        }]),
    )
    detector.detect_refinement = Mock(return_value=detector.detect_initial.return_value)
    backend = RecordedMotionObjectDetector(
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        detector, SimpleNamespace(recording_at=lambda *_: None), lambda: None,
        timestamped_live_frame_provider=camera._get_latest_detection_frame,
        timestamped_evidence_frame_provider=camera._get_evidence_detection_frame,
    )
    monkeypatch.setattr("survng.app.motion_pipeline.object_detection.time.time", lambda: 100)
    evidence = {"evidence_frame_at_epoch": 100, "evidence_frame_sequence": 7,
                "evidence_capture_generation": 8, "evidence_lifecycle_generation": 4}
    if path == "fallback":
        monkeypatch.setattr("survng.app.motion_pipeline.object_detection.RECORDED_EVENT_SETTLE_SECONDS", 0)
        monkeypatch.setattr("survng.app.motion_pipeline.object_detection.RECORDED_EVENT_RETRY_SECONDS", 0)
        result = backend.detect(datetime.fromtimestamp(100, timezone.utc))
        assert result.objects[0]["frame_source"] == "live_fallback"
    else:
        result = backend.detect_initial(datetime.fromtimestamp(100, timezone.utc), evidence if path == "qualified" else None)
        assert result.refinement_pending
    assert result.objects[0]["frame_pixel_format"] == sample.pixel_format
    store = EventStore(tmp_path)

    def save(frame, _at):
        assert cv2.imwrite(str(tmp_path / "snapshot.png"), frame)
        return "snapshot.png"

    handler = MotionDecisionHandler(
        camera_id="gate", events=store, detection_provider=lambda _at: result,
        snapshot_writer=save, object_serializer=json.dumps,
    )
    outcome = handler.handle("manual", "fixture", datetime.fromtimestamp(100, timezone.utc), {})
    assert outcome.event_id is not None
    # Reopen the database: only persisted metadata may authorize this crop.
    restored = EventStore(tmp_path)
    event = restored.get(outcome.event_id)
    assert json.loads(event["objects_json"])[0]["frame_pixel_format"] == sample.pixel_format
    index = AppearanceIndex(tmp_path / "survng.sqlite3")
    encoder = Mock(wraps=_Encoder())
    backfill = DeferredAppearanceBackfill(
        tmp_path / "survng.sqlite3", tmp_path,
        ObjectTrackingConfig(vehicle_reid_enabled=True, vehicle_reid_model_path="vehicle.xml",
                             deferred_reid_min_crop_pixels=256),
        restored, index, encoder,
    )
    state, count, reason = backfill.process_event(outcome.event_id)
    assert (state, count) == (("deferred", 0) if gray else ("completed", 1)), reason
    assert encoder.embed_for_label.call_count == (0 if gray else 1)
