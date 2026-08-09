from __future__ import annotations

import tempfile
import subprocess
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import cv2

from survng.app.camera import CameraWorker
from survng.app.camera_lifecycle import CAPTURE_STOP_TIMEOUT_SECONDS
from survng.app.camera_capture import (
    CAPTURE_OPEN_CONCURRENCY,
    CAPTURE_OPEN_TIMEOUT_MS,
    CAPTURE_READ_TIMEOUT_MS,
    FRAME_STALE_SECONDS,
    CapturedFrame,
)
from survng.app.config import CameraConfig, ImageStorageConfig, MotionQualificationConfig, ObjectTrackingConfig
from survng.app.detector import objects_to_json
from survng.app.image_storage import DurableImageWriter
from survng.app.motion import MotionQualificationResult
from survng.app.motion_analysis import FairMotionAnalysisLimiter
from survng.app.motion_events import MotionTrigger, MotionTriggerBatch
from survng.app.motion_decisions import (
    is_borderline_candidate,
    should_verify_suppression,
)
from survng.app.object_tracking import ObjectTrackingSessionFactory
from survng.app.motion_pipeline import (
    EVIDENCE_REPOSITORY_SERVICE,
    MotionDecisionHandlerFactory,
    MotionEvidenceRepository,
    MotionPipelineFactory,
    MotionStageDependencies,
    RecordedMotionObjectDetectorFactory,
    build_builtin_motion_registry,
    resolve_motion_pipeline_graphs,
)


class StubIncidentOutcome:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def as_dict(self) -> dict[str, object]:
        return dict(self.values)


def incident_outcome(**values: object) -> StubIncidentOutcome:
    return StubIncidentOutcome(values)


class DummyDetector:
    def __init__(self) -> None:
        self.calls = 0
        self.config = SimpleNamespace(confidence_threshold=0.5)

    def detect(self, frame, confidence_threshold=None):
        self.calls += 1
        return [
            {
                "label": "car",
                "confidence": 0.8,
                "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            }
        ]


class DummyEvents:
    def add_motion_audit(self, **payload):
        return {"id": 1, **payload}


class DummyRecorder:
    ffmpeg_path = "ffmpeg"
    hardware_acceleration = "none"

    def recording_at(self, camera_id: str, epoch: float):
        return None

    def recording_rows_between(self, camera_id, start_epoch, end_epoch, source="main"):
        return []


def make_worker(
    camera: CameraConfig,
    storage_dir: Path,
    detector=None,
    events=None,
    recorder=None,
    motion_config: MotionQualificationConfig | None = None,
    event_callback=None,
) -> CameraWorker:
    event_store = events or DummyEvents()
    detector_backend = detector or DummyDetector()
    recording_provider = recorder or DummyRecorder()
    effective_config = motion_config or MotionQualificationConfig()
    override = camera.motion_qualification
    graphs = resolve_motion_pipeline_graphs(effective_config, override)
    evidence = MotionEvidenceRepository(camera.id)
    factory = MotionPipelineFactory(
        build_builtin_motion_registry(),
        dependencies=MotionStageDependencies(
            services={EVIDENCE_REPOSITORY_SERVICE: evidence},
        ),
    )
    worker = CameraWorker(
        camera,
        storage_dir,
        motion_config,
        event_callback,
        motion_pipeline=factory.create(
            camera.id,
            graphs.qualification,
            required_artifacts={"scoring"},
        ),
        motion_observation_pipeline=factory.create(
            camera.id,
            graphs.observation,
            required_artifacts={"source_evidence"},
        ),
        motion_fusion_pipeline=factory.create(
            camera.id,
            graphs.fusion,
            initial_artifacts={"scoring"},
            required_artifacts={"scoring", "decision"},
        ),
        motion_evidence=evidence,
        motion_pipeline_origins=graphs.origins,
        motion_decision_handler_factory=MotionDecisionHandlerFactory(
            events=event_store,
            object_serializer=objects_to_json,
        ),
        motion_object_detector_factory=RecordedMotionObjectDetectorFactory(
            detector=detector_backend,
            recorder=recording_provider,
        ),
        object_tracking_session_factory=ObjectTrackingSessionFactory(
            config=ObjectTrackingConfig(enabled=False),
            detector=detector_backend,
            update_event=lambda _event_id, _tracking, _objects: None,
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
        ),
        motion_analysis_limiter=FairMotionAnalysisLimiter(2),
        image_writer=DurableImageWriter(ImageStorageConfig()),
    )
    # Most worker unit tests exercise motion ingress without starting camera
    # I/O. Production workers now begin in the explicit stopped/non-accepting
    # lifecycle phase, so opt this test fixture into ingress unless a test
    # deliberately disables it.
    worker.runtime_state.accepting_motion_events = True
    return worker


def prime_visual_backup_scene(worker: CameraWorker, started_at: float) -> None:
    stable = MotionQualificationResult(
        False,
        0.0,
        0.48,
        "no_motion_blobs",
        4,
        {},
    )
    warmup_end = started_at + worker.motion_config.visual_backup_warmup_seconds
    for captured_at in (started_at, warmup_end, warmup_end + 0.75, warmup_end + 1.5):
        worker.motion_analysis.visual_backup_readiness(stable, captured_at)
    assert worker.motion_analysis.visual_backup_scene_ready


def seed_capture_frame(
    worker: CameraWorker,
    source: str,
    frame: np.ndarray,
    captured_at: float,
    frame_clock: float,
) -> None:
    with worker.capture._lock:
        worker.capture._stop.clear()
        worker.capture._sequence += 1
        worker.capture._frames[source] = CapturedFrame(
            source=source,
            image=frame.copy(),
            captured_at_epoch=captured_at,
            captured_at_monotonic=frame_clock,
            captured_at_iso=datetime.fromtimestamp(
                captured_at, timezone.utc
            ).isoformat(),
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            sequence=worker.capture._sequence,
        )
        worker.capture._dimensions[source] = {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }


class CameraWorkerTest(unittest.TestCase):
    def test_scene_context_policy_uses_camera_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                CameraConfig(
                    id="gate",
                    name="Gate",
                    stream_url="rtsp://camera/main",
                    object_activity_attribution="shadow",
                ),
                Path(tmpdir),
            )

        self.assertEqual(
            worker.motion_decision_handler.activity_status()["mode"],
            "shadow",
        )

    def test_illumination_filter_uses_global_and_camera_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            inherited = make_worker(
                CameraConfig(
                    id="gate",
                    name="Gate",
                    stream_url="rtsp://camera/main",
                ),
                Path(tmpdir),
                motion_config=MotionQualificationConfig(
                    illumination_filter_enabled=True,
                ),
            )
            overridden = make_worker(
                CameraConfig.model_validate({
                    "id": "boiler",
                    "name": "Boiler",
                    "stream_url": "rtsp://camera/main",
                    "motion_qualification": {
                        "illumination_filter_enabled": False,
                    },
                }),
                Path(tmpdir),
                motion_config=MotionQualificationConfig(
                    illumination_filter_enabled=True,
                ),
            )

        self.assertTrue(inherited.motion_qualification.illumination_filter_enabled())
        self.assertFalse(overridden.motion_qualification.illumination_filter_enabled())

    def test_visual_backup_settings_use_camera_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                CameraConfig(
                    id="gate",
                    name="Gate",
                    stream_url="rtsp://camera/main",
                    motion_qualification={
                        "visual_backup_grace_seconds": 2.0,
                        "visual_backup_min_score": 0.75,
                        "visual_backup_min_consecutive": 4,
                        "visual_backup_cooldown_seconds": 30.0,
                        "visual_backup_max_triggers_5m": 2,
                    },
                ),
                Path(tmpdir),
            )

            self.assertEqual(worker.motion_qualification.visual_backup_settings(), {
                "grace_seconds": 2.0,
                "minimum_score": 0.75,
                "minimum_consecutive": 4,
                "cooldown_seconds": 30.0,
                "maximum_triggers_5m": 2,
            })

    def test_tracking_pause_and_resume_preserve_camera_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main"),
                Path(tmpdir),
            )
            tracking = Mock()
            tracking.stop.return_value = True
            worker.tracking_lifecycle.bind_for_compatibility(tracking)
            worker._stop.clear()

            worker.pause_object_tracking_session()
            worker.resume_object_tracking_session()

            tracking.stop.assert_called_once_with()
            tracking.set_accepting.assert_called_once_with(True)
            self.assertFalse(worker._stop.is_set())

    def test_detection_toggle_restores_previous_state_when_tracking_sync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main"),
                Path(tmpdir),
            )
            with patch.object(
                worker.tracking_lifecycle,
                "sync_accepting",
                side_effect=[RuntimeError("sync failed"), None],
            ) as sync:
                with self.assertRaisesRegex(RuntimeError, "sync failed"):
                    worker.set_detection_enabled(False)

            self.assertTrue(worker.runtime_state.detection_enabled)
            self.assertEqual(sync.call_count, 2)

    def test_tracking_session_swap_preserves_camera_and_resizes_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main"),
                Path(tmpdir),
            )
            previous = Mock()
            previous.stop.return_value = True
            replacement = Mock()
            replacement.config = ObjectTrackingConfig(sample_fps=3.0)
            worker.tracking_lifecycle.bind_for_compatibility(previous)
            worker._stop.clear()
            worker.tracking_frames.frames.extend([
                (1.0, np.zeros((2, 2, 3), dtype=np.uint8)),
                (2.0, np.zeros((2, 2, 3), dtype=np.uint8)),
            ])

            returned = worker.replace_object_tracking_session(replacement)

            self.assertIs(returned, previous)
            self.assertIs(worker.tracking_lifecycle.current(), replacement)
            previous.stop.assert_called_once_with()
            replacement.set_accepting.assert_called_once_with(True)
            self.assertEqual(worker.tracking_frames.frames.maxlen, 32)
            self.assertEqual(len(worker.tracking_frames.frames), 2)
            self.assertFalse(worker._stop.is_set())

    def test_tracking_session_swap_refuses_to_replace_busy_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main"),
                Path(tmpdir),
            )
            previous = Mock()
            previous.stop.return_value = False
            replacement = Mock()
            worker.tracking_lifecycle.bind_for_compatibility(previous)

            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                worker.replace_object_tracking_session(replacement)

            self.assertIs(worker.tracking_lifecycle.current(), previous)
            replacement.set_accepting.assert_not_called()

    def test_tracking_session_swap_restores_pointer_when_replacement_cleanup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main"),
                Path(tmpdir),
            )
            previous = Mock()
            previous.stop.return_value = True
            replacement = Mock()
            replacement.config = ObjectTrackingConfig(sample_fps=2.0)
            replacement.set_accepting.side_effect = RuntimeError("accept failed")
            replacement.stop.side_effect = RuntimeError("cleanup failed")
            worker.tracking_lifecycle.bind_for_compatibility(previous)
            worker._stop.clear()

            with self.assertRaisesRegex(RuntimeError, "accept failed"):
                worker.replace_object_tracking_session(replacement)

            self.assertIs(worker.tracking_lifecycle.current(), previous)
            previous.set_accepting.assert_called_once_with(True)

    def test_status_advertises_stable_dimensions_for_each_captured_source(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            seed_capture_frame(
                worker,
                "live",
                np.zeros((672, 896, 3), dtype=np.uint8),
                time.time(),
                time.monotonic(),
            )
            seed_capture_frame(
                worker,
                "main",
                np.zeros((1080, 1920, 3), dtype=np.uint8),
                time.time(),
                time.monotonic(),
            )

            dimensions = worker.status()["stream_dimensions"]

        self.assertEqual(dimensions["live"], {"width": 896, "height": 672})
        self.assertEqual(dimensions["main"], {"width": 1920, "height": 1080})

    def test_main_stream_buffer_bridges_unfinalized_recording_gap(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            worker.tracking_lifecycle.current().config = ObjectTrackingConfig(sample_fps=2.0)
            first = np.full((900, 1600, 3), 1, dtype=np.uint8)
            second = np.full((900, 1600, 3), 2, dtype=np.uint8)
            worker._remember_tracking_frame(first, 100.0)
            worker._remember_tracking_frame(second, 100.5)

            samples = list(worker._recorded_tracking_frames(99.9, 101.0, 2.0, 1280))

        self.assertEqual([sample[0] for sample in samples], [100.0, 100.5])
        self.assertEqual(samples[0][1].shape, (360, 640, 3))
        self.assertEqual(int(samples[0][1][0, 0, 0]), 1)
        self.assertEqual(int(samples[1][1][0, 0, 0]), 2)

    def test_main_stream_buffer_respects_tracking_sample_rate(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker.tracking_lifecycle.current().config = ObjectTrackingConfig(sample_fps=2.0)
            frame = np.zeros((90, 160, 3), dtype=np.uint8)
            worker._remember_tracking_frame(frame, 100.0)
            worker._remember_tracking_frame(frame, 100.1)
            worker._remember_tracking_frame(frame, 100.5)

        self.assertEqual([sample[0] for sample in worker.tracking_frames.frames], [100.0, 100.5])

    def test_capture_observer_preserves_timestamp_for_motion_and_tracking(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="adaptive", sample_fps=2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            worker.motion_runtime.start(worker._stop)
            live = CapturedFrame(
                source="live",
                image=np.zeros((90, 160, 3), dtype=np.uint8),
                captured_at_epoch=100.25,
                captured_at_monotonic=50.25,
                captured_at_iso="1970-01-01T00:01:40.250000+00:00",
                width=160,
                height=90,
                sequence=1,
            )
            main = CapturedFrame(
                source="main",
                image=np.zeros((90, 160, 3), dtype=np.uint8),
                captured_at_epoch=100.5,
                captured_at_monotonic=50.5,
                captured_at_iso="1970-01-01T00:01:40.500000+00:00",
                width=160,
                height=90,
                sequence=2,
            )

            worker._capture_frame(live)
            worker._capture_frame(main)

            deadline = time.monotonic() + 1
            while not worker.motion_analysis.frames and time.monotonic() < deadline:
                time.sleep(0.01)
            motion_timestamp = worker.motion_analysis.frames[-1][0]
            tracking_timestamp = worker.tracking_frames.frames[-1][0]
            worker._stop.set()
            worker.motion_runtime.request_stop()
            self.assertTrue(worker.motion_runtime.wait_stopped(
                analysis_timeout=1.0,
                decision_timeout=1.0,
            ))

        self.assertEqual(motion_timestamp, 100.25)
        self.assertEqual(tracking_timestamp, 100.5)

    def test_tracking_frame_includes_capture_time_and_rejects_stale_cache(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            frame = np.zeros((20, 30, 3), dtype=np.uint8)
            captured_at = time.time() - 0.1
            frame_clock = time.monotonic() - 0.1
            seed_capture_frame(worker, "main", frame, captured_at, frame_clock)
            with patch.object(
                worker.capture,
                "ensure_source",
                return_value=SimpleNamespace(running=True),
            ):
                sample = worker._get_latest_tracking_frame("main")
                with worker.capture._lock:
                    cached = worker.capture._frames["main"]
                    worker.capture._frames["main"] = CapturedFrame(
                        source=cached.source,
                        image=cached.image,
                        captured_at_epoch=cached.captured_at_epoch,
                        captured_at_monotonic=(
                            time.monotonic() - FRAME_STALE_SECONDS - 1.0
                        ),
                        captured_at_iso=cached.captured_at_iso,
                        width=cached.width,
                        height=cached.height,
                        sequence=cached.sequence,
                    )
                stale = worker._get_latest_tracking_frame("main")

        self.assertIsNotNone(sample)
        sampled_frame, sampled_at, sampled_clock = sample
        self.assertIsNot(sampled_frame, frame)
        self.assertEqual(sampled_at, captured_at)
        self.assertEqual(sampled_clock, frame_clock)
        self.assertIsNone(stale)

    def test_tracking_frame_falls_back_to_live_until_main_is_ready(self) -> None:
        camera = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            live_frame = np.full((20, 30, 3), 7, dtype=np.uint8)
            live_epoch = time.time() - 0.1
            live_clock = time.monotonic() - 0.1
            seed_capture_frame(worker, "live", live_frame, live_epoch, live_clock)
            with patch.object(
                worker.capture,
                "ensure_source",
                return_value=SimpleNamespace(running=True),
            ) as ensure_source:
                sample = worker._get_latest_tracking_frame_with_fallback()

        self.assertIsNotNone(sample)
        sampled_frame, sampled_at, sampled_clock = sample
        self.assertEqual(int(sampled_frame[0, 0, 0]), 7)
        self.assertEqual(sampled_at, live_epoch)
        self.assertEqual(sampled_clock, live_clock)
        self.assertEqual(
            [call.args[0] for call in ensure_source.call_args_list],
            ["main", "live"],
        )

    def test_tracking_frame_switches_to_main_when_main_is_ready(self) -> None:
        camera = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            now_epoch = time.time() - 0.1
            now_clock = time.monotonic() - 0.1
            seed_capture_frame(
                worker,
                "main",
                np.full((40, 60, 3), 9, dtype=np.uint8),
                now_epoch,
                now_clock,
            )
            seed_capture_frame(
                worker,
                "live",
                np.full((20, 30, 3), 7, dtype=np.uint8),
                now_epoch,
                now_clock,
            )
            with patch.object(
                worker.capture,
                "ensure_source",
                return_value=SimpleNamespace(running=True),
            ) as ensure_source:
                sample = worker._get_latest_tracking_frame_with_fallback()

        self.assertIsNotNone(sample)
        sampled_frame, _, _ = sample
        self.assertEqual(sampled_frame.shape, (40, 60, 3))
        self.assertEqual(int(sampled_frame[0, 0, 0]), 9)
        ensure_source.assert_called_once_with("main")

    def test_snapshot_filename_uses_event_time_not_processing_time(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        event_at = datetime(2026, 7, 27, 15, 56, 55, 123456, tzinfo=timezone.utc)
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            path = Path(worker._write_snapshot(frame, event_at))

            self.assertTrue(path.is_file())
            self.assertEqual(path.suffix, ".webp")
            self.assertTrue(path.name.startswith("20260727-155655-123456-"))

    def test_rejected_sample_keeps_valid_path_when_pruning_races(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        result = MotionQualificationResult(
            False,
            0.4,
            0.5,
            "low_score",
            2,
            {},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with (
                patch.object(
                    worker,
                    "_get_latest_frame",
                    return_value=np.zeros((20, 30, 3), dtype=np.uint8),
                ),
                patch.object(
                    worker.media.image_writer,
                    "stored_images",
                    side_effect=OSError("concurrent cleanup"),
                ),
            ):
                stored = worker.media.sample_rejected_motion(
                    datetime(2026, 8, 3, tzinfo=timezone.utc),
                    result,
                )

            self.assertTrue(stored)
            self.assertTrue(Path(stored).is_file())

    def test_isolated_qualification_telemetry_reports_replay_stage_metrics(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        frames = [
            np.zeros((90, 160), dtype=np.uint8),
            np.full((90, 160), 20, dtype=np.uint8),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))

            result = worker.motion_qualification.run_pipeline(
                frames,
                "balanced",
                2.0,
                [1.0, 2.0],
                isolated=True,
                capture_debug=False,
            )

        stage_metrics = result.telemetry["graphs"]["qualification"]["stage_metrics"]
        self.assertEqual(stage_metrics["preprocess"]["calls"], 1)
        self.assertEqual(worker.motion_pipeline.status()["stages"]["preprocess"]["calls"], 0)

    def test_capture_limits_ffmpeg_decoder_threads(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        capture = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()

            def stop_after_open(*_args):
                worker.capture._stop.set()
                return False

            capture.open.side_effect = stop_after_open
            with worker.capture._lock:
                worker.capture._stop.clear()
            with patch("survng.app.camera_capture.cv2.VideoCapture", return_value=capture):
                worker.capture._run_source("live", threading.Event())

        _, backend, options = capture.open.call_args.args
        self.assertEqual(backend, cv2.CAP_FFMPEG)
        self.assertEqual(options[options.index(cv2.CAP_PROP_N_THREADS) + 1], 1)
        self.assertEqual(
            options[options.index(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC) + 1],
            CAPTURE_OPEN_TIMEOUT_MS,
        )
        self.assertEqual(
            options[options.index(cv2.CAP_PROP_READ_TIMEOUT_MSEC) + 1],
            CAPTURE_READ_TIMEOUT_MS,
        )
        capture.release.assert_called_once()

    def test_capture_io_deadlines_fit_inside_shutdown_budget(self) -> None:
        shutdown_budget_ms = CAPTURE_STOP_TIMEOUT_SECONDS * 1000

        self.assertLess(CAPTURE_OPEN_TIMEOUT_MS, shutdown_budget_ms)
        self.assertLess(CAPTURE_READ_TIMEOUT_MS, shutdown_budget_ms)
        self.assertLess(
            CAPTURE_OPEN_TIMEOUT_MS * CAPTURE_OPEN_CONCURRENCY,
            shutdown_budget_ms,
        )

    def test_capture_waiting_for_global_open_slot_cancels_on_stop(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        stop_event = threading.Event()
        capture = Mock()
        result: list[bool] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            limiter = worker.capture.backend.limiter
            for _ in range(CAPTURE_OPEN_CONCURRENCY):
                limiter.acquire(timeout=0.1)
            try:
                handle = worker.capture.backend.create_handle()
                thread = threading.Thread(
                    target=lambda: result.append(
                        worker.capture.backend.open(
                            handle,
                            camera.source_url("live"),
                            lambda: stop_event.is_set(),
                        )
                    )
                )
                thread.start()
                time.sleep(0.02)
                stop_event.set()
                thread.join(timeout=1)
            finally:
                for _ in range(CAPTURE_OPEN_CONCURRENCY):
                    limiter.release()

        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [False])
        capture.open.assert_not_called()

    def test_borderline_candidate_is_rescued_by_eligible_object(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        events = Mock()
        events.add_event.return_value = {"id": 42}
        qualification = {
            "borderline_candidate": True,
            "effective_accepted": True,
            "would_suppress": True,
        }
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        objects = [{"label": "dog", "confidence": 0.8, "incident_eligible": True}]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), events=events)
            with (
                patch.object(worker, "_recorded_motion_frame", return_value=(frame, objects, "recording.mp4")),
                patch.object(worker, "_write_snapshot", return_value="snapshot.jpg"),
            ):
                outcome = worker.motion_incidents.process(
                    "motion", "message", datetime.now(timezone.utc), qualification
                ).as_dict()

        self.assertTrue(outcome["object_detected"])
        self.assertTrue(qualification["rescued_by_object"])
        self.assertTrue(qualification["effective_accepted"])
        self.assertFalse(qualification["would_suppress"])

    def test_tracking_seeds_timing_and_dimensions_from_the_event_snapshot(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        events = Mock()
        events.add_event.return_value = {"id": 42}
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        seed = np.ones((180, 320, 3), dtype=np.uint8)
        objects = [{"label": "person", "confidence": 0.9, "incident_eligible": True}]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), events=events)
            worker.tracking_lifecycle.current().config = ObjectTrackingConfig(
                enabled=True,
                reid_enabled=False,
            )
            with (
                patch.object(
                    worker,
                    "_recorded_motion_frame",
                    return_value=(frame, objects, "recording.mp4"),
                ),
                patch.object(worker, "_write_snapshot", return_value="snapshot.jpg"),
                patch("survng.app.camera_media.cv2.imread", return_value=seed) as imread,
                patch.object(worker, "_get_latest_tracking_frame", return_value=None) as prewarm,
                patch.object(worker.tracking_lifecycle.current(), "start", return_value=True) as start,
            ):
                worker.motion_incidents.process(
                    "motion",
                    "message",
                    datetime.now(timezone.utc),
                    {"effective_accepted": True},
                )

        imread.assert_called_once_with("snapshot.jpg")
        prewarm.assert_called_once_with("main")
        self.assertIs(start.call_args.args[3], seed)

    def test_borderline_candidate_without_object_remains_suppressed(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        events = Mock()
        events.add_event.return_value = {"id": 43}
        qualification = {
            "borderline_candidate": True,
            "effective_accepted": True,
            "would_suppress": False,
        }
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), events=events)
            with (
                patch.object(worker, "_recorded_motion_frame", return_value=(frame, [], "recording.mp4")),
                patch.object(worker, "_write_snapshot", return_value="snapshot.jpg"),
            ):
                outcome = worker.motion_incidents.process(
                    "motion", "message", datetime.now(timezone.utc), qualification
                ).as_dict()

        self.assertFalse(outcome["object_detected"])
        self.assertFalse(qualification["rescued_by_object"])
        self.assertFalse(qualification["effective_accepted"])
        self.assertTrue(qualification["would_suppress"])

    def test_motion_ring_uses_per_camera_analysis_width(self) -> None:
        camera = CameraConfig.model_validate({
            "id": "back-middle",
            "name": "Back Middle",
            "stream_url": "rtsp://example.invalid/main",
            "motion_qualification": {"frame_width": 480},
        })
        config = MotionQualificationConfig(frame_width=320, sample_fps=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            thread = worker.motion_analysis.thread = threading.Thread(
                target=worker.motion_analysis.run, args=(worker._stop,)
            )
            thread.start()
            worker.motion_analysis.remember_frame(
                np.zeros((720, 1280, 3), dtype=np.uint8),
                time.monotonic(),
                worker._stop,
            )
            deadline = time.monotonic() + 1
            while worker.motion_state._stats["continuous_frames"] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertEqual(worker.motion_qualification.settings(), ("camera", "balanced", 480))
            self.assertEqual(worker.status()["motion_qualification"]["frame_width"], 480)
            self.assertEqual(worker.status()["motion_qualification"]["frame_shape"], [270, 480])
            self.assertEqual(
                worker.status()["motion_qualification"]["color_frame_shape"],
                [270, 480, 3],
            )
            self.assertEqual(worker.motion_analysis.frames[-1][1].ndim, 2)
            self.assertEqual(worker.motion_analysis.color_frames[-1][1].ndim, 3)
            self.assertFalse(worker.status()["motion_qualification"]["mog2_audit_enabled"])
            self.assertIsNone(worker.status()["motion_qualification"]["mog2_last"])
            worker._stop.set()
            worker.motion_analysis.request_stop()
            thread.join(timeout=1)

    def test_no_visual_validators_skip_frame_motion_work(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig.model_validate({
            "mode": "camera",
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": "bypass", "sources": []},
                    },
                    {"stage_id": "event_state", "implementation": "score_event_state"},
                    {"stage_id": "trigger", "implementation": "score_trigger"},
                ],
            },
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)

            worker.motion_analysis.remember_frame(
                np.zeros((180, 320, 3), dtype=np.uint8),
                time.monotonic(),
                worker._stop,
            )

            self.assertEqual(len(worker.motion_analysis.frames), 0)
            self.assertTrue(worker.motion_analysis.queue.empty())

    def test_continuous_adaptive_transition_enqueues_prequalified_trigger(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        accepted = MotionQualificationResult(
            True, 0.8, 0.5, "qualified", 2, {}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(mode="enforce"),
            )
            worker._stop.clear()
            thread = worker.motion_analysis.thread = threading.Thread(
                target=worker.motion_analysis.run, args=(worker._stop,)
            )
            thread.start()
            with (
                patch.object(worker.motion_qualification, "run_pipeline", return_value=accepted) as analyze,
                patch.object(worker.motion_qualification, "with_source_evidence", return_value=accepted),
            ):
                worker.motion_analysis.remember_frame(
                    np.zeros((180, 320, 3), dtype=np.uint8), 1.0, worker._stop
                )
                worker.motion_analysis.remember_frame(
                    np.zeros((180, 320, 3), dtype=np.uint8), 2.0, worker._stop
                )
                deadline = time.monotonic() + 1
                while worker.motion_events.queue.empty() and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertGreaterEqual(analyze.call_count, 1)
            trigger = worker.motion_events.queue.get_nowait()
            self.assertEqual(trigger["topic"], "adaptive/motion")
            self.assertIs(trigger["prequalified"], accepted)
            worker._stop.set()
            worker.motion_analysis.request_stop()
            thread.join(timeout=1)

    def test_visual_backup_requires_persistent_strong_motion_without_onvif(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera_rescue",
            camera_mode_background_fps=2,
            visual_backup_warmup_seconds=0,
            visual_backup_grace_seconds=1.0,
            visual_backup_min_score=0.7,
            visual_backup_score_margin=0.1,
            visual_backup_min_consecutive=3,
            visual_backup_cooldown_seconds=5,
        )
        accepted = MotionQualificationResult(True, 0.82, 0.5, "qualified", 4, {})
        samples = [(100.0, np.zeros((90, 160), dtype=np.uint8))]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            prime_visual_backup_scene(worker, 90.0)
            with patch.object(worker.motion_qualification, "with_source_evidence", return_value=accepted):
                worker.motion_analysis.consider_visual_backup(accepted, samples, 100.0)
                worker.motion_analysis.consider_visual_backup(accepted, samples, 100.5)
                self.assertTrue(worker.motion_events.queue.empty())
                worker.motion_analysis.consider_visual_backup(accepted, samples, 101.0)

            trigger = worker.motion_events.queue.get_nowait()

        self.assertEqual(trigger["topic"], "adaptive/visual_backup")
        self.assertTrue(trigger["prequalified"].features["visual_backup"])

    def test_visual_backup_counts_rate_limited_promotion_without_audit(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera_rescue",
            visual_backup_warmup_seconds=0,
            visual_backup_grace_seconds=0,
            visual_backup_min_consecutive=2,
            visual_backup_cooldown_seconds=20,
        )
        accepted = MotionQualificationResult(True, 0.9, 0.48, "qualified", 4, {})
        samples = [(100.0, np.zeros((90, 160), dtype=np.uint8))]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            prime_visual_backup_scene(worker, 90.0)
            worker.motion_events.adaptive_trigger_pending = True
            worker.motion_analysis.consider_visual_backup(accepted, samples, 100.0)
            worker.motion_analysis.consider_visual_backup(accepted, samples, 100.5)
        self.assertEqual(worker.motion_state._stats["visual_backup_not_promoted"], 1)

    def test_visual_backup_yields_to_recent_onvif_notice(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera_rescue",
            camera_mode_background_fps=2,
            visual_backup_warmup_seconds=0,
            visual_backup_grace_seconds=1.0,
            visual_backup_min_consecutive=3,
            visual_backup_cooldown_seconds=5,
        )
        accepted = MotionQualificationResult(True, 0.82, 0.5, "qualified", 4, {})
        samples = [(100.0, np.zeros((90, 160), dtype=np.uint8))]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            prime_visual_backup_scene(worker, 90.0)
            worker.motion_events.camera_motion_times.append(100.75)
            for captured_at in (100.0, 100.5, 101.0):
                worker.motion_analysis.consider_visual_backup(accepted, samples, captured_at)
            for captured_at in (102.0, 102.5, 103.0):
                worker.motion_analysis.consider_visual_backup(accepted, samples, captured_at)

        self.assertTrue(worker.motion_events.queue.empty())
        self.assertEqual(worker.motion_state._stats["visual_backup_onvif_matches"], 1)
        self.assertEqual(worker.motion_state._stats["visual_backup_candidates"], 6)

    def test_visual_backup_waits_for_background_learning_after_startup(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera_rescue",
            camera_mode_background_fps=2,
            visual_backup_warmup_seconds=10,
            visual_backup_grace_seconds=0,
            visual_backup_min_consecutive=2,
        )
        accepted = MotionQualificationResult(True, 0.82, 0.5, "qualified", 4, {})
        samples = [(100.0, np.zeros((90, 160), dtype=np.uint8))]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            with patch.object(worker.motion_qualification, "with_source_evidence", return_value=accepted):
                for captured_at in (100.0, 100.5, 109.5):
                    worker.motion_analysis.consider_visual_backup(accepted, samples, captured_at)
                self.assertTrue(worker.motion_events.queue.empty())
                stable = MotionQualificationResult(False, 0.0, 0.48, "no_motion_blobs", 4, {})
                for captured_at in (110.0, 110.75, 111.5):
                    worker.motion_analysis.consider_visual_backup(stable, samples, captured_at)
                worker.motion_analysis.consider_visual_backup(accepted, samples, 112.0)
                worker.motion_analysis.consider_visual_backup(accepted, samples, 112.5)

            trigger = worker.motion_events.queue.get_nowait()

        self.assertEqual(trigger["topic"], "adaptive/visual_backup")

    def test_visual_backup_readiness_resets_until_post_warmup_scene_is_quiet(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera_rescue",
            visual_backup_warmup_seconds=10,
            visual_backup_grace_seconds=1.5,
            visual_backup_min_consecutive=3,
        )
        stable = MotionQualificationResult(False, 0.0, 0.48, "no_motion_blobs", 4, {})
        active = MotionQualificationResult(True, 0.82, 0.48, "qualified", 4, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            self.assertFalse(worker.motion_analysis.visual_backup_readiness(stable, 100.0))
            self.assertFalse(worker.motion_analysis.visual_backup_readiness(stable, 109.5))
            self.assertFalse(worker.motion_analysis.visual_backup_readiness(stable, 110.0))
            self.assertFalse(worker.motion_analysis.visual_backup_readiness(active, 110.75))
            self.assertFalse(worker.motion_analysis.visual_backup_readiness(stable, 111.0))
            self.assertFalse(worker.motion_analysis.visual_backup_readiness(stable, 111.75))
            self.assertTrue(worker.motion_analysis.visual_backup_readiness(stable, 112.5))

    def test_visual_trigger_modes_run_replaceable_non_adaptive_qualifiers_continuously(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        classic_pipeline = {
            "qualification": [{
                "stage_id": "qualification",
                "implementation": "legacy_qualifier",
            }],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            rescue = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig.model_validate({
                    "mode": "camera_rescue",
                    "pipeline": classic_pipeline,
                }),
            )
            visual = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig.model_validate({
                    "mode": "adaptive",
                    "pipeline": classic_pipeline,
                }),
            )
            camera_only = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig.model_validate({
                    "mode": "camera",
                    "pipeline": classic_pipeline,
                }),
            )

        self.assertFalse(rescue.motion_pipeline.continuous_analysis)
        self.assertTrue(rescue.motion_qualification.continuous_primary_required())
        self.assertTrue(visual.motion_qualification.continuous_primary_required())
        self.assertFalse(camera_only.motion_qualification.continuous_primary_required())

    def test_visual_backup_never_promotes_weak_or_known_noise_motion(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera_rescue",
            visual_backup_warmup_seconds=0,
            visual_backup_grace_seconds=0,
            visual_backup_min_consecutive=2,
        )
        weak = MotionQualificationResult(True, 0.60, 0.5, "qualified", 4, {})
        noisy = MotionQualificationResult(True, 0.95, 0.5, "global_illumination_change", 4, {})
        samples = [(100.0, np.zeros((90, 160), dtype=np.uint8))]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker.motion_analysis.consider_visual_backup(weak, samples, 100.0)
            worker.motion_analysis.consider_visual_backup(weak, samples, 100.5)
            worker.motion_analysis.consider_visual_backup(noisy, samples, 101.0)
            worker.motion_analysis.consider_visual_backup(noisy, samples, 101.5)

        self.assertTrue(worker.motion_events.queue.empty())
        self.assertEqual(worker.motion_state._stats["visual_backup_triggers"], 0)

    def test_visual_backup_periodically_verifies_filtered_illumination(self) -> None:
        camera = CameraConfig(
            id="boiler",
            name="Boiler",
            stream_url="rtsp://example.invalid/main",
        )
        config = MotionQualificationConfig(
            mode="camera_rescue",
            illumination_filter_enabled=True,
            suppression_verification_rate=1.0,
            visual_backup_warmup_seconds=0,
            visual_backup_grace_seconds=0,
            visual_backup_min_consecutive=2,
            visual_backup_cooldown_seconds=5,
        )
        filtered = MotionQualificationResult(
            False,
            0.82,
            0.5,
            "illumination_change",
            4,
            {"illumination_would_reject": True},
        )
        samples = [(100.0, np.zeros((90, 160, 3), dtype=np.uint8))]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            prime_visual_backup_scene(worker, 90.0)
            with patch.object(
                    worker.motion_qualification,
                    "with_source_evidence",
                side_effect=lambda result, *_args, **_kwargs: result,
            ):
                worker.motion_analysis.consider_visual_backup(filtered, samples, 100.0)
                worker.motion_analysis.consider_visual_backup(filtered, samples, 100.5)

            trigger = worker.motion_events.queue.get_nowait()

        self.assertEqual(trigger["topic"], "adaptive/visual_backup")
        self.assertEqual(
            trigger["prequalified"].reason,
            "illumination_verification_probe",
        )
        self.assertTrue(
            trigger["prequalified"].features["illumination_verification_probe"]
        )
        self.assertEqual(worker.motion_state._stats["illumination_verification_probes"], 1)

    def test_visual_backup_detection_requires_eligible_object_and_is_audited(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera_rescue",
            visual_backup_warmup_seconds=0,
            burst_quiet_seconds=0.1,
        )
        accepted = MotionQualificationResult(True, 0.82, 0.5, "qualified", 4, {
            "visual_backup": True,
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            observed_at = time.time()
            prequalified = worker.motion_qualification.with_source_evidence(
                accepted,
                observed_at - 1.0,
                observed_at,
                require_primary_trigger=True,
            )
            with (
                patch.object(
                    worker.motion_incidents,
                    "process",
                    return_value=incident_outcome(
                        event_id=None,
                        snapshot_path="checked.jpg",
                        object_detected=False,
                    ),
                ) as process_event,
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
            ):
                thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                worker.motion_ingress.enqueue({
                    "topic": "adaptive/visual_backup",
                    "message": "backup",
                    "event_at": datetime.fromtimestamp(observed_at, timezone.utc),
                    "received_at": observed_at,
                    "prequalified": prequalified,
                })
                deadline = time.monotonic() + 2
                while record_audit.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker.motion_events.queue.put_nowait(None)
                thread.join(timeout=2)

        process_event.assert_called_once()
        self.assertTrue(process_event.call_args.kwargs["require_eligible_object"])
        self.assertTrue(process_event.call_args.kwargs["require_motion_correlation"])
        self.assertEqual(record_audit.call_args.kwargs["category"], "visual_backup")
        self.assertEqual(record_audit.call_args.kwargs["reason"], "visual_backup_trigger")
        self.assertEqual(worker.motion_fusion_pipeline.runtime.generation, 2)

    def test_onvif_notice_merged_with_queued_backup_uses_camera_semantics(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera_rescue",
            visual_backup_warmup_seconds=0,
            burst_quiet_seconds=0.1,
        )
        accepted = MotionQualificationResult(True, 0.82, 0.5, "qualified", 4, {
            "visual_backup": True,
        })
        event_at = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            worker.motion_events.camera_motion_times.append(event_at.timestamp())
            with (
                patch.object(
                    worker.motion_incidents,
                    "process",
                    return_value=incident_outcome(
                        event_id=42,
                        snapshot_path="person.jpg",
                        object_detected=True,
                    ),
                ) as process_event,
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
            ):
                worker.motion_ingress.enqueue({
                    "topic": "adaptive/visual_backup",
                    "message": "backup",
                    "event_at": event_at,
                    "received_at": event_at.timestamp(),
                    "prequalified": accepted,
                })
                worker.motion_ingress.enqueue({
                    "topic": "onvif/motion",
                    "message": "camera notice",
                    "event_at": event_at,
                    "received_at": event_at.timestamp(),
                })
                thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                deadline = time.monotonic() + 2
                while process_event.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker.motion_events.queue.put_nowait(None)
                thread.join(timeout=2)

        process_event.assert_called_once()
        self.assertFalse(process_event.call_args.kwargs["require_eligible_object"])
        self.assertFalse(process_event.call_args.kwargs["require_motion_correlation"])
        record_audit.assert_not_called()
        self.assertEqual(worker.motion_state._stats["visual_backup_onvif_matches"], 1)

    def test_visual_backup_audits_uncorrelated_object_rejection(self) -> None:
        camera = CameraConfig(id="front-door", name="Front Door", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera_rescue",
            visual_backup_warmup_seconds=0,
            burst_quiet_seconds=0.1,
        )
        accepted = MotionQualificationResult(True, 0.82, 0.5, "qualified", 4, {
            "visual_backup": True,
            "motion_regions": [[0.1, 0.1, 0.3, 0.3]],
        })
        correlation = {
            "eligible_object_count": 1,
            "correlated_object_count": 0,
        }
        event_at = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            with (
                patch.object(
                    worker.motion_incidents,
                    "process",
                    return_value=incident_outcome(
                        event_id=None,
                        snapshot_path="checked.jpg",
                        object_detected=False,
                        rejection_reason="object_not_motion_correlated",
                        motion_correlation=correlation,
                    ),
                ),
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
            ):
                worker.motion_ingress.enqueue({
                    "topic": "adaptive/visual_backup",
                    "message": "backup",
                    "event_at": event_at,
                    "received_at": event_at.timestamp(),
                    "prequalified": accepted,
                })
                thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                deadline = time.monotonic() + 2
                while record_audit.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker.motion_events.queue.put_nowait(None)
                thread.join(timeout=2)

        self.assertEqual(record_audit.call_args.kwargs["reason"], "object_not_motion_correlated")
        self.assertEqual(record_audit.call_args.kwargs["features"]["motion_correlation"], correlation)
        self.assertEqual(worker.motion_state._stats["visual_backup_uncorrelated_objects"], 1)

    def test_audit_mode_learns_continuously_without_creating_events(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        accepted = MotionQualificationResult(True, 0.8, 0.5, "qualified", 2, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            with patch.object(worker.motion_qualification, "run_pipeline", return_value=accepted):
                with worker._frame_lock:
                    worker.motion_analysis.frames.extend([
                        (1.0, np.zeros((90, 160), dtype=np.uint8)),
                        (2.0, np.zeros((90, 160), dtype=np.uint8)),
                    ])
                worker.motion_analysis.analyze_continuous(2.0)

            self.assertTrue(worker.motion_events.queue.empty())
            self.assertEqual(worker.motion_state._stats["continuous_candidates"], 1)

    def test_adaptive_mode_records_onvif_without_enqueuing_detection(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(mode="adaptive"),
            )

            worker.handle_motion_event("onvif/motion", "generic")

            self.assertTrue(worker.motion_events.queue.empty())
            self.assertIsNotNone(worker.motion_evidence.last("onvif"))
            self.assertEqual(worker.motion_state._stats["triggers"], 0)

    def test_adaptive_mode_still_allows_manual_test_trigger(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(mode="adaptive"),
            )

            worker.handle_motion_event("manual/test", "manual GUI trigger")

            self.assertEqual(worker.motion_events.queue.get_nowait()["topic"], "manual/test")

    def test_continuous_analysis_never_blocks_camera_frame_delivery(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            thread = worker.motion_analysis.thread = threading.Thread(
                target=worker.motion_analysis.run, args=(worker._stop,)
            )
            with patch.object(worker.motion_analysis, "analyze_continuous", side_effect=lambda _at: time.sleep(0.2)):
                thread.start()
                started = time.perf_counter()
                worker.motion_analysis.remember_frame(
                    np.zeros((180, 320, 3), dtype=np.uint8), 1.0, worker._stop
                )
                worker.motion_analysis.remember_frame(
                    np.zeros((180, 320, 3), dtype=np.uint8), 2.0, worker._stop
                )
                elapsed = time.perf_counter() - started

            worker._stop.set()
            worker.motion_analysis.request_stop()
            thread.join(timeout=1)
            self.assertLess(elapsed, 0.1)

    def test_adaptive_trigger_pending_flag_bounds_queue_during_chatter(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        accepted = MotionQualificationResult(True, 0.8, 0.5, "qualified", 2, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(mode="enforce"),
            )
            worker._stop.clear()
            with worker._frame_lock:
                worker.motion_analysis.frames.extend([
                    (1.0, np.zeros((90, 160), dtype=np.uint8)),
                    (2.0, np.zeros((90, 160), dtype=np.uint8)),
                ])
            with (
                patch.object(worker.motion_qualification, "run_pipeline", return_value=accepted),
                patch.object(worker.motion_qualification, "with_source_evidence", return_value=accepted),
            ):
                for captured_at in range(10, 110, 5):
                    worker.motion_analysis.analyze_continuous(float(captured_at))

            self.assertEqual(worker.motion_events.queue.qsize(), 1)
            self.assertEqual(worker.motion_state._stats["triggers"], 1)
            self.assertEqual(worker.motion_state._stats["adaptive_triggers_deferred"], 19)

    def test_adaptive_trigger_queue_saturation_backs_off_without_publishing(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        accepted = MotionQualificationResult(True, 0.8, 0.5, "qualified", 2, {})
        published: list[tuple[str, dict]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(mode="enforce"),
                event_callback=lambda *args: published.append(args),
            )
            worker._stop.clear()
            now = datetime.now(timezone.utc)
            for index in range(worker.motion_events.queue.maxsize):
                worker.motion_events.queue.put_nowait({
                    "topic": "onvif/motion",
                    "message": str(index),
                    "event_at": now,
                    "received_at": now.timestamp(),
                })
            with worker._frame_lock:
                worker.motion_analysis.frames.extend([
                    (1.0, np.zeros((90, 160), dtype=np.uint8)),
                    (2.0, np.zeros((90, 160), dtype=np.uint8)),
                ])
            with (
                patch.object(worker.motion_qualification, "run_pipeline", return_value=accepted),
                patch.object(worker.motion_qualification, "with_source_evidence", return_value=accepted),
            ):
                worker.motion_analysis.analyze_continuous(10.0)
                worker.motion_analysis.analyze_continuous(11.0)

            self.assertEqual(worker.motion_events.queue.qsize(), worker.motion_events.queue.maxsize)
            self.assertEqual(worker.motion_state._stats["triggers"], 1)
            self.assertEqual(worker.motion_state._stats["dropped_triggers"], 1)
            self.assertEqual(worker.motion_state._stats["adaptive_triggers_deferred"], 1)
            self.assertFalse(worker.motion_events.adaptive_trigger_pending)
            self.assertEqual(published, [])

    def test_adaptive_rejection_updates_state_without_allowing_source_rescue(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        rejected = MotionQualificationResult(False, 0.2, 0.48, "noise", 2, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(mode="adaptive"),
            )
            worker._stop.clear()
            with worker._frame_lock:
                worker.motion_analysis.frames.extend([
                    (1.0, np.zeros((90, 160), dtype=np.uint8)),
                    (2.0, np.zeros((90, 160), dtype=np.uint8)),
                ])
            with (
                patch.object(worker.motion_qualification, "run_pipeline", return_value=rejected),
                patch.object(
                    worker.motion_qualification,
                    "with_source_evidence",
                    wraps=worker.motion_qualification.with_source_evidence,
                ) as fuse,
            ):
                worker.motion_analysis.analyze_continuous(2.0)

            self.assertTrue(fuse.call_args.kwargs["require_primary_trigger"])
            self.assertTrue(worker.motion_events.queue.empty())
            self.assertEqual(
                worker.motion_analysis.last_continuous_result.reason,
                "primary_trigger_rejected",
            )
            self.assertEqual(
                worker.motion_analysis.last_continuous_result.features["event_state_phase"],
                "rejected",
            )

    def test_camera_validation_pipeline_failure_fails_open(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera",
            post_trigger_seconds=0.5,
            window_seconds=0.8,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            received_at = time.time() - 1.0
            for index in range(5):
                worker.motion_analysis.frames.append((
                    received_at - 0.8 + index * 0.2,
                    np.zeros((90, 160), dtype=np.uint8),
                ))
            with patch.object(
                worker.motion_qualification,
                "run_pipeline",
                side_effect=RuntimeError("validator unavailable"),
            ):
                result, diagnostics = worker.motion_qualification.qualify_burst(
                    datetime.fromtimestamp(received_at, timezone.utc),
                    received_at,
                    "balanced",
                    worker.motion_analysis,
                )

            self.assertTrue(result.accepted)
            self.assertEqual(result.reason, "validation_unavailable_fail_open")
            self.assertTrue(result.features["validation_fail_open"])
            self.assertEqual(diagnostics["windows_evaluated"], 1)
            self.assertEqual(worker.motion_state._stats["validation_fail_opens"], 1)

    def test_external_confirmation_uses_strongest_full_event_window(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig.model_validate({
            "mode": "camera",
            "post_trigger_seconds": 0.5,
            "window_seconds": 0.8,
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {
                            "policy": "all",
                            "sources": ["mog2"],
                            "include_primary": True,
                        },
                    },
                    {"stage_id": "event_state", "implementation": "score_event_state"},
                    {"stage_id": "trigger", "implementation": "score_trigger"},
                ],
            },
        })
        accepted = MotionQualificationResult(True, 0.8, 0.48, "qualified", 4, {})
        received_at = time.time() - 1.0
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            for index in range(7):
                worker.motion_analysis.frames.append((
                    received_at - 0.8 + index * 0.2,
                    np.zeros((90, 160), dtype=np.uint8),
                ))
            with (
                patch.object(
                    worker.motion_qualification,
                    "run_pipeline",
                    return_value=accepted,
                ) as analyze,
                patch.object(
                    worker.motion_qualification,
                    "with_source_evidence",
                    return_value=accepted,
                ) as fuse,
            ):
                result, _diagnostics = worker.motion_qualification.qualify_burst(
                    datetime.fromtimestamp(received_at, timezone.utc),
                    received_at,
                    "balanced",
                    worker.motion_analysis,
                )

            self.assertTrue(result.accepted)
            self.assertGreaterEqual(analyze.call_count, 2)
            fuse.assert_called_once()

    def test_mog2_only_validation_waits_for_post_trigger_evidence(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig.model_validate({
            "mode": "camera",
            "post_trigger_seconds": 0.5,
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {
                            "policy": "all",
                            "sources": ["mog2"],
                            "include_primary": False,
                        },
                    },
                    {"stage_id": "event_state", "implementation": "score_event_state"},
                    {"stage_id": "trigger", "implementation": "score_trigger"},
                ],
            },
        })
        accepted = MotionQualificationResult(True, 0.8, 0.5, "fusion_all_accepted", 0, {})
        received_at = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            with (
                patch.object(worker._stop, "wait", return_value=False) as wait_for_evidence,
                patch.object(
                    worker.motion_qualification,
                    "with_source_evidence",
                    return_value=accepted,
                ) as fuse,
            ):
                result, diagnostics = worker.motion_qualification.qualify_burst(
                    datetime.fromtimestamp(received_at, timezone.utc),
                    received_at,
                    "balanced",
                    worker.motion_analysis,
                )

            self.assertTrue(result.accepted)
            wait_for_evidence.assert_called_once_with(0.5)
            fuse.assert_called_once()
            self.assertEqual(diagnostics["windows_evaluated"], 0)

    def test_scalar_external_source_still_requires_confirmation_window(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig.model_validate({
            "mode": "camera",
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {
                            "policy": "all",
                            "sources": "mog2",
                            "include_primary": False,
                        },
                    },
                    {"stage_id": "event_state", "implementation": "score_event_state"},
                    {"stage_id": "trigger", "implementation": "score_trigger"},
                ],
            },
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            self.assertTrue(worker.motion_qualification.external_confirmation_required())

    def test_blank_external_source_does_not_add_confirmation_delay(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig.model_validate({
            "mode": "camera",
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {
                            "policy": "all",
                            "sources": " ",
                            "include_primary": False,
                        },
                    },
                    {"stage_id": "event_state", "implementation": "score_event_state"},
                    {"stage_id": "trigger", "implementation": "score_trigger"},
                ],
            },
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            self.assertFalse(worker.motion_qualification.external_confirmation_required())

    def test_shutdown_during_validation_does_not_start_object_detection(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            event_at = datetime.now(timezone.utc)
            worker.motion_events.queue.put_nowait({
                "topic": "onvif/motion",
                "message": "motion",
                "event_at": event_at,
                "received_at": event_at.timestamp(),
            })
            worker._stop.clear()

            def stop_during_validation(*_args):
                worker._stop.set()
                return MotionQualificationResult(True, 0.8, 0.5, "qualified", 4, {}), {}

            with (
                patch.object(worker.motion_qualification, "qualify_burst", side_effect=stop_during_validation),
                patch.object(worker.motion_incidents, "process") as process_event,
            ):
                worker.motion_decisions.run_until_error(worker._stop)

            process_event.assert_not_called()
            self.assertIsNone(worker.motion_events.active_triggers)

    def test_fusion_pipeline_failure_fails_open(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        accepted = MotionQualificationResult(True, 0.8, 0.48, "qualified", 4, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with patch.object(
                worker.motion_fusion_pipeline,
                "process",
                side_effect=RuntimeError("fusion unavailable"),
            ):
                result = worker.motion_qualification.with_source_evidence(accepted, 1.0, 2.0)

            self.assertTrue(result.accepted)
            self.assertEqual(result.reason, "validation_unavailable_fail_open")
            self.assertEqual(worker.motion_state._stats["validation_fail_opens"], 1)

    def test_explicit_fail_closed_result_cannot_be_borderline_rescued(self) -> None:
        result = MotionQualificationResult(
            False,
            0.8,
            0.48,
            "validation_unavailable_fail_closed",
            4,
            {},
        )

        self.assertFalse(is_borderline_candidate(result, True, 0.03))

    def test_categorical_nuisance_rejection_cannot_be_borderline_rescued(self) -> None:
        result = MotionQualificationResult(
            False,
            0.8,
            0.48,
            "stationary_foreground",
            4,
            {},
        )

        self.assertFalse(is_borderline_candidate(result, True, 0.03))

    def test_low_score_within_margin_can_be_borderline_rescued(self) -> None:
        result = MotionQualificationResult(False, 0.46, 0.48, "low_score", 4, {})

        self.assertTrue(is_borderline_candidate(result, True, 0.03))

    def test_suppression_verification_sampling_is_stable_per_decision(self) -> None:
        decision_id = "fixed-decision"

        first = should_verify_suppression(decision_id, 0.5)

        self.assertEqual(
            first,
            should_verify_suppression(decision_id, 0.5),
        )
        self.assertFalse(should_verify_suppression(decision_id, 0.0))
        self.assertTrue(should_verify_suppression(decision_id, 1.0))

    def test_camera_mode_reduces_background_upkeep_without_throttling_adaptive_mode(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            camera_mode = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(
                    mode="camera",
                    sample_fps=5.0,
                    camera_mode_background_fps=2.0,
                ),
            )
            camera_mode.motion_analysis.primary_last_processed_at = 10.0
            self.assertFalse(
                camera_mode.motion_qualification.continuous_primary_due(10.2, 10.0)
            )
            self.assertTrue(
                camera_mode.motion_qualification.continuous_primary_due(10.5, 10.0)
            )

            adaptive_mode = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(
                    mode="adaptive",
                    camera_mode_background_fps=0.5,
                ),
            )
            adaptive_mode.motion_analysis.primary_last_processed_at = 10.0
            self.assertTrue(
                adaptive_mode.motion_qualification.continuous_primary_due(10.01, 10.0)
            )

    def test_sampled_visual_rejection_runs_detector_without_creating_empty_incident(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera",
            burst_quiet_seconds=0.1,
            suppression_verification_rate=1.0,
        )
        rejected = MotionQualificationResult(False, 0.2, 0.5, "stationary_foreground", 3, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            with (
                patch.object(worker.motion_qualification, "qualify_burst", return_value=(rejected, {})),
                patch.object(
                    worker.motion_incidents,
                    "process",
                    return_value=incident_outcome(
                        event_id=None,
                        snapshot_path="checked.jpg",
                        object_detected=False,
                    ),
                ) as process_event,
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
            ):
                thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                worker.handle_motion_event("onvif/motion", "motion")
                deadline = time.monotonic() + 2
                while record_audit.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker.motion_events.queue.put_nowait(None)
                thread.join(timeout=2)

            process_event.assert_called_once()
            self.assertTrue(process_event.call_args.kwargs["require_eligible_object"])
            record_audit.assert_called_once()
            self.assertIsNone(record_audit.call_args.kwargs["event_id"])
            self.assertTrue(record_audit.call_args.kwargs["features"]["suppression_verification"])
            self.assertEqual(worker.motion_state._stats["suppression_verification_checks"], 1)
            self.assertEqual(worker.motion_state._stats["suppression_verification_rescues"], 0)
            self.assertEqual(worker.motion_state._stats["suppressed"], 1)

    def test_sampled_visual_rejection_is_restored_when_detector_finds_object(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera",
            burst_quiet_seconds=0.1,
            suppression_verification_rate=1.0,
        )
        rejected = MotionQualificationResult(False, 0.2, 0.5, "micro_jitter", 3, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            with (
                patch.object(worker.motion_qualification, "qualify_burst", return_value=(rejected, {})),
                patch.object(
                    worker.motion_incidents,
                    "process",
                    return_value=incident_outcome(
                        event_id=9,
                        snapshot_path="person.jpg",
                        object_detected=True,
                    ),
                ),
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
            ):
                thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                worker.handle_motion_event("onvif/motion", "motion")
                deadline = time.monotonic() + 2
                while record_audit.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker.motion_events.queue.put_nowait(None)
                thread.join(timeout=2)

            self.assertEqual(record_audit.call_args.kwargs["event_id"], 9)
            self.assertEqual(worker.motion_state._stats["suppression_verification_checks"], 1)
            self.assertEqual(worker.motion_state._stats["suppression_verification_rescues"], 1)
            self.assertEqual(worker.motion_state._stats["suppressed"], 0)

    def test_fusion_failure_cannot_rescue_rejected_adaptive_trigger(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        rejected = MotionQualificationResult(False, 0.2, 0.48, "noise", 4, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(mode="adaptive"),
            )
            with patch.object(
                worker.motion_fusion_pipeline,
                "process",
                side_effect=RuntimeError("fusion unavailable"),
            ):
                result = worker.motion_qualification.with_source_evidence(
                    rejected,
                    1.0,
                    2.0,
                    require_primary_trigger=True,
                )

            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, "primary_trigger_rejected")
            self.assertFalse(result.features["validation_fail_open"])
            self.assertEqual(worker.motion_state._stats["validation_failures"], 1)
            self.assertEqual(worker.motion_state._stats["validation_fail_opens"], 0)

    def test_analysis_worker_survives_a_transient_cycle_failure(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            thread = worker.motion_analysis.thread = threading.Thread(
                target=worker.motion_analysis.run, args=(worker._stop,)
            )
            with patch.object(
                worker.motion_analysis,
                "analyze_continuous",
                side_effect=[RuntimeError("transient fusion failure"), None],
            ) as analyze:
                thread.start()
                with worker._frame_lock:
                    worker.motion_analysis.frames.append((1.0, np.zeros((90, 160), dtype=np.uint8)))
                worker.motion_analysis.schedule(1.0, worker._stop)
                deadline = time.monotonic() + 1
                while worker.motion_state._stats["analysis_worker_errors"] == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                with worker._frame_lock:
                    worker.motion_analysis.frames.append((2.0, np.zeros((90, 160), dtype=np.uint8)))
                worker.motion_analysis.schedule(2.0, worker._stop)
                deadline = time.monotonic() + 1
                while analyze.call_count < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertTrue(thread.is_alive())
            self.assertEqual(analyze.call_count, 2)
            self.assertEqual(worker.motion_state._stats["analysis_worker_errors"], 1)
            worker._stop.set()
            worker.motion_analysis.request_stop()
            thread.join(timeout=1)

    def test_event_worker_survives_a_transient_handler_failure(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="off", burst_quiet_seconds=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            with patch.object(
                    worker.motion_incidents,
                    "process",
                side_effect=[
                    RuntimeError("transient event failure"),
                    incident_outcome(event_id=2, snapshot_path="", object_detected=False),
                    incident_outcome(event_id=3, snapshot_path="", object_detected=False),
                ],
            ) as process_event:
                thread = worker.motion_decisions._thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                worker.handle_motion_event("onvif/motion", "first")
                deadline = time.monotonic() + 1
                while process_event.call_count < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker.handle_motion_event("onvif/motion", "second")
                deadline = time.monotonic() + 1
                while process_event.call_count < 3 and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertTrue(thread.is_alive())
            self.assertEqual(process_event.call_count, 3)
            self.assertEqual(worker.motion_state._stats["event_worker_errors"], 1)
            self.assertEqual(worker.motion_state._stats["event_retries"], 1)
            self.assertEqual(worker.motion_state._stats["bursts"], 2)
            self.assertEqual(worker.motion_state._stats["passed"], 2)
            worker._stop.set()
            worker.motion_events.queue.put_nowait(None)
            thread.join(timeout=1)

    def test_retry_batch_is_protected_from_normal_queue_eviction(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            now = datetime.now(timezone.utc)
            worker.motion_decisions.retry_batch([{
                "topic": "onvif/motion",
                "message": "retry me",
                "event_at": now,
                "received_at": now.timestamp(),
            }], worker._stop)
            for index in range(worker.motion_events.queue.maxsize):
                worker.motion_events.queue.put_nowait({
                    "topic": "onvif/motion",
                    "message": str(index),
                    "event_at": now,
                    "received_at": now.timestamp(),
                })

            worker.motion_ingress.enqueue({
                "topic": "onvif/motion",
                "message": "new priority event",
                "event_at": now,
                "received_at": now.timestamp(),
            })

            self.assertEqual(len(worker.motion_events.retry_batches), 1)
            self.assertEqual(
                worker.motion_events.retry_batches[0]["_retry_batch"][0]["message"],
                "retry me",
            )
            self.assertEqual(worker.motion_state._stats["event_retry_drops"], 0)

    def test_event_worker_bounds_retries_for_a_persistent_handler_failure(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="off", burst_quiet_seconds=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            with patch.object(
                    worker.motion_incidents,
                    "process",
                side_effect=RuntimeError("persistent event failure"),
            ) as process_event:
                thread = worker.motion_decisions._thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                worker.handle_motion_event("onvif/motion", "first")
                deadline = time.monotonic() + 3
                while worker.motion_state._stats["event_retry_drops"] == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertTrue(thread.is_alive())
            self.assertEqual(process_event.call_count, 3)
            self.assertEqual(worker.motion_state._stats["event_retries"], 2)
            self.assertEqual(worker.motion_state._stats["event_retry_drops"], 1)
            worker._stop.set()
            worker.motion_events.queue.put_nowait(None)
            thread.join(timeout=1)

    def test_failed_adaptive_batch_stays_reserved_until_terminal_retry_drop(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="off", burst_quiet_seconds=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            worker.motion_events.adaptive_trigger_pending = True
            with patch.object(
                    worker.motion_incidents,
                    "process",
                side_effect=RuntimeError("persistent event failure"),
            ):
                thread = worker.motion_decisions._thread = threading.Thread(
                    target=worker.motion_decisions.run, args=(worker._stop,)
                )
                thread.start()
                now = datetime.now(timezone.utc)
                worker.motion_ingress.enqueue({
                    "topic": "adaptive/motion",
                    "message": "adaptive",
                    "event_at": now,
                    "received_at": now.timestamp(),
                })
                deadline = time.monotonic() + 3
                while (
                    worker.motion_state._stats["event_retry_drops"] == 0
                    and time.monotonic() < deadline
                ):
                    self.assertTrue(worker.motion_events.adaptive_trigger_pending)
                    time.sleep(0.01)
                while worker.motion_events.adaptive_trigger_pending and time.monotonic() < deadline:
                    time.sleep(0.005)

            self.assertEqual(worker.motion_state._stats["event_retry_drops"], 1)
            self.assertFalse(worker.motion_events.adaptive_trigger_pending)
            worker._stop.set()
            worker.motion_events.queue.put_nowait(None)
            thread.join(timeout=1)

    def test_enforce_retry_reuses_accepted_qualification_without_advancing_fusion(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="enforce", burst_quiet_seconds=0.1)
        accepted = MotionQualificationResult(True, 0.8, 0.5, "qualified", 4, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()

            def qualify(*_args):
                return worker.motion_qualification.with_source_evidence(
                    accepted,
                    time.time() - 1,
                    time.time(),
                ), {
                    "windows_evaluated": 1,
                    "event_receipt_delta_seconds": 0.0,
                }

            with (
                patch.object(worker.motion_qualification, "qualify_burst", side_effect=qualify),
                patch.object(
                    worker.motion_incidents,
                    "process",
                    side_effect=[
                        RuntimeError("transient event failure"),
                        incident_outcome(
                            event_id=2,
                            snapshot_path="",
                            object_detected=False,
                        ),
                    ],
                ) as process_event,
            ):
                thread = worker.motion_decisions._thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                worker.handle_motion_event("onvif/motion", "generic")
                deadline = time.monotonic() + 2
                while process_event.call_count < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertTrue(thread.is_alive())
            self.assertEqual(process_event.call_count, 2)
            self.assertEqual(worker.motion_state._stats["event_retries"], 1)
            self.assertEqual(worker.motion_state._stats["suppressed"], 0)
            self.assertTrue(worker.motion_state._stats["last_result"]["effective_accepted"])
            worker._stop.set()
            worker.motion_events.queue.put_nowait(None)
            thread.join(timeout=1)

    def test_enforce_retry_reuses_rejected_qualification_without_advancing_fusion(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig.model_validate({
            "mode": "enforce",
            "burst_quiet_seconds": 0.1,
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": "audit"},
                    },
                    {
                        "stage_id": "event_state",
                        "implementation": "score_event_state",
                        "options": {"activation_frames": 2},
                    },
                    {
                        "stage_id": "trigger",
                        "implementation": "score_trigger",
                    },
                ],
            },
        })
        accepted = MotionQualificationResult(True, 0.8, 0.5, "qualified", 4, {})
        published: list[tuple[str, dict]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=config,
                event_callback=lambda *args: published.append(args),
            )
            worker._stop.clear()

            def qualify(*_args):
                return worker.motion_qualification.with_source_evidence(
                    accepted,
                    time.time() - 1,
                    time.time(),
                ), {
                    "windows_evaluated": 1,
                    "event_receipt_delta_seconds": 0.0,
                }

            with (
                patch.object(worker.motion_qualification, "qualify_burst", side_effect=qualify),
                patch.object(
                    worker.media,
                    "sample_rejected_motion",
                    return_value="",
                ) as sample_rejected,
                patch.object(
                    worker.motion_decision_handler,
                    "record_audit",
                    side_effect=[RuntimeError("audit write failed"), {}],
                ) as record_audit,
                patch.object(worker.motion_incidents, "process") as process_event,
            ):
                thread = worker.motion_decisions._thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                worker.handle_motion_event("onvif/motion", "generic")
                deadline = time.monotonic() + 2
                while record_audit.call_count < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertTrue(thread.is_alive())
            self.assertEqual(record_audit.call_count, 2)
            self.assertEqual(sample_rejected.call_count, 1)
            first_audit = record_audit.call_args_list[0].kwargs
            retry_audit = record_audit.call_args_list[1].kwargs
            self.assertEqual(first_audit["decision_id"], retry_audit["decision_id"])
            self.assertEqual(first_audit["snapshot_path"], "")
            self.assertEqual(retry_audit["snapshot_path"], "")
            process_event.assert_not_called()
            self.assertEqual(worker.motion_state._stats["event_retries"], 1)
            self.assertEqual(worker.motion_state._stats["bursts"], 1)
            self.assertEqual(worker.motion_state._stats["suppressed"], 1)
            self.assertEqual(
                worker.motion_state._stats["last_result"]["reason"],
                "event_state_candidate",
            )
            self.assertFalse(worker.motion_state._stats["last_result"]["effective_accepted"])
            qualifications = [
                payload
                for event_type, payload in published
                if event_type == "motion_qualification"
            ]
            self.assertEqual(len(qualifications), 1)
            worker._stop.set()
            worker.motion_events.queue.put_nowait(None)
            thread.join(timeout=1)

    def test_post_persistence_audit_failure_does_not_duplicate_incident(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="audit", burst_quiet_seconds=0.1)
        rejected = MotionQualificationResult(False, 0.1, 0.5, "rejected", 2, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            with (
                patch.object(
                    worker.motion_incidents,
                    "process",
                    return_value=incident_outcome(
                        event_id=1,
                        snapshot_path="",
                        object_detected=False,
                    ),
                ) as process_event,
                patch.object(
                    worker.motion_decision_handler,
                    "record_audit",
                    side_effect=RuntimeError("audit write failed"),
                ),
            ):
                thread = worker.motion_decisions._thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                now = datetime.now(timezone.utc)
                worker.motion_ingress.enqueue({
                    "topic": "adaptive/motion",
                    "message": "adaptive",
                    "event_at": now,
                    "received_at": now.timestamp(),
                    "prequalified": rejected,
                })
                deadline = time.monotonic() + 1
                while worker.motion_state._stats["event_worker_errors"] == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertTrue(thread.is_alive())
            process_event.assert_called_once()
            self.assertEqual(worker.motion_state._stats["event_retries"], 0)
            worker._stop.set()
            worker.motion_events.queue.put_nowait(None)
            thread.join(timeout=1)

    def test_fusion_pipeline_calls_are_serialized_across_workers(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        accepted = MotionQualificationResult(True, 0.8, 0.5, "qualified", 2, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            original_process = worker.motion_fusion_pipeline.process
            state_lock = threading.Lock()
            active = 0
            maximum_active = 0

            def slow_process(context):
                nonlocal active, maximum_active
                with state_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                try:
                    time.sleep(0.03)
                    return original_process(context)
                finally:
                    with state_lock:
                        active -= 1

            errors: list[Exception] = []

            def fuse(end_epoch: float) -> None:
                try:
                    worker.motion_qualification.with_source_evidence(accepted, end_epoch - 1, end_epoch)
                except Exception as error:
                    errors.append(error)

            with patch.object(worker.motion_fusion_pipeline, "process", side_effect=slow_process):
                threads = [threading.Thread(target=fuse, args=(end_epoch,)) for end_epoch in (20.0, 10.0)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=1)

            self.assertEqual(errors, [])
            self.assertEqual(maximum_active, 1)

    def test_stale_fusion_timestamp_uses_clock_reset_instead_of_being_retimestamped(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        accepted = MotionQualificationResult(True, 0.8, 0.5, "qualified", 2, {})
        rejected = MotionQualificationResult(False, 0.1, 0.5, "rejected", 2, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker.motion_qualification.with_source_evidence(accepted, 19.0, 20.0)

            stale = worker.motion_qualification.with_source_evidence(rejected, 18.0, 19.0)
            still_active = worker.motion_qualification.with_source_evidence(accepted, 19.5, 20.5)
            clock_rollback = worker.motion_qualification.with_source_evidence(rejected, 9.0, 10.0)

            self.assertEqual(stale.reason, "stale_fusion_evidence")
            self.assertEqual(stale.score, 0.0)
            self.assertEqual(stale.threshold, 1.0)
            self.assertEqual(worker.motion_state._stats["stale_fusion_samples"], 1)
            self.assertEqual(still_active.features["event_state_phase"], "active")
            self.assertEqual(clock_rollback.features["event_state_phase"], "rejected")

    def test_priority_onvif_bypasses_active_adaptive_event_state(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="audit", burst_quiet_seconds=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            active = MotionQualificationResult(True, 0.8, 0.5, "qualified", 2, {})
            self.assertTrue(worker.motion_qualification.with_source_evidence(active, 1.0, 2.0).accepted)
            worker._stop.clear()
            with patch.object(
                    worker.motion_incidents,
                    "process",
                return_value=incident_outcome(
                    event_id=1,
                    snapshot_path="",
                    object_detected=False,
                ),
            ) as process_event:
                thread = worker.motion_decisions._thread = threading.Thread(
                    target=worker.motion_decisions.run, args=(worker._stop,)
                )
                thread.start()
                worker.handle_motion_event("onvif/person", "person")
                deadline = time.monotonic() + 1
                while process_event.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker.motion_events.queue.put_nowait(None)
                thread.join(timeout=1)

            process_event.assert_called_once()
            qualification = process_event.call_args.args[3]
            self.assertEqual(qualification["reason"], "priority_topic")
            self.assertTrue(qualification["effective_accepted"])

    def test_priority_onvif_defers_the_next_adaptive_trigger_in_enforce_mode(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="enforce", burst_quiet_seconds=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            with patch.object(
                    worker.motion_incidents,
                    "process",
                return_value=incident_outcome(
                    event_id=1,
                    snapshot_path="",
                    object_detected=True,
                ),
            ) as process_event:
                thread = worker.motion_decisions._thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                worker.handle_motion_event("onvif/person", "person")
                deadline = time.monotonic() + 1
                while process_event.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)

                self.assertFalse(worker.motion_analysis.reserve_adaptive_trigger(time.time()))
                distinct_event_at = time.time() + worker.motion_analysis._priority_dedup_seconds() + 0.1
                self.assertTrue(worker.motion_analysis.reserve_adaptive_trigger(distinct_event_at))
                worker._stop.set()
                worker.motion_events.queue.put_nowait(None)
                thread.join(timeout=1)

            process_event.assert_called_once()
            self.assertGreater(worker.motion_state._stats["adaptive_triggers_deferred"], 0)

    def test_priority_results_still_use_event_state_deduplication(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            priority = MotionQualificationResult(
                True, 1.0, 0.0, "priority_topic", 0, {"primary_motion_source": "onvif_priority"}
            )

            first = worker.motion_qualification.with_source_evidence(priority, 10.0, 11.0)
            second = worker.motion_qualification.with_source_evidence(priority, 11.0, 12.0)

            self.assertTrue(first.accepted)
            self.assertFalse(second.accepted)
            self.assertEqual(second.reason, "event_state_active")

    def test_active_and_cooldown_results_link_to_the_active_incident_event(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker.runtime_state.active_incident_event_id = 42

            active = MotionQualificationResult(False, 0.7, 0.5, "event_state_active", 2, {})
            cooldown = MotionQualificationResult(False, 0.7, 0.5, "event_state_cooldown", 2, {})
            unrelated = MotionQualificationResult(False, 0.2, 0.5, "low_score", 2, {})

            self.assertEqual(worker.motion_decisions._related_incident_event_id(active), 42)
            self.assertEqual(worker.motion_decisions._related_incident_event_id(cooldown), 42)
            self.assertIsNone(worker.motion_decisions._related_incident_event_id(unrelated))

    def test_active_incident_activity_is_linked_without_saving_duplicate_image(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(
            mode="camera",
            burst_quiet_seconds=0.1,
            suppression_verification_rate=0.0,
        )
        active = MotionQualificationResult(False, 0.7, 0.5, "event_state_active", 2, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker.runtime_state.active_incident_event_id = 42
            worker._stop.clear()
            with (
                patch.object(worker.motion_qualification, "qualify_burst", return_value=(active, {})),
                patch.object(worker.media, "sample_rejected_motion") as sample_rejected,
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
                patch.object(worker.motion_incidents, "process") as process_event,
            ):
                thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                worker.handle_motion_event("onvif/motion", "continued activity")
                deadline = time.monotonic() + 2
                while record_audit.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker.motion_events.queue.put_nowait(None)
                thread.join(timeout=2)

            record_audit.assert_called_once()
            self.assertEqual(record_audit.call_args.kwargs["snapshot_path"], "")
            self.assertEqual(record_audit.call_args.kwargs["related_event_id"], 42)
            sample_rejected.assert_not_called()
            process_event.assert_not_called()

    def test_powered_off_worker_does_not_start_snapshot_source(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with patch.object(
                worker.capture,
                "ensure_source",
                wraps=worker.capture.ensure_source,
            ) as ensure_source:
                self.assertIsNone(worker.snapshot("main"))

        ensure_source.assert_not_called()

    def test_status_separates_power_capture_and_frame_freshness(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker.runtime_state.enabled = True
            worker._stop.clear()
            seed_capture_frame(
                worker,
                "live",
                np.zeros((10, 10, 3), dtype=np.uint8),
                time.time(),
                time.monotonic(),
            )
            with worker.capture._lock:
                worker.capture._threads["live"] = Mock(is_alive=lambda: True)
            status = worker.status()

        self.assertTrue(status["running"])
        self.assertTrue(status["capture_running"])
        self.assertTrue(status["connected"])
        self.assertTrue(status["frame_fresh"])

    def test_only_main_source_expires_when_idle(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with worker.capture._lock:
                worker.capture._last_access["main"] = time.monotonic() - 60

            self.assertTrue(worker.capture.source_is_idle("main"))
            self.assertFalse(worker.capture.source_is_idle("live"))

    def test_snapshot_rejects_a_stale_cached_frame(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            seed_capture_frame(
                worker,
                "live",
                np.zeros((10, 10, 3), dtype=np.uint8),
                time.time() - 60,
                time.monotonic() - 60,
            )
            with patch.object(
                worker.capture,
                "ensure_source",
                return_value=SimpleNamespace(running=True),
            ):
                self.assertIsNone(worker.snapshot())

    def test_start_source_replaces_a_stopping_capture_thread(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            old_thread = Mock(is_alive=lambda: True)
            old_stop = threading.Event()
            old_stop.set()
            with worker.capture._lock:
                worker.capture._stop.clear()
                worker.capture._threads["live"] = old_thread
                worker.capture._source_stops["live"] = old_stop
            with patch.object(worker.capture, "_run_source_when_released"):
                self.assertTrue(worker.capture.ensure_source("live").running)
                replacement = worker.capture.threads()["live"]
                replacement.join(timeout=1)

        self.assertIsNot(replacement, old_thread)
        with worker.capture._lock:
            self.assertIsNot(worker.capture._source_stops.get("live"), old_stop)

    def test_capture_does_not_publish_a_read_completed_after_stop(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        capture = Mock()
        capture.open.return_value = True
        capture.isOpened.return_value = True
        stop_event = threading.Event()

        def stop_during_read():
            stop_event.set()
            return True, np.ones((10, 10, 3), dtype=np.uint8)

        capture.read.side_effect = stop_during_read
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            with worker.capture._lock:
                worker.capture._stop.clear()
            with patch("survng.app.camera_capture.cv2.VideoCapture", return_value=capture):
                worker.capture._run_source("live", stop_event)

        with worker.capture._lock:
            self.assertNotIn("live", worker.capture._frames)
        self.assertIsNone(worker.capture.latest("live"))

    def test_start_rolls_back_workers_when_capture_start_fails(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with patch.object(worker.capture, "start", side_effect=RuntimeError("no threads")):
                with self.assertRaisesRegex(RuntimeError, "no threads"):
                    worker.start()

            self.assertFalse(worker.status()["running"])
            self.assertIsNone(worker.motion_analysis.thread)
            self.assertIsNone(worker.motion_decisions._thread)

    def test_start_clears_stale_events_before_starting_producers(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        order: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with (
                patch.object(
                    worker.tracking_lifecycle,
                    "sync_accepting",
                    side_effect=lambda: order.append("tracking"),
                ),
                patch.object(
                    worker.motion_events,
                    "clear",
                    side_effect=lambda: order.append("clear"),
                ),
                patch.object(
                    worker.motion_analysis,
                    "start",
                    side_effect=lambda _stop: order.append("analysis"),
                ),
                patch.object(
                    worker.capture,
                    "start",
                    side_effect=lambda: order.append("capture") or False,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "camera source did not start"):
                    worker.start()

        self.assertEqual(order[:4], ["tracking", "clear", "analysis", "capture"])

    def test_start_rolls_back_when_tracking_eligibility_sync_fails(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with patch.object(
                worker.tracking_lifecycle,
                "sync_accepting",
                side_effect=RuntimeError("tracking unavailable"),
            ):
                with self.assertRaisesRegex(RuntimeError, "tracking unavailable"):
                    worker.start()

            self.assertFalse(worker.status()["running"])
            self.assertTrue(worker._stop.is_set())
            self.assertIsNone(worker.motion_analysis.thread)
            self.assertIsNone(worker.motion_decisions._thread)

    def test_start_preserves_original_error_when_rollback_raises_base_exception(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with (
                patch.object(worker.motion_analysis, "start"),
                patch.object(
                    worker.capture,
                    "start",
                    side_effect=RuntimeError("startup failed"),
                ),
                patch.object(
                    worker.lifecycle,
                    "stop",
                    side_effect=KeyboardInterrupt("rollback interrupted"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "startup failed"):
                    worker.start()

    def test_stop_releases_camera_io_before_waiting_for_tracking(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        order: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with (
                patch.object(
                    worker.capture,
                    "request_stop",
                    side_effect=lambda: order.append("capture"),
                ),
                patch.object(
                    worker.onvif,
                    "request_stop",
                    side_effect=lambda: order.append("onvif"),
                ),
                patch.object(
                    worker.tracking_lifecycle,
                    "request_stop",
                    side_effect=lambda: order.append("tracking"),
                ),
            ):
                worker.stop()

        self.assertEqual(order[:3], ["capture", "onvif", "tracking"])

    def test_tracking_stop_failure_does_not_skip_remaining_camera_cleanup(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        order: list[str] = []

        def fail_tracking_stop() -> None:
            order.append("tracking")
            raise RuntimeError("tracking stop failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with (
                patch.object(worker.capture, "request_stop", side_effect=lambda: order.append("capture")),
                patch.object(worker.onvif, "request_stop", side_effect=lambda: order.append("onvif")),
                patch.object(
                    worker.tracking_lifecycle,
                    "request_stop",
                    side_effect=fail_tracking_stop,
                ),
                patch.object(
                    worker.motion_analysis,
                    "request_stop",
                    side_effect=lambda: order.append("analysis"),
                ),
                patch.object(
                    worker.motion_events,
                    "signal_stop",
                    side_effect=lambda: order.append("events"),
                ),
                patch.object(worker.tracking_lifecycle, "running", return_value=False),
            ):
                with self.assertRaisesRegex(RuntimeError, "object tracking cleanup"):
                    worker.stop()

        self.assertEqual(order, ["capture", "onvif", "tracking", "analysis", "events"])

    def test_stop_continues_reset_after_capture_wait_raises(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with (
                patch.object(
                    worker.capture,
                    "wait_stopped",
                    side_effect=RuntimeError("capture join failed"),
                ),
                patch.object(worker.tracking_frames, "clear") as clear_frames,
                patch.object(worker.motion_analysis, "reset") as reset_analysis,
                patch.object(worker.motion_events, "reset") as reset_events,
            ):
                with self.assertRaisesRegex(RuntimeError, "capture wait cleanup"):
                    worker.stop()

        clear_frames.assert_called_once_with()
        reset_analysis.assert_called_once_with()
        reset_events.assert_called_once_with()

    def test_stop_preserves_motion_state_when_workers_do_not_exit(self) -> None:
        camera = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://example.invalid/main",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker.motion_analysis.frames.append(
                (100.0, np.zeros((90, 160), dtype=np.uint8))
            )
            active = MotionTriggerBatch((MotionTrigger(
                topic="adaptive/motion",
                message="motion",
                event_at=datetime.now(timezone.utc),
                received_at=100.0,
            ),))
            worker.motion_events.set_active(active)
            stuck_event_worker = Mock()
            stuck_event_worker.is_alive.return_value = True
            worker.motion_decisions._thread = stuck_event_worker

            with (
                patch.object(worker.motion_analysis, "wait_stopped", return_value=False),
                patch.object(worker.motion_analysis, "reset") as reset_analysis,
                patch.object(worker.motion_evidence, "clear") as clear_evidence,
                patch.object(worker.capture, "wait_stopped", return_value={}),
            ):
                with self.assertRaisesRegex(RuntimeError, "motion workers"):
                    worker.stop()

            reset_analysis.assert_not_called()
            clear_evidence.assert_not_called()
            self.assertEqual(len(worker.motion_analysis.frames), 1)
            self.assertIs(worker.motion_events.active_triggers, active)
            with self.assertRaisesRegex(RuntimeError, "stale workers"):
                worker.start()
            self.assertTrue(worker._stop.is_set())

    def test_close_refuses_to_race_an_active_motion_worker(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker.motion_decisions._thread = Mock(is_alive=lambda: True)

            with self.assertRaisesRegex(RuntimeError, "motion events"):
                worker.close()

    def test_close_attempts_every_motion_pipeline_after_a_failure(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker.motion_pipeline = Mock()
            worker.motion_observation_pipeline = Mock()
            worker.motion_fusion_pipeline = Mock()
            worker.motion_runtime.pipelines = (
                ("qualification", worker.motion_pipeline),
                ("observation", worker.motion_observation_pipeline),
                ("fusion", worker.motion_fusion_pipeline),
            )
            worker.motion_pipeline.close.side_effect = RuntimeError("close failed")

            with self.assertRaisesRegex(RuntimeError, "failed to close"):
                worker.close()

        worker.motion_observation_pipeline.close.assert_called_once_with()
        worker.motion_fusion_pipeline.close.assert_called_once_with()

    def test_disabled_detection_ignores_motion_event(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        detector = DummyDetector()
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), detector=detector)
            worker.set_detection_enabled(False)
            worker.handle_motion_event("onvif/motion", "motion")

        self.assertEqual(worker.motion_state.last_motion_at(), "")
        self.assertEqual(detector.calls, 0)
        self.assertIsNone(worker.motion_evidence.last("onvif"))

    def test_late_motion_callback_is_ignored_after_shutdown_starts(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
        )
        published: list[tuple[str, dict]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                event_callback=lambda *args: published.append(args),
            )
            worker.runtime_state.accepting_motion_events = False

            worker.handle_motion_event("onvif/motion", "late callback")

        self.assertEqual(worker.motion_state.last_motion_at(), "")
        self.assertIsNone(worker.motion_evidence.last("onvif"))
        self.assertTrue(worker.motion_events.queue.empty())
        self.assertEqual(published, [])

    def test_motion_handler_enqueues_without_running_detection_inline(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with patch.object(worker, "_recorded_motion_frame") as recorded_frame:
                worker.handle_motion_event("onvif/motion", "motion")

            self.assertEqual(worker.motion_events.queue.qsize(), 1)
            evidence = worker.motion_evidence.last("onvif")
            self.assertIsNotNone(evidence)
            self.assertEqual(evidence.values["event_source"], "onvif")
            self.assertEqual(evidence.values["score"], 0.55)
            recorded_frame.assert_not_called()

    def test_motion_worker_coalesces_a_trigger_burst(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="off", burst_quiet_seconds=0.1, window_seconds=0.8)
        published = []
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=config,
                event_callback=lambda event_type, payload: published.append((event_type, payload)),
            )
            worker._stop.clear()
            with patch.object(
                    worker.motion_incidents,
                    "process",
                return_value=incident_outcome(
                    event_id=1,
                    snapshot_path="",
                    object_detected=False,
                ),
            ) as process_event:
                thread = worker.motion_decisions._thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                now = datetime.now(timezone.utc)
                worker.handle_motion_event("onvif/motion", "first", now)
                worker.handle_motion_event("onvif/motion", "second", now)
                deadline = time.monotonic() + 2
                while process_event.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.02)
                worker._stop.set()
                worker.motion_events.queue.put_nowait(None)
                thread.join(timeout=2)

            self.assertEqual(process_event.call_count, 1)
            qualifications = [payload for event_type, payload in published if event_type == "motion_qualification"]
            self.assertEqual(len(qualifications), 1)
            self.assertEqual(qualifications[0]["trigger_count"], 2)

    def test_audit_records_detector_failure_as_not_run(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="audit", burst_quiet_seconds=0.1)
        rejected = MotionQualificationResult(False, 0.2, 0.5, "low_score", 3, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            with (
                patch.object(worker.motion_qualification, "qualify_burst", return_value=(rejected, {})),
                patch.object(
                    worker.motion_incidents,
                    "process",
                    return_value=incident_outcome(
                        event_id=5,
                        snapshot_path="snapshot.jpg",
                        object_detected=None,
                    ),
                ),
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
            ):
                thread = worker.motion_decisions._thread = threading.Thread(target=worker.motion_decisions.run, args=(worker._stop,))
                thread.start()
                worker.handle_motion_event("onvif/motion", "motion")
                deadline = time.monotonic() + 2
                while record_audit.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker.motion_events.queue.put_nowait(None)
                thread.join(timeout=2)

            record_audit.assert_called_once()
            self.assertIsNone(record_audit.call_args.kwargs["object_detected"])

    def test_all_zero_rolling_windows_fail_open_as_inconclusive(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(post_trigger_seconds=0.5, window_seconds=0.8)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            received_at = time.time() - 1.0
            for index in range(5):
                worker.motion_analysis.frames.append((received_at - 0.8 + index * 0.2, np.zeros((90, 160), dtype=np.uint8)))

            result, diagnostics = worker.motion_qualification.qualify_burst(
                datetime.fromtimestamp(received_at, timezone.utc),
                received_at,
                "balanced",
                worker.motion_analysis,
            )

            self.assertTrue(result.accepted)
            self.assertEqual(result.reason, "no_temporal_signal")
            self.assertGreater(diagnostics["windows_evaluated"], 0)
            self.assertNotIn("mog2_warmed", result.features)
            self.assertEqual(result.features["event_state_phase"], "active")

    def test_worker_uses_final_state_machine_trigger_decision(self) -> None:
        camera = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://example.invalid/main",
        )
        config = MotionQualificationConfig.model_validate({
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": "audit"},
                    },
                    {
                        "stage_id": "event_state",
                        "implementation": "score_event_state",
                        "options": {"activation_frames": 2},
                    },
                    {
                        "stage_id": "trigger",
                        "implementation": "score_trigger",
                    },
                ],
            },
        })
        score = MotionQualificationResult(
            accepted=True,
            score=0.8,
            threshold=0.48,
            reason="qualified",
            frame_count=9,
            features={},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)

            candidate = worker.motion_qualification.with_source_evidence(score, 9.0, 10.0)
            active = worker.motion_qualification.with_source_evidence(score, 10.0, 11.0)

        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.reason, "event_state_candidate")
        self.assertTrue(active.accepted)
        self.assertEqual(active.reason, "qualified")
        self.assertEqual(active.telemetry["schema_version"], 1)
        self.assertEqual(
            set(active.telemetry["graphs"]),
            {"qualification", "observation", "fusion"},
        )
        self.assertIn(
            "event_state",
            active.telemetry["graphs"]["fusion"]["invocation_timings"],
        )

    def test_motion_event_runs_detection_on_live_fallback(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        detector = DummyDetector()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), detector=detector)
            with (
                patch("survng.app.motion_pipeline.object_detection.RECORDED_EVENT_SETTLE_SECONDS", 0.0),
                patch("survng.app.motion_pipeline.object_detection.RECORDED_EVENT_RETRY_SECONDS", 0.0),
                patch.object(worker, "_get_latest_frame", lambda source="live": frame.copy()),
            ):
                fallback, objects, recording_path = worker._recorded_motion_frame(
                    datetime(2026, 7, 11, 15, 36, 57, tzinfo=timezone.utc)
                )

        self.assertIsNotNone(fallback)
        self.assertEqual(recording_path, "")
        self.assertEqual(detector.calls, 1)
        self.assertEqual(objects[0]["label"], "car")
        self.assertEqual(objects[0]["frame_source"], "live_fallback")
        self.assertEqual(objects[0]["recording_status"], "no_recorded_frame")

    def test_recorded_frame_retry_deadline_bounds_ffmpeg_attempts(self) -> None:
        camera = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://example.invalid/main",
        )
        detector = DummyDetector()
        fallback = np.zeros((10, 10, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            recording = Path(tmpdir) / "segment.mp4"
            recording.touch()
            recorder = DummyRecorder()
            recorder.recording_at = Mock(return_value={
                "path": str(recording),
                "start_epoch": time.time() - 1.0,
            })
            worker = make_worker(
                camera,
                Path(tmpdir),
                detector=detector,
                recorder=recorder,
            )

            timeouts: list[float] = []

            def timeout_ffmpeg(command, **kwargs):
                timeout = float(kwargs["timeout"])
                timeouts.append(timeout)
                time.sleep(timeout)
                raise subprocess.TimeoutExpired(command, timeout)

            started = time.monotonic()
            with (
                patch("survng.app.motion_pipeline.object_detection.RECORDED_EVENT_SETTLE_SECONDS", 0.0),
                patch("survng.app.motion_pipeline.object_detection.RECORDED_EVENT_RETRY_SECONDS", 0.04),
                patch("survng.app.motion_pipeline.object_detection.subprocess.run", side_effect=timeout_ffmpeg),
                patch.object(worker, "_get_latest_frame", return_value=fallback.copy()),
            ):
                frame, objects, recording_path = worker._recorded_motion_frame(
                    datetime.fromtimestamp(time.time() - 2.0, timezone.utc)
                )
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.15)
        self.assertEqual(len(timeouts), 1)
        self.assertLessEqual(timeouts[0], 0.04)
        self.assertIsNotNone(frame)
        self.assertEqual(recording_path, "")
        self.assertEqual(objects[0]["frame_source"], "live_fallback")


if __name__ == "__main__":
    unittest.main()
