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

from survng.app.camera import (
    CAPTURE_OPEN_TIMEOUT_MS,
    CAPTURE_OPEN_CONCURRENCY,
    CAPTURE_OPEN_SLOTS,
    CAPTURE_READ_TIMEOUT_MS,
    CAPTURE_STOP_TIMEOUT_SECONDS,
    CameraWorker,
    FRAME_STALE_SECONDS,
)
from survng.app.config import CameraConfig, MotionQualificationConfig, ObjectTrackingConfig
from survng.app.detector import objects_to_json
from survng.app.motion import MotionQualificationResult
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
    pass


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
    return CameraWorker(
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
        motion_analysis_limiter=threading.BoundedSemaphore(2),
    )


class CameraWorkerTest(unittest.TestCase):
    def test_main_stream_buffer_bridges_unfinalized_recording_gap(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker.object_tracking.config = ObjectTrackingConfig(sample_fps=2.0)
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
            worker.object_tracking.config = ObjectTrackingConfig(sample_fps=2.0)
            frame = np.zeros((90, 160, 3), dtype=np.uint8)
            worker._remember_tracking_frame(frame, 100.0)
            worker._remember_tracking_frame(frame, 100.1)
            worker._remember_tracking_frame(frame, 100.5)

        self.assertEqual([sample[0] for sample in worker._tracking_frames], [100.0, 100.5])

    def test_tracking_frame_includes_capture_time_and_rejects_stale_cache(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            frame = np.zeros((20, 30, 3), dtype=np.uint8)
            captured_at = time.time() - 0.1
            frame_clock = time.monotonic() - 0.1
            with worker._frame_lock:
                worker._source_frames["main"] = frame
                worker._source_frame_epoch["main"] = captured_at
                worker._source_frame_monotonic["main"] = frame_clock
            with patch.object(worker, "_start_source", return_value=True):
                sample = worker._get_latest_tracking_frame("main")
                with worker._frame_lock:
                    worker._source_frame_monotonic["main"] = (
                        time.monotonic() - FRAME_STALE_SECONDS - 1.0
                    )
                stale = worker._get_latest_tracking_frame("main")

        self.assertIsNotNone(sample)
        sampled_frame, sampled_at, sampled_clock = sample
        self.assertIsNot(sampled_frame, frame)
        self.assertEqual(sampled_at, captured_at)
        self.assertEqual(sampled_clock, frame_clock)
        self.assertIsNone(stale)

    def test_snapshot_filename_uses_event_time_not_processing_time(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        event_at = datetime(2026, 7, 27, 15, 56, 55, 123456, tzinfo=timezone.utc)
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            path = Path(worker._write_snapshot(frame, event_at))

            self.assertTrue(path.is_file())
            self.assertTrue(path.name.startswith("20260727-155655-123456-"))

    def test_isolated_qualification_telemetry_reports_replay_stage_metrics(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        frames = [
            np.zeros((90, 160), dtype=np.uint8),
            np.full((90, 160), 20, dtype=np.uint8),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))

            result = worker._run_motion_pipeline(
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
                worker._stop.set()
                return False

            capture.open.side_effect = stop_after_open
            with patch("survng.app.camera.cv2.VideoCapture", return_value=capture):
                worker._run_source("live", threading.Event())

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
            for _ in range(CAPTURE_OPEN_CONCURRENCY):
                CAPTURE_OPEN_SLOTS.acquire()
            try:
                thread = threading.Thread(
                    target=lambda: result.append(
                        worker._open_capture(capture, "live", stop_event)
                    )
                )
                thread.start()
                time.sleep(0.02)
                stop_event.set()
                thread.join(timeout=1)
            finally:
                for _ in range(CAPTURE_OPEN_CONCURRENCY):
                    CAPTURE_OPEN_SLOTS.release()

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
                outcome = worker._process_motion_event("motion", "message", datetime.now(timezone.utc), qualification)

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
            worker.object_tracking.config = ObjectTrackingConfig(
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
                patch("survng.app.camera.cv2.imread", return_value=seed) as imread,
                patch.object(worker, "_get_latest_tracking_frame", return_value=None) as prewarm,
                patch.object(worker.object_tracking, "start", return_value=True) as start,
            ):
                worker._process_motion_event(
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
                outcome = worker._process_motion_event("motion", "message", datetime.now(timezone.utc), qualification)

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
            thread = worker._motion_analysis_thread = threading.Thread(
                target=worker._run_motion_analysis
            )
            thread.start()
            worker._remember_motion_frame(np.zeros((720, 1280, 3), dtype=np.uint8), time.monotonic())
            deadline = time.monotonic() + 1
            while worker._motion_stats["continuous_frames"] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertEqual(worker._motion_settings(), ("camera", "balanced", 480))
            self.assertEqual(worker.status()["motion_qualification"]["frame_width"], 480)
            self.assertEqual(worker.status()["motion_qualification"]["frame_shape"], [270, 480])
            self.assertFalse(worker.status()["motion_qualification"]["mog2_audit_enabled"])
            self.assertIsNone(worker.status()["motion_qualification"]["mog2_last"])
            worker._stop.set()
            worker._signal_motion_analysis_stop()
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

            worker._remember_motion_frame(
                np.zeros((180, 320, 3), dtype=np.uint8),
                time.monotonic(),
            )

            self.assertEqual(len(worker._motion_frames), 0)
            self.assertTrue(worker._motion_analysis_queue.empty())

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
            thread = worker._motion_analysis_thread = threading.Thread(
                target=worker._run_motion_analysis
            )
            thread.start()
            with (
                patch.object(worker, "_run_motion_pipeline", return_value=accepted) as analyze,
                patch.object(worker, "_with_source_evidence", return_value=accepted),
            ):
                worker._remember_motion_frame(np.zeros((180, 320, 3), dtype=np.uint8), 1.0)
                worker._remember_motion_frame(np.zeros((180, 320, 3), dtype=np.uint8), 2.0)
                deadline = time.monotonic() + 1
                while worker._motion_queue.empty() and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertGreaterEqual(analyze.call_count, 1)
            trigger = worker._motion_queue.get_nowait()
            self.assertEqual(trigger["topic"], "adaptive/motion")
            self.assertIs(trigger["prequalified"], accepted)
            worker._stop.set()
            worker._signal_motion_analysis_stop()
            thread.join(timeout=1)

    def test_audit_mode_learns_continuously_without_creating_events(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        accepted = MotionQualificationResult(True, 0.8, 0.5, "qualified", 2, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            with patch.object(worker, "_run_motion_pipeline", return_value=accepted):
                with worker._frame_lock:
                    worker._motion_frames.extend([
                        (1.0, np.zeros((90, 160), dtype=np.uint8)),
                        (2.0, np.zeros((90, 160), dtype=np.uint8)),
                    ])
                worker._analyze_continuous_motion(2.0)

            self.assertTrue(worker._motion_queue.empty())
            self.assertEqual(worker._motion_stats["continuous_candidates"], 1)

    def test_adaptive_mode_records_onvif_without_enqueuing_detection(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(mode="adaptive"),
            )

            worker.handle_motion_event("onvif/motion", "generic")

            self.assertTrue(worker._motion_queue.empty())
            self.assertIsNotNone(worker.motion_evidence.last("onvif"))
            self.assertEqual(worker._motion_stats["triggers"], 0)

    def test_adaptive_mode_still_allows_manual_test_trigger(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(mode="adaptive"),
            )

            worker.handle_motion_event("manual/test", "manual GUI trigger")

            self.assertEqual(worker._motion_queue.get_nowait()["topic"], "manual/test")

    def test_continuous_analysis_never_blocks_camera_frame_delivery(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            thread = worker._motion_analysis_thread = threading.Thread(
                target=worker._run_motion_analysis
            )
            with patch.object(worker, "_analyze_continuous_motion", side_effect=lambda _at: time.sleep(0.2)):
                thread.start()
                started = time.perf_counter()
                worker._remember_motion_frame(np.zeros((180, 320, 3), dtype=np.uint8), 1.0)
                worker._remember_motion_frame(np.zeros((180, 320, 3), dtype=np.uint8), 2.0)
                elapsed = time.perf_counter() - started

            worker._stop.set()
            worker._signal_motion_analysis_stop()
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
                worker._motion_frames.extend([
                    (1.0, np.zeros((90, 160), dtype=np.uint8)),
                    (2.0, np.zeros((90, 160), dtype=np.uint8)),
                ])
            with (
                patch.object(worker, "_run_motion_pipeline", return_value=accepted),
                patch.object(worker, "_with_source_evidence", return_value=accepted),
            ):
                for captured_at in range(10, 110, 5):
                    worker._analyze_continuous_motion(float(captured_at))

            self.assertEqual(worker._motion_queue.qsize(), 1)
            self.assertEqual(worker._motion_stats["triggers"], 1)
            self.assertEqual(worker._motion_stats["adaptive_triggers_deferred"], 19)

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
            for index in range(worker._motion_queue.maxsize):
                worker._motion_queue.put_nowait({
                    "topic": "onvif/motion",
                    "message": str(index),
                    "event_at": now,
                    "received_at": now.timestamp(),
                })
            with worker._frame_lock:
                worker._motion_frames.extend([
                    (1.0, np.zeros((90, 160), dtype=np.uint8)),
                    (2.0, np.zeros((90, 160), dtype=np.uint8)),
                ])
            with (
                patch.object(worker, "_run_motion_pipeline", return_value=accepted),
                patch.object(worker, "_with_source_evidence", return_value=accepted),
            ):
                worker._analyze_continuous_motion(10.0)
                worker._analyze_continuous_motion(11.0)

            self.assertEqual(worker._motion_queue.qsize(), worker._motion_queue.maxsize)
            self.assertEqual(worker._motion_stats["triggers"], 1)
            self.assertEqual(worker._motion_stats["dropped_triggers"], 1)
            self.assertEqual(worker._motion_stats["adaptive_triggers_deferred"], 1)
            self.assertFalse(worker._adaptive_trigger_pending)
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
                worker._motion_frames.extend([
                    (1.0, np.zeros((90, 160), dtype=np.uint8)),
                    (2.0, np.zeros((90, 160), dtype=np.uint8)),
                ])
            with (
                patch.object(worker, "_run_motion_pipeline", return_value=rejected),
                patch.object(
                    worker,
                    "_with_source_evidence",
                    wraps=worker._with_source_evidence,
                ) as fuse,
            ):
                worker._analyze_continuous_motion(2.0)

            self.assertTrue(fuse.call_args.kwargs["require_primary_trigger"])
            self.assertTrue(worker._motion_queue.empty())
            self.assertEqual(
                worker._motion_last_continuous_result.reason,
                "primary_trigger_rejected",
            )
            self.assertEqual(
                worker._motion_last_continuous_result.features["event_state_phase"],
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
                worker._motion_frames.append((
                    received_at - 0.8 + index * 0.2,
                    np.zeros((90, 160), dtype=np.uint8),
                ))
            with patch.object(
                worker,
                "_run_motion_pipeline",
                side_effect=RuntimeError("validator unavailable"),
            ):
                result, diagnostics = worker._qualify_motion_burst(
                    datetime.fromtimestamp(received_at, timezone.utc),
                    received_at,
                    "balanced",
                )

            self.assertTrue(result.accepted)
            self.assertEqual(result.reason, "validation_unavailable_fail_open")
            self.assertTrue(result.features["validation_fail_open"])
            self.assertEqual(diagnostics["windows_evaluated"], 1)
            self.assertEqual(worker._motion_stats["validation_fail_opens"], 1)

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
                worker._motion_frames.append((
                    received_at - 0.8 + index * 0.2,
                    np.zeros((90, 160), dtype=np.uint8),
                ))
            with (
                patch.object(worker, "_run_motion_pipeline", return_value=accepted) as analyze,
                patch.object(worker, "_with_source_evidence", return_value=accepted) as fuse,
            ):
                result, _diagnostics = worker._qualify_motion_burst(
                    datetime.fromtimestamp(received_at, timezone.utc),
                    received_at,
                    "balanced",
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
                patch.object(worker, "_with_source_evidence", return_value=accepted) as fuse,
            ):
                result, diagnostics = worker._qualify_motion_burst(
                    datetime.fromtimestamp(received_at, timezone.utc),
                    received_at,
                    "balanced",
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
            self.assertTrue(worker._external_confirmation_required())

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
            self.assertFalse(worker._external_confirmation_required())

    def test_shutdown_during_validation_does_not_start_object_detection(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            event_at = datetime.now(timezone.utc)
            worker._motion_queue.put_nowait({
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
                patch.object(worker, "_qualify_motion_burst", side_effect=stop_during_validation),
                patch.object(worker, "_process_motion_event") as process_event,
            ):
                worker._run_motion_events_until_error()

            process_event.assert_not_called()
            self.assertIsNone(worker._active_motion_triggers)

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
                result = worker._with_source_evidence(accepted, 1.0, 2.0)

            self.assertTrue(result.accepted)
            self.assertEqual(result.reason, "validation_unavailable_fail_open")
            self.assertEqual(worker._motion_stats["validation_fail_opens"], 1)

    def test_explicit_fail_closed_result_cannot_be_borderline_rescued(self) -> None:
        result = MotionQualificationResult(
            False,
            0.8,
            0.48,
            "validation_unavailable_fail_closed",
            4,
            {},
        )

        self.assertFalse(CameraWorker._is_borderline_candidate(result, True, 0.03))

    def test_categorical_nuisance_rejection_cannot_be_borderline_rescued(self) -> None:
        result = MotionQualificationResult(
            False,
            0.8,
            0.48,
            "stationary_foreground",
            4,
            {},
        )

        self.assertFalse(CameraWorker._is_borderline_candidate(result, True, 0.03))

    def test_low_score_within_margin_can_be_borderline_rescued(self) -> None:
        result = MotionQualificationResult(False, 0.46, 0.48, "low_score", 4, {})

        self.assertTrue(CameraWorker._is_borderline_candidate(result, True, 0.03))

    def test_suppression_verification_sampling_is_stable_per_decision(self) -> None:
        decision_id = "fixed-decision"

        first = CameraWorker._should_verify_suppression(decision_id, 0.5)

        self.assertEqual(
            first,
            CameraWorker._should_verify_suppression(decision_id, 0.5),
        )
        self.assertFalse(CameraWorker._should_verify_suppression(decision_id, 0.0))
        self.assertTrue(CameraWorker._should_verify_suppression(decision_id, 1.0))

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
            camera_mode._motion_primary_last_processed_at = 10.0
            self.assertFalse(camera_mode._continuous_primary_analysis_due(10.2))
            self.assertTrue(camera_mode._continuous_primary_analysis_due(10.5))

            adaptive_mode = make_worker(
                camera,
                Path(tmpdir),
                motion_config=MotionQualificationConfig(
                    mode="adaptive",
                    camera_mode_background_fps=0.5,
                ),
            )
            adaptive_mode._motion_primary_last_processed_at = 10.0
            self.assertTrue(adaptive_mode._continuous_primary_analysis_due(10.01))

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
                patch.object(worker, "_qualify_motion_burst", return_value=(rejected, {})),
                patch.object(
                    worker,
                    "_process_motion_event",
                    return_value={"event_id": None, "snapshot_path": "checked.jpg", "object_detected": False},
                ) as process_event,
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
            ):
                thread = threading.Thread(target=worker._run_motion_events)
                thread.start()
                worker.handle_motion_event("onvif/motion", "motion")
                deadline = time.monotonic() + 2
                while record_audit.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker._motion_queue.put_nowait(None)
                thread.join(timeout=2)

            process_event.assert_called_once()
            self.assertTrue(process_event.call_args.kwargs["require_eligible_object"])
            record_audit.assert_called_once()
            self.assertIsNone(record_audit.call_args.kwargs["event_id"])
            self.assertTrue(record_audit.call_args.kwargs["features"]["suppression_verification"])
            self.assertEqual(worker._motion_stats["suppression_verification_checks"], 1)
            self.assertEqual(worker._motion_stats["suppression_verification_rescues"], 0)
            self.assertEqual(worker._motion_stats["suppressed"], 1)

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
                patch.object(worker, "_qualify_motion_burst", return_value=(rejected, {})),
                patch.object(
                    worker,
                    "_process_motion_event",
                    return_value={"event_id": 9, "snapshot_path": "person.jpg", "object_detected": True},
                ),
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
            ):
                thread = threading.Thread(target=worker._run_motion_events)
                thread.start()
                worker.handle_motion_event("onvif/motion", "motion")
                deadline = time.monotonic() + 2
                while record_audit.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker._motion_queue.put_nowait(None)
                thread.join(timeout=2)

            self.assertEqual(record_audit.call_args.kwargs["event_id"], 9)
            self.assertEqual(worker._motion_stats["suppression_verification_checks"], 1)
            self.assertEqual(worker._motion_stats["suppression_verification_rescues"], 1)
            self.assertEqual(worker._motion_stats["suppressed"], 0)

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
                result = worker._with_source_evidence(
                    rejected,
                    1.0,
                    2.0,
                    require_primary_trigger=True,
                )

            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, "primary_trigger_rejected")
            self.assertFalse(result.features["validation_fail_open"])
            self.assertEqual(worker._motion_stats["validation_failures"], 1)
            self.assertEqual(worker._motion_stats["validation_fail_opens"], 0)

    def test_analysis_worker_survives_a_transient_cycle_failure(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            thread = worker._motion_analysis_thread = threading.Thread(
                target=worker._run_motion_analysis
            )
            with patch.object(
                worker,
                "_analyze_continuous_motion",
                side_effect=[RuntimeError("transient fusion failure"), None],
            ) as analyze:
                thread.start()
                with worker._frame_lock:
                    worker._motion_frames.append((1.0, np.zeros((90, 160), dtype=np.uint8)))
                worker._schedule_motion_analysis(1.0)
                deadline = time.monotonic() + 1
                while worker._motion_stats["analysis_worker_errors"] == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                with worker._frame_lock:
                    worker._motion_frames.append((2.0, np.zeros((90, 160), dtype=np.uint8)))
                worker._schedule_motion_analysis(2.0)
                deadline = time.monotonic() + 1
                while analyze.call_count < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertTrue(thread.is_alive())
            self.assertEqual(analyze.call_count, 2)
            self.assertEqual(worker._motion_stats["analysis_worker_errors"], 1)
            worker._stop.set()
            worker._signal_motion_analysis_stop()
            thread.join(timeout=1)

    def test_event_worker_survives_a_transient_handler_failure(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="off", burst_quiet_seconds=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            with patch.object(
                worker,
                "_process_motion_event",
                side_effect=[
                    RuntimeError("transient event failure"),
                    {"event_id": 2, "snapshot_path": "", "object_detected": False},
                    {"event_id": 3, "snapshot_path": "", "object_detected": False},
                ],
            ) as process_event:
                thread = worker._motion_thread = threading.Thread(target=worker._run_motion_events)
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
            self.assertEqual(worker._motion_stats["event_worker_errors"], 1)
            self.assertEqual(worker._motion_stats["event_retries"], 1)
            self.assertEqual(worker._motion_stats["bursts"], 2)
            self.assertEqual(worker._motion_stats["passed"], 2)
            worker._stop.set()
            worker._motion_queue.put_nowait(None)
            thread.join(timeout=1)

    def test_retry_batch_is_protected_from_normal_queue_eviction(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            now = datetime.now(timezone.utc)
            worker._retry_motion_trigger_batch([{
                "topic": "onvif/motion",
                "message": "retry me",
                "event_at": now,
                "received_at": now.timestamp(),
            }])
            for index in range(worker._motion_queue.maxsize):
                worker._motion_queue.put_nowait({
                    "topic": "onvif/motion",
                    "message": str(index),
                    "event_at": now,
                    "received_at": now.timestamp(),
                })

            worker._enqueue_motion_trigger({
                "topic": "onvif/motion",
                "message": "new priority event",
                "event_at": now,
                "received_at": now.timestamp(),
            })

            self.assertEqual(len(worker._motion_retry_batches), 1)
            self.assertEqual(
                worker._motion_retry_batches[0]["_retry_batch"][0]["message"],
                "retry me",
            )
            self.assertEqual(worker._motion_stats["event_retry_drops"], 0)

    def test_event_worker_bounds_retries_for_a_persistent_handler_failure(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="off", burst_quiet_seconds=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()
            with patch.object(
                worker,
                "_process_motion_event",
                side_effect=RuntimeError("persistent event failure"),
            ) as process_event:
                thread = worker._motion_thread = threading.Thread(target=worker._run_motion_events)
                thread.start()
                worker.handle_motion_event("onvif/motion", "first")
                deadline = time.monotonic() + 3
                while worker._motion_stats["event_retry_drops"] == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertTrue(thread.is_alive())
            self.assertEqual(process_event.call_count, 3)
            self.assertEqual(worker._motion_stats["event_retries"], 2)
            self.assertEqual(worker._motion_stats["event_retry_drops"], 1)
            worker._stop.set()
            worker._motion_queue.put_nowait(None)
            thread.join(timeout=1)

    def test_enforce_retry_reuses_accepted_qualification_without_advancing_fusion(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="enforce", burst_quiet_seconds=0.1)
        accepted = MotionQualificationResult(True, 0.8, 0.5, "qualified", 4, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            worker._stop.clear()

            def qualify(*_args):
                return worker._with_source_evidence(
                    accepted,
                    time.time() - 1,
                    time.time(),
                ), {
                    "windows_evaluated": 1,
                    "event_receipt_delta_seconds": 0.0,
                }

            with (
                patch.object(worker, "_qualify_motion_burst", side_effect=qualify),
                patch.object(
                    worker,
                    "_process_motion_event",
                    side_effect=[
                        RuntimeError("transient event failure"),
                        {"event_id": 2, "snapshot_path": "", "object_detected": False},
                    ],
                ) as process_event,
            ):
                thread = worker._motion_thread = threading.Thread(target=worker._run_motion_events)
                thread.start()
                worker.handle_motion_event("onvif/motion", "generic")
                deadline = time.monotonic() + 2
                while process_event.call_count < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertTrue(thread.is_alive())
            self.assertEqual(process_event.call_count, 2)
            self.assertEqual(worker._motion_stats["event_retries"], 1)
            self.assertEqual(worker._motion_stats["suppressed"], 0)
            self.assertTrue(worker._motion_stats["last_result"]["effective_accepted"])
            worker._stop.set()
            worker._motion_queue.put_nowait(None)
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
                return worker._with_source_evidence(
                    accepted,
                    time.time() - 1,
                    time.time(),
                ), {
                    "windows_evaluated": 1,
                    "event_receipt_delta_seconds": 0.0,
                }

            with (
                patch.object(worker, "_qualify_motion_burst", side_effect=qualify),
                patch.object(
                    worker,
                    "_sample_rejected_motion",
                    return_value="sample.jpg",
                ) as sample_rejected,
                patch.object(
                    worker.motion_decision_handler,
                    "record_audit",
                    side_effect=[RuntimeError("audit write failed"), {}],
                ) as record_audit,
                patch.object(worker, "_process_motion_event") as process_event,
            ):
                thread = worker._motion_thread = threading.Thread(target=worker._run_motion_events)
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
            self.assertEqual(first_audit["snapshot_path"], "sample.jpg")
            self.assertEqual(retry_audit["snapshot_path"], "sample.jpg")
            process_event.assert_not_called()
            self.assertEqual(worker._motion_stats["event_retries"], 1)
            self.assertEqual(worker._motion_stats["bursts"], 1)
            self.assertEqual(worker._motion_stats["suppressed"], 1)
            self.assertEqual(
                worker._motion_stats["last_result"]["reason"],
                "event_state_candidate",
            )
            self.assertFalse(worker._motion_stats["last_result"]["effective_accepted"])
            qualifications = [
                payload
                for event_type, payload in published
                if event_type == "motion_qualification"
            ]
            self.assertEqual(len(qualifications), 1)
            worker._stop.set()
            worker._motion_queue.put_nowait(None)
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
                    worker,
                    "_process_motion_event",
                    return_value={"event_id": 1, "snapshot_path": "", "object_detected": False},
                ) as process_event,
                patch.object(
                    worker.motion_decision_handler,
                    "record_audit",
                    side_effect=RuntimeError("audit write failed"),
                ),
            ):
                thread = worker._motion_thread = threading.Thread(target=worker._run_motion_events)
                thread.start()
                now = datetime.now(timezone.utc)
                worker._enqueue_motion_trigger({
                    "topic": "adaptive/motion",
                    "message": "adaptive",
                    "event_at": now,
                    "received_at": now.timestamp(),
                    "prequalified": rejected,
                })
                deadline = time.monotonic() + 1
                while worker._motion_stats["event_worker_errors"] == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertTrue(thread.is_alive())
            process_event.assert_called_once()
            self.assertEqual(worker._motion_stats["event_retries"], 0)
            worker._stop.set()
            worker._motion_queue.put_nowait(None)
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
                    worker._with_source_evidence(accepted, end_epoch - 1, end_epoch)
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
            worker._with_source_evidence(accepted, 19.0, 20.0)

            stale = worker._with_source_evidence(rejected, 18.0, 19.0)
            still_active = worker._with_source_evidence(accepted, 19.5, 20.5)
            clock_rollback = worker._with_source_evidence(rejected, 9.0, 10.0)

            self.assertEqual(stale.reason, "stale_fusion_evidence")
            self.assertEqual(stale.score, 0.0)
            self.assertEqual(stale.threshold, 1.0)
            self.assertEqual(worker._motion_stats["stale_fusion_samples"], 1)
            self.assertEqual(still_active.features["event_state_phase"], "active")
            self.assertEqual(clock_rollback.features["event_state_phase"], "rejected")

    def test_priority_onvif_bypasses_active_adaptive_event_state(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="audit", burst_quiet_seconds=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir), motion_config=config)
            active = MotionQualificationResult(True, 0.8, 0.5, "qualified", 2, {})
            self.assertTrue(worker._with_source_evidence(active, 1.0, 2.0).accepted)
            worker._stop.clear()
            with patch.object(
                worker,
                "_process_motion_event",
                return_value={"event_id": 1, "snapshot_path": "", "object_detected": False},
            ) as process_event:
                thread = worker._motion_thread = threading.Thread(
                    target=worker._run_motion_events
                )
                thread.start()
                worker.handle_motion_event("onvif/person", "person")
                deadline = time.monotonic() + 1
                while process_event.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker._motion_queue.put_nowait(None)
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
                worker,
                "_process_motion_event",
                return_value={"event_id": 1, "snapshot_path": "", "object_detected": True},
            ) as process_event:
                thread = worker._motion_thread = threading.Thread(target=worker._run_motion_events)
                thread.start()
                worker.handle_motion_event("onvif/person", "person")
                deadline = time.monotonic() + 1
                while process_event.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)

                self.assertFalse(worker._reserve_adaptive_trigger(time.time()))
                distinct_event_at = time.time() + worker._priority_dedup_seconds() + 0.1
                self.assertTrue(worker._reserve_adaptive_trigger(distinct_event_at))
                worker._stop.set()
                worker._motion_queue.put_nowait(None)
                thread.join(timeout=1)

            process_event.assert_called_once()
            self.assertGreater(worker._motion_stats["adaptive_triggers_deferred"], 0)

    def test_priority_results_still_use_event_state_deduplication(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            priority = MotionQualificationResult(
                True, 1.0, 0.0, "priority_topic", 0, {"primary_motion_source": "onvif_priority"}
            )

            first = worker._with_source_evidence(priority, 10.0, 11.0)
            second = worker._with_source_evidence(priority, 11.0, 12.0)

            self.assertTrue(first.accepted)
            self.assertFalse(second.accepted)
            self.assertEqual(second.reason, "event_state_active")

    def test_active_and_cooldown_results_link_to_the_active_incident_event(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._active_incident_event_id = 42

            active = MotionQualificationResult(False, 0.7, 0.5, "event_state_active", 2, {})
            cooldown = MotionQualificationResult(False, 0.7, 0.5, "event_state_cooldown", 2, {})
            unrelated = MotionQualificationResult(False, 0.2, 0.5, "low_score", 2, {})

            self.assertEqual(worker._related_incident_event_id(active), 42)
            self.assertEqual(worker._related_incident_event_id(cooldown), 42)
            self.assertIsNone(worker._related_incident_event_id(unrelated))

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
            worker._active_incident_event_id = 42
            worker._stop.clear()
            with (
                patch.object(worker, "_qualify_motion_burst", return_value=(active, {})),
                patch.object(worker, "_sample_rejected_motion") as sample_rejected,
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
                patch.object(worker, "_process_motion_event") as process_event,
            ):
                thread = threading.Thread(target=worker._run_motion_events)
                thread.start()
                worker.handle_motion_event("onvif/motion", "continued activity")
                deadline = time.monotonic() + 2
                while record_audit.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker._motion_queue.put_nowait(None)
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
            with patch.object(worker, "_start_source", wraps=worker._start_source) as start_source:
                self.assertIsNone(worker.snapshot("main"))

        start_source.assert_not_called()

    def test_status_separates_power_capture_and_frame_freshness(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._enabled = True
            worker._stop.clear()
            worker._source_threads["live"] = Mock(is_alive=lambda: True)
            worker._source_frame_at["live"] = "2026-07-14T12:00:00+00:00"
            worker._source_frame_monotonic["live"] = time.monotonic()
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
            worker._source_last_access["main"] = time.monotonic() - 60

            self.assertTrue(worker._source_is_idle("main"))
            self.assertFalse(worker._source_is_idle("live"))

    def test_snapshot_rejects_a_stale_cached_frame(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            worker._source_threads["live"] = Mock(is_alive=lambda: True)
            worker._source_stops["live"] = threading.Event()
            worker._source_frames["live"] = np.zeros((10, 10, 3), dtype=np.uint8)
            worker._source_frame_monotonic["live"] = time.monotonic() - 60

            self.assertIsNone(worker.snapshot())

    def test_start_source_replaces_a_stopping_capture_thread(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._stop.clear()
            old_thread = Mock(is_alive=lambda: True)
            old_stop = threading.Event()
            old_stop.set()
            worker._source_threads["live"] = old_thread
            worker._source_stops["live"] = old_stop
            with patch.object(worker, "_run_source"):
                self.assertTrue(worker._start_source("live"))
                replacement = worker._source_threads["live"]
                replacement.join(timeout=1)

        self.assertIsNot(replacement, old_thread)
        self.assertIsNot(worker._source_stops.get("live"), old_stop)

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
            with patch("survng.app.camera.cv2.VideoCapture", return_value=capture):
                worker._run_source("live", stop_event)

        self.assertNotIn("live", worker._source_frames)
        self.assertEqual(worker.last_frame_at, "")

    def test_start_rolls_back_workers_when_capture_start_fails(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with patch.object(worker, "_start_source", side_effect=RuntimeError("no threads")):
                with self.assertRaisesRegex(RuntimeError, "no threads"):
                    worker.start()

            self.assertFalse(worker.status()["running"])
            self.assertIsNone(worker._motion_analysis_thread)
            self.assertIsNone(worker._motion_thread)

    def test_close_refuses_to_race_an_active_motion_worker(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker._motion_thread = Mock(is_alive=lambda: True)

            with self.assertRaisesRegex(RuntimeError, "motion events"):
                worker.close()

    def test_close_attempts_every_motion_pipeline_after_a_failure(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            worker.motion_pipeline = Mock()
            worker.motion_observation_pipeline = Mock()
            worker.motion_fusion_pipeline = Mock()
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

        self.assertEqual(worker.last_motion_at, "")
        self.assertEqual(detector.calls, 0)
        self.assertIsNone(worker.motion_evidence.last("onvif"))

    def test_motion_handler_enqueues_without_running_detection_inline(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = make_worker(camera, Path(tmpdir))
            with patch.object(worker, "_recorded_motion_frame") as recorded_frame:
                worker.handle_motion_event("onvif/motion", "motion")

            self.assertEqual(worker._motion_queue.qsize(), 1)
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
                worker,
                "_process_motion_event",
                return_value={"event_id": 1, "snapshot_path": "", "object_detected": False},
            ) as process_event:
                thread = worker._motion_thread = threading.Thread(target=worker._run_motion_events)
                thread.start()
                now = datetime.now(timezone.utc)
                worker.handle_motion_event("onvif/motion", "first", now)
                worker.handle_motion_event("onvif/motion", "second", now)
                deadline = time.monotonic() + 2
                while process_event.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.02)
                worker._stop.set()
                worker._motion_queue.put_nowait(None)
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
                patch.object(worker, "_qualify_motion_burst", return_value=(rejected, {})),
                patch.object(
                    worker,
                    "_process_motion_event",
                    return_value={"event_id": 5, "snapshot_path": "snapshot.jpg", "object_detected": None},
                ),
                patch.object(worker.motion_decision_handler, "record_audit") as record_audit,
            ):
                thread = worker._motion_thread = threading.Thread(target=worker._run_motion_events)
                thread.start()
                worker.handle_motion_event("onvif/motion", "motion")
                deadline = time.monotonic() + 2
                while record_audit.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                worker._stop.set()
                worker._motion_queue.put_nowait(None)
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
                worker._motion_frames.append((received_at - 0.8 + index * 0.2, np.zeros((90, 160), dtype=np.uint8)))

            result, diagnostics = worker._qualify_motion_burst(
                datetime.fromtimestamp(received_at, timezone.utc),
                received_at,
                "balanced",
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

            candidate = worker._with_source_evidence(score, 9.0, 10.0)
            active = worker._with_source_evidence(score, 10.0, 11.0)

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
