from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .camera_capture import (
    CaptureBackend,
    CameraCaptureService,
    CapturedFrame,
    CaptureOpenLimiter,
    OpenCvFfmpegCaptureBackend,
)
from .camera_status import CameraStatusService
from .camera_media import CameraMediaService
from .camera_lifecycle import CameraLifecycleService, CameraRuntimeState
from .config import CameraConfig, DetectionZone, MotionQualificationConfig
from .image_storage import DurableImageWriter
from .onvif_events import OnvifEventListener
from .motion_analysis import FairMotionAnalysisLimiter
from .motion_analysis_service import MotionAnalysisService
from .motion_qualification_service import MotionQualificationService
from .motion_coordinator import VisualBackupCoordinator
from .motion_events import MotionEventCoordinator
from .motion_decisions import MotionDecisionOrchestrator
from .motion_incidents import MotionIncidentService
from .motion_ingress import MotionEventIngressService
from .motion_runtime import CameraMotionState, MotionRuntimeService
from .object_tracking import ObjectTrackingSession, ObjectTrackingSessionFactory
from .object_tracking_lifecycle import ObjectTrackingLifecycle
from .tracking_frames import TrackingFrameService
from .motion_pipeline import (
    MotionDecisionHandlerFactory,
    MotionDebugSnapshotStore,
    MotionEvidenceRepository,
    MotionPipeline,
    RecordedMotionObjectDetectorFactory,
)

MOTION_QUEUE_SIZE = 32
MOTION_ANALYSIS_QUEUE_SIZE = 1
MOTION_EVENT_MAX_RETRIES = 2


class CameraWorker:
    def __init__(
        self,
        camera: CameraConfig,
        storage_dir: Path,
        motion_config: MotionQualificationConfig | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        motion_pipeline: MotionPipeline,
        motion_observation_pipeline: MotionPipeline,
        motion_fusion_pipeline: MotionPipeline,
        motion_evidence: MotionEvidenceRepository,
        motion_pipeline_origins: dict[str, str],
        motion_decision_handler_factory: MotionDecisionHandlerFactory,
        motion_object_detector_factory: RecordedMotionObjectDetectorFactory,
        object_tracking_session_factory: ObjectTrackingSessionFactory,
        motion_analysis_limiter: FairMotionAnalysisLimiter,
        image_writer: DurableImageWriter,
        onvif_cache_dir: Path | None = None,
        capture_backend: CaptureBackend | None = None,
    ) -> None:
        self.camera = camera
        self.storage_dir = storage_dir
        self.motion_config = motion_config or MotionQualificationConfig()
        self.motion_pipeline = motion_pipeline
        self.motion_observation_pipeline = motion_observation_pipeline
        self.motion_fusion_pipeline = motion_fusion_pipeline
        self.motion_evidence = motion_evidence
        self.motion_pipeline_origins = dict(motion_pipeline_origins)
        self.runtime_state = CameraRuntimeState()
        self.motion_state = CameraMotionState(
            camera_id=camera.id,
            camera_state=self.runtime_state,
            event_callback=event_callback,
        )
        # Runtime-state reads are independent from tracking-session operations;
        # neither lock is held while camera lifecycle I/O is blocking.
        self._stop = self.runtime_state.stop_event
        self._lifecycle_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        effective_capture_backend = capture_backend or OpenCvFfmpegCaptureBackend(
            CaptureOpenLimiter()
        )
        self.motion_debug = MotionDebugSnapshotStore()
        self.motion_object_detector = motion_object_detector_factory.create(
            camera=camera,
            live_frame_provider=lambda: self._get_latest_frame(),
        )
        self.media = CameraMediaService(
            camera=camera,
            storage_dir=storage_dir,
            image_writer=image_writer,
            motion_detector=self.motion_object_detector,
            frame_provider=lambda source: self._get_latest_frame(source),
            rejected_sample_rate=lambda: self.motion_config.rejected_sample_rate,
            stop_requested=self._stop.is_set,
        )
        self.tracking_lifecycle = ObjectTrackingLifecycle(
            camera=camera,
            factory=object_tracking_session_factory,
            frame_provider=self._get_latest_tracking_frame_with_fallback,
            catchup_frame_provider=self._recorded_tracking_frames,
            prewarm_frame_provider=lambda: self._get_latest_tracking_frame("main"),
            history=lambda: self.tracking_frames,
            accepting=lambda: (
                self.motion_state.detection_enabled() and not self._stop.is_set()
            ),
            lifecycle_lock=self._lifecycle_lock,
        )
        self.motion_decision_handler = motion_decision_handler_factory.create(
            camera_id=camera.id,
            detection_provider=lambda event_at: self._recorded_motion_frame(event_at),
            snapshot_writer=lambda frame, event_at: self._write_snapshot(frame, event_at),
            event_callback=(
                self.motion_state.publish_event if event_callback is not None else None
            ),
        )
        self.motion_incidents = MotionIncidentService(
            camera_id=camera.id,
            decision_processor=self.motion_decision_handler,
            tracking_enabled=self.tracking_lifecycle.enabled,
            has_trackable_objects=self.tracking_lifecycle.has_trackable_objects,
            start_tracking=self.tracking_lifecycle.start_incident,
            prewarm_tracking=self.tracking_lifecycle.prewarm,
            image_reader=self.media.read_image,
        )
        ring_size = max(
            12,
            round(
                self.motion_config.sample_fps
                * (self.motion_config.window_seconds + self.motion_config.post_trigger_seconds + 3.0)
            ),
        )
        self.motion_events = MotionEventCoordinator(
            queue_size=MOTION_QUEUE_SIZE,
            retry_limit=MOTION_EVENT_MAX_RETRIES,
        )
        self.visual_backup = VisualBackupCoordinator()
        self.motion_qualification = MotionQualificationService(
            camera=camera,
            config=self.motion_config,
            qualification_pipeline=self.motion_pipeline,
            observation_pipeline=self.motion_observation_pipeline,
            fusion_pipeline=self.motion_fusion_pipeline,
            pipeline_origins=self.motion_pipeline_origins,
            debug_store=self.motion_debug,
            stop_event=self._stop,
            state=self.motion_state,
        )
        self.motion_analysis = MotionAnalysisService(
            camera_id=camera.id,
            frame_lock=self._frame_lock,
            analysis_lock=self.motion_qualification.analysis_lock,
            ring_size=ring_size,
            queue_size=MOTION_ANALYSIS_QUEUE_SIZE,
            limiter=motion_analysis_limiter,
            events=self.motion_events,
            evidence=self.motion_evidence,
            visual_backup=self.visual_backup,
            audit_recorder=self.motion_decision_handler,
            debug_store=self.motion_debug,
            config=self.motion_config,
            qualification=self.motion_qualification,
            media=self.media,
            state=self.motion_state,
        )
        self.motion_decisions = MotionDecisionOrchestrator(
            camera_id=camera.id,
            events=self.motion_events,
            audit_recorder=self.motion_decision_handler,
            config=self.motion_config,
            qualification=self.motion_qualification,
            incidents=self.motion_incidents,
            media=self.media,
            analysis=self.motion_analysis,
            state=self.motion_state,
        )
        self.motion_ingress = MotionEventIngressService(
            camera_id=camera.id,
            events=self.motion_events,
            qualification=self.motion_qualification,
            state=self.motion_state,
        )
        self.motion_runtime = MotionRuntimeService(
            camera_id=camera.id,
            state=self.motion_state,
            events=self.motion_events,
            analysis=self.motion_analysis,
            decisions=self.motion_decisions,
            ingress=self.motion_ingress,
            qualification=self.motion_qualification,
            evidence=self.motion_evidence,
            pipelines=(
                ("qualification", self.motion_pipeline),
                ("observation", self.motion_observation_pipeline),
                ("fusion", self.motion_fusion_pipeline),
            ),
        )
        self.capture = CameraCaptureService(
            camera_id=camera.id,
            source_url=camera.source_url,
            backend=effective_capture_backend,
            frame_observer=self._capture_frame,
            source_started_observer=self._capture_source_started,
            source_stopped_observer=self._capture_source_stopped,
        )
        self.tracking_frames = TrackingFrameService(
            camera=camera,
            capture=self.capture,
            recorder=self.motion_object_detector.recorder,
            stop_event=self._stop,
            sample_fps=self.tracking_lifecycle.sample_fps,
        )
        self.onvif = OnvifEventListener(
            camera,
            self.handle_motion_event,
            cache_dir=onvif_cache_dir or storage_dir / "onvif",
        )
        self.lifecycle = CameraLifecycleService(
            camera_id=camera.id,
            state=self.runtime_state,
            capture=self.capture,
            onvif=self.onvif,
            tracking=self.tracking_lifecycle,
            motion_runtime=self.motion_runtime,
            tracking_frames=self.tracking_frames,
        )
        self.status_reporter = CameraStatusService(
            camera=camera,
            motion_config=self.motion_config,
            capture=self.capture,
            motion_analysis=self.motion_analysis,
            motion_evidence=self.motion_evidence,
            onvif=self.onvif,
            qualification_pipeline=self.motion_pipeline,
            observation_pipeline=self.motion_observation_pipeline,
            fusion_pipeline=self.motion_fusion_pipeline,
            debug_store=self.motion_debug,
            pipeline_origins=self.motion_pipeline_origins,
            runtime_state=self.runtime_state,
            motion_state=self.motion_state,
            qualification=self.motion_qualification,
            motion_runtime=self.motion_runtime,
            object_tracking=self.tracking_lifecycle,
            incidents=self.motion_incidents,
            lifecycle=self.lifecycle,
        )

    def start(self) -> None:
        self.lifecycle.start()

    def stop(self) -> None:
        self.lifecycle.stop()

    def request_stop(self) -> None:
        self.lifecycle.request_stop()

    def wait_stopped(self, deadline: float) -> bool:
        return self.lifecycle.wait_stopped(deadline)

    def active_workers(self) -> list[str]:
        return self.lifecycle.active_workers()

    def stop_onvif_events(self) -> None:
        """Release the camera's ONVIF subscription without stopping video."""
        self.lifecycle.stop_onvif_events()

    def request_onvif_stop(self) -> None:
        self.lifecycle.request_onvif_stop()

    def wait_onvif_stopped(self, deadline: float) -> bool:
        return self.lifecycle.wait_onvif_stopped(deadline)

    def close(self) -> None:
        self.lifecycle.close()

    def status(self) -> dict[str, Any]:
        return self.status_reporter.snapshot()

    def live_capture_ready(self) -> bool:
        """Expose capture readiness to lifecycle orchestration without a frame copy."""
        return self.capture.frame_ready("live")

    def update_zones(self, zones: list[DetectionZone]) -> None:
        next_zones = [zone.model_copy(deep=True) for zone in zones]
        with self._lifecycle_lock:
            self.camera.zones = next_zones

    def set_detection_enabled(self, enabled: bool) -> None:
        self.lifecycle.set_detection_enabled(enabled)

    def create_object_tracking_session(
        self,
        factory: ObjectTrackingSessionFactory,
    ) -> ObjectTrackingSession:
        """Build a replacement tracking session without changing camera I/O."""
        return self.tracking_lifecycle.create(factory)

    def pause_object_tracking_session(self) -> None:
        """Quiesce tracking before an inference engine transition."""
        self.tracking_lifecycle.pause()

    def resume_object_tracking_session(self) -> None:
        """Restore tracking eligibility after a cancelled transition."""
        self.tracking_lifecycle.sync_accepting()

    def replace_object_tracking_session(
        self,
        replacement: ObjectTrackingSession,
    ) -> ObjectTrackingSession:
        """Atomically replace tracking while preserving capture and ONVIF state."""
        return self.tracking_lifecycle.replace(replacement)

    def snapshot(self, source: str = "live") -> bytes | None:
        return self.media.snapshot(source)

    def mjpeg_frames(self, fps: float = 4.0, source: str = "live") -> Iterator[bytes]:
        yield from self.media.mjpeg_frames(fps, source)

    def handle_motion_event(
        self,
        topic: str = "manual",
        message: str = "",
        event_at: datetime | None = None,
    ) -> None:
        self.motion_runtime.handle_event(topic, message, event_at)

    def set_motion_debug_enabled(self, enabled: bool) -> None:
        self.motion_qualification.set_debug_enabled(enabled)

    def motion_debug_status(self) -> dict[str, Any]:
        return self.motion_qualification.debug_status()

    def motion_debug_image(self, layer: str) -> bytes | None:
        return self.motion_qualification.debug_image(layer)

    def _capture_frame(self, frame: CapturedFrame) -> None:
        if frame.source == "live":
            self.motion_runtime.submit_frame(
                frame.image,
                frame.captured_at_monotonic,
                frame.captured_at_epoch,
            )
        elif frame.source == "main":
            self._remember_tracking_frame(frame.image, frame.captured_at_epoch)

    def _capture_source_started(self, source: str) -> None:
        if source == "main":
            self.tracking_frames.clear()

    def _capture_source_stopped(self, source: str) -> None:
        if source == "main":
            self.tracking_frames.clear()

    def _get_latest_frame(self, source: str = "live") -> Any:
        source = self.camera.normalized_source(source)
        if self._stop.is_set():
            return None
        frame = self.capture.request_frame(source)
        return frame.image if frame is not None else None

    def _get_latest_tracking_frame(
        self,
        source: str = "main",
    ) -> tuple[np.ndarray, float, float] | None:
        return self.tracking_frames.latest(source)

    def _get_latest_tracking_frame_with_fallback(
        self,
    ) -> tuple[np.ndarray, float, float] | None:
        return self._get_latest_tracking_frame("main") or self._get_latest_tracking_frame(
            "live"
        )

    def _remember_tracking_frame(self, frame: np.ndarray, captured_at: float) -> None:
        self.tracking_frames.remember(frame, captured_at)

    def _recorded_tracking_frames(
        self,
        start_epoch: float,
        end_epoch: float,
        sample_fps: float,
        frame_width: int,
    ) -> Iterator[tuple[float, np.ndarray]]:
        yield from self.tracking_frames.recorded_frames(
            start_epoch,
            end_epoch,
            sample_fps,
            frame_width,
        )

    def _recorded_motion_frame(
        self,
        event_at: datetime,
    ) -> tuple[Any | None, list[dict[str, Any]], str]:
        return self.media.detect_recorded_motion(event_at)

    def _write_snapshot(self, frame: Any, event_at: datetime | None = None) -> str:
        return self.media.write_snapshot(frame, event_at)
