from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import cv2

from .activity_events import ActivityEventBus
from .camera_capture import (
    CaptureBackend,
    CameraCaptureService,
    CapturedFrame,
    CaptureOpenLimiter,
    FfmpegCaptureBackend,
)
from .camera_status import CameraStatusService
from .camera_media import CameraMediaService
from .camera_lifecycle import (
    CameraLifecycleService,
    CameraLifecyclePhase,
    CameraRuntimeState,
    CameraStopTicket,
)
from .config import CameraConfig, DetectionZone, MotionQualificationConfig
from .image_storage import DurableImageWriter
from .media_storage import MediaStorageRegistry
from .media_sessions import MediaSessionManager
from .onvif_events import OnvifEventListener, OnvifStopTicket
from .motion_analysis import FairMotionAnalysisLimiter
from .motion_analysis_service import MotionAnalysisService
from .motion_qualification_service import MotionQualificationService
from .motion_events import MotionEventCoordinator
from .motion_decisions import MotionDecisionOrchestrator
from .motion_incidents import MotionIncidentService
from .motion_ingress import MotionEventIngressService
from .security import redact_secret_text
from .motion_runtime import CameraMotionState, MotionRuntimeService
from .object_tracking import ObjectTrackingSession, ObjectTrackingSessionFactory
from .object_tracking_lifecycle import ObjectTrackingLifecycle
from .object_activity import AttributionMode, ObjectActivityAttributor
from .tracking_frames import CameraFrameTimeline, TrackingFrameBatch
from .motion_pipeline import (
    MotionDecisionHandlerFactory,
    MotionDebugSnapshotStore,
    MotionEvidenceRepository,
    MotionPipeline,
    RecordedMotionObjectDetectorFactory,
)
from .motion_pipeline.object_detection import TimestampedLiveFrame

MOTION_QUEUE_SIZE = 32
MOTION_ANALYSIS_QUEUE_SIZE = 1
MOTION_EVENT_MAX_RETRIES = 2
# Wait for live + a stable recording still before scoring. Do not wake main capture.
SPATIAL_ALIGNMENT_STARTUP_READY_SECONDS = 45.0
SPATIAL_ALIGNMENT_STARTUP_SCORE_SECONDS = 20.0
SPATIAL_ALIGNMENT_STARTUP_POLL_SECONDS = 0.5
SPATIAL_ALIGNMENT_REQUIRED_STABLE_SAMPLES = 3
SPATIAL_ALIGNMENT_FAILURE_LIMIT = 3
SPATIAL_ALIGNMENT_ATTEMPT_INTERVAL_SECONDS = 3.0
SPATIAL_ALIGNMENT_RECORDING_FINALIZE_GRACE_SECONDS = 2.0
LOGGER = logging.getLogger(__name__)


def _decode_recording_still(
    path: Path,
    *,
    ffmpeg_path: str,
    seek_seconds: float = 1.0,
) -> np.ndarray | None:
    """Decode one JPEG still from a closed recording segment via FFmpeg."""
    if not ffmpeg_path or not path.is_file():
        return None
    command = [
        ffmpeg_path,
        "-nostdin",
        "-v", "error",
        "-ss", f"{max(0.0, float(seek_seconds)):.3f}",
        "-i", str(path),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=12.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or getattr(image, "size", 0) <= 0:
        return None
    return image


class _AutoStreamAlignment:
    """Bounded live-to-main registration for one camera.

    Prefer a still from the latest closed main recording. Never wake main
    capture for FOV. Boot-time open races should not burn untrusted strikes.
    """

    def __init__(self, camera: CameraConfig) -> None:
        self.enabled = (
            camera.motion_qualification.spatial_alignment.mode == "auto"
            and camera.live_url() != camera.stream_url
        )
        self._frames: dict[str, CapturedFrame] = {}
        self._samples: deque[tuple[float, float, float, float]] = deque(maxlen=4)
        self._last_attempt = 0.0
        self._failures = 0
        self._streams_ready = False
        self._reference_source = ""
        self._reference_width = 0
        self._reference_height = 0

    @property
    def streams_ready(self) -> bool:
        return self._streams_ready

    def status(self, alignment: dict[str, Any]) -> dict[str, Any]:
        reference_width = int(alignment.get("reference_width") or self._reference_width or 0)
        reference_height = int(alignment.get("reference_height") or self._reference_height or 0)
        return {
            **alignment,
            "stable_samples": len(self._samples),
            "failed_samples": self._failures,
            "streams_ready": self._streams_ready,
            "reference_source": (
                alignment.get("reference_source")
                or self._reference_source
                or None
            ),
            "reference_width": reference_width or None,
            "reference_height": reference_height or None,
            "last_attempt_seconds_ago": round(max(0.0, time.monotonic() - self._last_attempt), 3)
            if self._last_attempt else None,
        }

    def is_pending(self, alignment: dict[str, Any]) -> bool:
        """True until auto FOV calibration reaches trusted or untrusted."""
        if not self.enabled:
            return False
        if bool(alignment.get("reliable")):
            return False
        return str(alignment.get("mode") or "") != "untrusted"

    def reset_attempt_state(self) -> None:
        """Clear sample/failure counters for a fresh calibration attempt."""
        self._samples.clear()
        self._failures = 0
        self._last_attempt = 0.0
        self._reference_source = ""
        self._reference_width = 0
        self._reference_height = 0

    @staticmethod
    def _frame_usable(frame: CapturedFrame | None) -> bool:
        if frame is None:
            return False
        if int(frame.width) <= 0 or int(frame.height) <= 0:
            return False
        image = frame.image
        if image is None or getattr(image, "size", 0) <= 0:
            return False
        height, width = image.shape[:2]
        return int(height) > 0 and int(width) > 0

    def _accept_estimate(
        self,
        estimate: tuple[float, float, float, float] | None,
        *,
        reference_source: str,
    ) -> dict[str, Any] | None:
        self._reference_source = reference_source
        if estimate is None:
            self._samples.clear()
            self._failures += 1
            if self._failures >= SPATIAL_ALIGNMENT_FAILURE_LIMIT:
                return {"mode": "untrusted", "reliable": False, "confidence": 0.0,
                        "scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0,
                        "reference_source": reference_source}
            return None
        self._failures = 0
        self._samples.append(estimate)
        if len(self._samples) < SPATIAL_ALIGNMENT_REQUIRED_STABLE_SAMPLES:
            return None
        values = np.asarray(self._samples, dtype=np.float64)
        median = np.median(values, axis=0)
        # A stable same-camera view has negligible normalized drift between
        # independent live/main decoder pairs.
        if float(np.max(np.ptp(values, axis=0))) > 0.025:
            self._samples.clear()
            return None
        result = {
            "mode": "affine",
            "reliable": True,
            "confidence": 0.9,
            "scale_x": round(float(median[0]), 5),
            "scale_y": round(float(median[1]), 5),
            "offset_x": round(float(median[2]), 5),
            "offset_y": round(float(median[3]), 5),
            "reference_source": reference_source,
        }
        if self._reference_width > 0 and self._reference_height > 0:
            result["reference_width"] = self._reference_width
            result["reference_height"] = self._reference_height
        return result

    def calibrate_with_main_image(
        self,
        live: CapturedFrame,
        main_image: np.ndarray,
        *,
        reference_source: str = "recording",
    ) -> dict[str, Any] | None:
        """Score live against a main-space still (usually a closed recording frame).

        Skips capture-time sync: FOV geometry does not need contemporaneous frames.
        """
        if not self.enabled or not self._frame_usable(live):
            return None
        if main_image is None or getattr(main_image, "size", 0) <= 0:
            return None
        height, width = main_image.shape[:2]
        if int(height) <= 0 or int(width) <= 0:
            return None
        self._frames["live"] = live
        self._streams_ready = True
        self._reference_width = int(width)
        self._reference_height = int(height)
        now = time.monotonic()
        if now - self._last_attempt < SPATIAL_ALIGNMENT_ATTEMPT_INTERVAL_SECONDS:
            return None
        self._last_attempt = now
        return self._accept_estimate(
            self._estimate(live.image, main_image),
            reference_source=reference_source,
        )

    def observe(self, frame: CapturedFrame) -> dict[str, Any] | None:
        if not self.enabled or frame.source not in {"main", "live"}:
            return None
        self._frames[frame.source] = frame
        main, live = self._frames.get("main"), self._frames.get("live")
        if not (self._frame_usable(main) and self._frame_usable(live)):
            # Boot/open races: keep Waiting/Checking without burning strikes.
            return None
        self._streams_ready = True
        self._reference_width = int(main.width)
        self._reference_height = int(main.height)
        now = time.monotonic()
        if (
            now - self._last_attempt < SPATIAL_ALIGNMENT_ATTEMPT_INTERVAL_SECONDS
            or abs(main.captured_at_epoch - live.captured_at_epoch) > 0.75
        ):
            return None
        self._last_attempt = now
        return self._accept_estimate(
            self._estimate(live.image, main.image),
            reference_source="capture",
        )

    @staticmethod
    def _estimate(live: np.ndarray, main: np.ndarray) -> tuple[float, float, float, float] | None:
        if live.size == 0 or main.size == 0:
            return None
        def gray(image: np.ndarray) -> np.ndarray:
            height, width = image.shape[:2]
            scale = min(1.0, 640.0 / max(height, width))
            return cv2.cvtColor(cv2.resize(image, (round(width * scale), round(height * scale))), cv2.COLOR_BGR2GRAY)
        live_gray, main_gray = gray(live), gray(main)
        orb = cv2.ORB_create(nfeatures=600)
        live_keypoints, live_descriptors = orb.detectAndCompute(live_gray, None)
        main_keypoints, main_descriptors = orb.detectAndCompute(main_gray, None)
        if live_descriptors is None or main_descriptors is None:
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(live_descriptors, main_descriptors)
        matches = sorted(matches, key=lambda item: item.distance)[:80]
        if len(matches) < 18:
            return None
        source = np.float32([live_keypoints[item.queryIdx].pt for item in matches])
        target = np.float32([main_keypoints[item.trainIdx].pt for item in matches])
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if matrix is None or inliers is None or int(inliers.sum()) < 18:
            return None
        # Only accept scale/translation: this configuration model deliberately
        # does not represent rotation, shear, or a changed perspective.
        if abs(float(matrix[0, 1])) > 0.02 or abs(float(matrix[1, 0])) > 0.02:
            return None
        live_h, live_w = live_gray.shape[:2]
        main_h, main_w = main_gray.shape[:2]
        return (
            float(matrix[0, 0]) * live_w / main_w,
            float(matrix[1, 1]) * live_h / main_h,
            float(matrix[0, 2]) / main_w,
            float(matrix[1, 2]) / main_h,
        )


class CameraWorker:
    def __init__(
        self,
        camera: CameraConfig,
        storage_dir: Path,
        motion_config: MotionQualificationConfig | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        activity_events: ActivityEventBus | None = None,
        media_sessions: MediaSessionManager | None = None,
        media_session_generation: str | int | None = None,
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
        media_storage: MediaStorageRegistry | None = None,
        route_detection_watch: Callable[[str, float], Any | None] | None = None,
        route_target_admitted: Callable[[str, str, int], object] | None = None,
        record_ema_route_candidate: (
            Callable[[str, float, dict[str, Any]], object] | None
        ) = None,
        load_ema_route_candidates: (
            Callable[[str, float, float], list[tuple[float, dict[str, Any]]]] | None
        ) = None,
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
            activity_events=activity_events,
        )
        # Runtime-state reads are independent from tracking-session operations;
        # neither lock is held while camera lifecycle I/O is blocking.
        self._stop = self.runtime_state.stop_event
        self._lifecycle_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._stream_alignment = _AutoStreamAlignment(camera)
        self._effective_spatial_alignment = self._motion_spatial_alignment(camera)
        self._spatial_alignment_startup_active = False
        self._spatial_alignment_startup_lock = threading.Lock()
        self._spatial_alignment_recheck_armed = False
        self._spatial_alignment_recheck_used = False
        effective_capture_backend = capture_backend or FfmpegCaptureBackend(
            CaptureOpenLimiter()
        )
        self.motion_debug = MotionDebugSnapshotStore()
        self.motion_object_detector = motion_object_detector_factory.create(
            camera=camera,
            live_frame_provider=lambda: self._get_latest_frame(),
            timestamped_live_frame_provider=self._get_latest_detection_frame,
            timestamped_evidence_frame_provider=self._get_evidence_detection_frame,
            stop_requested=lambda: (
                self._stop.is_set()
                and self.runtime_state.phase is not CameraLifecyclePhase.STOPPED
            ),
        )
        self.media = CameraMediaService(
            camera=camera,
            storage_dir=storage_dir,
            image_writer=image_writer,
            motion_detector=self.motion_object_detector,
            frame_provider=lambda source: self._get_latest_frame(source),
            rejected_sample_rate=lambda: self.motion_config.rejected_sample_rate,
            stop_requested=self._stop.is_set,
            media_storage=media_storage,
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
            cover_frame_provider=lambda captured_at, width, reference: (
                self.tracking_frames.recorded_frame_at(
                    captured_at,
                    width,
                    reference,
                )
            ),
            snapshot_writer=lambda frame, event_at: self.media.write_snapshot(
                frame,
                event_at,
            ),
        )
        self.motion_decision_handler = motion_decision_handler_factory.create(
            camera_id=camera.id,
            detection_provider=lambda event_at, qualification=None: self._recorded_motion_frame(
                event_at, qualification=qualification
            ),
            initial_detection_provider=(
                lambda event_at: self._recorded_motion_frame(event_at, initial=True)
            ),
            initial_evidence_detection_provider=(
                lambda event_at, evidence: self._recorded_motion_frame(
                    event_at,
                    initial=True,
                    evidence=evidence,
                )
            ),
            snapshot_writer=lambda frame, event_at: self._write_snapshot(frame, event_at),
            event_callback=(
                self.motion_state.publish_event if event_callback is not None else None
            ),
            activity_attributor=ObjectActivityAttributor(
                mode=(
                    getattr(
                        motion_object_detector_factory.detector.config,
                        "object_activity_attribution",
                        "enforce",
                    )
                    if camera.object_activity_attribution == "inherit"
                    else camera.object_activity_attribution
                ),
                stationary_tolerance=(
                    self.motion_config.stationary_object_tolerance
                    if camera.motion_qualification.stationary_object_tolerance == "inherit"
                    else camera.motion_qualification.stationary_object_tolerance
                ),
            ),
            spatial_alignment=self._effective_spatial_alignment,
            route_admission_callback=route_target_admitted,
        )
        self.motion_incidents = MotionIncidentService(
            camera_id=camera.id,
            decision_processor=self.motion_decision_handler,
            tracking_enabled=self.tracking_lifecycle.enabled,
            has_trackable_objects=self.tracking_lifecycle.has_trackable_objects,
            start_tracking=self.tracking_lifecycle.start_incident,
            prewarm_tracking=self.tracking_lifecycle.prewarm,
            image_reader=self.media.read_image,
            refinement_store=(
                motion_decision_handler_factory.events
                if hasattr(
                    motion_decision_handler_factory.events,
                    "enqueue_detection_job",
                )
                else None
            ),
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
            camera_id=camera.id,
            durable_store=(
                motion_decision_handler_factory.events
                if hasattr(
                    motion_decision_handler_factory.events,
                    "enqueue_motion_trigger",
                )
                else None
            ),
        )
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
            model_labels=lambda: list(
                getattr(self.motion_object_detector.detector, "labels", [])
            ),
        )
        self.motion_ingress = MotionEventIngressService(
            camera_id=camera.id,
            events=self.motion_events,
            qualification=self.motion_qualification,
            state=self.motion_state,
            model_labels=lambda: list(
                getattr(self.motion_object_detector.detector, "labels", [])
            ),
        )
        self.motion_runtime = MotionRuntimeService(
            camera_id=camera.id,
            state=self.motion_state,
            events=self.motion_events,
            analysis=self.motion_analysis,
            decisions=self.motion_decisions,
            incidents=self.motion_incidents,
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
            media_sessions=media_sessions,
            owner_generation=media_session_generation,
        )
        self.tracking_frames = CameraFrameTimeline(
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
        self.motion_analysis.set_onvif_effectiveness_observer(
            self.onvif.record_ema_observation
        )
        self.motion_analysis.set_security_verification_context(
            onvif_effectiveness=self.onvif.effectiveness_snapshot,
            route_watch=route_detection_watch,
            record_ema_candidate=record_ema_route_candidate,
            load_ema_candidates=load_ema_route_candidates,
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
            spatial_alignment=lambda: {
                **self._stream_alignment.status(dict(self._effective_spatial_alignment)),
                "startup_calibration": self._spatial_alignment_startup_active,
            },
        )

    @staticmethod
    def _motion_spatial_alignment(camera: CameraConfig) -> dict[str, Any]:
        configured = camera.motion_qualification.spatial_alignment
        mode = configured.mode
        same_stream = camera.live_url() == camera.stream_url
        confidence = float(configured.confidence)
        reliable = bool(
            mode == "identity"
            or (mode == "affine" and confidence >= 0.8)
            or (mode == "auto" and same_stream)
        )
        identity_transform = mode == "identity" or (mode == "auto" and same_stream)
        return {
            "mode": mode,
            "reliable": reliable,
            "confidence": (
                0.0
                if mode == "untrusted"
                else 1.0 if mode == "identity" or same_stream else confidence
            ),
            "scale_x": 1.0 if identity_transform else configured.scale_x,
            "scale_y": 1.0 if identity_transform else configured.scale_y,
            "offset_x": 0.0 if identity_transform else configured.offset_x,
            "offset_y": 0.0 if identity_transform else configured.offset_y,
        }

    def start(self) -> None:
        with self.runtime_state.lock:
            already_running = self.runtime_state.phase is CameraLifecyclePhase.RUNNING
            detection_enabled = self.runtime_state.detection_enabled
        self.lifecycle.start()
        if not already_running and detection_enabled:
            self._spawn_startup_spatial_alignment()

    def consider_route_detection_watch(self, watch: Any) -> bool:
        return self.motion_analysis.consider_route_watch(watch)

    def _request_spatial_alignment_calibration(self, *, reason: str) -> None:
        """Start (or restart) FOV calibration when geometry is still unresolved."""
        with self.runtime_state.detection_work() as admitted:
            if not admitted:
                return
            if not self._stream_alignment.enabled:
                return
            if bool(self._effective_spatial_alignment.get("reliable")):
                return
            # Detection may turn on after a detection-off boot burned the one-shot
            # recheck. Allow a fresh attempt for this enablement.
            self._spatial_alignment_recheck_armed = False
            self._spatial_alignment_recheck_used = False
            self._stream_alignment.reset_attempt_state()
            if str(self._effective_spatial_alignment.get("mode") or "") == "untrusted":
                pending = {
                    "mode": "auto",
                    "reliable": False,
                    "confidence": 0.0,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "offset_x": 0.0,
                    "offset_y": 0.0,
                }
                self._effective_spatial_alignment = pending
                self.motion_decision_handler.spatial_alignment = dict(pending)
            with self.runtime_state.lock:
                phase = self.runtime_state.phase
                detection_enabled = bool(self.runtime_state.detection_enabled)
            if phase is not CameraLifecyclePhase.RUNNING or not detection_enabled:
                # Fleet applies detection prefs before start(); start() will spawn.
                return
            LOGGER.info(
                "FOV alignment requested for %s (%s)",
                self.camera.id,
                reason,
            )
            self._spawn_startup_spatial_alignment()

    def _spawn_startup_spatial_alignment(self) -> None:
        """Calibrate FOV at startup (or one healthy recheck) without periodic wake.

        Use a still from the latest closed main recording segment. Never open
        main capture here: that second ffmpeg competes with the recorder.
        """
        with self.runtime_state.detection_work() as admitted:
            if not admitted:
                return
            if not self.motion_state.detection_enabled():
                return
            if not self._stream_alignment.is_pending(self._effective_spatial_alignment):
                return
            with self._spatial_alignment_startup_lock:
                if self._spatial_alignment_startup_active:
                    return
                self._spatial_alignment_startup_active = True
            generation = self.runtime_state.generation
            thread = threading.Thread(
                target=self._run_startup_spatial_alignment,
                args=(generation, self.runtime_state.detection_generation),
                name=f"camera-{self.camera.id}-fov-align",
                daemon=True,
            )
            thread.start()

    def _apply_spatial_alignment(self, calibrated: dict[str, Any]) -> None:
        self._effective_spatial_alignment = calibrated
        self.motion_decision_handler.spatial_alignment = dict(calibrated)
        if bool(calibrated.get("reliable")) or str(calibrated.get("mode") or "") == "untrusted":
            self._spatial_alignment_recheck_armed = False

    def _stable_recording_candidates(self) -> list[tuple[Path, float]]:
        """Newest-first closed main recording paths with a seek offset."""
        recorder = getattr(self.motion_object_detector, "recorder", None)
        if recorder is None:
            return []
        rows_fn = getattr(recorder, "recording_rows", None)
        if not callable(rows_fn):
            return []
        try:
            rows = list(rows_fn(self.camera.id, limit=8, source="main") or [])
        except Exception as error:
            LOGGER.debug(
                "FOV alignment could not list recordings for %s: %s: %s",
                self.camera.id,
                type(error).__name__,
                redact_secret_text(error)[:200],
            )
            return []
        now_epoch = time.time()
        candidates: list[tuple[Path, float]] = []
        for row in reversed(rows):
            path = Path(str(row.get("path") or ""))
            if not path.is_file():
                continue
            start_epoch = float(row.get("start_epoch") or 0.0)
            duration = float(row.get("duration_seconds") or 0.0)
            end_epoch = float(row.get("end_epoch") or (start_epoch + duration))
            if end_epoch and now_epoch < end_epoch + SPATIAL_ALIGNMENT_RECORDING_FINALIZE_GRACE_SECONDS:
                continue
            is_stable = getattr(recorder, "_recording_file_is_stable", None)
            if callable(is_stable):
                try:
                    if not bool(is_stable(path)):
                        continue
                except Exception:
                    continue
            seek = 1.0
            if duration > 0:
                seek = max(0.5, min(duration * 0.5, max(0.0, duration - 0.5)))
            candidates.append((path, seek))
        return candidates

    def _latest_stable_recording_still(self) -> np.ndarray | None:
        """Return one still from the newest closed, playable main recording."""
        recorder = getattr(self.motion_object_detector, "recorder", None)
        ffmpeg_path = str(getattr(recorder, "ffmpeg_path", "") or "ffmpeg")
        for path, seek in self._stable_recording_candidates():
            if not self.motion_state.detection_enabled():
                return None
            still = _decode_recording_still(
                path,
                ffmpeg_path=ffmpeg_path,
                seek_seconds=seek,
            )
            if still is not None:
                return still
        return None

    def _run_startup_spatial_alignment(
        self, generation: int, detection_generation: int | None = None,
    ) -> None:
        if detection_generation is None:
            detection_generation = self.runtime_state.detection_generation
        ready_deadline = time.monotonic() + SPATIAL_ALIGNMENT_STARTUP_READY_SECONDS
        score_deadline: float | None = None
        recording_still: np.ndarray | None = None
        try:
            while True:
                now = time.monotonic()
                with self.runtime_state.detection_work() as admitted:
                    if not admitted:
                        return
                    if (
                        self._stop.is_set()
                        or self.runtime_state.generation != generation
                        or self.runtime_state.detection_generation != detection_generation
                        or not self.motion_state.detection_enabled()
                    ):
                        return
                    if not self._stream_alignment.is_pending(self._effective_spatial_alignment):
                        return
                    live: CapturedFrame | None = None
                    try:
                        live = self.capture.request_frame("live")
                    except Exception as error:
                        LOGGER.warning(
                            "startup FOV alignment could not sample live for %s: %s: %s",
                            self.camera.id,
                            type(error).__name__,
                            redact_secret_text(error)[:300],
                        )
                    if recording_still is None:
                        recording_still = self._latest_stable_recording_still()
                    if (
                        self.motion_state.detection_enabled()
                        and self.runtime_state.detection_generation == detection_generation
                        and recording_still is not None
                        and live is not None
                        and _AutoStreamAlignment._frame_usable(live)
                    ):
                        calibrated = self._stream_alignment.calibrate_with_main_image(
                            live,
                            recording_still,
                            reference_source="recording",
                        )
                        if calibrated is not None:
                            self._apply_spatial_alignment(calibrated)
                            if not self._stream_alignment.is_pending(calibrated):
                                return
                if not self._stream_alignment.streams_ready:
                    if now >= ready_deadline:
                        break
                    time.sleep(SPATIAL_ALIGNMENT_STARTUP_POLL_SECONDS)
                    continue
                if score_deadline is None:
                    score_deadline = now + SPATIAL_ALIGNMENT_STARTUP_SCORE_SECONDS
                if now >= score_deadline:
                    break
                time.sleep(SPATIAL_ALIGNMENT_STARTUP_POLL_SECONDS)
            if (
                self._stop.is_set()
                or self.runtime_state.generation != generation
                or self.runtime_state.detection_generation != detection_generation
                or not self.motion_state.detection_enabled()
                or not self._stream_alignment.is_pending(self._effective_spatial_alignment)
            ):
                return
            # Soft leave as Checking. Boot-time FOV is brittle; do not sticky-untrust
            # on timeout. Arm at most one recheck once a recording/live pair is healthy.
            if not self._spatial_alignment_recheck_used:
                self._spatial_alignment_recheck_armed = True
            LOGGER.info(
                "FOV alignment attempt timed out for %s; leaving Checking%s%s",
                self.camera.id,
                " (recording still unavailable)"
                if recording_still is None
                else "",
                " (will recheck once when live/recording are healthy)"
                if self._spatial_alignment_recheck_armed
                else "",
            )
        finally:
            with self._spatial_alignment_startup_lock:
                self._spatial_alignment_startup_active = False
            if (
                self.runtime_state.detection_generation != detection_generation
                and self.motion_state.detection_enabled()
                and self.runtime_state.phase is CameraLifecyclePhase.RUNNING
            ):
                self._spawn_startup_spatial_alignment()
            else:
                self._maybe_spawn_healthy_spatial_recheck()

    def _maybe_spawn_healthy_spatial_recheck(self) -> None:
        """One-shot follow-up after live/recording become healthy — not periodic."""
        if not self._spatial_alignment_recheck_armed:
            return
        if self._spatial_alignment_recheck_used:
            return
        if not self._stream_alignment.is_pending(self._effective_spatial_alignment):
            self._spatial_alignment_recheck_armed = False
            return
        live = self._stream_alignment._frames.get("live")
        live_ok = _AutoStreamAlignment._frame_usable(live)
        recording_ready = bool(self._stable_recording_candidates())
        if not (
            self._stream_alignment.streams_ready
            or live_ok
            or recording_ready
        ):
            return
        with self._spatial_alignment_startup_lock:
            if self._spatial_alignment_startup_active:
                return
            if self._spatial_alignment_recheck_used:
                return
            self._spatial_alignment_recheck_armed = False
            self._spatial_alignment_recheck_used = True
        self._stream_alignment.reset_attempt_state()
        LOGGER.info(
            "FOV alignment rechecking once for %s after live/recording became healthy",
            self.camera.id,
        )
        self._spawn_startup_spatial_alignment()

    def stop(self) -> None:
        self.lifecycle.stop()

    def request_stop(self) -> CameraStopTicket | None:
        return self.lifecycle.request_stop()

    def wait_stopped(
        self,
        deadline: float,
        ticket: CameraStopTicket | None = None,
    ) -> bool:
        return self.lifecycle.wait_stopped(deadline, ticket)

    def active_workers(self) -> list[str]:
        return self.lifecycle.active_workers()

    def stop_onvif_events(self) -> None:
        """Release the camera's ONVIF subscription without stopping video."""
        self.lifecycle.stop_onvif_events()

    def request_onvif_stop(self) -> OnvifStopTicket:
        return self.lifecycle.request_onvif_stop()

    def wait_onvif_stopped(
        self,
        deadline: float,
        ticket: OnvifStopTicket | None = None,
    ) -> bool:
        return self.lifecycle.wait_onvif_stopped(deadline, ticket)

    def close(self) -> None:
        self.lifecycle.close()

    def status(self) -> dict[str, Any]:
        return self.status_reporter.snapshot()

    def reconfigure_object_activity_attribution(self, mode: AttributionMode) -> None:
        self.motion_decision_handler.reconfigure_activity_attribution(mode)

    def reconfigure_motion_policy(
        self,
        config: MotionQualificationConfig,
        camera: CameraConfig,
    ) -> None:
        """Hot-apply non-structural EMA policy without interrupting capture."""
        next_global = config.model_copy(deep=True)
        next_override = camera.motion_qualification.model_copy(deep=True)
        with self._lifecycle_lock, self.motion_qualification.analysis_lock:
            self.motion_config = next_global
            next_camera = camera.model_copy(deep=True)
            next_camera.motion_qualification = next_override
            self.motion_qualification.reconfigure_policy(next_global, next_camera)
            self.motion_analysis.config = next_global
            self.motion_decisions.config = next_global
            # Do not carry scene-readiness or persistence accumulated under a
            # different threshold/warmup policy into the new policy.
            self.motion_analysis.reset_ema_policy_state()
            self.motion_decision_handler.reconfigure_stationary_tolerance(
                self.motion_qualification.stationary_object_tolerance()
            )

    def live_capture_ready(self) -> bool:
        """Expose capture readiness to lifecycle orchestration without a frame copy."""
        return self.capture.frame_ready("live")

    def update_zones(self, zones: list[DetectionZone]) -> None:
        next_zones = [zone.model_copy(deep=True) for zone in zones]
        with self._lifecycle_lock:
            self.camera.zones = next_zones
            self.motion_qualification.update_zones(next_zones)

    def set_detection_enabled(self, enabled: bool) -> None:
        with self.runtime_state.lock:
            previously_enabled = bool(self.runtime_state.detection_enabled)
        self.lifecycle.set_detection_enabled(enabled)
        if bool(enabled) and not previously_enabled:
            self._request_spatial_alignment_calibration(reason="detection_enabled")

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
        with self.runtime_state.detection_work() as admitted:
            if not admitted:
                return
            with self.runtime_state.lock:
                if not self.runtime_state.detection_enabled:
                    return
            # FOV is one-shot (startup / detection-on / one recheck). Do not keep
            # re-scoring when demand-driven main capture later publishes frames.
            if self._stream_alignment.is_pending(self._effective_spatial_alignment):
                calibrated = self._stream_alignment.observe(frame)
                if calibrated is not None:
                    self._apply_spatial_alignment(calibrated)
                else:
                    self._maybe_spawn_healthy_spatial_recheck()
            if frame.source == "live":
                with self.runtime_state.lock:
                    lifecycle_generation = self.runtime_state.generation
                self.motion_runtime.submit_frame(
                    frame.image,
                    frame.captured_at_monotonic,
                    frame.captured_at_epoch,
                    capture_sequence=frame.sequence,
                    capture_generation=frame.generation,
                    lifecycle_generation=lifecycle_generation,
                )
                # Keep timestamped live history for bridging open main segments.
                self._remember_tracking_frame(
                    frame.image,
                    frame.captured_at_epoch,
                    source="live",
                )
            elif frame.source == "main":
                self._remember_tracking_frame(
                    frame.image,
                    frame.captured_at_epoch,
                    source="main",
                )

    def _capture_source_started(self, source: str) -> None:
        if source in {"main", "live"}:
            self.tracking_frames.clear(source)

    def _capture_source_stopped(self, source: str) -> None:
        if source in {"main", "live"}:
            self.tracking_frames.clear(source)

    def _get_latest_frame(self, source: str = "live") -> Any:
        source = self.camera.normalized_source(source)
        if self._stop.is_set():
            return None
        frame = self.capture.request_frame(source)
        return frame.image if frame is not None else None

    def _get_latest_detection_frame(self) -> TimestampedLiveFrame | None:
        frame = self.tracking_frames.captured("live")
        if frame is None:
            return None
        with self.runtime_state.lock:
            generation = self.runtime_state.generation
        alignment = self._effective_spatial_alignment
        height, width = frame.image.shape[:2]
        return TimestampedLiveFrame(
            frame=frame.image,
            captured_at_epoch=frame.captured_at_epoch,
            captured_at_monotonic=frame.captured_at_monotonic,
            sequence=frame.sequence,
            camera_generation=generation,
            capture_generation=frame.generation,
            source=frame.source,
            geometry_trusted=bool(alignment.get("reliable", False)),
            width=width,
            height=height,
        )

    def _get_evidence_detection_frame(
        self,
        evidence: dict[str, Any],
    ) -> TimestampedLiveFrame | None:
        expected_lifecycle = int(evidence.get("evidence_lifecycle_generation") or 0)
        expected_capture = int(evidence.get("evidence_capture_generation") or 0)
        expected_sequence = int(evidence.get("evidence_frame_sequence") or 0)
        expected_epoch = evidence.get("evidence_frame_at_epoch")
        if not isinstance(expected_epoch, (int, float)):
            return None
        with self.runtime_state.lock:
            current_lifecycle = self.runtime_state.generation
        if (
            expected_lifecycle <= 0
            or expected_capture <= 0
            or current_lifecycle != expected_lifecycle
        ):
            return None
        selected = self.motion_analysis.evidence_frame_near(
            float(expected_epoch),
            sequence=expected_sequence,
            capture_generation=expected_capture,
            lifecycle_generation=expected_lifecycle,
        )
        if selected is None:
            return None
        height, width = selected.image.shape[:2]
        alignment = self._effective_spatial_alignment
        return TimestampedLiveFrame(
            frame=selected.image,
            captured_at_epoch=selected.captured_at_epoch,
            captured_at_monotonic=selected.captured_at_monotonic,
            sequence=selected.sequence,
            camera_generation=selected.lifecycle_generation,
            capture_generation=selected.capture_generation,
            source="live",
            geometry_trusted=bool(alignment.get("reliable", False)),
            width=width,
            height=height,
        )

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

    def _remember_tracking_frame(
        self,
        frame: np.ndarray,
        captured_at: float,
        *,
        source: str = "main",
    ) -> None:
        self.tracking_frames.remember(frame, captured_at, source=source)

    def _recorded_tracking_frames(
        self,
        start_epoch: float,
        end_epoch: float,
        sample_fps: float,
        frame_width: int,
        *,
        after_epoch: float | None = None,
    ) -> TrackingFrameBatch:
        return self.tracking_frames.read_recorded_frames(
            start_epoch,
            end_epoch,
            sample_fps,
            frame_width,
            after_epoch=after_epoch,
        )

    def _recorded_motion_frame(
        self,
        event_at: datetime,
        *,
        initial: bool = False,
        evidence: dict[str, Any] | None = None,
        qualification: dict[str, Any] | None = None,
    ) -> Any:
        if initial:
            return self.media.detect_initial_recorded_motion(event_at, evidence)
        return self.media.detect_recorded_motion(event_at, qualification)

    def _write_snapshot(self, frame: Any, event_at: datetime | None = None) -> str:
        return self.media.write_snapshot(frame, event_at)
