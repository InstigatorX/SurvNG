from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .camera import CameraWorker
from .camera_capture import CaptureOpenLimiter, OpenCvFfmpegCaptureBackend
from .camera_control import CameraControlService
from .camera_fleet import CameraFleetLifecycle, CameraFleetOperationError
from .camera_startup import (
    CAMERA_STARTUP_FIRST_FRAME_TIMEOUT_SECONDS,
    CAMERA_STARTUP_MAX_CONCURRENCY,
    CAMERA_STARTUP_RECORDER_SETTLE_SECONDS,
    CameraStartupCoordinator,
)
from .appearance_backfill import DeferredAppearanceBackfill
from .appearance_index import AppearanceIndex
from .config import (
    AppConfig,
    CameraConfig,
    CameraMotionQualificationConfig,
    DetectorConfig,
    DetectionZone,
    ImageStorageConfig,
    MqttConfig,
    SemanticSearchConfig,
)
from .events import EventStore
from .detector import objects_to_json
from .go2rtc import Go2RtcAdapter
from .inference_lifecycle import InferenceLifecycle
from .image_cache import LocalImageCache
from .image_storage import DurableImageWriter
from .media_storage import MediaStorageRegistry
from .mqtt import MqttService
from .mqtt_lifecycle import MqttLifecycle
from .runtime_monitor import (
    ApplicationRuntimeMonitor,
    process_rss_bytes,
    system_memory_usage,
)
from .semantic_search import DisabledSemanticSearch, SemanticIndex
from .motion_pipeline import (
    LoggingMotionPipelineObserver,
    EVIDENCE_REPOSITORY_SERVICE,
    MotionDecisionHandlerFactory,
    MotionEvidenceRepository,
    MotionPipeline,
    MotionPipelineFactory,
    MotionStageDependencies,
    RecordedMotionObjectDetectorFactory,
    build_builtin_motion_registry,
    resolve_motion_pipeline_graphs,
)
from .motion_analysis import FairMotionAnalysisLimiter
from .recording_lifecycle import RecordingLifecycle
from .state_events import StateEventBroker
from .security import redact_secret_text


LOGGER = logging.getLogger("uvicorn.error")


class ManagerShutdownIncompleteError(RuntimeError):
    """Camera-owned work remains active, so shared services must stay alive."""

    def __init__(self, error: CameraFleetOperationError) -> None:
        self.fleet_error = error
        residuals = ", ".join(error.residual_camera_ids) or "unknown cameras"
        super().__init__(f"camera shutdown remains active: {residuals}")


def validate_motion_pipeline_configuration(config: AppConfig) -> None:
    def selected_sources(value: object) -> set[str]:
        if isinstance(value, str):
            values = (value,)
        elif isinstance(value, (list, tuple)):
            values = value
        else:
            return set()
        return {
            str(source).strip().lower()
            for source in values
            if str(source).strip()
        }

    def reject_retired_mog2(scope: str, motion_config: object) -> None:
        pipeline = getattr(motion_config, "pipeline", None)
        for graph_name in ("qualification", "observation", "fusion"):
            stages = getattr(pipeline, graph_name, None) if pipeline is not None else None
            for stage in stages or ():
                implementation = str(stage.implementation).strip().lower()
                sources = selected_sources(stage.options.get("sources"))
                if implementation in {"opencv_mog2", "opencv_mog2_evidence"} or "mog2" in sources:
                    raise ValueError(
                        f"invalid motion pipeline for {scope}: MOG2 stage/source "
                        f"in {graph_name} has been retired; use Enhanced Motion "
                        "Analysis (EMA) validation"
                    )

    reject_retired_mog2("global configuration", config.motion_qualification)
    for camera in config.cameras:
        reject_retired_mog2(f"camera {camera.id!r}", camera.motion_qualification)

    registry = build_builtin_motion_registry()
    targets = [("global", CameraMotionQualificationConfig())] + [
        (camera.id, camera.motion_qualification)
        for camera in config.cameras
    ]
    for camera_id, camera_config in targets:
        evidence = MotionEvidenceRepository(camera_id)
        factory = MotionPipelineFactory(
            registry=registry,
            dependencies=MotionStageDependencies(
                services={EVIDENCE_REPOSITORY_SERVICE: evidence},
            ),
        )
        pipelines: list[MotionPipeline] = []
        try:
            graphs = resolve_motion_pipeline_graphs(
                config.motion_qualification,
                camera_config,
            )
            pipelines.append(factory.create(
                camera_id,
                graphs.qualification,
                required_artifacts={"scoring"},
            ))
            pipelines.append(factory.create(
                camera_id,
                graphs.observation,
                required_artifacts={"source_evidence"},
            ))
            pipelines.append(factory.create(
                camera_id,
                graphs.fusion,
                initial_artifacts={"scoring"},
                required_artifacts={"scoring"},
            ))
        except ValueError as error:
            raise ValueError(
                f"invalid motion pipeline for camera {camera_id!r}: {error}"
            ) from error
        finally:
            for pipeline in reversed(pipelines):
                pipeline.close()


class AppManager:
    def __init__(self, config: AppConfig) -> None:
        validate_motion_pipeline_configuration(config)
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.database_dir = Path(config.database_dir) if config.database_dir else self.storage_dir
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache = LocalImageCache(self.database_dir / "image-cache")
        self.image_writer = DurableImageWriter(config.image_storage)
        self.media_storage = MediaStorageRegistry(self.storage_dir, config.media_storage)
        self.events = EventStore(self.storage_dir, database_dir=self.database_dir)
        self.appearance_index = AppearanceIndex(self.events.db_path)
        self.semantic_index = SemanticIndex(self.events.db_path)
        self.recording = RecordingLifecycle(
            config=config,
            storage_dir=self.storage_dir,
            protected_recording_paths=self.events.protected_recording_paths,
            media_storage=self.media_storage,
        )
        # Compatibility handle for media APIs and camera dependencies. Shared
        # lifecycle/reconfiguration ownership lives in ``self.recording``.
        self.recorder = self.recording.recorder
        self.go2rtc = Go2RtcAdapter()
        # Camera startup pacing is an internal safety policy. Keep native
        # OpenCV admission and the startup coordinator on the same fixed cap.
        self._capture_open_limiter = CaptureOpenLimiter(
            CAMERA_STARTUP_MAX_CONCURRENCY
        )
        self.capture_backend = OpenCvFfmpegCaptureBackend(
            self._capture_open_limiter
        )
        self.state_events = StateEventBroker()
        try:
            self.inference = InferenceLifecycle(
                config=config.detector,
                semantic_config=config.semantic_search,
                storage_dir=self.storage_dir,
                events=self.events,
                appearance_index=self.appearance_index,
                semantic_index=self.semantic_index,
                event_publisher=self.publish_event,
                tracking_burst_guard=self._tracking_burst_available,
                database_dir=self.database_dir,
            )
        except BaseException:
            for label, operation in (
                ("recording lifecycle", self.recording.close),
                ("state event broker", self.state_events.close),
            ):
                try:
                    operation()
                except Exception as error:
                    LOGGER.error(
                        "%s cleanup failed during inference construction: %s",
                        label,
                        redact_secret_text(error),
                    )
            raise
        # Stable compatibility handles used throughout the application. The
        # lifecycle replaces generations behind these supervisors, not the
        # supervisor objects themselves.
        self.detector = self.inference.detector
        self.face_recognizer = self.inference.face_recognizer
        self.person_reidentifier = self.inference.person_reidentifier
        self.faces = self.inference.faces
        self.motion_pipeline_registry = build_builtin_motion_registry()
        self.motion_decision_handler_factory = MotionDecisionHandlerFactory(
            events=self.events,
            object_serializer=objects_to_json,
            face_candidate_sink=self.faces.ingest_candidates,
        )
        self.motion_object_detector_factory = RecordedMotionObjectDetectorFactory(
            detector=self.detector,
            recorder=self.recorder,
        )
        try:
            self.mqtt = MqttLifecycle(config.mqtt, self._build_mqtt_service)
        except BaseException:
            for label, operation in (
                ("inference lifecycle", self.inference.close),
                ("recording lifecycle", self.recording.close),
                ("state event broker", self.state_events.close),
            ):
                try:
                    operation()
                except Exception as error:
                    LOGGER.error(
                        "%s cleanup failed during MQTT construction: %s",
                        label,
                        redact_secret_text(error),
                    )
            raise
        self._process_started_monotonic = time.monotonic()
        self._process_started_at = datetime.now(timezone.utc).isoformat()
        self._lifecycle_lock = threading.RLock()
        # Camera, recording, and detection preferences are restored by the
        # control service. In-process reloads explicitly transfer the same
        # snapshot before the replacement manager starts.
        self._stopping = False
        self._started = False
        self._closed = False
        self._startup_services_ready = False
        self._startup_timings: dict[str, float] = {}
        camera_startup = CameraStartupCoordinator(
            max_concurrency=CAMERA_STARTUP_MAX_CONCURRENCY,
            readiness_timeout_seconds=CAMERA_STARTUP_FIRST_FRAME_TIMEOUT_SECONDS,
            recorder_settle_seconds=CAMERA_STARTUP_RECORDER_SETTLE_SECONDS,
        )
        self.motion_evidence: dict[str, MotionEvidenceRepository] = {}
        # Keep the established two-camera CPU ceiling, but dispatch those slots
        # fairly so continuous EMA work from one camera cannot starve another.
        self._motion_analysis_limiter = FairMotionAnalysisLimiter(2)
        workers: dict[str, CameraWorker] = {}
        try:
            for camera in self._unique_cameras():
                workers[camera.id] = self._create_camera_worker(camera)
        except BaseException:
            for worker in reversed(tuple(workers.values())):
                try:
                    worker.close()
                except Exception as error:
                    LOGGER.error(
                        "camera cleanup failed during manager construction: %s",
                        redact_secret_text(error),
                    )
            for label, callback in (
                ("MQTT", self.mqtt.stop),
                ("inference lifecycle", self.inference.close),
                ("recording lifecycle", self.recording.close),
                ("state event broker", self.state_events.close),
            ):
                try:
                    callback()
                except Exception as error:
                    LOGGER.error(
                        "%s cleanup failed during manager construction: %s",
                        label,
                        redact_secret_text(error),
                    )
            raise
        self.workers = workers
        self.inference.bind_workers(self.workers)
        self.camera_fleet = CameraFleetLifecycle(
            cameras=tuple(self._unique_cameras()),
            workers=self.workers,
            recorder=self.recorder,
            startup=camera_startup,
            state_publisher=self.mqtt,
        )
        self.runtime_monitor = ApplicationRuntimeMonitor(
            inference=self.inference,
            events=self.events,
            state_events=self.state_events,
            camera_statuses=self.statuses,
        )
        self.camera_controls = CameraControlService(
            cameras=tuple(self._unique_cameras()),
            workers=self.workers,
            recording=self.recording,
            fleet=self.camera_fleet,
            mqtt=self.mqtt,
            runtime_monitor=self.runtime_monitor,
            state_path=self.database_dir / "runtime_state.json",
            legacy_state_paths=(self.storage_dir / "runtime_state.json",),
        )

    def _tracking_burst_available(self) -> bool:
        """Allow the optional extra tracker only while inference and memory are healthy."""
        if getattr(self, "_stopping", False) or getattr(self, "_closed", False):
            return False
        detector = getattr(self, "detector", None)
        if detector is None:
            # The limiter is assembled with the inference lifecycle. Deny a
            # burst if a future implementation evaluates the guard before the
            # manager has published its stable detector handle.
            return False
        try:
            runtime = detector.cached_object_status().get("runtime") or {}
            _total, _used, memory_percent = system_memory_usage()
            return (
                int(runtime.get("queue_depth") or 0) == 0
                and int(runtime.get("pending_frames") or 0) == 0
                and int(runtime.get("active_inferences") or 0) <= 1
                and (memory_percent <= 0.0 or memory_percent < 85.0)
            )
        except Exception:
            LOGGER.exception("could not evaluate adaptive tracking burst capacity")
            return False

    def _create_camera_worker(self, camera: CameraConfig) -> CameraWorker:
        motion_config = self.config.motion_qualification
        override = camera.motion_qualification
        graphs = resolve_motion_pipeline_graphs(motion_config, override)
        ring_size = max(
            12,
            round(
                motion_config.sample_fps
                * (
                    motion_config.window_seconds
                    + motion_config.post_trigger_seconds
                    + 3.0
                )
            ),
        )
        evidence = MotionEvidenceRepository(camera.id, max_samples_per_source=ring_size)
        self.motion_evidence[camera.id] = evidence
        dependencies = MotionStageDependencies(
            services={EVIDENCE_REPOSITORY_SERVICE: evidence},
        )
        factory = MotionPipelineFactory(
            registry=self.motion_pipeline_registry,
            dependencies=dependencies,
            observer=LoggingMotionPipelineObserver(),
        )
        pipelines: list[MotionPipeline] = []
        try:
            qualification_pipeline = factory.create(
                camera.id,
                graphs.qualification,
                required_artifacts={"scoring"},
            )
            pipelines.append(qualification_pipeline)
            observation_pipeline = factory.create(
                camera.id,
                graphs.observation,
                required_artifacts={"source_evidence"},
            )
            pipelines.append(observation_pipeline)
            fusion_pipeline = factory.create(
                camera.id,
                graphs.fusion,
                initial_artifacts={"scoring"},
                required_artifacts={"scoring"},
            )
            pipelines.append(fusion_pipeline)
            return CameraWorker(
                camera,
                self.storage_dir,
                motion_config,
                self.publish_event,
                motion_pipeline=qualification_pipeline,
                motion_observation_pipeline=observation_pipeline,
                motion_fusion_pipeline=fusion_pipeline,
                motion_evidence=evidence,
                motion_pipeline_origins=graphs.origins,
                motion_decision_handler_factory=self.motion_decision_handler_factory,
                motion_object_detector_factory=self.motion_object_detector_factory,
                object_tracking_session_factory=self.inference.tracking_factory,
                motion_analysis_limiter=self._motion_analysis_limiter,
                image_writer=self.image_writer,
                onvif_cache_dir=self.database_dir / "onvif",
                capture_backend=self.capture_backend,
            )
        except BaseException:
            for pipeline in reversed(pipelines):
                pipeline.close()
            raise

    def _unique_cameras(self):
        seen: set[str] = set()
        for camera in self.config.cameras:
            if camera.id in seen:
                continue
            seen.add(camera.id)
            yield camera

    def recording_enabled(self, camera_id: str) -> bool:
        return self.camera_controls.recording_enabled(camera_id)

    def detection_enabled(self, camera_id: str) -> bool:
        return self.camera_controls.detection_enabled(camera_id)

    def set_recording(self, camera_id: str, enabled: bool) -> bool:
        return self.camera_controls.set_recording(camera_id, enabled)

    def set_detection(self, camera_id: str, enabled: bool) -> bool:
        return self.camera_controls.set_detection(camera_id, enabled)

    def runtime_preferences(self) -> dict[str, dict[str, bool]]:
        return self.camera_controls.snapshot()

    def apply_runtime_preferences(
        self,
        preferences: dict[str, dict[str, bool]],
        *,
        persist: bool = False,
    ) -> None:
        self.camera_controls.apply(preferences, persist=persist)

    def start_all(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("application manager is closed")
            if self._started:
                return
            self._stopping = False
            self._startup_services_ready = False
            startup_started = time.monotonic()
            phase_started = startup_started
            try:
                # Rewrite the restored or explicitly transferred snapshot so
                # removed cameras are pruned before startup admission.
                self.camera_controls.persist()
                self.inference.start_core()
                self._startup_timings = {
                    "inference_seconds": round(time.monotonic() - phase_started, 3),
                }
                cameras = list(self._unique_cameras())
                recording_timings = self.recording.start_services(
                    cameras,
                    self.camera_fleet.recorder_keys(),
                )
                self._startup_timings["recorder_cleanup_seconds"] = round(
                    recording_timings.cleanup_seconds, 3
                )
                preferences = self.camera_controls.startup_preferences()
                startup_tasks = self.camera_fleet.prepare_startup(
                    camera_enabled=preferences["camera_enabled"],
                    recording_enabled=preferences["recording_enabled"],
                    detection_enabled=preferences["detection_enabled"],
                    recording_is_enabled=self.recording_enabled,
                )
                self._startup_timings["recorder_services_seconds"] = round(
                    recording_timings.services_seconds, 3
                )
                # Publish discovery only after persisted recording/detection preferences
                # have been applied to every worker.
                phase_started = time.monotonic()
                self.mqtt.start()
                self.mqtt.set_server_lifecycle("starting")
                self.runtime_monitor.start()
                self._startup_timings["mqtt_seconds"] = round(
                    time.monotonic() - phase_started,
                    3,
                )
                self._started = True
                self.camera_fleet.start_admission(
                    startup_tasks,
                    on_complete=self._camera_startup_completed,
                )
                phase_started = time.monotonic()
                self.inference.start_auxiliary()
                self._startup_timings["auxiliary_services_seconds"] = round(
                    time.monotonic() - phase_started,
                    3,
                )
                self._startup_timings["application_ready_seconds"] = round(
                    time.monotonic() - startup_started,
                    3,
                )
                self._startup_services_ready = True
                self._mark_running_if_startup_complete()
                LOGGER.info(
                    "SurvNG application ready in %.2fs; camera admission continues in background (%s)",
                    self._startup_timings["application_ready_seconds"],
                    self._startup_timings,
                )
            except BaseException:
                self._stopping = True
                try:
                    self._shutdown_components()
                except Exception:
                    LOGGER.exception("application startup rollback was incomplete")
                self._closed = True
                raise

    def _camera_startup_completed(self) -> None:
        self._mark_running_if_startup_complete()

    def _mark_running_if_startup_complete(self) -> None:
        if (
            self._stopping
            or self._closed
            or not self._started
            or not self._startup_services_ready
            or not self.camera_fleet.status().get("complete")
        ):
            return
        self.mqtt.set_server_lifecycle("running")

    def wait_for_camera_startup(self, timeout: float | None = None) -> bool:
        return self.camera_fleet.wait(timeout)

    def camera_startup_status(self) -> dict[str, object]:
        return {
            **self.camera_fleet.status(),
            "application_startup": dict(self._startup_timings),
        }

    def stop_all(self) -> None:
        self.stop_all_with_runtime_preferences()

    def stop_all_with_runtime_preferences(self) -> dict[str, dict[str, bool]]:
        with self._lifecycle_lock:
            preferences = self.runtime_preferences()
            if self._closed:
                return preferences
            self._stopping = True
            try:
                self._shutdown_components()
            except ManagerShutdownIncompleteError:
                # Keep the manager retryable and, critically, keep shared
                # inference/recording dependencies alive beneath residual
                # camera workers. The process supervisor owns the hard limit.
                self._started = False
                raise
            except BaseException:
                self._started = False
                self._closed = True
                raise
            self._started = False
            self._closed = True
            return preferences

    def release_onvif_subscriptions(self) -> None:
        """Stop ONVIF listeners early while leaving video and recorders running."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self.camera_fleet.quiesce_onvif()

    def _shutdown_components(self) -> None:
        errors: list[tuple[str, Exception]] = []

        def attempt(label: str, callback) -> None:
            try:
                callback()
            except Exception as exc:
                errors.append((label, exc))
                LOGGER.exception("SurvNG shutdown step failed: %s", label)

        started = time.monotonic()
        self.camera_controls.quiesce()
        self.mqtt.set_server_lifecycle("stopping", refresh_status=False)
        LOGGER.info(
            "SurvNG shutdown: cancelling camera admission and releasing ONVIF subscriptions"
        )
        try:
            self.release_onvif_subscriptions()
        except CameraFleetOperationError as error:
            if error.residual_camera_ids:
                raise ManagerShutdownIncompleteError(error) from None
            errors.extend(
                (f"camera ONVIF {failure.label}", failure.error)
                for failure in error.failures
            )
        attempt("runtime monitor", self.runtime_monitor.stop)
        LOGGER.info("SurvNG shutdown: stopping MQTT command intake")
        attempt("MQTT", self.mqtt.stop)
        LOGGER.info("SurvNG shutdown: stopping camera and ONVIF workers")
        try:
            self.camera_fleet.stop_workers()
        except CameraFleetOperationError as error:
            if error.residual_camera_ids:
                raise ManagerShutdownIncompleteError(error) from None
            errors.extend(
                (f"camera {failure.label}", failure.error)
                for failure in error.failures
            )

        LOGGER.info("SurvNG shutdown: stopping inference lifecycle")
        attempt("inference lifecycle", self.inference.close)

        LOGGER.info("SurvNG shutdown: stopping recorder processes")
        attempt("recording lifecycle", self.recording.close)
        attempt("state event broker", self.state_events.close)
        LOGGER.info("SurvNG shutdown complete in %.2fs", time.monotonic() - started)
        if errors:
            labels = ", ".join(label for label, _exc in errors)
            raise RuntimeError(
                f"one or more shutdown steps failed: {labels}"
            ) from None


    def camera(self, camera_id: str):
        return self.camera_fleet.camera(camera_id)

    def start_camera(self, camera_id: str) -> bool:
        return self.camera_controls.start_camera(camera_id)

    def stop_camera(self, camera_id: str) -> bool:
        return self.camera_controls.stop_camera(camera_id)

    def update_camera_zones(
        self,
        camera_id: str,
        zones: list[DetectionZone],
        previous_zones: list[dict],
    ) -> bool:
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        worker.update_zones(zones)
        self.mqtt.remove_zone_discovery(camera_id, previous_zones, self.detector.labels)
        self._mqtt_connected()
        return True

    def _mqtt_power_command(self, camera_id: str, turn_on: bool) -> bool:
        return self.start_camera(camera_id) if turn_on else self.stop_camera(camera_id)

    def _build_mqtt_service(self, mqtt_config: MqttConfig) -> MqttService:
        return MqttService(
            mqtt_config,
            self._mqtt_power_command,
            self.set_recording,
            self.set_detection,
            self._mqtt_connected,
            self._mqtt_server_status,
        )

    def reconfigure_mqtt(self, mqtt_config: MqttConfig) -> None:
        """Replace only the MQTT runtime, leaving cameras and recorders untouched."""
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                raise RuntimeError("application manager is stopping")
            running = self._started
        # Do not hold the manager lock while the old MQTT command worker
        # drains: a command already in flight may need that same lock to finish.
        # MqttLifecycle serializes cutover against stop(), and this guard is
        # evaluated only after that lifecycle lock has been acquired.
        self.mqtt.reconfigure(
            mqtt_config,
            running=running,
            allowed=lambda: not self._stopping and not self._closed,
        )

    @staticmethod
    def _detector_runtime_ready(status: dict[str, object]) -> bool:
        """Normalize readiness across legacy and isolated detector payloads."""
        explicit_ready = status.get("ready")
        if explicit_ready is not None:
            return bool(explicit_ready)
        if status.get("enabled") is False:
            return False
        backend_loaded = bool(
            status.get("loaded_backend")
            or status.get("openvino_loaded")
            or status.get("opencv_loaded")
            or status.get("coreml_loaded")
        )
        isolation = status.get("isolation")
        if isinstance(isolation, dict) and isolation.get("enabled"):
            return backend_loaded and bool(isolation.get("worker_alive"))
        return backend_loaded

    def _mqtt_server_status(self) -> dict[str, dict[str, object]]:
        """Build a bounded MQTT snapshot without scanning recording storage."""
        statuses = self.statuses()
        startup = self.camera_fleet.status()
        startup_counts = dict(startup.get("counts") or {})
        startup_active = bool(startup.get("active"))
        enabled_ids = {
            camera.id
            for camera in self._unique_cameras()
            if self.camera_controls.camera_enabled(camera.id)
        }
        running_cameras = sum(
            1 for status in statuses
            if status.get("id") in enabled_ids and status.get("running")
        )
        recorder_total = 0
        recorders_running = 0
        for status in statuses:
            camera = self.camera(str(status.get("id") or ""))
            if camera is None or camera.id not in enabled_ids or not self.recording_enabled(camera.id):
                continue
            if camera.record:
                recorder_total += 1
                recorders_running += int(bool(status.get("recording")))
            if camera.record_sub and camera.live_stream_url:
                recorder_total += 1
                recorders_running += int(bool(status.get("sub_recording")))

        detector_state = "disabled"
        detector_device = str(self.config.detector.device or "")
        object_queue_depth = 0
        detector_ready = True
        detector_degraded = False
        object_workers_configured = 0
        object_workers_alive = 0
        if self.config.detector.enabled:
            try:
                detector = self.detector_status()
                runtime = dict(detector.get("runtime") or {})
                isolation = dict(detector.get("isolation") or {})
                detector_ready = self._detector_runtime_ready(detector)
                object_workers_configured = int(
                    isolation.get("configured_workers")
                    or int(bool(isolation.get("enabled")))
                )
                object_workers_alive = int(
                    isolation.get("alive_workers")
                    or int(bool(isolation.get("worker_alive")))
                )
                detector_degraded = bool(
                    detector_ready
                    and object_workers_configured > 0
                    and object_workers_alive < object_workers_configured
                )
                detector_state = (
                    "degraded"
                    if detector_degraded
                    else "ready" if detector_ready else "unavailable"
                )
                detector_device = str(detector.get("configured_device") or detector_device)
                object_queue_depth = int(runtime.get("queue_depth") or 0)
            except Exception:
                detector_ready = False
                detector_state = "unavailable"
                LOGGER.warning("failed to sample detector status for MQTT", exc_info=True)

        retention = self.recorder.retention_status()
        retention_state = str(retention.get("state") or "idle")
        if retention_state == "planning":
            activity = "planning"
        elif retention_state in {"queued", "cleaning", "waiting"}:
            activity = "cleaning"
        else:
            activity = "idle"
        if startup_active and activity == "idle":
            activity = "starting_cameras"
        plan = dict(retention.get("plan") or {})
        storage = dict(plan.get("storage") or {})

        health = "ok"
        if self._started and not self._stopping:
            if self.config.detector.enabled and not detector_ready:
                health = "fault"
            elif detector_degraded:
                health = "degraded"
            elif startup_active:
                health = "degraded"
            elif enabled_ids and running_cameras == 0:
                health = "fault"
            elif running_cameras < len(enabled_ids) or recorders_running < recorder_total:
                health = "degraded"
            if bool(storage.get("emergency")):
                health = "fault"

        memory_total, memory_used, memory_percent = system_memory_usage()
        cpu_count = os.cpu_count() or 1
        try:
            load_1m = os.getloadavg()[0]
        except OSError:
            load_1m = 0.0
        storage_free_percent = storage.get("free_percent")
        return {
            "state": {
                "health": health,
                "activity": activity,
                "started_at": self._process_started_at,
                "uptime_seconds": round(
                    max(0.0, time.monotonic() - self._process_started_monotonic),
                    1,
                ),
            },
            "metrics": {
                "cameras_running": running_cameras,
                "cameras_total": len(enabled_ids),
                "recorders_running": recorders_running,
                "recorders_total": recorder_total,
                "cpu_load_percent": round(min(100.0, (load_1m / cpu_count) * 100.0), 1),
                "memory_used_percent": memory_percent,
                "memory_used_bytes": memory_used,
                "memory_total_bytes": memory_total,
                "process_rss_bytes": process_rss_bytes(),
                "storage_free_percent": storage_free_percent,
                "storage_free_bytes": storage.get("free_bytes"),
                "storage_total_bytes": storage.get("total_bytes"),
                "detector_state": detector_state,
                "detector_device": detector_device,
                "object_queue_depth": object_queue_depth,
                "object_workers_configured": object_workers_configured,
                "object_workers_alive": object_workers_alive,
                "retention_state": retention_state,
                "camera_startup_active": startup_active,
                "camera_startup_ready": int(startup_counts.get("ready") or 0),
                "camera_startup_degraded": int(startup_counts.get("degraded") or 0),
                "camera_startup_failed": int(startup_counts.get("failed") or 0),
                "camera_startup_queued": int(startup_counts.get("queued") or 0),
            },
        }

    def reconfigure_recorders(self, next_config: AppConfig) -> None:
        """Restart recorder processes only; camera capture workers remain active."""
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                raise RuntimeError("application manager is stopping")
            self.camera_controls.reconfigure_recorders(
                next_config,
                restart_recorders=self._started,
            )

    def reconfigure_recording_retention(self, next_config: AppConfig) -> None:
        self.recording.reconfigure_retention(
            next_config.retention,
            list(self._unique_cameras()),
        )

    def reconfigure_image_storage(self, config: ImageStorageConfig) -> None:
        self.image_writer.reconfigure(config)

    def reconfigure_detector_policy(self, config: AppConfig) -> None:
        """Apply policy-only detector settings without disturbing camera workers."""
        self.inference.reconfigure_policy(config.detector)
        cameras = {camera.id: camera for camera in config.cameras}
        for worker in self.camera_fleet.workers.values():
            camera = cameras.get(worker.camera.id)
            mode = (
                config.detector.object_activity_attribution
                if camera is None or camera.object_activity_attribution == "inherit"
                else camera.object_activity_attribution
            )
            worker.reconfigure_object_activity_attribution(
                mode
            )

    def reconfigure_motion(
        self,
        config: AppConfig,
        *,
        restart_camera_ids: set[str],
        hot_camera_ids: set[str],
    ) -> None:
        """Apply EMA policy live and replace only structurally affected cameras."""
        cameras = {camera.id: camera for camera in config.cameras}
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                raise RuntimeError("application manager is stopping")
            for camera_id in sorted(hot_camera_ids):
                worker = self.workers.get(camera_id)
                camera = cameras.get(camera_id)
                if worker is not None and camera is not None:
                    worker.reconfigure_motion_policy(
                        config.motion_qualification,
                        camera,
                    )
            if not restart_camera_ids:
                return

            replacements: dict[str, CameraWorker] = {}
            previous: dict[str, CameraWorker] = {}
            previous_evidence: dict[str, MotionEvidenceRepository] = {}
            enabled: dict[str, bool] = {}
            detection: dict[str, bool] = {}
            published: set[str] = set()
            try:
                for camera_id in sorted(restart_camera_ids):
                    camera = cameras.get(camera_id)
                    old_worker = self.workers.get(camera_id)
                    if camera is None or old_worker is None:
                        raise RuntimeError(f"camera {camera_id} cannot be reconfigured")
                    previous[camera_id] = old_worker
                    previous_evidence[camera_id] = self.motion_evidence[camera_id]
                    enabled[camera_id] = self.camera_controls.camera_enabled(camera_id)
                    detection[camera_id] = self.camera_controls.detection_enabled(camera_id)
                    replacements[camera_id] = self._create_camera_worker(camera)

                for camera_id, old_worker in previous.items():
                    old_worker.stop()
                    replacement = replacements[camera_id]
                    replacement.set_detection_enabled(detection[camera_id])
                    if enabled[camera_id]:
                        replacement.start()
                    camera = cameras[camera_id]
                    self.workers[camera_id] = replacement
                    self.camera_fleet.replace_worker(camera, replacement)
                    self.camera_controls.replace_worker(camera, replacement)
                    self.inference.replace_worker(camera_id, replacement)
                    published.add(camera_id)
            except BaseException:
                for camera_id, replacement in replacements.items():
                    try:
                        replacement.close()
                    except Exception:
                        LOGGER.exception(
                            "failed to close replacement camera %s during rollback",
                            camera_id,
                        )
                for camera_id in published:
                    old_worker = previous[camera_id]
                    old_camera = old_worker.camera
                    self.workers[camera_id] = old_worker
                    self.camera_fleet.replace_worker(old_camera, old_worker)
                    self.camera_controls.replace_worker(old_camera, old_worker)
                    self.inference.replace_worker(camera_id, old_worker)
                    self.motion_evidence[camera_id] = previous_evidence[camera_id]
                for camera_id in previous.keys() - published:
                    self.motion_evidence[camera_id] = previous_evidence[camera_id]
                for camera_id, old_worker in previous.items():
                    if enabled.get(camera_id):
                        try:
                            old_worker.start()
                        except Exception:
                            LOGGER.exception(
                                "failed to restart camera %s after motion rollback",
                                camera_id,
                            )
                raise

            for camera_id, old_worker in previous.items():
                try:
                    old_worker.close()
                except Exception:
                    LOGGER.exception(
                        "retired camera %s did not close after motion reconfiguration",
                        camera_id,
                    )
            try:
                self._mqtt_connected()
            except Exception:
                # Camera replacement is already committed. Discovery refresh
                # is observational and must not turn a healthy cutover into a
                # second camera-generation rollback.
                LOGGER.exception(
                    "MQTT discovery refresh failed after motion reconfiguration"
                )

    def reconfigure_object_tracking(self, config: DetectorConfig) -> None:
        """Replace tracking sessions without restarting camera-owned services."""
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                raise RuntimeError("application manager is stopping")
            self.inference.reconfigure_tracking(config)

    def reconfigure_inference(
        self,
        config: DetectorConfig,
        roles: set[str],
        *,
        refresh_tracking: bool = False,
    ) -> None:
        """Restart selected inference roles without restarting camera workers."""
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                raise RuntimeError("application manager is stopping")
            self.inference.reconfigure_roles(
                config,
                roles,
                refresh_tracking=refresh_tracking,
            )
            if "object" in roles:
                try:
                    self._mqtt_connected()
                except Exception:
                    LOGGER.exception(
                        "MQTT discovery refresh failed after detector reconfiguration"
                    )

    def reconfigure_semantic_search(self, config: SemanticSearchConfig) -> None:
        """Replace semantic search independently of cameras and object inference."""
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                raise RuntimeError("application manager is stopping")
            self.inference.reconfigure_semantic_search(config)

    def semantic_search_status(self) -> dict:
        return self.inference.semantic_search.status()

    @property
    def semantic_search(self) -> DisabledSemanticSearch:
        return self.inference.semantic_search

    @property
    def appearance_backfill(self) -> DeferredAppearanceBackfill:
        return self.inference.appearance_backfill

    def _mqtt_connected(self) -> None:
        self.mqtt.publish_discovery([
            {
                "id": camera.id,
                "name": camera.name,
                "model_classes": self.detector.labels,
                "recording_configured": bool(camera.record or camera.record_sub),
                "zones": [
                    {
                        "name": zone.name,
                        "enabled": zone.enabled,
                        "object_classes": zone.object_classes or self.detector.labels,
                    }
                    for zone in camera.zones
                ],
            }
            for camera in self._unique_cameras()
        ])
        for status in self.statuses():
            camera_id = str(status.get("id") or "")
            self.mqtt.publish_camera_state(camera_id, bool(status.get("running")))
            self.mqtt.publish_camera_feature_state(camera_id, "recording", bool(status.get("recording_enabled")))
            self.mqtt.publish_camera_feature_state(camera_id, "detection", bool(status.get("detection_enabled")))

    def publish_event(self, event_type: str, payload: dict) -> None:
        camera_id = str(payload.get("camera_id") or "")
        if not camera_id:
            return
        if event_type == "incident" or (event_type == "object" and payload.get("source") == "manual_openvino"):
            event_id = int(payload.get("event_id") or 0)
            event = self.events.get(event_id) if event_id else None
            camera = self.camera(camera_id)
            if event is not None:
                self.mqtt.track_incident(
                    event,
                    camera.name if camera is not None else camera_id,
                    self.config.base_path,
                    allow_new=event_type == "incident",
                )
        if event_type == "object":
            objects = payload.get("objects") or []
            incident_objects = payload.get("incident_objects")
            alert_objects = (
                incident_objects if isinstance(incident_objects, list) else objects
            )
            event_id = payload.get("event_id")
            if event_id:
                event = self.events.get(int(event_id))
                if event:
                    self.faces.ingest_events([event])
                    self.semantic_search.queue_event(event)
                    self.appearance_backfill.enqueue(int(event_id), camera_id)
            payload = {
                **payload,
                "classes": sorted({str(item.get("label")) for item in alert_objects if item.get("label")}),
                "zones": sorted({str(zone) for item in alert_objects for zone in item.get("zones", []) if zone}),
            }
        self.mqtt.publish(f"camera/{camera_id}/{event_type}", payload)
        self.state_events.publish(event_type, payload)
        if event_type == "object_tracking" and payload.get("state") != "active":
            # Existing incident clients already use this event to coalesce refreshes.
            self.state_events.publish("incident", {
                "event_id": payload.get("event_id"),
                "camera_id": camera_id,
                "updated": True,
            })
        if event_type == "object":
            camera = self.camera(camera_id)
            if camera is not None:
                self.mqtt.publish_zone_objects(
                    camera_id,
                    [
                        {
                            "name": zone.name,
                            "enabled": zone.enabled,
                            "object_classes": zone.object_classes or self.detector.labels,
                        }
                        for zone in camera.zones
                    ],
                    {**payload, "objects": alert_objects},
                )

    def mqtt_status(self) -> dict:
        return self.mqtt.status()

    def statuses(self) -> list[dict]:
        recording_keys = set()
        camera_config = {camera.id: camera for camera in self._unique_cameras()}
        for camera in camera_config.values():
            if camera.record:
                recording_keys.add((camera.id, "main"))
            if camera.record_sub and camera.live_stream_url:
                recording_keys.add((camera.id, "live"))
        recordings = self.recorder.status(recording_keys)
        timestamp_health = self.recorder.timestamp_health()
        startup_cameras = dict(self.camera_fleet.status().get("cameras") or {})
        return [
            {
                **worker.status(),
                "startup": dict(startup_cameras.get(camera_id) or {}),
                "expected_enabled": self.camera_controls.camera_enabled(camera_id),
                "recording": recordings.get((camera_id, "main"), False),
                "sub_recording": recordings.get((camera_id, "live"), False),
                "recording_enabled": self.recording_enabled(camera_id),
                "recording_configured": bool(
                    camera_config.get(camera_id)
                    and (camera_config[camera_id].record or camera_config[camera_id].record_sub)
                ),
                "record_sub_enabled": bool(camera_config.get(camera_id) and camera_config[camera_id].record_sub),
                "recording_timestamp_health": {
                    source: dict(timestamp_health[(camera_id, source)])
                    for source in ("main", "live")
                    if (camera_id, source) in timestamp_health
                },
            }
            for camera_id, worker in self.workers.items()
        ]

    def detector_status(self) -> dict:
        return {
            **self.detector.status(),
            "lifecycle": self.inference.status(),
        }

    def go2rtc_status(self) -> dict:
        return self.go2rtc.status(list(self._unique_cameras()))
