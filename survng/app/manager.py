from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .camera import CameraWorker
from .appearance_index import AppearanceIndex
from .config import (
    AppConfig,
    CameraConfig,
    CameraMotionQualificationConfig,
    DetectorConfig,
    DetectionZone,
    ImageStorageConfig,
    MqttConfig,
    ObjectTrackingConfig,
)
from .events import EventStore
from .faces import FaceStore
from .detector import objects_to_json
from .go2rtc import Go2RtcAdapter
from .inference import InferenceSupervisor, IsolatedFaceRecognizer, IsolatedPersonReidentifier
from .image_cache import LocalImageCache
from .image_storage import DurableImageWriter
from .mqtt import MqttService
from .object_tracking import ObjectTrackingSession, ObjectTrackingSessionFactory
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
from .recorder import Recorder
from .state_events import StateEventBroker


LOGGER = logging.getLogger("uvicorn.error")
ONVIF_RELEASE_TIMEOUT_SECONDS = 15.0
CAMERA_SHUTDOWN_TIMEOUT_SECONDS = 30.0


def validate_motion_pipeline_configuration(config: AppConfig) -> None:
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
                required_artifacts={"scoring", "decision"},
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
        self.events = EventStore(self.storage_dir, database_dir=self.database_dir)
        self.appearance_index = AppearanceIndex(self.events.db_path)
        self.detector = InferenceSupervisor(config.detector)
        self.face_recognizer = IsolatedFaceRecognizer(self.detector)
        self.person_reidentifier = IsolatedPersonReidentifier(self.detector)
        self.faces = FaceStore(
            self.storage_dir,
            config.detector.face_max_observations,
            self.face_recognizer,
            start_recognition=False,
            database_dir=self.database_dir,
        )
        recording_index_dir = Path(config.recording_index_dir) if config.recording_index_dir else None
        self.recorder = Recorder(
            config.ffmpeg_path,
            self.storage_dir,
            config.recording_segment_seconds,
            config.hardware_acceleration,
            index_dir=recording_index_dir,
            retention_config=config.retention,
            protected_recording_paths=self.events.protected_recording_paths,
        )
        self.go2rtc = Go2RtcAdapter()
        self.state_events = StateEventBroker()
        self._object_tracking_limiter = threading.BoundedSemaphore(
            config.detector.tracking.max_active_cameras
        )
        self.motion_pipeline_registry = build_builtin_motion_registry()
        self.motion_decision_handler_factory = MotionDecisionHandlerFactory(
            events=self.events,
            object_serializer=objects_to_json,
        )
        self.motion_object_detector_factory = RecordedMotionObjectDetectorFactory(
            detector=self.detector,
            recorder=self.recorder,
        )
        self.object_tracking_session_factory = ObjectTrackingSessionFactory(
            config=config.detector.tracking,
            detector=self.detector,
            update_event=self.events.update_object_tracking,
            publisher=self.publish_event,
            limiter=self._object_tracking_limiter,
            appearance_encoder=self.person_reidentifier,
            appearance_indexer=self.appearance_index.replace_event,
        )
        self.mqtt = self._build_mqtt_service(config.mqtt)
        self._process_started_monotonic = time.monotonic()
        self._process_started_at = datetime.now(timezone.utc).isoformat()
        self._lifecycle_lock = threading.RLock()
        self._runtime_state_lock = threading.Lock()
        self._runtime_state_path = self.storage_dir / "runtime_state.json"
        # A fresh SurvNG process always starts from the safe, deterministic
        # operational default. Home-automation policies can reapply their
        # desired runtime state after MQTT reports that the server is running.
        # In-process configuration reloads preserve current preferences through
        # apply_runtime_preferences() before the replacement manager starts.
        self._recording_enabled: dict[str, bool] = {}
        self._detection_enabled: dict[str, bool] = {}
        self._camera_enabled = {
            camera.id: True
            for camera in config.cameras
        }
        self._stopping = False
        self._started = False
        self._closed = False
        self._state_monitor_stop = threading.Event()
        self._state_monitor_thread: threading.Thread | None = None
        self.motion_evidence: dict[str, MotionEvidenceRepository] = {}
        self._motion_analysis_limiter = threading.BoundedSemaphore(2)
        workers: dict[str, CameraWorker] = {}
        try:
            for camera in self._unique_cameras():
                workers[camera.id] = self._create_camera_worker(camera)
        except BaseException:
            for worker in reversed(tuple(workers.values())):
                try:
                    worker.close()
                except Exception:
                    LOGGER.exception("camera cleanup failed during manager construction")
            for label, callback in (
                ("MQTT", self.mqtt.stop),
                ("face recognition", self.faces.close),
                ("inference", self.detector.stop),
                ("recorder", self.recorder.stop_all),
                ("state event broker", self.state_events.close),
            ):
                try:
                    callback()
                except Exception:
                    LOGGER.exception("%s cleanup failed during manager construction", label)
            raise
        self.workers = workers

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
                required_artifacts={"scoring", "decision"},
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
                object_tracking_session_factory=self.object_tracking_session_factory,
                motion_analysis_limiter=self._motion_analysis_limiter,
                image_writer=self.image_writer,
                onvif_cache_dir=self.database_dir / "onvif",
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

    def _save_runtime_state(self) -> None:
        with self._runtime_state_lock:
            payload = {
                "recording_enabled": self._recording_enabled,
                "detection_enabled": self._detection_enabled,
                "camera_enabled": self._camera_enabled,
            }
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self._runtime_state_path.parent,
                    prefix=f".{self._runtime_state_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    json.dump(payload, handle, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._runtime_state_path)
                temporary = None
                try:
                    directory_fd = os.open(self._runtime_state_path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError:
                    pass
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass

    def recording_enabled(self, camera_id: str) -> bool:
        return self._recording_enabled.get(camera_id, True)

    def detection_enabled(self, camera_id: str) -> bool:
        return self._detection_enabled.get(camera_id, True)

    def _recorder_should_run(self, camera_id: str) -> bool:
        return self._camera_enabled.get(camera_id, True) and self.recording_enabled(camera_id)

    def _start_configured_recorders(self, camera) -> None:
        if camera.record:
            self.recorder.start(camera, "main")
        if camera.record_sub and camera.live_stream_url:
            self.recorder.start(camera, "live")

    def set_recording(self, camera_id: str, enabled: bool) -> bool:
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                return False
            camera = self.camera(camera_id)
            if camera is None:
                return False
            had_previous = camera_id in self._recording_enabled
            previous = self._recording_enabled.get(camera_id)
            self._recording_enabled[camera_id] = bool(enabled)
            try:
                self._save_runtime_state()
            except Exception:
                if had_previous:
                    self._recording_enabled[camera_id] = bool(previous)
                else:
                    self._recording_enabled.pop(camera_id, None)
                raise
            should_run = self._recorder_should_run(camera_id)
            self.recorder.set_camera_enabled(camera_id, should_run)
            if should_run:
                self._start_configured_recorders(camera)
            self.mqtt.publish_camera_feature_state(camera_id, "recording", bool(enabled))
            self._publish_camera_status(camera_id)
            return True

    def set_detection(self, camera_id: str, enabled: bool) -> bool:
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                return False
            worker = self.workers.get(camera_id)
            if worker is None:
                return False
            had_previous = camera_id in self._detection_enabled
            previous = self._detection_enabled.get(camera_id)
            self._detection_enabled[camera_id] = bool(enabled)
            try:
                self._save_runtime_state()
            except Exception:
                if had_previous:
                    self._detection_enabled[camera_id] = bool(previous)
                else:
                    self._detection_enabled.pop(camera_id, None)
                raise
            worker.set_detection_enabled(enabled)
            self.mqtt.publish_camera_feature_state(camera_id, "detection", bool(enabled))
            self._publish_camera_status(camera_id)
            return True

    def runtime_preferences(self) -> dict[str, dict[str, bool]]:
        with self._lifecycle_lock:
            return {
                "recording_enabled": dict(self._recording_enabled),
                "detection_enabled": dict(self._detection_enabled),
                "camera_enabled": dict(self._camera_enabled),
            }

    def apply_runtime_preferences(
        self,
        preferences: dict[str, dict[str, bool]],
        *,
        persist: bool = False,
    ) -> None:
        with self._lifecycle_lock:
            previous = (
                self._recording_enabled,
                self._detection_enabled,
                self._camera_enabled,
            )
            camera_ids = set(self.workers)
            self._recording_enabled = {
                camera_id: bool(enabled)
                for camera_id, enabled in preferences.get("recording_enabled", {}).items()
                if camera_id in camera_ids
            }
            self._detection_enabled = {
                camera_id: bool(enabled)
                for camera_id, enabled in preferences.get("detection_enabled", {}).items()
                if camera_id in camera_ids
            }
            self._camera_enabled = {
                camera_id: bool(preferences.get("camera_enabled", {}).get(camera_id, True))
                for camera_id in camera_ids
            }
            try:
                if persist:
                    self._save_runtime_state()
            except BaseException:
                (
                    self._recording_enabled,
                    self._detection_enabled,
                    self._camera_enabled,
                ) = previous
                raise

    def start_all(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("application manager is closed")
            if self._started:
                return
            self._stopping = False
            try:
                # Replace any state left by the previous process with the
                # preferences this manager will actually apply. On a full
                # service start these are all-on defaults; reload candidates
                # receive the active manager's preferences before this point.
                self._save_runtime_state()
                self.detector.start()
                self.faces.start()
                cameras = list(self._unique_cameras())
                recorder_keys = set()
                for camera in cameras:
                    if camera.record:
                        recorder_keys.add((camera.id, "main"))
                    if camera.record_sub and camera.live_stream_url:
                        recorder_keys.add((camera.id, "live"))
                self.recorder.cleanup_stale_recorders(recorder_keys)
                for camera in cameras:
                    camera_enabled = self._camera_enabled.get(camera.id, True)
                    self.recorder.set_camera_enabled(
                        camera.id,
                        camera_enabled and self.recording_enabled(camera.id),
                    )
                    self.workers[camera.id].set_detection_enabled(self.detection_enabled(camera.id))
                    if camera_enabled:
                        self.workers[camera.id].start()
                        if self.recording_enabled(camera.id):
                            self._start_configured_recorders(camera)
                    self.mqtt.publish_camera_state(camera.id, camera_enabled)
                self.recorder.start_indexer(cameras)
                self.recorder.start_watchdog(cameras)
                # Publish discovery only after persisted recording/detection preferences
                # have been applied to every worker.
                self.mqtt.start()
                self._start_state_monitor()
                self._started = True
                self.mqtt.set_server_lifecycle("running")
            except BaseException:
                self._stopping = True
                try:
                    self._shutdown_components()
                except Exception:
                    LOGGER.exception("application startup rollback was incomplete")
                self._closed = True
                raise

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
            finally:
                self._started = False
                self._closed = True
            return preferences

    def release_onvif_subscriptions(self) -> None:
        """Stop ONVIF listeners early while leaving video and recorders running."""
        with self._lifecycle_lock:
            if self._closed:
                return
            failures: list[tuple[str, Exception]] = []
            failures_lock = threading.Lock()

            def release(camera_id: str, worker: CameraWorker) -> None:
                try:
                    worker.stop_onvif_events()
                except Exception as exc:
                    with failures_lock:
                        failures.append((camera_id, exc))
                    LOGGER.exception(
                        "early ONVIF subscription release failed for %s",
                        camera_id,
                    )

            releases = [
                (camera_id, threading.Thread(
                    target=release,
                    args=(camera_id, worker),
                    name=f"release-onvif-{camera_id}",
                    daemon=True,
                ))
                for camera_id, worker in self.workers.items()
            ]
            for _camera_id, thread in releases:
                thread.start()
            deadline = time.monotonic() + ONVIF_RELEASE_TIMEOUT_SECONDS
            for _camera_id, thread in releases:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            with failures_lock:
                for camera_id, thread in releases:
                    if thread.is_alive():
                        failures.append((
                            camera_id,
                            RuntimeError("ONVIF subscription release timed out"),
                        ))
                failure_snapshot = list(failures)
            if failure_snapshot:
                cameras = ", ".join(camera_id for camera_id, _exc in failure_snapshot)
                raise RuntimeError(
                    f"failed to release ONVIF subscriptions for: {cameras}"
                ) from failure_snapshot[0][1]

    def _shutdown_components(self) -> None:
        errors: list[tuple[str, Exception]] = []

        def attempt(label: str, callback) -> None:
            try:
                callback()
            except Exception as exc:
                errors.append((label, exc))
                LOGGER.exception("SurvNG shutdown step failed: %s", label)

        started = time.monotonic()
        self.mqtt.set_server_lifecycle("stopping", refresh_status=False)
        LOGGER.info("SurvNG shutdown: releasing ONVIF subscriptions")
        attempt("ONVIF subscriptions", self.release_onvif_subscriptions)
        attempt("state monitor", self._stop_state_monitor)
        LOGGER.info("SurvNG shutdown: stopping MQTT command intake")
        attempt("MQTT", self.mqtt.stop)
        LOGGER.info("SurvNG shutdown: stopping camera and ONVIF workers")
        camera_errors: list[tuple[str, Exception]] = []
        camera_errors_lock = threading.Lock()

        def stop_worker(camera_id: str, worker: CameraWorker) -> None:
            try:
                worker.stop()
            except Exception as exc:
                with camera_errors_lock:
                    camera_errors.append((camera_id, exc))
                LOGGER.exception("camera shutdown failed for %s", camera_id)

        camera_shutdowns = [
            (camera_id, threading.Thread(
                target=stop_worker,
                args=(camera_id, worker),
                name=f"stop-camera-{camera_id}",
                daemon=True,
            ))
            for camera_id, worker in self.workers.items()
        ]
        for _camera_id, thread in camera_shutdowns:
            thread.start()
        deadline = time.monotonic() + CAMERA_SHUTDOWN_TIMEOUT_SECONDS
        for _camera_id, thread in camera_shutdowns:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        timed_out_cameras = {
            camera_id for camera_id, thread in camera_shutdowns if thread.is_alive()
        }
        with camera_errors_lock:
            for camera_id in sorted(timed_out_cameras):
                camera_errors.append((camera_id, RuntimeError("camera shutdown timed out")))
            camera_error_snapshot = list(camera_errors)
        errors.extend((f"camera {camera_id}", exc) for camera_id, exc in camera_error_snapshot)
        for camera_id, worker in self.workers.items():
            # close() is the final idempotent resource sweep and must still run
            # when stop() reports a partial teardown failure. Do not race a
            # stop() call that exceeded the manager-level deadline.
            if camera_id not in timed_out_cameras:
                attempt(f"camera {camera_id} resources", worker.close)

        LOGGER.info("SurvNG shutdown: stopping face recognition")
        attempt("face recognition", self.faces.close)

        LOGGER.info("SurvNG shutdown: stopping isolated inference workers")
        attempt("inference", self.detector.stop)

        LOGGER.info("SurvNG shutdown: stopping recorder processes")
        attempt("recorders", self.recorder.stop_all)
        attempt("state event broker", self.state_events.close)
        LOGGER.info("SurvNG shutdown complete in %.2fs", time.monotonic() - started)
        if errors:
            labels = ", ".join(label for label, _exc in errors)
            raise RuntimeError(f"one or more shutdown steps failed: {labels}") from errors[0][1]


    def camera(self, camera_id: str):
        return next((camera for camera in self.config.cameras if camera.id == camera_id), None)

    def start_camera(self, camera_id: str) -> bool:
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                return False
            camera = self.camera(camera_id)
            worker = self.workers.get(camera_id)
            if camera is None or worker is None:
                return False
            previous = self._camera_enabled.get(camera_id, True)
            self._camera_enabled[camera_id] = True
            try:
                self._save_runtime_state()
                self.recorder.set_camera_enabled(camera_id, self.recording_enabled(camera_id))
                worker.start()
            except Exception:
                self._camera_enabled[camera_id] = previous
                try:
                    self._save_runtime_state()
                except Exception:
                    LOGGER.exception("failed to roll back camera power state for %s", camera_id)
                self.recorder.set_camera_enabled(
                    camera_id,
                    previous and self.recording_enabled(camera_id),
                )
                raise
            if self.recording_enabled(camera_id):
                self._start_configured_recorders(camera)
            self.mqtt.publish_camera_state(camera_id, True)
            self._publish_camera_status(camera_id)
            return True

    def stop_camera(self, camera_id: str) -> bool:
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                return False
            worker = self.workers.get(camera_id)
            if worker is None:
                return False
            previous = self._camera_enabled.get(camera_id, True)
            self._camera_enabled[camera_id] = False
            try:
                self._save_runtime_state()
            except Exception:
                self._camera_enabled[camera_id] = previous
                raise
            self.recorder.set_camera_enabled(camera_id, False)
            worker.stop()
            self.mqtt.publish_camera_state(camera_id, False)
            self._publish_camera_status(camera_id)
            return True

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
            previous = self.mqtt
            replacement = self._build_mqtt_service(mqtt_config)
            previous.stop(lifecycle="restarting")
            self.mqtt = replacement
            try:
                replacement.start()
                if self._started:
                    replacement.set_server_lifecycle("running")
            except BaseException:
                self.mqtt = previous
                try:
                    replacement.stop(lifecycle="restarting")
                finally:
                    previous.start()
                    if self._started:
                        previous.set_server_lifecycle("running")
                raise

    @staticmethod
    def _memory_usage() -> tuple[int, int, float]:
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return 0, 0, 0.0
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", values.get("MemFree", 0))
        used = max(0, total - available)
        return total, used, round((used / total) * 100.0, 1) if total else 0.0

    @staticmethod
    def _process_rss_bytes() -> int:
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return 0

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
        enabled_ids = {
            camera.id
            for camera in self._unique_cameras()
            if self._camera_enabled.get(camera.id, True)
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
        if self.config.detector.enabled:
            try:
                detector = self.detector_status()
                runtime = dict(detector.get("runtime") or {})
                detector_ready = self._detector_runtime_ready(detector)
                detector_state = "ready" if detector_ready else "unavailable"
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
        plan = dict(retention.get("plan") or {})
        storage = dict(plan.get("storage") or {})

        health = "ok"
        if self._started and not self._stopping:
            if self.config.detector.enabled and not detector_ready:
                health = "fault"
            elif enabled_ids and running_cameras == 0:
                health = "fault"
            elif running_cameras < len(enabled_ids) or recorders_running < recorder_total:
                health = "degraded"
            if bool(storage.get("emergency")):
                health = "fault"

        memory_total, memory_used, memory_percent = self._memory_usage()
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
                "process_rss_bytes": self._process_rss_bytes(),
                "storage_free_percent": storage_free_percent,
                "storage_free_bytes": storage.get("free_bytes"),
                "storage_total_bytes": storage.get("total_bytes"),
                "detector_state": detector_state,
                "detector_device": detector_device,
                "object_queue_depth": object_queue_depth,
                "retention_state": retention_state,
            },
        }

    def reconfigure_recorders(self, next_config: AppConfig) -> None:
        """Restart recorder processes only; camera capture workers remain active."""
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                raise RuntimeError("application manager is stopping")
            cameras = list(self._unique_cameras())
            desired_enabled = {
                camera.id: (
                    self._camera_enabled.get(camera.id, True)
                    and self.recording_enabled(camera.id)
                )
                for camera in cameras
            }
            for camera in cameras:
                self.recorder.set_camera_enabled(camera.id, False)
            try:
                self.recorder.reconfigure_runtime(
                    ffmpeg_path=next_config.ffmpeg_path,
                    hardware_acceleration=next_config.hardware_acceleration,
                    segment_seconds=next_config.recording_segment_seconds,
                )
            finally:
                for camera in cameras:
                    self.recorder.set_camera_enabled(
                        camera.id,
                        desired_enabled[camera.id],
                    )
            if not self._started:
                return
            for camera in cameras:
                if desired_enabled[camera.id]:
                    self._start_configured_recorders(camera)

    def reconfigure_image_storage(self, config: ImageStorageConfig) -> None:
        self.image_writer.reconfigure(config)

    def reconfigure_detector_policy(self, config: DetectorConfig) -> None:
        """Apply policy-only detector settings without disturbing camera workers."""
        self.detector.update_runtime_config(config)
        self.face_recognizer.config = self.detector.config
        self.faces.reconfigure_max_observations(config.face_max_observations)

    def reconfigure_object_tracking(self, config: DetectorConfig) -> None:
        """Replace tracking sessions without restarting camera-owned services."""
        with self._lifecycle_lock:
            if self._stopping or self._closed:
                raise RuntimeError("application manager is stopping")
            tracking = config.tracking.model_copy(deep=True)
            next_limiter = threading.BoundedSemaphore(tracking.max_active_cameras)
            next_factory = ObjectTrackingSessionFactory(
                config=tracking,
                detector=self.detector,
                update_event=self.events.update_object_tracking,
                publisher=self.publish_event,
                limiter=next_limiter,
                appearance_encoder=self.person_reidentifier,
                appearance_indexer=self.appearance_index.replace_event,
            )
            workers = list(self.workers.values())
            replacements = [
                worker.create_object_tracking_session(next_factory)
                for worker in workers
            ]
            previous_sessions: list[tuple[CameraWorker, ObjectTrackingSession]] = []
            previous_config = self.detector.config
            previous_factory = self.object_tracking_session_factory
            previous_limiter = self._object_tracking_limiter
            try:
                for worker, replacement in zip(workers, replacements, strict=True):
                    previous = worker.replace_object_tracking_session(replacement)
                    previous_sessions.append((worker, previous))
                self.detector.update_runtime_config(config)
                self.face_recognizer.config = self.detector.config
                self.person_reidentifier.config = self.detector.config.tracking
                self.object_tracking_session_factory = next_factory
                self._object_tracking_limiter = next_limiter
            except BaseException:
                for worker, previous in reversed(previous_sessions):
                    try:
                        worker.replace_object_tracking_session(previous)
                    except Exception:
                        LOGGER.exception(
                            "failed to roll back object tracking for %s",
                            worker.camera.id,
                        )
                self.detector.update_runtime_config(previous_config)
                self.face_recognizer.config = self.detector.config
                self.person_reidentifier.config = self.detector.config.tracking
                self.object_tracking_session_factory = previous_factory
                self._object_tracking_limiter = previous_limiter
                raise

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
            event_id = payload.get("event_id")
            if event_id:
                event = self.events.get(int(event_id))
                if event:
                    self.faces.ingest_events([event])
            payload = {
                **payload,
                "classes": sorted({str(item.get("label")) for item in objects if item.get("label")}),
                "zones": sorted({str(zone) for item in objects for zone in item.get("zones", []) if zone}),
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
                    payload,
                )

    @staticmethod
    def _camera_state_fingerprint(status: dict) -> tuple:
        keys = (
            "running", "connected", "capture_running", "frame_fresh", "main_running",
            "main_frame_fresh", "last_error", "main_last_error", "onvif_connected",
            "onvif_last_event_at", "onvif_last_motion_event_at", "onvif_last_error",
            "onvif_last_poll_success_at", "onvif_last_poll_error_at",
            "onvif_notifications_received", "onvif_motion_events_received",
            "onvif_renewals", "onvif_renewal_errors", "last_motion_at",
            "detection_enabled", "recording",
            "sub_recording", "recording_enabled", "recording_configured",
            "stream_dimensions",
        )
        motion = status.get("motion_qualification") or {}
        return tuple(status.get(key) for key in keys) + (
            motion.get("passed"),
            motion.get("audit_rejected"),
            motion.get("suppressed"),
            motion.get("last_decision_at"),
        )

    def _publish_camera_status(self, camera_id: str) -> None:
        status = next((item for item in self.statuses() if item.get("id") == camera_id), None)
        if status is not None:
            self.state_events.publish("camera_state", status)

    def _start_state_monitor(self) -> None:
        if self._state_monitor_thread is not None and self._state_monitor_thread.is_alive():
            return
        self._state_monitor_stop.clear()

        def monitor() -> None:
            previous: dict[str, tuple] = {}
            telemetry_sample_at = 0.0
            while not self._state_monitor_stop.is_set():
                try:
                    statuses = self.statuses()
                    for status in statuses:
                        camera_id = str(status.get("id") or "")
                        fingerprint = self._camera_state_fingerprint(status)
                        if camera_id and previous.get(camera_id) != fingerprint:
                            previous[camera_id] = fingerprint
                            self.state_events.publish("camera_state", status)
                    now = time.monotonic()
                    if now - telemetry_sample_at >= 60.0:
                        self.events.record_runtime_telemetry(statuses)
                        telemetry_sample_at = now
                except Exception:
                    LOGGER.exception("camera state monitor failed")
                self._state_monitor_stop.wait(1.0)

        self._state_monitor_thread = threading.Thread(target=monitor, name="camera-state-monitor", daemon=False)
        self._state_monitor_thread.start()

    def _stop_state_monitor(self) -> None:
        self._state_monitor_stop.set()
        thread = self._state_monitor_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError("camera state monitor did not stop")
        self._state_monitor_thread = None

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
        return [
            {
                **worker.status(),
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
        return self.detector.status()

    def go2rtc_status(self) -> dict:
        return self.go2rtc.status(list(self._unique_cameras()))
