from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
import time
from typing import Any

import numpy as np

from ..config import DetectorConfig
from .process import load_detector_labels, stop_multiprocessing_resource_tracker
from .types import (
    INCIDENT_INITIAL_ADMISSION_TIMEOUT_SECONDS,
    INCIDENT_INITIAL_WORKER_TIMEOUT_SECONDS,
    INFERENCE_REQUEST_TIMEOUT_SECONDS,
    INFERENCE_STATUS_TIMEOUT_SECONDS,
    LOGGER,
    MAX_INFERENCE_FRAME_BYTES,
    PERSON_REID_REQUEST_TIMEOUT_SECONDS,
    InferenceRollbackIncomplete,
    InferenceUnavailable,
    InferenceWorkload,
)
from .worker import _InferenceWorker


class InferenceSupervisor:
    def __init__(self, config: DetectorConfig) -> None:
        self._config_lock = threading.RLock()
        self.config = config
        self.labels = load_detector_labels(config)
        self.enabled = bool(
            config.enabled
            and (config.resolved_model_path() or config.resolved_coreml_model_path())
        )
        self._object_route_lock = threading.Lock()
        self._object_route_cursor = 0
        self._device_condition = threading.Condition(threading.Lock())
        self._device_accepting = True
        self._security_waiting = 0
        self._security_active = 0
        self._initial_waiting = 0
        self._initial_active = 0
        self._refinement_active = 0
        self._optional_active = 0
        self._offline_active = 0
        self._device_workload_stats: dict[InferenceWorkload, dict[str, float | int]] = {
            workload: {
                "admitted": 0,
                "completed": 0,
                "shed": 0,
                "timed_out": 0,
                "wait_total_ms": 0.0,
                "wait_max_ms": 0.0,
            }
            for workload in InferenceWorkload
        }
        self._object_workers = self._build_object_workers(config)
        # Preserve the long-standing primary-worker handle for compatibility
        # with lifecycle diagnostics and focused tests.
        self._object = self._object_workers[0]
        self._face = _InferenceWorker(
            config,
            "face",
            self._base_face_status(),
            start_enabled=bool(config.face_recognition_enabled),
        )
        self._reid = _InferenceWorker(
            config,
            "reid",
            self._base_reid_status(),
            start_enabled=bool(config.tracking.appearance_reid_enabled),
        )

    @staticmethod
    def _effective_object_worker_count(config: DetectorConfig) -> int:
        return config.object_worker_count if config.backend == "openvino" else 1

    def _build_object_workers(self, config: DetectorConfig) -> list[_InferenceWorker]:
        return [
            _InferenceWorker(config, "object", self._base_detector_status())
            for _index in range(self._effective_object_worker_count(config))
        ]

    def _ordered_object_workers(
        self,
        workload: InferenceWorkload = InferenceWorkload.INCIDENT_INITIAL,
    ) -> list[_InferenceWorker]:
        """Order workers by pressure while rotating equal-load workers fairly."""
        with self._config_lock:
            workers = list(self._object_workers)
        with self._object_route_lock:
            if workload >= InferenceWorkload.TRACKING and len(workers) > 1:
                # Worker zero is the protected incident lane. Lower-priority
                # work remains on the rest of the pool and cannot queue ahead
                # of a newly received security event.
                workers = workers[1:]
            count = len(workers)
            start = self._object_route_cursor % count
            ordered = [
                (start + offset) % count
                for offset in range(count)
            ]
            ordered.sort(
                key=lambda index: workers[index].pending_requests()
            )
            self._object_route_cursor = (ordered[0] + 1) % count
            return [workers[index] for index in ordered]

    @staticmethod
    def _security_workload(workload: InferenceWorkload) -> bool:
        return workload <= InferenceWorkload.INCIDENT_REFINEMENT

    def _max_concurrent_refinements(self) -> int:
        with self._config_lock:
            return int(self.config.max_concurrent_refinements)

    def _enter_device_workload(
        self,
        workload: InferenceWorkload,
        *,
        shed_optional: bool = True,
        timeout: float = INFERENCE_REQUEST_TIMEOUT_SECONDS,
    ) -> bool:
        """Cooperatively keep optional GPU work behind security inference."""
        started = time.monotonic()
        deadline = started + max(0.0, timeout)
        security = self._security_workload(workload)
        with self._device_condition:
            if not self._device_accepting:
                self._device_workload_stats[workload]["shed"] += 1
                return False
            if security:
                self._security_waiting += 1
                initial = workload is InferenceWorkload.INCIDENT_INITIAL
                if initial:
                    self._initial_waiting += 1
                try:
                    # Initials may enter while refinements are already active so
                    # they can reach the worker priority heap. Refinements still
                    # yield to waiting/active initials and are capped so a burst
                    # cannot monopolize device admission.
                    while True:
                        if not self._device_accepting:
                            self._device_workload_stats[workload]["shed"] += 1
                            return False
                        blocked_by_optional = self._optional_active > 0
                        blocked_by_initial = (not initial) and (
                            self._initial_waiting > 0 or self._initial_active > 0
                        )
                        blocked_by_refinement_cap = (not initial) and (
                            self._refinement_active
                            >= self._max_concurrent_refinements()
                        )
                        if not (
                            blocked_by_optional
                            or blocked_by_initial
                            or blocked_by_refinement_cap
                        ):
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self._device_workload_stats[workload]["timed_out"] += 1
                            return False
                        self._device_condition.wait(remaining)
                    self._security_active += 1
                    if initial:
                        self._initial_active += 1
                    else:
                        self._refinement_active += 1
                finally:
                    self._security_waiting = max(0, self._security_waiting - 1)
                    if initial:
                        self._initial_waiting = max(0, self._initial_waiting - 1)
            else:
                offline = workload is InferenceWorkload.OFFLINE
                if shed_optional and (
                    self._security_waiting
                    or self._security_active
                    or self._offline_active
                ):
                    self._device_workload_stats[workload]["shed"] += 1
                    return False
                while (
                    self._security_waiting
                    or self._security_active
                    or (offline and self._optional_active)
                    or (not offline and self._offline_active)
                ):
                    if not self._device_accepting:
                        self._device_workload_stats[workload]["shed"] += 1
                        return False
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._device_workload_stats[workload]["timed_out"] += 1
                        return False
                    self._device_condition.wait(remaining)
                self._optional_active += 1
                if offline:
                    self._offline_active += 1
            wait_ms = max(0.0, (time.monotonic() - started) * 1000.0)
            stats = self._device_workload_stats[workload]
            stats["admitted"] += 1
            stats["wait_total_ms"] += wait_ms
            stats["wait_max_ms"] = max(float(stats["wait_max_ms"]), wait_ms)
            return True

    def _leave_device_workload(self, workload: InferenceWorkload) -> None:
        with self._device_condition:
            if self._security_workload(workload):
                self._security_active = max(0, self._security_active - 1)
                if workload is InferenceWorkload.INCIDENT_INITIAL:
                    self._initial_active = max(0, self._initial_active - 1)
                else:
                    self._refinement_active = max(0, self._refinement_active - 1)
            else:
                self._optional_active = max(0, self._optional_active - 1)
                if workload is InferenceWorkload.OFFLINE:
                    self._offline_active = max(0, self._offline_active - 1)
            self._device_workload_stats[workload]["completed"] += 1
            self._device_condition.notify_all()

    @contextmanager
    def offline_device_lease(self):
        """Serialize unmanaged offline inference behind security workloads."""
        workload = InferenceWorkload.OFFLINE
        if not self._enter_device_workload(workload, shed_optional=False, timeout=60.0):
            raise InferenceUnavailable("offline inference timed out waiting for production")
        try:
            yield
        finally:
            self._leave_device_workload(workload)

    def workload_status(self) -> dict[str, Any]:
        with self._device_condition:
            result = {
                "security_waiting": self._security_waiting,
                "security_active": self._security_active,
                "initial_waiting": self._initial_waiting,
                "initial_active": self._initial_active,
                "refinement_active": self._refinement_active,
                "max_concurrent_refinements": self._max_concurrent_refinements(),
                "optional_active": self._optional_active,
                "offline_active": self._offline_active,
                "accepting": self._device_accepting,
                "classes": {},
            }
            for workload, raw in self._device_workload_stats.items():
                admitted = int(raw["admitted"])
                result["classes"][workload.name.lower()] = {
                    "admitted": admitted,
                    "completed": int(raw["completed"]),
                    "shed": int(raw["shed"]),
                    "timed_out": int(raw["timed_out"]),
                    "average_wait_ms": round(
                        float(raw["wait_total_ms"]) / max(1, admitted), 3
                    ),
                    "max_wait_ms": round(float(raw["wait_max_ms"]), 3),
                }
            return result

    def _select_object_worker(self) -> _InferenceWorker:
        return self._ordered_object_workers()[0]

    def _base_detector_status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured_backend": self.config.backend,
            "loaded_backend": "",
            "configured_device": self.config.device,
            "loaded_device": "",
            "input_shape": [],
            "output_format": "unknown",
            "labels": len(self.labels),
            "coreml_loaded": False,
            "coreml_input_name": "",
            "coreml_image_input": False,
            "openvino_loaded": False,
            "opencv_loaded": False,
            "cache_enabled": self.config.cache_enabled,
            "cache_dir": "",
            "mmap_enabled": True,
            "performance_hint": "",
            "num_streams": None,
            "model_load_ms": None,
            "warmup_enabled": self.config.warmup_enabled,
            "warmup_ms": None,
            "warmup_error": "",
            "runtime": {
                "last_inference_ms": None,
                "average_inference_ms": None,
                "detection_fps": 0.0,
                "queue_depth": 0,
                "pending_frames": 0,
                "active_inferences": 0,
                "last_inference_at": "",
                "last_detection_at": "",
                "last_detection_age_seconds": None,
                "last_detection_labels": [],
                "total_inferences": 0,
                "failed_inferences": 0,
                "object_hit_inferences": 0,
                "stages": {"last_ms": {}, "average_ms": {}},
            },
        }

    def update_runtime_config(self, config: DetectorConfig) -> None:
        """Hot-swap policy data without restarting active inference workers."""
        next_config = config.model_copy(deep=True)
        next_labels = load_detector_labels(next_config)
        next_enabled = bool(
            next_config.enabled
            and (
                next_config.resolved_model_path()
                or next_config.resolved_coreml_model_path()
            )
        )
        with self._config_lock:
            for worker in (*self._object_workers, self._face, self._reid):
                worker.update_config_reference(next_config)
            self.config = next_config
            self.labels = next_labels
            self.enabled = next_enabled

    def reconfigure_roles(
        self,
        config: DetectorConfig,
        roles: set[str],
    ) -> None:
        """Fence new inference while replacing process generations."""
        with self._device_condition:
            previously_accepting = self._device_accepting
            self._device_accepting = False
            self._device_condition.notify_all()
        resume_admission = previously_accepting
        try:
            self._reconfigure_roles_unfenced(config, roles)
        except InferenceRollbackIncomplete:
            # A partially restored pool must remain fenced until an explicit
            # successful restart/reconfiguration proves it is healthy.
            resume_admission = False
            raise
        finally:
            with self._device_condition:
                self._device_accepting = resume_admission
                self._device_condition.notify_all()

    def _reconfigure_roles_unfenced(
        self,
        config: DetectorConfig,
        roles: set[str],
    ) -> None:
        """Restart only selected inference processes, transactionally."""
        invalid = roles - {"object", "face", "reid"}
        if invalid:
            raise ValueError(f"unknown inference roles: {', '.join(sorted(invalid))}")
        if not roles:
            self.update_runtime_config(config)
            return
        next_config = config.model_copy(deep=True)
        # Validate all file-backed metadata before changing supervisor or worker
        # references. A rejected labels file must leave the active runtime intact.
        next_labels = load_detector_labels(next_config)
        next_enabled = bool(
            next_config.enabled
            and (
                next_config.resolved_model_path()
                or next_config.resolved_coreml_model_path()
            )
        )

        with self._config_lock:
            previous_config = self.config.model_copy(deep=True)
            previous_labels = list(self.labels)
            previous_enabled = self.enabled
            completed: list[str] = []

            def apply_supervisor_config(
                value: DetectorConfig,
                labels: list[str],
                enabled: bool,
            ) -> None:
                self.config = value
                self.labels = labels
                self.enabled = enabled

            def role_settings(role: str) -> tuple[_InferenceWorker, dict[str, Any], bool]:
                if role == "object":
                    return self._object, self._base_detector_status(), True
                if role == "face":
                    return (
                        self._face,
                        self._base_face_status(),
                        bool(self.config.face_recognition_enabled),
                    )
                return (
                    self._reid,
                    self._base_reid_status(),
                    bool(self.config.tracking.appearance_reid_enabled),
                )

            apply_supervisor_config(next_config, next_labels, next_enabled)
            try:
                for role in ("object", "face", "reid"):
                    if role not in roles:
                        continue
                    if role == "object":
                        self._reconfigure_object_workers(next_config)
                    else:
                        worker, status, start_enabled = role_settings(role)
                        worker.reconfigure(
                            next_config,
                            status,
                            start_enabled=start_enabled,
                        )
                    completed.append(role)
                if "object" not in roles:
                    for worker in self._object_workers:
                        worker.update_config_reference(next_config)
                if "face" not in roles:
                    self._face.update_config_reference(next_config)
                if "reid" not in roles:
                    self._reid.update_config_reference(next_config)
            except BaseException as reconfigure_error:
                apply_supervisor_config(
                    previous_config,
                    previous_labels,
                    previous_enabled,
                )
                rollback_failures: list[str] = []
                for role in reversed(completed):
                    try:
                        if role == "object":
                            self._reconfigure_object_workers(previous_config)
                        else:
                            worker, status, start_enabled = role_settings(role)
                            worker.reconfigure(
                                previous_config,
                                status,
                                start_enabled=start_enabled,
                            )
                    except Exception as exc:
                        rollback_failures.append(f"{role}: {exc}")
                        LOGGER.exception("failed to roll back %s inference role", role)
                if "object" not in completed:
                    for worker in self._object_workers:
                        worker.update_config_reference(previous_config)
                if "face" not in completed:
                    self._face.update_config_reference(previous_config)
                if "reid" not in completed:
                    self._reid.update_config_reference(previous_config)
                if rollback_failures:
                    raise InferenceRollbackIncomplete(
                        "inference rollback incomplete: "
                        + "; ".join(rollback_failures)
                    ) from reconfigure_error
                raise

    def _reconfigure_object_workers(self, config: DetectorConfig) -> None:
        """Transactionally apply object-engine settings, including pool size."""
        expected = self._effective_object_worker_count(config)
        current = list(self._object_workers)
        if len(current) == expected:
            previous_config = current[0].config
            completed: list[_InferenceWorker] = []
            try:
                for worker in current:
                    worker.reconfigure(
                        config,
                        self._base_detector_status(),
                        start_enabled=True,
                    )
                    completed.append(worker)
            except BaseException as error:
                rollback_errors: list[BaseException] = []
                for worker in reversed(completed):
                    try:
                        worker.reconfigure(
                            previous_config,
                            self._base_detector_status(),
                            start_enabled=True,
                        )
                    except BaseException as rollback_error:
                        rollback_errors.append(rollback_error)
                if rollback_errors:
                    raise InferenceRollbackIncomplete(
                        "object inference pool rollback failed: "
                        + "; ".join(str(item) for item in rollback_errors)
                    ) from error
                raise
            return

        previous_config = current[0].config
        stopped: list[_InferenceWorker] = []
        try:
            for worker in reversed(current):
                worker.stop()
                stopped.append(worker)
        except BaseException as stop_error:
            restore_errors: list[BaseException] = []
            for worker in reversed(stopped):
                try:
                    if not worker.start():
                        raise InferenceUnavailable(
                            worker.isolation_status().get("last_error")
                            or "object inference worker failed to restore after stop failure"
                        )
                except BaseException as restore_error:
                    restore_errors.append(restore_error)
            if restore_errors:
                raise InferenceRollbackIncomplete(
                    "object inference pool stop rollback failed: "
                    + "; ".join(str(item) for item in restore_errors)
                ) from stop_error
            raise
        replacements = [
            _InferenceWorker(config, "object", self._base_detector_status())
            for _index in range(expected)
        ]
        try:
            for worker in replacements:
                if not worker.start():
                    raise InferenceUnavailable(
                        worker.isolation_status().get("last_error")
                        or "object inference worker failed to start"
                    )
        except BaseException as error:
            for worker in reversed(replacements):
                try:
                    worker.stop()
                except Exception:
                    LOGGER.exception("replacement object inference cleanup failed")
            restore_errors: list[BaseException] = []
            for worker in current:
                worker.update_config_reference(previous_config)
                try:
                    if not worker.start():
                        raise InferenceUnavailable(
                            worker.isolation_status().get("last_error")
                            or "object inference worker failed to restore"
                        )
                except BaseException as restore_error:
                    restore_errors.append(restore_error)
            if restore_errors:
                raise InferenceRollbackIncomplete(
                    "object inference pool restore failed: "
                    + "; ".join(str(item) for item in restore_errors)
                ) from error
            raise
        self._object_workers = replacements
        self._object = replacements[0]
        self._object_route_cursor = 0

    def _base_face_status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.face_recognition_enabled),
            "ready": False,
            "error": "Face inference worker has not started.",
            "device": self.config.face_recognition_device,
            "model_path": self.config.face_embedding_model_path,
            "landmark_model_path": self.config.face_landmark_model_path,
            "detector": {
                "enabled": bool(self.config.face_detection_model_path),
                "ready": False,
                "error": "Face inference worker has not started.",
                "model_path": self.config.face_detection_model_path,
                "threshold": self.config.face_detection_threshold,
            },
            "alignment_enabled": False,
            "landmark_input_shape": [],
            "model_fingerprint": "",
            "input_shape": [],
            "input_color_order": "",
            "embedding_size": 0,
            "model_load_ms": None,
            "match_threshold": self.config.face_match_threshold,
        }

    def _base_reid_status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.tracking.appearance_reid_enabled),
            "ready": False,
            "error": "Person ReID inference worker has not started.",
            "device": self.config.tracking.reid_device,
            "model_path": self.config.tracking.reid_model_path,
            "model_fingerprint": "",
            "input_shape": [],
            "embedding_size": 0,
            "model_load_ms": None,
            "match_threshold": self.config.tracking.reid_match_threshold,
            "person": {
                "enabled": bool(self.config.tracking.reid_enabled),
                "ready": False,
                "model_path": self.config.tracking.reid_model_path,
            },
            "vehicle": {
                "enabled": bool(self.config.tracking.vehicle_reid_enabled),
                "ready": False,
                "device": self.config.tracking.vehicle_reid_device,
                "model_path": self.config.tracking.vehicle_reid_model_path,
                "labels": list(self.config.tracking.vehicle_reid_labels),
                "match_threshold": self.config.tracking.vehicle_reid_match_threshold,
            },
        }

    def start(self) -> bool:
        object_ready = True
        for worker in self._object_workers:
            object_ready = worker.start() and object_ready
        face_ready = self._face.start()
        reid_ready = self._reid.start()
        ready = object_ready and face_ready and reid_ready
        with self._device_condition:
            self._device_accepting = bool(ready)
            self._device_condition.notify_all()
        return ready

    def stop(self) -> None:
        with self._device_condition:
            self._device_accepting = False
            self._device_condition.notify_all()
        failures: list[tuple[str, BaseException]] = []
        workers = [
            ("reid", self._reid),
            ("face", self._face),
            *(
                (f"object-{index + 1}", worker)
                for index, worker in reversed(list(enumerate(self._object_workers)))
            ),
        ]
        for role, worker in workers:
            try:
                worker.stop()
            except BaseException as exc:
                failures.append((role, exc))
                LOGGER.exception("%s inference worker shutdown failed", role.capitalize())
        if failures:
            roles = ", ".join(role for role, _exc in failures)
            first_error = failures[0][1]
            if not isinstance(first_error, Exception):
                raise first_error
            raise RuntimeError(f"inference worker shutdown failed for: {roles}") from first_error

    def stop_resource_tracker(self) -> bool:
        alive = [
            role
            for role, status in self.worker_status().items()
            if status.get("worker_alive")
        ]
        if alive:
            LOGGER.error(
                "not stopping multiprocessing resource tracker while inference workers remain alive: %s",
                ", ".join(alive),
            )
            return False
        return stop_multiprocessing_resource_tracker()

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Interactive compatibility boundary; security callers must be explicit."""
        return self._detect_for_workload(
            frame,
            confidence_threshold,
            InferenceWorkload.INTERACTIVE,
        )

    def detect_initial(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._detect_for_workload(
            frame,
            confidence_threshold,
            InferenceWorkload.INCIDENT_INITIAL,
        )

    def detect_refinement(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._detect_for_workload(
            frame,
            confidence_threshold,
            InferenceWorkload.INCIDENT_REFINEMENT,
        )

    def detect_interactive(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        return self.detect(frame, confidence_threshold)

    def detect_offline(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._detect_for_workload(
            frame,
            confidence_threshold,
            InferenceWorkload.OFFLINE,
        )

    def detect_tracking(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._detect_for_workload(
            frame,
            confidence_threshold,
            InferenceWorkload.TRACKING,
        )

    def detect_enrichment(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._detect_for_workload(
            frame,
            confidence_threshold,
            InferenceWorkload.ENRICHMENT,
        )

    def _detect_for_workload(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None,
        workload: InferenceWorkload,
    ) -> list[dict[str, Any]]:
        shed_optional = workload in {
            InferenceWorkload.TRACKING,
            InferenceWorkload.ENRICHMENT,
        }
        if not self._enter_device_workload(
            workload,
            shed_optional=shed_optional,
        ):
            # Optional consumers process timestamped samples. Once security
            # work is pending, retaining that old sample is worse than
            # dropping it and allowing the next fresh sample to recover.
            if workload >= InferenceWorkload.TRACKING:
                return [{
                    "status": "inference_deferred",
                    "error": f"{workload.name.lower()} inference deferred for security work",
                }]
            return [{
                "status": "detector_unavailable",
                "error": f"{workload.name.lower()} inference admission timed out",
            }]
        unavailable: list[str] = []
        try:
            workers = self._ordered_object_workers(workload)
            fast_failover = bool(
                workload is InferenceWorkload.INCIDENT_INITIAL
                and len(workers) > 1
            )
            for worker in workers:
                try:
                    return list(
                        worker.request(
                            "detect",
                            frame=frame,
                            confidence_threshold=confidence_threshold,
                            workload=workload,
                            timeout=(
                                INCIDENT_INITIAL_WORKER_TIMEOUT_SECONDS
                                if fast_failover
                                else INFERENCE_REQUEST_TIMEOUT_SECONDS
                            ),
                            admission_timeout=(
                                INCIDENT_INITIAL_ADMISSION_TIMEOUT_SECONDS
                                if fast_failover
                                else None
                            ),
                        )
                        or []
                    )
                except InferenceUnavailable as exc:
                    unavailable.append(str(exc))
                    continue
                except Exception as exc:
                    LOGGER.error("Isolated object detection unavailable: %s", exc)
                    return [{"status": "detector_unavailable", "error": str(exc)}]
            error = "; ".join(dict.fromkeys(unavailable)) or "all object inference workers unavailable"
            LOGGER.error("Isolated object detection unavailable: %s", error)
            return [{"status": "detector_unavailable", "error": error}]
        finally:
            self._leave_device_workload(workload)

    def embed(self, face: np.ndarray) -> np.ndarray:
        workload = InferenceWorkload.ENRICHMENT
        if not self._enter_device_workload(workload):
            raise InferenceUnavailable("face embedding shed for incident inference")
        try:
            result = self._face.request("embed", frame=face, workload=workload)
            return np.asarray(result, dtype=np.float32)
        finally:
            self._leave_device_workload(workload)

    def detect_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if (
            not self.config.face_recognition_enabled
            or not self.config.face_detection_model_path
        ):
            return []
        workload = InferenceWorkload.ENRICHMENT
        if not self._enter_device_workload(workload):
            return []
        try:
            return list(
                self._face.request(
                    "detect_faces",
                    frame=frame,
                    confidence_threshold=self.config.face_detection_threshold,
                    workload=workload,
                )
                or []
            )
        except Exception as exc:
            LOGGER.warning("Dedicated face detection unavailable: %s", exc)
            return []
        finally:
            self._leave_device_workload(workload)

    def embed_person(self, person: np.ndarray) -> np.ndarray:
        workload = InferenceWorkload.ENRICHMENT
        if not self._enter_device_workload(workload):
            raise InferenceUnavailable("person ReID shed for incident inference")
        try:
            result = self._reid.request(
                "embed_person",
                frame=person,
                timeout=PERSON_REID_REQUEST_TIMEOUT_SECONDS,
                workload=workload,
            )
            return np.asarray(result, dtype=np.float32)
        finally:
            self._leave_device_workload(workload)

    def embed_reid(self, label: str, crop: np.ndarray) -> np.ndarray:
        workload = InferenceWorkload.ENRICHMENT
        if not self._enter_device_workload(workload):
            raise InferenceUnavailable("object ReID shed for incident inference")
        try:
            result = self._reid.request(
                "embed_reid",
                frame=crop,
                label=str(label or "").strip().lower(),
                timeout=PERSON_REID_REQUEST_TIMEOUT_SECONDS,
                workload=workload,
            )
            return np.asarray(result, dtype=np.float32)
        finally:
            self._leave_device_workload(workload)

    @staticmethod
    def _aggregate_object_status(statuses: list[dict[str, Any]]) -> dict[str, Any]:
        if not statuses:
            return {}
        status = dict(statuses[0])
        runtimes = [dict(item.get("runtime") or {}) for item in statuses]
        total_inferences = sum(int(item.get("total_inferences") or 0) for item in runtimes)

        def average(key: str) -> float | None:
            values = [float(item[key]) for item in runtimes if item.get(key) is not None]
            return round(sum(values) / len(values), 2) if values else None

        latest_runtime = max(
            runtimes,
            key=lambda item: str(item.get("last_inference_at") or ""),
            default={},
        )
        latest_detection = max(
            runtimes,
            key=lambda item: str(item.get("last_detection_at") or ""),
            default={},
        )
        stage_names = {
            str(name)
            for runtime in runtimes
            for name in dict((runtime.get("stages") or {}).get("average_ms") or {})
        }
        candidate_names = {
            str(name)
            for runtime in runtimes
            for name in dict((runtime.get("candidates") or {}).get("average") or {})
        }
        runtime = {
            "last_inference_ms": latest_runtime.get("last_inference_ms"),
            "average_inference_ms": average("average_inference_ms"),
            "detection_fps": round(sum(float(item.get("detection_fps") or 0.0) for item in runtimes), 2),
            "queue_depth": sum(int(item.get("queue_depth") or 0) for item in runtimes),
            "pending_frames": sum(int(item.get("pending_frames") or 0) for item in runtimes),
            "active_inferences": sum(int(item.get("active_inferences") or 0) for item in runtimes),
            "last_inference_at": latest_runtime.get("last_inference_at") or "",
            "last_detection_at": latest_detection.get("last_detection_at") or "",
            "last_detection_age_seconds": latest_detection.get("last_detection_age_seconds"),
            "last_detection_labels": list(latest_detection.get("last_detection_labels") or []),
            "total_inferences": total_inferences,
            "failed_inferences": sum(int(item.get("failed_inferences") or 0) for item in runtimes),
            "object_hit_inferences": sum(int(item.get("object_hit_inferences") or 0) for item in runtimes),
            "stages": {
                "last_ms": dict((latest_runtime.get("stages") or {}).get("last_ms") or {}),
                "average_ms": {
                    name: (
                        round(sum(values) / len(values), 2)
                        if (values := [
                            float(value)
                            for item in runtimes
                            if (value := dict((item.get("stages") or {}).get("average_ms") or {}).get(name)) is not None
                        ])
                        else None
                    )
                    for name in stage_names
                },
                "percentiles_ms": dict(
                    (latest_runtime.get("stages") or {}).get("percentiles_ms") or {}
                ),
            },
            "candidates": {
                "last": dict((latest_runtime.get("candidates") or {}).get("last") or {}),
                "average": {
                    name: (
                        round(sum(values) / len(values), 1)
                        if (values := [
                            float(value)
                            for item in runtimes
                            if (value := dict((item.get("candidates") or {}).get("average") or {}).get(name)) is not None
                        ])
                        else None
                    )
                    for name in candidate_names
                },
            },
            "workers": [
                {
                    "index": index + 1,
                    "last_inference_ms": item.get("last_inference_ms"),
                    "average_inference_ms": item.get("average_inference_ms"),
                    "detection_fps": item.get("detection_fps"),
                    "queue_depth": item.get("queue_depth"),
                    "active_inferences": item.get("active_inferences"),
                    "total_inferences": item.get("total_inferences"),
                    "failed_inferences": item.get("failed_inferences"),
                }
                for index, item in enumerate(runtimes)
            ],
        }
        status["runtime"] = runtime
        status["openvino_loaded"] = any(bool(item.get("openvino_loaded")) for item in statuses)
        status["opencv_loaded"] = any(bool(item.get("opencv_loaded")) for item in statuses)
        status["coreml_loaded"] = any(bool(item.get("coreml_loaded")) for item in statuses)
        status["object_worker_count"] = len(statuses)
        return status

    def status(self) -> dict[str, Any]:
        # Each isolated worker owns its IPC lock and timeout. Poll the bounded
        # pool concurrently so one unhealthy process costs one timeout window,
        # rather than multiplying that delay by the configured worker count.
        with self._config_lock:
            object_workers = tuple(self._object_workers)
            reid_worker = self._reid
        with ThreadPoolExecutor(
            max_workers=len(object_workers) + 1,
            thread_name_prefix="inference-status",
        ) as executor:
            object_futures = [executor.submit(worker.status) for worker in object_workers]
            reid_future = executor.submit(reid_worker.status)
            statuses = [future.result() for future in object_futures]
            reid_status = reid_future.result()
        status = self._aggregate_object_status(statuses)
        runtime = dict(status.get("runtime") or {})
        isolation = self.isolation_status()
        pending = isolation["pending_requests"]
        runtime["queue_depth"] = max(int(runtime.get("queue_depth") or 0), pending)
        runtime["pending_frames"] = runtime["queue_depth"]
        runtime["workloads"] = self.workload_status()
        status["runtime"] = runtime
        status["isolation"] = isolation
        status["configured_device"] = self.config.device
        status["reid"] = self._current_reid_status(reid_status)
        status["workers"] = self.worker_status()
        return status

    def cached_object_status(self) -> dict[str, Any]:
        """Read the last object-worker status without IPC from scheduling hot paths."""
        statuses = [worker.cached_status() for worker in self._object_workers]
        status = self._aggregate_object_status(statuses)
        runtime = dict(status.get("runtime") or {})
        pending = self.isolation_status()["pending_requests"]
        runtime["queue_depth"] = max(int(runtime.get("queue_depth") or 0), pending)
        runtime["pending_frames"] = runtime["queue_depth"]
        runtime["workloads"] = self.workload_status()
        status["runtime"] = runtime
        return status

    def face_status(self) -> dict[str, Any]:
        status = self._face.status()
        status["enabled"] = bool(self.config.face_recognition_enabled)
        status["match_threshold"] = self.config.face_match_threshold
        detector = dict(status.get("detector") or {})
        detector["threshold"] = self.config.face_detection_threshold
        status["detector"] = detector
        return status

    def reid_status(self) -> dict[str, Any]:
        return self._current_reid_status(self._reid.status())

    def cached_reid_status(self) -> dict[str, Any]:
        return self._current_reid_status(self._reid.cached_status())

    def _current_reid_status(self, status: dict[str, Any]) -> dict[str, Any]:
        """Overlay hot matching policy on engine-owned worker status."""
        current = dict(status)
        current["enabled"] = bool(self.config.tracking.appearance_reid_enabled)
        current["match_threshold"] = self.config.tracking.reid_match_threshold
        person = current.get("person")
        if isinstance(person, dict):
            current["person"] = {
                **person,
                "enabled": bool(self.config.tracking.reid_enabled),
                "match_threshold": self.config.tracking.reid_match_threshold,
            }
        vehicle = current.get("vehicle")
        if isinstance(vehicle, dict):
            current["vehicle"] = {
                **vehicle,
                "enabled": bool(self.config.tracking.vehicle_reid_enabled),
                "labels": list(self.config.tracking.vehicle_reid_labels),
                "match_threshold": self.config.tracking.vehicle_reid_match_threshold,
            }
        return current

    def probe_devices(self) -> dict[str, Any]:
        try:
            with self.offline_device_lease():
                worker = self._ordered_object_workers(InferenceWorkload.OFFLINE)[0]
                return dict(
                    worker.request(
                        "probe_devices",
                        timeout=INFERENCE_STATUS_TIMEOUT_SECONDS,
                        workload=InferenceWorkload.OFFLINE,
                    )
                    or {}
                )
        except Exception as exc:
            return {"devices": [], "error": str(exc)}

    def inspect_model(self, path: str) -> dict[str, Any]:
        try:
            with self.offline_device_lease():
                worker = self._ordered_object_workers(InferenceWorkload.OFFLINE)[0]
                return dict(
                    worker.request(
                        "inspect_model",
                        path=path,
                        timeout=10.0,
                        workload=InferenceWorkload.OFFLINE,
                    )
                    or {}
                )
        except Exception as exc:
            return {"input_shape": [], "output_shapes": [], "error": str(exc)}

    def isolation_status(self) -> dict[str, Any]:
        instances = [worker.isolation_status() for worker in self._object_workers]
        alive = [item for item in instances if item.get("worker_alive")]
        errors = [str(item.get("last_error") or "") for item in instances]
        return {
            "enabled": any(bool(item.get("enabled")) for item in instances),
            "role": "object",
            "configured_device": self.config.device,
            "worker_pid": alive[0].get("worker_pid") if alive else None,
            "worker_pids": [item.get("worker_pid") for item in alive],
            "worker_alive": bool(alive),
            "all_workers_alive": len(alive) == len(instances),
            "configured_workers": len(instances),
            "alive_workers": len(alive),
            "generation": max((int(item.get("generation") or 0) for item in instances), default=0),
            "restart_count": sum(int(item.get("restart_count") or 0) for item in instances),
            "crash_count": sum(int(item.get("crash_count") or 0) for item in instances),
            "last_exit_code": next((item.get("last_exit_code") for item in reversed(instances) if item.get("last_exit_code") is not None), None),
            "last_exit_at": max((str(item.get("last_exit_at") or "") for item in instances), default=""),
            "last_error": "; ".join(dict.fromkeys(error for error in errors if error)),
            "pending_requests": sum(int(item.get("pending_requests") or 0) for item in instances),
            "fallback_active": any(bool(item.get("fallback_active")) for item in instances),
            "fallback_seconds_remaining": max((float(item.get("fallback_seconds_remaining") or 0.0) for item in instances), default=0.0),
            "request_timeout_seconds": INFERENCE_REQUEST_TIMEOUT_SECONDS,
            "max_frame_bytes": MAX_INFERENCE_FRAME_BYTES,
            "instances": [
                {"index": index + 1, **item}
                for index, item in enumerate(instances)
            ],
        }

    def worker_status(self) -> dict[str, dict[str, Any]]:
        return {
            "object": self.isolation_status(),
            "face": self._face.isolation_status(),
            "reid": self._reid.isolation_status(),
        }
