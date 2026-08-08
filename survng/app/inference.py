from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import multiprocessing
import os
from pathlib import Path
import resource
import threading
import time
from typing import Any

import numpy as np

from .config import DetectorConfig


LOGGER = logging.getLogger("uvicorn.error")
MAX_INFERENCE_FRAME_BYTES = 64 * 1024 * 1024
INFERENCE_START_TIMEOUT_SECONDS = 30.0
INFERENCE_REQUEST_TIMEOUT_SECONDS = 15.0
PERSON_REID_REQUEST_TIMEOUT_SECONDS = 3.0
INFERENCE_STATUS_TIMEOUT_SECONDS = 5.0
INFERENCE_RESTART_DELAY_SECONDS = 1.0
INFERENCE_CRASH_WINDOW_SECONDS = 10 * 60.0
INFERENCE_GPU_FALLBACK_CRASHES = 3
INFERENCE_GPU_FALLBACK_SECONDS = 30 * 60.0
RESOURCE_TRACKER_STOP_TIMEOUT_SECONDS = 2.0


class InferenceUnavailable(RuntimeError):
    pass


def stop_multiprocessing_resource_tracker(
    timeout: float = RESOURCE_TRACKER_STOP_TIMEOUT_SECONDS,
) -> bool:
    """Reap Python's tracker after all SurvNG multiprocessing users stop."""
    try:
        from multiprocessing import resource_tracker

        tracker = resource_tracker._resource_tracker
        pid = getattr(tracker, "_pid", None)
        if pid is None:
            return True
        failure: list[BaseException] = []

        def stop_tracker() -> None:
            try:
                tracker._stop()
            except BaseException as exc:
                failure.append(exc)

        thread = threading.Thread(
            target=stop_tracker,
            name="stop-multiprocessing-resource-tracker",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=max(0.0, float(timeout)))
        if thread.is_alive():
            LOGGER.warning(
                "multiprocessing resource tracker pid=%s did not stop within %.1fs; systemd will clean it up",
                pid,
                timeout,
            )
            return False
        if failure:
            LOGGER.warning(
                "multiprocessing resource tracker pid=%s cleanup failed: %s",
                pid,
                failure[0],
            )
            return False
        LOGGER.info("multiprocessing resource tracker stopped pid=%s", pid)
        return True
    except (AttributeError, ImportError, RuntimeError) as exc:
        LOGGER.warning("multiprocessing resource tracker cleanup unavailable: %s", exc)
        return False


def load_detector_labels(config: DetectorConfig) -> list[str]:
    labels = list(config.labels)
    if config.labels_path:
        labels_path = Path(config.labels_path)
        if labels_path.exists():
            labels = [
                line.strip()
                for line in labels_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    if labels:
        return labels

    model_path_text = config.resolved_model_path()
    model_path = Path(model_path_text) if model_path_text else None
    metadata_path = model_path.parent / "metadata.yaml" if model_path else None
    if metadata_path is None or not metadata_path.exists():
        return []
    try:
        import yaml

        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        names = metadata.get("names") or {}
        if isinstance(names, dict):
            return [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
        if isinstance(names, list):
            return [str(value) for value in names]
    except Exception:
        LOGGER.exception("Failed to load detector labels from %s", metadata_path)
    return []


def _disable_worker_core_dumps() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass


def _set_worker_process_name(role: str) -> str:
    name = f"survng-{role}"[:15]
    multiprocessing.current_process().name = name
    try:
        Path("/proc/self/comm").write_text(name, encoding="utf-8")
    except OSError:
        pass
    return name


def _openvino_devices() -> dict[str, Any]:
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core

        return {"devices": list(Core().available_devices), "error": ""}
    except Exception as exc:
        return {"devices": [], "error": str(exc) or "OpenVINO device probe failed"}


def _inspect_openvino_model(path_text: str) -> dict[str, Any]:
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core

        model = Core().read_model(model=path_text)
        return {
            "input_shape": [int(value) for value in model.input(0).shape],
            "output_shapes": [[int(value) for value in output.shape] for output in model.outputs],
            "error": "",
        }
    except Exception as exc:
        return {"input_shape": [], "output_shapes": [], "error": str(exc)}


def _inference_worker_main(
    connection,
    frame_buffer,
    config_payload: dict[str, Any],
    role: str,
) -> None:
    _set_worker_process_name(role)
    _disable_worker_core_dumps()
    config = DetectorConfig.model_validate(config_payload)
    try:
        if role == "object":
            from .detector import OpenVinoDetector

            engine = OpenVinoDetector(config)
        elif role == "face":
            from .face_detection import OpenVinoFaceDetector
            from .face_recognition import OpenVinoFaceRecognizer

            engine = OpenVinoFaceRecognizer(config)
            face_detector = OpenVinoFaceDetector(config)
        elif role == "reid":
            from .person_reidentification import OpenVinoAppearanceReidentifier

            engine = OpenVinoAppearanceReidentifier(config)
        else:
            raise ValueError(f"unknown inference worker role: {role}")
        engine_status = engine.status()
        if role == "face":
            engine_status["detector"] = face_detector.status()
        connection.send({
            "type": "ready",
            "pid": os.getpid(),
            "role": role,
            "status": engine_status,
        })
    except BaseException as exc:
        try:
            connection.send({"type": "fatal", "error": str(exc) or type(exc).__name__})
        finally:
            connection.close()
        return

    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            request_id = request.get("id")
            operation = request.get("op")
            if operation == "shutdown":
                connection.send({"id": request_id, "ok": True, "result": {"stopped": True}})
                return
            try:
                if operation in {"detect", "detect_faces", "embed", "embed_person", "embed_reid"}:
                    shape = tuple(int(value) for value in request.get("shape") or ())
                    dtype_name = str(request.get("dtype") or "")
                    byte_count = int(request.get("byte_count") or 0)
                    if (
                        dtype_name != "uint8"
                        or len(shape) != 3
                        or shape[2] != 3
                        or any(value <= 0 for value in shape)
                        or byte_count != math.prod(shape)
                        or byte_count > len(frame_buffer)
                    ):
                        raise ValueError("invalid shared inference frame")
                    view = np.frombuffer(frame_buffer, dtype=np.uint8, count=byte_count)
                    frame = view.reshape(shape)
                    if operation == "detect" and role == "object":
                        result = engine.detect(
                            frame,
                            confidence_threshold=request.get("confidence_threshold"),
                        )
                    elif operation == "embed" and role == "face":
                        result = engine.embed(frame).astype(np.float32).tolist()
                    elif operation == "detect_faces" and role == "face":
                        result = face_detector.detect(
                            frame,
                            threshold=request.get("confidence_threshold"),
                        )
                    elif operation == "embed_person" and role == "reid":
                        result = engine.embed(frame).astype(np.float32).tolist()
                    elif operation == "embed_reid" and role == "reid":
                        result = engine.embed_for_label(
                            str(request.get("label") or ""),
                            frame,
                        ).astype(np.float32).tolist()
                    else:
                        raise ValueError(f"{operation} is unavailable in the {role} worker")
                elif operation == "status":
                    result = engine.status()
                    if role == "face":
                        result["detector"] = face_detector.status()
                elif operation == "probe_devices" and role == "object":
                    result = _openvino_devices()
                elif operation == "inspect_model" and role == "object":
                    result = _inspect_openvino_model(str(request.get("path") or ""))
                else:
                    raise ValueError(f"unknown inference operation: {operation}")
                connection.send({"id": request_id, "ok": True, "result": result})
            except BaseException as exc:
                connection.send({
                    "id": request_id,
                    "ok": False,
                    "error": str(exc) or type(exc).__name__,
                })
    finally:
        connection.close()


class _InferenceWorker:
    def __init__(
        self,
        config: DetectorConfig,
        role: str,
        initial_status: dict[str, Any],
        *,
        start_enabled: bool = True,
    ) -> None:
        self.config = config
        self.role = role
        self.start_enabled = start_enabled
        self._status = dict(initial_status)
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._pending_requests = 0
        self._frame_buffer = None
        self._connection = None
        self._process = None
        self._stopping = False
        self._request_id = 0
        self._generation = 0
        self._restart_count = 0
        self._crash_times: deque[float] = deque()
        self._fallback_until = 0.0
        self._next_restart_at = 0.0
        self._last_exit_code: int | None = None
        self._last_exit_at = ""
        self._last_error = ""

    @property
    def configured_device(self) -> str:
        if self.role == "face":
            return self.config.face_recognition_device
        if self.role == "reid":
            devices = {
                self.config.tracking.reid_device
                if self.config.tracking.reid_enabled
                else "",
                self.config.tracking.vehicle_reid_device
                if self.config.tracking.vehicle_reid_enabled
                else "",
            } - {""}
            return next(iter(devices)) if len(devices) == 1 else "AUTO"
        return self.config.device

    def update_config_reference(self, config: DetectorConfig) -> None:
        """Update configuration used by status and any future worker respawn."""
        with self._lock:
            self.config = config

    def reconfigure(
        self,
        config: DetectorConfig,
        initial_status: dict[str, Any],
        *,
        start_enabled: bool,
    ) -> None:
        """Restart this inference role and restore its prior runtime on failure."""
        with self._lock:
            previous_config = self.config
            previous_status = dict(self._status)
            previous_start_enabled = self.start_enabled
            previous_alive = bool(
                self._process is not None and self._process.is_alive()
            )
            try:
                self.stop()
            except BaseException:
                # A failed stop can leave the prior process alive. Keep its
                # automatic recovery policy active because no config changed.
                self._stopping = False
                raise
            self.config = config
            self.start_enabled = start_enabled
            self._status = dict(initial_status)
            self._stopping = False
            try:
                if start_enabled and not self.start():
                    raise InferenceUnavailable(
                        self._last_error
                        or f"{self.role} inference worker failed to restart"
                    )
            except BaseException as reconfigure_error:
                rollback_errors: list[BaseException] = []
                try:
                    self.stop()
                except BaseException as exc:
                    rollback_errors.append(exc)
                finally:
                    self.config = previous_config
                    self.start_enabled = previous_start_enabled
                    self._status = previous_status
                    self._stopping = False
                    if previous_alive:
                        try:
                            if not self.start():
                                rollback_errors.append(InferenceUnavailable(
                                    self._last_error
                                    or f"{self.role} inference worker failed to restore"
                                ))
                        except BaseException as exc:
                            rollback_errors.append(exc)
                if rollback_errors:
                    details = "; ".join(str(exc) for exc in rollback_errors)
                    raise RuntimeError(
                        f"{self.role} inference rollback failed: {details}"
                    ) from reconfigure_error
                raise

    def start(self) -> bool:
        if not self.start_enabled:
            return True
        with self._lock:
            self._stopping = False
            return self._ensure_worker_locked(force=True)

    def stop(self) -> None:
        stubborn_process = None
        with self._lock:
            self._stopping = True
            process = self._process
            connection = self._connection
            if process is not None and process.is_alive() and connection is not None:
                try:
                    self._request_id += 1
                    connection.send({"id": self._request_id, "op": "shutdown"})
                    if connection.poll(5.0):
                        connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            if process is not None:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2.0)
                if process.is_alive():
                    stubborn_process = process
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    LOGGER.exception(
                        "%s inference connection cleanup failed",
                        self.role.capitalize(),
                    )
            self._connection = None
            if stubborn_process is None:
                self._process = None
                self._frame_buffer = None
            else:
                self._process = stubborn_process
        if stubborn_process is not None:
            raise RuntimeError(
                f"{self.role} inference worker did not stop after forced termination"
            )

    def _active_config_payload(self) -> dict[str, Any]:
        payload = self.config.model_dump(mode="json")
        if self.role == "object":
            payload["face_recognition_enabled"] = False
            if time.monotonic() < self._fallback_until:
                payload["device"] = "CPU"
        elif self.role == "face":
            payload["enabled"] = False
            if time.monotonic() < self._fallback_until:
                payload["face_recognition_device"] = "CPU"
        else:
            payload["enabled"] = False
            payload["face_recognition_enabled"] = False
            if time.monotonic() < self._fallback_until:
                payload["tracking"]["reid_device"] = "CPU"
                payload["tracking"]["vehicle_reid_device"] = "CPU"
        return payload

    def _ensure_worker_locked(
        self,
        force: bool = False,
        startup_timeout: float = INFERENCE_START_TIMEOUT_SECONDS,
    ) -> bool:
        if not self.start_enabled:
            return False
        process = self._process
        if process is not None and process.is_alive():
            return True
        if process is not None:
            self._record_dead_worker_locked(process.exitcode)
        if self._stopping:
            return False
        if not force and time.monotonic() < self._next_restart_at:
            return False
        return self._spawn_worker_locked(startup_timeout)

    def _spawn_worker_locked(self, startup_timeout: float) -> bool:
        parent = None
        child = None
        process = None
        try:
            if self._frame_buffer is None:
                self._frame_buffer = self._context.RawArray("B", MAX_INFERENCE_FRAME_BYTES)
            parent, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_inference_worker_main,
                args=(child, self._frame_buffer, self._active_config_payload(), self.role),
                name=f"survng-{self.role}-inference",
                daemon=False,
            )
            process.start()
            child.close()
            child = None
        except BaseException as exc:
            if child is not None:
                child.close()
            if parent is not None:
                parent.close()
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=3.0)
            self._last_error = f"{self.role} inference worker failed to start: {exc}"
            self._next_restart_at = time.monotonic() + INFERENCE_RESTART_DELAY_SECONDS
            LOGGER.exception("%s", self._last_error)
            return False
        self._process = process
        self._connection = parent
        self._generation += 1
        if self._generation > 1:
            self._restart_count += 1
        if startup_timeout <= 0 or not parent.poll(startup_timeout):
            self._terminate_failed_worker_locked(f"{self.role} inference worker startup timed out")
            return False
        try:
            message = parent.recv()
        except (EOFError, OSError) as exc:
            self._terminate_failed_worker_locked(
                f"{self.role} inference worker startup failed: {exc}"
            )
            return False
        if message.get("type") != "ready":
            self._terminate_failed_worker_locked(
                str(message.get("error") or f"{self.role} inference worker failed during startup")
            )
            return False
        self._status = dict(message.get("status") or self._status)
        self._last_error = ""
        self._next_restart_at = 0.0
        loaded_device = self._status.get("loaded_device") or self._status.get("device")
        LOGGER.info(
            "%s inference worker ready pid=%s generation=%s device=%s",
            self.role.capitalize(),
            process.pid,
            self._generation,
            loaded_device or "unavailable",
        )
        return True

    def _record_dead_worker_locked(self, exit_code: int | None) -> None:
        process = self._process
        connection = self._connection
        if process is not None:
            process.join(timeout=0.1)
        if connection is not None:
            try:
                connection.close()
            except OSError:
                LOGGER.exception("%s inference connection cleanup failed", self.role.capitalize())
        self._process = None
        self._connection = None
        self._last_exit_code = exit_code
        self._last_exit_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self._stopping and exit_code not in {0, None}:
            now = time.monotonic()
            self._crash_times.append(now)
            while self._crash_times and now - self._crash_times[0] > INFERENCE_CRASH_WINDOW_SECONDS:
                self._crash_times.popleft()
            if (
                self.configured_device.upper() != "CPU"
                and len(self._crash_times) >= INFERENCE_GPU_FALLBACK_CRASHES
            ):
                self._fallback_until = now + INFERENCE_GPU_FALLBACK_SECONDS
                LOGGER.error("%s inference worker entered CPU fallback", self.role.capitalize())
            self._next_restart_at = now + INFERENCE_RESTART_DELAY_SECONDS
            self._last_error = f"{self.role} inference worker exited with code {exit_code}"
            LOGGER.error("%s", self._last_error)

    def _terminate_failed_worker_locked(self, error: str) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            if process.is_alive():
                connection = self._connection
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        LOGGER.exception(
                            "%s inference connection cleanup failed",
                            self.role.capitalize(),
                        )
                self._connection = None
                self._last_error = error
                self._next_restart_at = time.monotonic() + INFERENCE_RESTART_DELAY_SECONDS
                LOGGER.error(
                    "%s; worker remained alive after forced termination",
                    error,
                )
                return
        exit_code = process.exitcode if process is not None else None
        self._record_dead_worker_locked(exit_code if exit_code not in {0, None} else -1)
        self._last_error = error
        LOGGER.error("%s", error)

    def _write_frame_locked(self, frame: np.ndarray) -> dict[str, Any]:
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or any(value <= 0 for value in frame.shape)
        ):
            raise InferenceUnavailable("inference frame must be a non-empty uint8 BGR image")
        contiguous = np.ascontiguousarray(frame)
        byte_count = int(contiguous.nbytes)
        if byte_count <= 0 or byte_count > MAX_INFERENCE_FRAME_BYTES:
            raise InferenceUnavailable(
                f"inference frame is {byte_count} bytes; maximum is {MAX_INFERENCE_FRAME_BYTES}"
            )
        target = np.frombuffer(self._frame_buffer, dtype=np.uint8, count=byte_count)
        target[:] = contiguous.view(np.uint8).reshape(-1)
        return {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "byte_count": byte_count,
        }

    def request(
        self,
        operation: str,
        *,
        frame: np.ndarray | None = None,
        timeout: float = INFERENCE_REQUEST_TIMEOUT_SECONDS,
        **payload: Any,
    ) -> Any:
        if timeout <= 0:
            raise ValueError("inference timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._pending_lock:
            self._pending_requests += 1
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._lock.acquire(timeout=remaining):
                raise InferenceUnavailable(
                    f"{self.role} {operation} timed out waiting for the inference worker"
                )
            try:
                remaining = deadline - time.monotonic()
                if not self._ensure_worker_locked(startup_timeout=max(0.0, remaining)):
                    raise InferenceUnavailable(
                        self._last_error or f"{self.role} inference worker is unavailable"
                    )
                connection = self._connection
                if connection is None:
                    raise InferenceUnavailable(
                        f"{self.role} inference worker connection is unavailable"
                    )
                self._request_id += 1
                request = {"id": self._request_id, "op": operation, **payload}
                if frame is not None:
                    request.update(self._write_frame_locked(frame))
                try:
                    connection.send(request)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not connection.poll(remaining):
                        self._terminate_failed_worker_locked(
                            f"{self.role} {operation} timed out after {timeout:.1f}s"
                        )
                        raise InferenceUnavailable(self._last_error)
                    response = connection.recv()
                except (BrokenPipeError, EOFError, OSError) as exc:
                    error = f"{self.role} inference worker connection failed: {exc}"
                    self._terminate_failed_worker_locked(
                        error
                    )
                    raise InferenceUnavailable(
                        error
                    ) from exc
                if response.get("id") != self._request_id:
                    self._terminate_failed_worker_locked(
                        f"{self.role} inference worker response was out of sequence"
                    )
                    raise InferenceUnavailable(
                        self._last_error
                    )
                if not response.get("ok"):
                    raise RuntimeError(str(response.get("error") or f"{operation} failed"))
                self._last_error = ""
                return response.get("result")
            finally:
                self._lock.release()
        finally:
            with self._pending_lock:
                self._pending_requests = max(0, self._pending_requests - 1)

    def status(self) -> dict[str, Any]:
        if self.start_enabled:
            try:
                next_status = dict(
                    self.request("status", timeout=INFERENCE_STATUS_TIMEOUT_SECONDS) or {}
                )
                with self._lock:
                    self._status = next_status
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
        with self._lock:
            status = dict(self._status)
        status["isolation"] = self.isolation_status()
        return status

    def cached_status(self) -> dict[str, Any]:
        """Return worker readiness without adding an IPC request to a hot path."""
        with self._lock:
            status = dict(self._status)
            process = self._process
            alive = bool(process is not None and process.is_alive())
        if alive:
            return status
        status["ready"] = False
        for key in ("person", "vehicle"):
            child = status.get(key)
            if isinstance(child, dict):
                status[key] = {**child, "ready": False}
        return status

    def pending_requests(self) -> int:
        """Return pool-routing pressure without waiting for worker IPC."""
        with self._pending_lock:
            return self._pending_requests

    def isolation_status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            while self._crash_times and now - self._crash_times[0] > INFERENCE_CRASH_WINDOW_SECONDS:
                self._crash_times.popleft()
            process = self._process
            worker_alive = bool(process is not None and process.is_alive())
            with self._pending_lock:
                pending = self._pending_requests
            return {
                "enabled": self.start_enabled,
                "role": self.role,
                "configured_device": self.configured_device,
                "worker_pid": process.pid if worker_alive else None,
                "worker_alive": worker_alive,
                "generation": self._generation,
                "restart_count": self._restart_count,
                "crash_count": len(self._crash_times),
                "last_exit_code": self._last_exit_code,
                "last_exit_at": self._last_exit_at,
                "last_error": self._last_error,
                "pending_requests": pending,
                "fallback_active": now < self._fallback_until,
                "fallback_seconds_remaining": round(max(0.0, self._fallback_until - now), 1),
                "request_timeout_seconds": (
                    PERSON_REID_REQUEST_TIMEOUT_SECONDS
                    if self.role == "reid"
                    else INFERENCE_REQUEST_TIMEOUT_SECONDS
                ),
                "max_frame_bytes": MAX_INFERENCE_FRAME_BYTES,
            }


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

    def _ordered_object_workers(self) -> list[_InferenceWorker]:
        """Order workers by pressure while rotating equal-load workers fairly."""
        with self._object_route_lock:
            count = len(self._object_workers)
            start = self._object_route_cursor % count
            ordered = [
                (start + offset) % count
                for offset in range(count)
            ]
            ordered.sort(
                key=lambda index: self._object_workers[index].pending_requests()
            )
            self._object_route_cursor = (ordered[0] + 1) % count
            return [self._object_workers[index] for index in ordered]

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
                    raise RuntimeError(
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
                    raise RuntimeError(
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
                raise RuntimeError(
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
                raise RuntimeError(
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
        return object_ready and face_ready and reid_ready

    def stop(self) -> None:
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
        unavailable: list[str] = []
        for worker in self._ordered_object_workers():
            try:
                return list(
                    worker.request(
                        "detect",
                        frame=frame,
                        confidence_threshold=confidence_threshold,
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

    def embed(self, face: np.ndarray) -> np.ndarray:
        result = self._face.request("embed", frame=face)
        return np.asarray(result, dtype=np.float32)

    def detect_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if (
            not self.config.face_recognition_enabled
            or not self.config.face_detection_model_path
        ):
            return []
        try:
            return list(
                self._face.request(
                    "detect_faces",
                    frame=frame,
                    confidence_threshold=self.config.face_detection_threshold,
                )
                or []
            )
        except Exception as exc:
            LOGGER.warning("Dedicated face detection unavailable: %s", exc)
            return []

    def embed_person(self, person: np.ndarray) -> np.ndarray:
        result = self._reid.request(
            "embed_person",
            frame=person,
            timeout=PERSON_REID_REQUEST_TIMEOUT_SECONDS,
        )
        return np.asarray(result, dtype=np.float32)

    def embed_reid(self, label: str, crop: np.ndarray) -> np.ndarray:
        result = self._reid.request(
            "embed_reid",
            frame=crop,
            label=str(label or "").strip().lower(),
            timeout=PERSON_REID_REQUEST_TIMEOUT_SECONDS,
        )
        return np.asarray(result, dtype=np.float32)

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
            return dict(
                self._select_object_worker().request(
                    "probe_devices",
                    timeout=INFERENCE_STATUS_TIMEOUT_SECONDS,
                )
                or {}
            )
        except Exception as exc:
            return {"devices": [], "error": str(exc)}

    def inspect_model(self, path: str) -> dict[str, Any]:
        try:
            return dict(
                self._select_object_worker().request(
                    "inspect_model", path=path, timeout=10.0
                ) or {}
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


class IsolatedFaceRecognizer:
    def __init__(self, supervisor: InferenceSupervisor) -> None:
        self.supervisor = supervisor

    @property
    def config(self) -> DetectorConfig:
        return self.supervisor.config

    @property
    def enabled(self) -> bool:
        return bool(self.config.face_recognition_enabled)

    @property
    def ready(self) -> bool:
        return bool(self.status().get("ready"))

    @property
    def model_fingerprint(self) -> str:
        return str(self.status().get("model_fingerprint") or "")

    def embed(self, face: np.ndarray) -> np.ndarray:
        return self.supervisor.embed(face)

    def status(self) -> dict[str, Any]:
        return self.supervisor.face_status()


class IsolatedPersonReidentifier:
    def __init__(self, supervisor: InferenceSupervisor) -> None:
        self.supervisor = supervisor
        self.config = supervisor.config.tracking

    @property
    def enabled(self) -> bool:
        return bool(self.config.appearance_reid_enabled)

    @property
    def ready(self) -> bool:
        return bool(self.status().get("ready"))

    def embed(self, person: np.ndarray) -> np.ndarray:
        return self.supervisor.embed_person(person)

    def supports_label(self, label: str) -> bool:
        normalized = str(label or "").strip().lower()
        if not self.config.reid_enabled_for_label(normalized):
            return False
        status = self.supervisor.cached_reid_status()
        if normalized == "person":
            engine = status.get("person", status)
        else:
            engine = status.get("vehicle", {})
        return bool(isinstance(engine, dict) and engine.get("ready"))

    def embed_for_label(self, label: str, crop: np.ndarray) -> np.ndarray:
        return self.supervisor.embed_reid(label, crop)

    def model_identity_for_label(self, label: str) -> dict[str, Any] | None:
        normalized = str(label or "").strip().lower()
        status = self.supervisor.cached_reid_status()
        if normalized == "person":
            engine = status.get("person", status)
            model_kind = "person"
        elif normalized in self.config.vehicle_reid_labels:
            engine = status.get("vehicle", {})
            model_kind = "vehicle"
        else:
            return None
        if not isinstance(engine, dict) or not engine.get("ready"):
            return None
        fingerprint = str(engine.get("model_fingerprint") or "")
        if not fingerprint:
            return None
        return {
            "model_kind": model_kind,
            "model_fingerprint": fingerprint,
            "embedding_size": int(engine.get("embedding_size") or 0),
            "match_threshold": self.config.reid_threshold_for_label(normalized),
        }

    def status(self) -> dict[str, Any]:
        return self.supervisor.reid_status()
