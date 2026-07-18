from __future__ import annotations

from collections import deque
import logging
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
INFERENCE_STATUS_TIMEOUT_SECONDS = 5.0
INFERENCE_RESTART_DELAY_SECONDS = 1.0
INFERENCE_CRASH_WINDOW_SECONDS = 10 * 60.0
INFERENCE_GPU_FALLBACK_CRASHES = 3
INFERENCE_GPU_FALLBACK_SECONDS = 30 * 60.0


class InferenceUnavailable(RuntimeError):
    pass


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


def _inference_worker_main(connection, frame_buffer, config_payload: dict[str, Any]) -> None:
    _disable_worker_core_dumps()
    config = DetectorConfig.model_validate(config_payload)
    detector = None
    face_recognizer = None
    try:
        from .detector import OpenVinoDetector
        from .face_recognition import OpenVinoFaceRecognizer

        detector = OpenVinoDetector(config)
        face_recognizer = OpenVinoFaceRecognizer(config)
        connection.send({
            "type": "ready",
            "pid": os.getpid(),
            "detector": detector.status(),
            "face": face_recognizer.status(),
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
                if operation in {"detect", "embed"}:
                    shape = tuple(int(value) for value in request.get("shape") or ())
                    dtype = np.dtype(str(request.get("dtype") or "uint8"))
                    byte_count = int(request.get("byte_count") or 0)
                    if not shape or byte_count <= 0 or byte_count > len(frame_buffer):
                        raise ValueError("invalid shared inference frame")
                    view = np.frombuffer(frame_buffer, dtype=np.uint8, count=byte_count)
                    frame = view.view(dtype).reshape(shape)
                    if operation == "detect":
                        result = detector.detect(
                            frame,
                            confidence_threshold=request.get("confidence_threshold"),
                        )
                    else:
                        result = face_recognizer.embed(frame).astype(np.float32).tolist()
                elif operation == "detector_status":
                    result = detector.status()
                elif operation == "face_status":
                    result = face_recognizer.status()
                elif operation == "probe_devices":
                    result = _openvino_devices()
                elif operation == "inspect_model":
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


class InferenceSupervisor:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.labels = load_detector_labels(config)
        self.enabled = bool(
            config.enabled
            and (config.resolved_model_path() or config.resolved_coreml_model_path())
        )
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
        self._detector_status = self._base_detector_status()
        self._face_status = self._base_face_status()

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

    def _base_face_status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.face_recognition_enabled),
            "ready": False,
            "error": "Inference worker has not started.",
            "device": self.config.face_recognition_device,
            "model_path": self.config.face_embedding_model_path,
            "landmark_model_path": self.config.face_landmark_model_path,
            "alignment_enabled": False,
            "landmark_input_shape": [],
            "model_fingerprint": "",
            "input_shape": [],
            "input_color_order": "",
            "embedding_size": 0,
            "model_load_ms": None,
            "match_threshold": self.config.face_match_threshold,
        }

    def start(self) -> bool:
        with self._lock:
            self._stopping = False
            return self._ensure_worker_locked(force=True)

    def stop(self) -> None:
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
            if connection is not None:
                connection.close()
            self._connection = None
            self._process = None
            self._frame_buffer = None

    def _active_config_payload(self) -> dict[str, Any]:
        payload = self.config.model_dump(mode="json")
        if time.monotonic() < self._fallback_until:
            payload["device"] = "CPU"
            payload["face_recognition_device"] = "CPU"
        return payload

    def _ensure_worker_locked(self, force: bool = False) -> bool:
        process = self._process
        if process is not None and process.is_alive():
            return True
        if process is not None:
            self._record_dead_worker_locked(process.exitcode)
        if self._stopping:
            return False
        if not force and time.monotonic() < self._next_restart_at:
            return False
        return self._spawn_worker_locked()

    def _spawn_worker_locked(self) -> bool:
        if self._frame_buffer is None:
            self._frame_buffer = self._context.RawArray("B", MAX_INFERENCE_FRAME_BYTES)
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_inference_worker_main,
            args=(child, self._frame_buffer, self._active_config_payload()),
            name="survng-inference",
            daemon=False,
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        self._generation += 1
        if self._generation > 1:
            self._restart_count += 1
        if not parent.poll(INFERENCE_START_TIMEOUT_SECONDS):
            self._terminate_failed_worker_locked("inference worker startup timed out")
            return False
        try:
            message = parent.recv()
        except (EOFError, OSError) as exc:
            self._terminate_failed_worker_locked(f"inference worker startup failed: {exc}")
            return False
        if message.get("type") != "ready":
            self._terminate_failed_worker_locked(
                str(message.get("error") or "inference worker failed during startup")
            )
            return False
        self._detector_status = dict(message.get("detector") or self._base_detector_status())
        self._detector_status["configured_device"] = self.config.device
        self._face_status = dict(message.get("face") or self._base_face_status())
        self._last_error = ""
        self._next_restart_at = 0.0
        LOGGER.info(
            "Inference worker ready pid=%s generation=%s detector=%s face=%s",
            process.pid,
            self._generation,
            self._detector_status.get("loaded_device") or "unavailable",
            self._face_status.get("device") if self._face_status.get("ready") else "unavailable",
        )
        return True

    def _record_dead_worker_locked(self, exit_code: int | None) -> None:
        process = self._process
        connection = self._connection
        if process is not None:
            process.join(timeout=0.1)
        if connection is not None:
            connection.close()
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
                self.config.device.upper() != "CPU"
                and len(self._crash_times) >= INFERENCE_GPU_FALLBACK_CRASHES
            ):
                self._fallback_until = now + INFERENCE_GPU_FALLBACK_SECONDS
                LOGGER.error("Inference worker entered CPU fallback after repeated crashes")
            self._next_restart_at = now + INFERENCE_RESTART_DELAY_SECONDS
            self._last_error = f"inference worker exited with code {exit_code}"
            LOGGER.error("%s", self._last_error)

    def _terminate_failed_worker_locked(self, error: str) -> None:
        self._last_error = error
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        exit_code = process.exitcode if process is not None else None
        self._record_dead_worker_locked(exit_code if exit_code not in {0, None} else -1)

    def _write_frame_locked(self, frame: np.ndarray) -> dict[str, Any]:
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

    def _request_locked(
        self,
        operation: str,
        *,
        frame: np.ndarray | None = None,
        timeout: float = INFERENCE_REQUEST_TIMEOUT_SECONDS,
        **payload: Any,
    ) -> Any:
        if not self._ensure_worker_locked():
            raise InferenceUnavailable(self._last_error or "inference worker is unavailable")
        connection = self._connection
        if connection is None:
            raise InferenceUnavailable("inference worker connection is unavailable")
        self._request_id += 1
        request = {"id": self._request_id, "op": operation, **payload}
        if frame is not None:
            request.update(self._write_frame_locked(frame))
        try:
            connection.send(request)
            if not connection.poll(timeout):
                self._terminate_failed_worker_locked(f"{operation} timed out after {timeout:.1f}s")
                raise InferenceUnavailable(self._last_error)
            response = connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            process = self._process
            self._record_dead_worker_locked(process.exitcode if process is not None else -1)
            raise InferenceUnavailable(f"inference worker connection failed: {exc}") from exc
        if response.get("id") != self._request_id:
            raise InferenceUnavailable("inference worker response was out of sequence")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or f"{operation} failed"))
        return response.get("result")

    def _begin_request(self) -> None:
        with self._pending_lock:
            self._pending_requests += 1

    def _finish_request(self) -> None:
        with self._pending_lock:
            self._pending_requests = max(0, self._pending_requests - 1)

    def detect(self, frame: np.ndarray, confidence_threshold: float | None = None) -> list[dict[str, Any]]:
        self._begin_request()
        try:
            with self._lock:
                return list(self._request_locked(
                    "detect",
                    frame=frame,
                    confidence_threshold=confidence_threshold,
                ) or [])
        except Exception as exc:
            LOGGER.error("Isolated object detection unavailable: %s", exc)
            return [{"status": "detector_unavailable", "error": str(exc)}]
        finally:
            self._finish_request()

    def embed(self, face: np.ndarray) -> np.ndarray:
        self._begin_request()
        try:
            with self._lock:
                result = self._request_locked("embed", frame=face)
            return np.asarray(result, dtype=np.float32)
        finally:
            self._finish_request()

    def status(self) -> dict[str, Any]:
        with self._lock:
            try:
                status = dict(self._request_locked(
                    "detector_status",
                    timeout=INFERENCE_STATUS_TIMEOUT_SECONDS,
                ) or {})
                status["configured_device"] = self.config.device
                self._detector_status = status
            except Exception as exc:
                self._last_error = str(exc)
            status = dict(self._detector_status)
            runtime = dict(status.get("runtime") or {})
            with self._pending_lock:
                pending = self._pending_requests
            runtime["queue_depth"] = max(int(runtime.get("queue_depth") or 0), pending)
            runtime["pending_frames"] = runtime["queue_depth"]
            status["runtime"] = runtime
            status["isolation"] = self.isolation_status()
            return status

    def face_status(self) -> dict[str, Any]:
        with self._lock:
            try:
                self._face_status = dict(self._request_locked(
                    "face_status",
                    timeout=INFERENCE_STATUS_TIMEOUT_SECONDS,
                ) or {})
            except Exception as exc:
                self._last_error = str(exc)
            return dict(self._face_status)

    def probe_devices(self) -> dict[str, Any]:
        with self._lock:
            try:
                return dict(self._request_locked(
                    "probe_devices",
                    timeout=INFERENCE_STATUS_TIMEOUT_SECONDS,
                ) or {})
            except Exception as exc:
                return {"devices": [], "error": str(exc)}

    def inspect_model(self, path: str) -> dict[str, Any]:
        with self._lock:
            try:
                return dict(self._request_locked("inspect_model", path=path, timeout=10.0) or {})
            except Exception as exc:
                return {"input_shape": [], "output_shapes": [], "error": str(exc)}

    def isolation_status(self) -> dict[str, Any]:
        process = self._process
        now = time.monotonic()
        return {
            "enabled": True,
            "worker_pid": process.pid if process is not None and process.is_alive() else None,
            "worker_alive": bool(process is not None and process.is_alive()),
            "generation": self._generation,
            "restart_count": self._restart_count,
            "crash_count": len(self._crash_times),
            "last_exit_code": self._last_exit_code,
            "last_exit_at": self._last_exit_at,
            "last_error": self._last_error,
            "fallback_active": now < self._fallback_until,
            "fallback_seconds_remaining": round(max(0.0, self._fallback_until - now), 1),
            "request_timeout_seconds": INFERENCE_REQUEST_TIMEOUT_SECONDS,
            "max_frame_bytes": MAX_INFERENCE_FRAME_BYTES,
        }


class IsolatedFaceRecognizer:
    def __init__(self, supervisor: InferenceSupervisor) -> None:
        self.supervisor = supervisor
        self.config = supervisor.config

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
