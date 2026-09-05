from __future__ import annotations

from collections import deque
import heapq
import multiprocessing
import threading
import time
from typing import Any

import cv2
import numpy as np

from ..config import DetectorConfig
from ..perf_samples import RollingLatencySamples
from .process import _inference_worker_main
from .types import (
    INFERENCE_CRASH_WINDOW_SECONDS,
    INFERENCE_GPU_FALLBACK_CRASHES,
    INFERENCE_GPU_FALLBACK_SECONDS,
    INFERENCE_RESTART_DELAY_SECONDS,
    INFERENCE_REQUEST_TIMEOUT_SECONDS,
    INFERENCE_START_TIMEOUT_SECONDS,
    INFERENCE_STATUS_TIMEOUT_SECONDS,
    LOGGER,
    MAX_INFERENCE_FRAME_BYTES,
    PERSON_REID_REQUEST_TIMEOUT_SECONDS,
    InferenceRollbackIncomplete,
    InferenceUnavailable,
    InferenceWorkload,
)


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
        self._admission = threading.Condition(threading.Lock())
        self._admission_waiters: list[tuple[int, int, object, float]] = []
        self._admission_sequence = 0
        self._admission_active = False
        # Preserve lazy-start behavior. Disabled roles still fail their normal
        # start-enabled check in _ensure_worker_locked; only stop/reconfigure
        # explicitly closes admission for an existing generation.
        self._admission_open = True
        self._active_workload: InferenceWorkload | None = None
        self._workload_stats: dict[InferenceWorkload, dict[str, float | int]] = {
            workload: {
                "queued": 0,
                "admitted": 0,
                "completed": 0,
                "failed": 0,
                "timed_out": 0,
                "wait_total_ms": 0.0,
                "wait_max_ms": 0.0,
            }
            for workload in InferenceWorkload
        }
        self._admission_wait_samples = {
            workload: RollingLatencySamples()
            for workload in InferenceWorkload
        }
        self._ipc_frame_copy_bytes_total = 0
        self._ipc_frame_copy_samples = RollingLatencySamples()
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
        if self.role == "depth":
            return self.config.depth.device
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
                    raise InferenceRollbackIncomplete(
                        f"{self.role} inference rollback failed."
                    ) from reconfigure_error
                raise

    def start(self) -> bool:
        if not self.start_enabled:
            return True
        with self._lock:
            self._stopping = False
            started = self._ensure_worker_locked(force=True)
        with self._admission:
            self._admission_open = bool(started)
            self._admission.notify_all()
        return started

    def stop(self) -> None:
        # Close admission before waiting for the active IPC request. Queued
        # callers wake immediately instead of unwinding one-by-one against a
        # worker generation that is deliberately stopping.
        with self._admission:
            self._admission_open = False
            self._admission.notify_all()
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
        elif self.role == "depth":
            payload["enabled"] = False
            payload["face_recognition_enabled"] = False
            if time.monotonic() < self._fallback_until:
                payload["depth"]["device"] = "CPU"
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
            self._last_error = f"{self.role} inference worker failed to start."
            self._next_restart_at = time.monotonic() + INFERENCE_RESTART_DELAY_SECONDS
            LOGGER.error(
                "%s (%s)",
                self._last_error,
                type(exc).__name__,
            )
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
                f"{self.role} inference worker startup failed."
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
        copy_started = time.perf_counter_ns()
        target[:] = contiguous.view(np.uint8).reshape(-1)
        copy_ms = (time.perf_counter_ns() - copy_started) / 1_000_000.0
        self._ipc_frame_copy_bytes_total += byte_count
        self._ipc_frame_copy_samples.add(copy_ms)
        return {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "byte_count": byte_count,
        }

    def _prepare_object_frame_locked(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float] | None]:
        """Bound object-detection IPC while leaving backend preprocessing isolated."""
        if (
            self.role != "object"
            or not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or any(value <= 0 for value in frame.shape)
        ):
            return frame, None
        input_shape = self._status.get("input_shape")
        if (
            not isinstance(input_shape, (list, tuple))
            or len(input_shape) != 2
        ):
            return frame, None
        try:
            model_side = max(int(input_shape[0]), int(input_shape[1]))
        except (TypeError, ValueError):
            return frame, None
        source_height, source_width = frame.shape[:2]
        max_ipc_side = model_side * 2
        source_side = max(source_width, source_height)
        if model_side <= 0 or source_side <= max_ipc_side:
            return frame, None
        scale = max_ipc_side / float(source_side)
        target_width = max(1, int(round(source_width * scale)))
        target_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        return resized, (
            source_width / float(target_width),
            source_height / float(target_height),
        )

    @staticmethod
    def _restore_object_boxes(
        result: Any,
        scale: tuple[float, float] | None,
    ) -> Any:
        if scale is None or not isinstance(result, list):
            return result
        scale_x, scale_y = scale
        for item in result:
            if not isinstance(item, dict):
                continue
            box = item.get("box")
            if isinstance(box, dict):
                for key, factor in (
                    ("x1", scale_x),
                    ("x2", scale_x),
                    ("y1", scale_y),
                    ("y2", scale_y),
                ):
                    try:
                        box[key] = float(box[key]) * factor
                    except (KeyError, TypeError, ValueError):
                        continue
            elif isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    item["box"] = [
                        float(box[0]) * scale_x,
                        float(box[1]) * scale_y,
                        float(box[2]) * scale_x,
                        float(box[3]) * scale_y,
                        *[value for value in box[4:]],
                    ]
                except (TypeError, ValueError):
                    continue
        return result

    def request(
        self,
        operation: str,
        *,
        frame: np.ndarray | None = None,
        timeout: float = INFERENCE_REQUEST_TIMEOUT_SECONDS,
        admission_timeout: float | None = None,
        workload: InferenceWorkload = InferenceWorkload.INTERACTIVE,
        **payload: Any,
    ) -> Any:
        if timeout <= 0:
            raise ValueError("inference timeout must be positive")
        workload = InferenceWorkload(workload)
        deadline = time.monotonic() + timeout
        admission_deadline = (
            deadline
            if admission_timeout is None
            else min(deadline, time.monotonic() + max(0.0, admission_timeout))
        )
        queued_at = time.monotonic()
        token = object()
        admitted = False
        completed = False
        timed_out = False
        with self._pending_lock:
            self._pending_requests += 1
        try:
            with self._admission:
                self._admission_sequence += 1
                waiter = (int(workload), self._admission_sequence, token, queued_at)
                heapq.heappush(self._admission_waiters, waiter)
                self._workload_stats[workload]["queued"] += 1
                while True:
                    if not self._admission_open:
                        self._admission_waiters.remove(waiter)
                        heapq.heapify(self._admission_waiters)
                        self._admission.notify_all()
                        raise InferenceUnavailable(
                            f"{self.role} inference worker is stopping"
                        )
                    remaining = admission_deadline - time.monotonic()
                    if remaining <= 0:
                        self._admission_waiters.remove(waiter)
                        heapq.heapify(self._admission_waiters)
                        self._workload_stats[workload]["timed_out"] += 1
                        timed_out = True
                        self._admission.notify_all()
                        raise InferenceUnavailable(
                            f"{self.role} {operation} timed out waiting for priority admission"
                        )
                    if not self._admission_active and self._admission_waiters[0] is waiter:
                        heapq.heappop(self._admission_waiters)
                        self._admission_active = True
                        self._active_workload = workload
                        admitted = True
                        wait_ms = max(0.0, (time.monotonic() - queued_at) * 1000.0)
                        stats = self._workload_stats[workload]
                        stats["admitted"] += 1
                        stats["wait_total_ms"] += wait_ms
                        stats["wait_max_ms"] = max(float(stats["wait_max_ms"]), wait_ms)
                        self._admission_wait_samples[workload].add(wait_ms)
                        break
                    self._admission.wait(remaining)
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
                box_scale = None
                if frame is not None:
                    ipc_frame = frame
                    if operation == "detect":
                        ipc_frame, box_scale = self._prepare_object_frame_locked(frame)
                    request.update(self._write_frame_locked(ipc_frame))
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
                    error = f"{self.role} inference worker connection failed."
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
                completed = True
                return self._restore_object_boxes(response.get("result"), box_scale)
            finally:
                self._lock.release()
        finally:
            with self._admission:
                if admitted:
                    self._admission_active = False
                    self._active_workload = None
                else:
                    retained = [item for item in self._admission_waiters if item[2] is not token]
                    if len(retained) != len(self._admission_waiters):
                        self._admission_waiters[:] = retained
                        heapq.heapify(self._admission_waiters)
                stats = self._workload_stats[workload]
                if completed:
                    stats["completed"] += 1
                elif not timed_out:
                    stats["failed"] += 1
                self._admission.notify_all()
            with self._pending_lock:
                self._pending_requests = max(0, self._pending_requests - 1)

    def status(
        self,
        workload: InferenceWorkload = InferenceWorkload.OFFLINE,
    ) -> dict[str, Any]:
        if self.start_enabled:
            try:
                next_status = dict(
                    self.request(
                        "status",
                        timeout=INFERENCE_STATUS_TIMEOUT_SECONDS,
                        workload=workload,
                    ) or {}
                )
                with self._lock:
                    self._status = next_status
            except Exception as exc:
                with self._lock:
                    self._last_error = (
                        f"{self.role} inference status is unavailable."
                    )
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
        with self._admission:
            queued_by_workload = {
                workload.name.lower(): sum(
                    1
                    for priority, _sequence, _token, _queued_at
                    in self._admission_waiters
                    if priority == int(workload)
                )
                for workload in InferenceWorkload
            }
            workload_status = {}
            for workload, raw in self._workload_stats.items():
                admitted = int(raw["admitted"])
                wait_samples = self._admission_wait_samples[workload]
                workload_status[workload.name.lower()] = {
                    "queued": queued_by_workload[workload.name.lower()],
                    "active": int(self._active_workload is workload),
                    "submitted": int(raw["queued"]),
                    "admitted": admitted,
                    "completed": int(raw["completed"]),
                    "failed": int(raw["failed"]),
                    "timed_out": int(raw["timed_out"]),
                    "average_wait_ms": round(
                        float(raw["wait_total_ms"]) / max(1, admitted), 3
                    ),
                    "max_wait_ms": round(float(raw["wait_max_ms"]), 3),
                    "admission_wait_ms_p95": wait_samples.percentile(95),
                    "admission_wait_ms_p99": wait_samples.percentile(99),
                    "oldest_wait_ms": round(
                        max(
                            [
                                max(0.0, (now - queued_at) * 1000.0)
                                for priority, _sequence, _token, queued_at
                                in self._admission_waiters
                                if priority == int(workload)
                            ]
                            or [0.0]
                        ),
                        3,
                    ),
                }
            admission_open = self._admission_open
        with self._lock:
            copy_samples = self._ipc_frame_copy_samples.snapshot()
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
                "admission_open": admission_open,
                "workloads": workload_status,
                "ipc": {
                    "frame_copy_bytes_total": self._ipc_frame_copy_bytes_total,
                    "frame_copy_ms_p95": copy_samples["p95_ms"],
                    "frame_copy_ms_p99": copy_samples["p99_ms"],
                    "samples": copy_samples["samples"],
                },
                "fallback_active": now < self._fallback_until,
                "fallback_seconds_remaining": round(max(0.0, self._fallback_until - now), 1),
                "request_timeout_seconds": (
                    PERSON_REID_REQUEST_TIMEOUT_SECONDS
                    if self.role == "reid"
                    else INFERENCE_REQUEST_TIMEOUT_SECONDS
                ),
                "max_frame_bytes": MAX_INFERENCE_FRAME_BYTES,
            }
