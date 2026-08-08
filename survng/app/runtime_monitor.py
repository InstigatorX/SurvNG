from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .events import EventStore
from .inference_lifecycle import InferenceLifecycle
from .process_memory import (
    AllocatorMemoryTrimmer,
    process_memory_status,
    process_memory_status_for_pid,
)
from .state_events import StateEventBroker


LOGGER = logging.getLogger("uvicorn.error")


def system_memory_usage() -> tuple[int, int, float]:
    """Return host memory totals without coupling callers to /proc parsing."""
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


def process_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


class ApplicationRuntimeMonitor:
    """Own periodic camera-state publication and runtime memory maintenance.

    The monitor is the sole owner of its thread and allocator-reclamation state.
    Camera status is supplied as a snapshot boundary so the monitor observes the
    application without taking ownership of camera or recorder lifecycles.
    """

    def __init__(
        self,
        *,
        inference: InferenceLifecycle,
        events: EventStore,
        state_events: StateEventBroker,
        camera_statuses: Callable[[], list[dict[str, Any]]],
        sample_interval_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
        memory_trimmer: AllocatorMemoryTrimmer | None = None,
    ) -> None:
        self._inference = inference
        self._events = events
        self._state_events = state_events
        self._camera_statuses = camera_statuses
        self._sample_interval_seconds = max(0.01, float(sample_interval_seconds))
        self._poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        self._memory_trimmer = memory_trimmer or AllocatorMemoryTrimmer()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.running:
                return
            self._stop.clear()
            thread = threading.Thread(
                target=self._run,
                name="survng-runtime-monitor",
                daemon=False,
            )
            self._thread = thread
            thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError("application runtime monitor did not stop")
        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None

    def publish_camera_status(self, camera_id: str) -> None:
        status = next(
            (item for item in self._camera_statuses() if item.get("id") == camera_id),
            None,
        )
        if status is not None:
            self._state_events.publish("camera_state", status)

    def memory_maintenance_status(self) -> dict[str, Any]:
        return self._memory_trimmer.status()

    def worker_memory_status(
        self,
        *,
        detector_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return current isolated-inference worker memory without shelling out."""
        detector = detector_status or self._detector_status()
        workers = dict(detector.get("workers") or {})
        semantic = self._inference.semantic_search.status()
        workers["semantic"] = {
            "worker_pid": semantic.get("worker_pid"),
            "worker_alive": semantic.get("state") == "ready",
        }
        result: dict[str, dict[str, Any]] = {}
        total_rss = 0
        total_pss = 0
        seen_pids: set[int] = set()
        for role, worker in workers.items():
            if not isinstance(worker, dict) or not worker.get("worker_alive"):
                continue
            pids = worker.get("worker_pids") or [worker.get("worker_pid")]
            for index, raw_pid in enumerate(pids):
                pid = int(raw_pid or 0)
                if pid <= 0 or pid in seen_pids:
                    continue
                seen_pids.add(pid)
                memory = process_memory_status_for_pid(pid)
                rss = int(memory.get("rss_bytes") or 0)
                pss = int(memory.get("pss_bytes") or 0)
                key = str(role) if len(pids) == 1 else f"{role}-{index + 1}"
                result[key] = {
                    "pid": pid,
                    "rss_bytes": rss,
                    "pss_bytes": pss,
                    "threads": int(memory.get("threads") or 0),
                    "file_descriptors": int(memory.get("file_descriptors") or 0),
                }
                total_rss += rss
                total_pss += pss
        return {
            "total_rss_bytes": total_rss,
            "total_pss_bytes": total_pss,
            "workers": result,
        }

    def _detector_status(self) -> dict[str, Any]:
        return {
            **self._inference.detector.status(),
            "lifecycle": self._inference.status(),
        }

    def _run(self) -> None:
        previous: dict[str, tuple[Any, ...]] = {}
        telemetry_sample_at = 0.0
        while not self._stop.is_set():
            try:
                statuses = self._camera_statuses()
                for status in statuses:
                    camera_id = str(status.get("id") or "")
                    fingerprint = self.camera_state_fingerprint(status)
                    if camera_id and previous.get(camera_id) != fingerprint:
                        previous[camera_id] = fingerprint
                        self._state_events.publish("camera_state", status)
                now = time.monotonic()
                detector_status = self._detector_status()
                detector_runtime = dict(detector_status.get("runtime") or {})
                allocator_idle = self.allocator_trim_safe(statuses, detector_runtime)
                self._memory_trimmer.observe_idle(allocator_idle, now=now)
                if now - telemetry_sample_at >= self._sample_interval_seconds:
                    self._inference.maintain()
                    process_memory = process_memory_status()
                    self._memory_trimmer.maybe_trim(process_memory, now=now)
                    _memory_total, _memory_used, memory_used_percent = system_memory_usage()
                    try:
                        load_1m = os.getloadavg()[0]
                    except OSError:
                        load_1m = 0.0
                    self._events.record_runtime_telemetry(
                        statuses,
                        process_memory=process_memory,
                        worker_memory=self.worker_memory_status(
                            detector_status=detector_status,
                        ),
                        memory_maintenance=self.memory_maintenance_status(),
                        system_runtime={
                            "cpu_load_percent": round(
                                min(100.0, (load_1m / max(1, os.cpu_count() or 1)) * 100.0),
                                2,
                            ),
                            "memory_used_percent": memory_used_percent,
                            "inference_ms": detector_runtime.get("average_inference_ms"),
                        },
                    )
                    telemetry_sample_at = now
            except Exception:
                LOGGER.exception("application runtime monitor failed")
            self._stop.wait(self._poll_interval_seconds)

    @staticmethod
    def allocator_trim_safe(
        statuses: list[dict[str, Any]],
        detector_runtime: dict[str, Any],
    ) -> bool:
        """Allow arena reclamation outside inference and tracking work."""
        tracking_busy = any(
            bool((status.get("object_tracking") or {}).get("active"))
            or bool((status.get("object_tracking") or {}).get("worker_running"))
            for status in statuses
        )
        inference_busy = (
            int(detector_runtime.get("queue_depth") or 0) > 0
            or int(detector_runtime.get("pending_frames") or 0) > 0
            or int(detector_runtime.get("active_inferences") or 0) > 0
        )
        return not tracking_busy and not inference_busy

    @staticmethod
    def camera_state_fingerprint(status: dict[str, Any]) -> tuple[Any, ...]:
        keys = (
            "running", "connected", "capture_running", "frame_fresh", "main_running",
            "main_frame_fresh", "last_error", "main_last_error", "onvif_connected",
            "onvif_last_event_at", "onvif_last_motion_event_at", "onvif_last_error",
            "onvif_last_poll_success_at", "onvif_last_poll_error_at",
            "onvif_notifications_received", "onvif_motion_events_received",
            "onvif_renewals", "onvif_renewal_errors", "last_motion_at",
            "detection_enabled", "recording", "sub_recording", "recording_enabled",
            "recording_configured", "stream_dimensions",
        )
        motion = status.get("motion_qualification") or {}
        return tuple(status.get(key) for key in keys) + (
            motion.get("passed"),
            motion.get("audit_rejected"),
            motion.get("suppressed"),
            motion.get("last_decision_at"),
        )
