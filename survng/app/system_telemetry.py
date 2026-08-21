from __future__ import annotations

import logging
import os
import secrets
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import AppConfig
from .manager import AppManager
from .performance_health import camera_performance_health
from .process_memory import process_memory_status
from .product_update import ProductUpdateService
from .telemetry_interruptions import (
    classify_telemetry_interruptions,
    summarize_interruptions,
)
from survng import __version__ as SURVNG_VERSION

LOGGER = logging.getLogger(__name__)


class DiagnosticSessionRequest(BaseModel):
    scope: str = Field(default="system", max_length=32)
    camera_id: str = Field(default="", max_length=128)
    duration_seconds: int


@dataclass(frozen=True, slots=True)
class SystemTelemetryDependencies:
    get_manager: Callable[[], AppManager]
    get_config: Callable[[], AppConfig]


@dataclass(slots=True)
class SystemTelemetryService:
    """Own system sampling state independently of camera-manager generations."""

    process_started_monotonic: float = field(default_factory=time.monotonic)
    process_instance_id: str = field(default_factory=lambda: secrets.token_hex(12))
    _gpu_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _gpu_sample: dict[str, object] = field(
        default_factory=lambda: {"at": 0.0, "pids": (), "engines": {}}, init=False
    )
    _history_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _history: deque[dict[str, object]] = field(
        default_factory=lambda: deque(maxlen=360), init=False
    )
    _last_history_sample_at: float = field(default=0.0, init=False)
    _persisted_cache_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _persisted_source: object | None = field(default=None, init=False)
    _persisted_cache: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def system_status(self, manager: AppManager) -> dict[str, Any]:
        # The recording store may be an NFS mount.  Never stat it from an API
        # request: a stalled mount would otherwise block health consumers and
        # integrations even though cameras and the application are healthy.
        recorder = getattr(manager, "recorder", None)
        retention_reader = getattr(recorder, "retention_status", None)
        retention = retention_reader() if callable(retention_reader) else {}
        plan = retention.get("plan") if isinstance(retention, dict) else None
        storage = dict(plan.get("storage") or {}) if isinstance(plan, dict) else {}
        storage_total = int(storage.get("total_bytes") or 0)
        storage_free = int(storage.get("free_bytes") or 0)
        storage_used = max(0, int(storage.get("used_bytes") or storage_total - storage_free))
        cameras = manager.statuses()
        mqtt = manager.mqtt_status()
        service_memory = self.cgroup_memory_status()
        application_memory_bytes = int(service_memory.get("application_bytes") or 0)
        if application_memory_bytes <= 0:
            application_memory_bytes = int(process_memory_status().get("rss_bytes") or 0)
        cpu_count = os.cpu_count() or 1
        try:
            load_1m = os.getloadavg()[0]
        except OSError:
            load_1m = 0.0
        enabled_cameras = [camera for camera in cameras if camera.get("expected_enabled", True)]
        expected_recordings = [
            camera
            for camera in enabled_cameras
            if camera.get("recording_configured") and camera.get("recording_enabled", True)
        ]
        payload = {
            "instance_id": self.process_instance_id,
            "version": SURVNG_VERSION,
            "lifecycle": str((mqtt or {}).get("server_lifecycle") or "running"),
            "uptime_seconds": round(
                max(0.0, time.monotonic() - self.process_started_monotonic), 1
            ),
            "version": self.product_version(),
            "resources": {
                "cpu_load_percent": round(
                    min(100.0, max(0.0, (load_1m / cpu_count) * 100.0)), 1
                ),
                "application_memory_bytes": application_memory_bytes,
            },
            "storage": {
                "total_bytes": storage_total,
                "used_bytes": storage_used,
                "free_bytes": storage_free,
                "used_percent": round((storage_used / storage_total) * 100, 1)
                if storage_total
                else 0,
                "sampled_at": retention.get("last_plan_at")
                if isinstance(retention, dict)
                else None,
                "available": bool(storage_total),
            },
            "detector": manager.detector_status(),
            "cameras": {
                "total": len(cameras),
                "enabled": len(enabled_cameras),
                "online": sum(1 for camera in enabled_cameras if camera.get("running")),
                "recording_expected": len(expected_recordings),
                "recording": sum(1 for camera in expected_recordings if camera.get("recording")),
            },
            "mqtt": mqtt,
            "go2rtc": manager.go2rtc_status(),
            "camera_startup": manager.camera_startup_status(),
        }
        return payload

    @staticmethod
    def product_version() -> dict[str, Any]:
        snapshot = ProductUpdateService().status(refresh_remote=False)
        return {
            "deployment_mode": snapshot.get("deployment_mode"),
            "sha": snapshot.get("current_sha"),
            "short_sha": snapshot.get("current_short_sha"),
            "branch": snapshot.get("branch"),
        }

    @staticmethod
    def linux_memory_status() -> dict[str, int | float]:
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return {
                "total_bytes": 0,
                "available_bytes": 0,
                "used_bytes": 0,
                "used_percent": 0.0,
            }
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", values.get("MemFree", 0))
        used = max(0, total - available)
        return {
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": used,
            "used_percent": round((used / total) * 100, 1) if total else 0.0,
        }

    @staticmethod
    def cgroup_memory_status(
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        process_cgroup: Path = Path("/proc/self/cgroup"),
    ) -> dict[str, int]:
        empty = {
            "total_bytes": 0,
            "working_set_bytes": 0,
            "application_bytes": 0,
            "file_cache_bytes": 0,
            "reclaimable_file_cache_bytes": 0,
            "kernel_bytes": 0,
        }
        try:
            relative = next(
                line.split(":", 2)[2].strip().lstrip("/")
                for line in process_cgroup.read_text(encoding="utf-8").splitlines()
                if line.startswith("0::")
            )
            base = cgroup_root / relative
            total = int((base / "memory.current").read_text(encoding="utf-8").strip())
            stats = {
                key: int(value)
                for key, value in (
                    line.split(maxsplit=1)
                    for line in (base / "memory.stat").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
        except (OSError, StopIteration, ValueError):
            return empty
        shmem = max(0, stats.get("shmem", 0))
        file_cache = max(0, stats.get("file", 0) - shmem)
        reclaimable = min(file_cache, max(0, stats.get("inactive_file", 0) - shmem))
        return {
            "total_bytes": max(0, total),
            "working_set_bytes": max(0, total - reclaimable),
            "application_bytes": max(0, stats.get("anon", 0)) + shmem,
            "file_cache_bytes": file_cache,
            "reclaimable_file_cache_bytes": reclaimable,
            "kernel_bytes": max(0, stats.get("kernel", 0)),
        }

    @staticmethod
    def database_bytes(database_dir: Path) -> int:
        total = 0
        for path in database_dir.glob("*.sqlite3*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    @staticmethod
    def read_integer(path: Path, *, scale: int = 1) -> int | None:
        try:
            return int(path.read_text(encoding="utf-8").strip(), 0) * scale
        except (OSError, ValueError):
            return None

    @staticmethod
    def drm_worker_counters(pid: int) -> dict[str, object]:
        engines: dict[str, int] = {}
        allocated_bytes = 0
        resident_bytes = 0
        clients: set[str] = set()
        driver = ""
        try:
            fdinfo_paths = list(Path(f"/proc/{pid}/fdinfo").iterdir())
        except OSError:
            return {
                "engines": engines,
                "allocated_bytes": 0,
                "resident_bytes": 0,
                "driver": "",
            }
        for path in fdinfo_paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            values: dict[str, str] = {}
            for line in lines:
                if line.startswith("drm-") and ":" in line:
                    key, value = line.split(":", 1)
                    values[key] = value.strip()
            if not values:
                continue
            driver = values.get("drm-driver", driver)
            client_id = values.get("drm-client-id", str(path))
            for key, value in values.items():
                if not key.startswith("drm-engine-"):
                    continue
                try:
                    nanoseconds = int(value.split()[0])
                except (ValueError, IndexError):
                    continue
                name = key.removeprefix("drm-engine-")
                engines[name] = engines.get(name, 0) + nanoseconds
            if client_id in clients:
                continue
            clients.add(client_id)
            for key, target in (
                ("drm-total-system0", "allocated"),
                ("drm-resident-system0", "resident"),
            ):
                try:
                    byte_count = int(values.get(key, "0").split()[0]) * 1024
                except (ValueError, IndexError):
                    byte_count = 0
                if target == "allocated":
                    allocated_bytes += byte_count
                else:
                    resident_bytes += byte_count
        return {
            "engines": engines,
            "allocated_bytes": allocated_bytes,
            "resident_bytes": resident_bytes,
            "driver": driver,
        }

    def gpu_status(self, detector: dict[str, Any]) -> dict[str, object]:
        workers = detector.get("workers") or {}
        worker_pids: set[int] = set()
        if isinstance(workers, dict):
            for worker in workers.values():
                if not isinstance(worker, dict) or not worker.get("worker_alive"):
                    continue
                candidates = worker.get("worker_pids") or [worker.get("worker_pid")]
                worker_pids.update(
                    int(pid)
                    for pid in candidates
                    if pid is not None and int(pid) > 0
                )
        pids = (
            tuple(sorted(worker_pids))
            if isinstance(workers, dict) else ()
        )
        engines: dict[str, int] = {}
        allocated_bytes = 0
        resident_bytes = 0
        driver = ""
        for pid in pids:
            counters = self.drm_worker_counters(pid)
            driver = str(counters.get("driver") or driver)
            allocated_bytes += int(counters.get("allocated_bytes") or 0)
            resident_bytes += int(counters.get("resident_bytes") or 0)
            for name, value in dict(counters.get("engines") or {}).items():
                engines[str(name)] = engines.get(str(name), 0) + int(value)

        sampled_at = time.monotonic()
        engine_usage: dict[str, float] = {}
        with self._gpu_lock:
            previous_at = float(self._gpu_sample.get("at") or 0.0)
            previous_pids = tuple(self._gpu_sample.get("pids") or ())
            previous_engines = dict(self._gpu_sample.get("engines") or {})
            elapsed_ns = (sampled_at - previous_at) * 1_000_000_000
            if pids and pids == previous_pids and elapsed_ns > 0:
                for name, value in engines.items():
                    delta = value - int(previous_engines.get(name, value))
                    if delta >= 0:
                        engine_usage[name] = round(
                            min(100.0, (delta / elapsed_ns) * 100.0), 1
                        )
            self._gpu_sample.update({"at": sampled_at, "pids": pids, "engines": engines})

        card = Path("/sys/class/drm/card0")
        vendor_id = self.read_integer(card / "device/vendor")
        device_id = self.read_integer(card / "device/device")
        current_frequency = self.read_integer(card / "gt_act_freq_mhz")
        maximum_frequency = self.read_integer(card / "gt_max_freq_mhz")
        temperature_millidegrees = next(
            (
                value
                for value in (
                    self.read_integer(path)
                    for path in (card / "device/hwmon").glob("hwmon*/temp1_input")
                )
                if value is not None
            ),
            None,
        )
        vendor_names = {0x8086: "Intel", 0x1002: "AMD", 0x10DE: "NVIDIA"}
        vendor = vendor_names.get(
            vendor_id,
            f"0x{vendor_id:04x}" if vendor_id is not None else "Unknown",
        )
        utilization = (
            round(min(100.0, sum(engine_usage.values())), 1) if engine_usage else None
        )
        return {
            "available": bool(card.exists() or engines),
            "scope": "SurvNG inference workers",
            "vendor": vendor,
            "device_id": f"0x{device_id:04x}" if device_id is not None else "",
            "driver": driver,
            "worker_pids": list(pids),
            "utilization_percent": utilization,
            "engine_utilization": engine_usage,
            "allocated_bytes": allocated_bytes,
            "resident_bytes": resident_bytes,
            "current_frequency_mhz": current_frequency,
            "maximum_frequency_mhz": maximum_frequency,
            "temperature_celsius": round(temperature_millidegrees / 1000.0, 1)
            if temperature_millidegrees is not None
            else None,
            "sample_ready": utilization is not None,
        }

    def record_history(
        self, sample: dict[str, object], sampled_at: float
    ) -> list[dict[str, object]]:
        with self._history_lock:
            if not self._history or sampled_at - self._last_history_sample_at >= 5.0:
                self._history.append(sample)
                self._last_history_sample_at = sampled_at
            else:
                self._history[-1] = sample
            return [dict(item) for item in self._history]

    def persisted_history(self, manager: AppManager, camera_id: str) -> dict[str, Any]:
        event_store = manager.events
        telemetry_store = manager.telemetry
        now = time.monotonic()
        with self._persisted_cache_lock:
            if self._persisted_source is not telemetry_store:
                self._persisted_source = telemetry_store
                self._persisted_cache.clear()
            cached = self._persisted_cache.get(camera_id)
            if cached is not None and now - float(cached["at"]) < 55.0:
                return cached["value"]
        interruptions: list[dict[str, Any]] = []
        if not camera_id:
            try:
                interruptions = classify_telemetry_interruptions(
                    telemetry_store.sample_times(hours=168),
                    telemetry_store.lifecycle_events(hours=168),
                    observed_at=datetime.now(timezone.utc),
                )
            except Exception:
                LOGGER.exception("could not load telemetry interruption annotations")
        value = {
            "runtime": {
                "short": telemetry_store.operational_history(
                    hours=2, bucket_minutes=1, camera_id=camera_id
                ),
                "long": telemetry_store.operational_history(
                    hours=168, bucket_minutes=15, camera_id=camera_id
                ),
            },
            "tracking": {
                "short": event_store.tracking_capacity_activity(
                    hours=2, bucket_minutes=1, camera_id=camera_id
                ),
                "long": event_store.tracking_capacity_activity(
                    hours=168, bucket_minutes=15, camera_id=camera_id
                ),
            },
            "memory": (
                {
                    "short": telemetry_store.memory_history(
                        hours=24, bucket_minutes=5
                    ),
                    "long": telemetry_store.memory_history(
                        hours=168, bucket_minutes=15
                    ),
                }
                if not camera_id
                else {"short": [], "long": []}
            ),
            "interruptions": interruptions,
            "interruption_summary": summarize_interruptions(interruptions, hours=24),
        }
        with self._persisted_cache_lock:
            if self._persisted_source is telemetry_store:
                if len(self._persisted_cache) >= 32:
                    self._persisted_cache.clear()
                self._persisted_cache[camera_id] = {"at": now, "value": value}
        return value

    @staticmethod
    def _camera_payload(
        status: dict[str, Any], per_camera_activity: dict[str, Any],
        per_camera_storage: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        camera_id = str(status.get("id") or "")
        motion = status.get("motion_qualification") or {}
        tracking = status.get("object_tracking") or {}
        payload = {
            "id": camera_id,
            "name": status.get("name") or camera_id,
            "connected": bool(status.get("connected")),
            "expected_enabled": bool(status.get("expected_enabled", True)),
            "frame_fresh": bool(status.get("frame_fresh")),
            "last_frame_age_seconds": status.get("last_frame_age_seconds"),
            "recording": bool(status.get("recording") or status.get("sub_recording")),
            "recording_timestamps": dict(status.get("recording_timestamp_health") or {}),
            "detection_enabled": bool(status.get("detection_enabled")),
            "onvif": {
                "enabled": bool(status.get("onvif_enabled")),
                "connected": bool(status.get("onvif_connected")),
                "notifications": int(status.get("onvif_notifications_received") or 0),
                "motion_events": int(status.get("onvif_motion_events_received") or 0),
                "poll_errors": int(status.get("onvif_poll_errors") or 0),
                "poll_timeouts": int(status.get("onvif_poll_timeouts") or 0),
                "renewals": int(status.get("onvif_renewals") or 0),
                "renewal_errors": int(status.get("onvif_renewal_errors") or 0),
                "last_motion_at": status.get("onvif_last_motion_event_at"),
                "last_error": status.get("onvif_last_error") or "",
                "signal_effectiveness_status": (
                    status.get("onvif_signal_effectiveness_status")
                    or "insufficient_data"
                ),
                "signal_degraded": bool(status.get("onvif_signal_degraded")),
                "recognized_notifications": int(
                    status.get("onvif_recognized_notifications") or 0
                ),
                "notification_recognition_rate": status.get(
                    "onvif_notification_recognition_rate"
                ),
                "active_motion_rate": status.get("onvif_active_motion_rate"),
                "ema_qualified_observations": int(
                    status.get("onvif_ema_qualified_observations") or 0
                ),
                "ema_onvif_matches": int(
                    status.get("onvif_ema_onvif_matches") or 0
                ),
                "ema_without_onvif": int(
                    status.get("onvif_ema_without_onvif") or 0
                ),
                "ema_window_observations": int(
                    status.get("onvif_ema_window_observations") or 0
                ),
                "ema_window_onvif_matches": int(
                    status.get("onvif_ema_window_onvif_matches") or 0
                ),
                "ema_window_without_onvif": int(
                    status.get("onvif_ema_window_without_onvif") or 0
                ),
                "ema_window_match_rate": status.get(
                    "onvif_ema_window_match_rate"
                ),
                "last_ema_observation_at": status.get(
                    "onvif_last_ema_observation_at"
                ),
                "last_ema_onvif_match_at": status.get(
                    "onvif_last_ema_onvif_match_at"
                ),
                "last_ema_without_onvif_at": status.get(
                    "onvif_last_ema_without_onvif_at"
                ),
                "unknown_notification_samples": list(
                    status.get("onvif_unknown_notification_samples") or []
                ),
            },
            "motion": {
                "mode": motion.get("mode") or "",
                "sample_fps": float(motion.get("sample_fps") or 0.0),
                "triggers": int(motion.get("triggers") or 0),
                "bursts": int(motion.get("bursts") or 0),
                "passed": int(motion.get("passed") or 0),
                "rejected": int(motion.get("audit_rejected") or 0),
                "suppressed": int(motion.get("suppressed") or 0),
                "dropped": int(motion.get("dropped_triggers") or 0),
                "analysis_frames_dropped": int(motion.get("analysis_frames_dropped") or 0),
                "analysis_wait_ms_total": round(float(motion.get("analysis_wait_ms_total") or 0.0), 1),
                "analysis_wait_ms_max": round(float(motion.get("analysis_wait_ms_max") or 0.0), 1),
                "analysis_wait_ms_p95": round(
                    float(motion.get("analysis_wait_ms_p95") or 0.0),
                    1,
                ),
                "queue_depth": int(motion.get("queue_depth") or 0),
                "analysis_runtime": dict(motion.get("analysis_runtime") or {}),
                "event_runtime": dict(motion.get("event_runtime") or {}),
                "visual_backup_candidates": int(motion.get("visual_backup_candidates") or 0),
                "visual_backup_triggers": int(motion.get("visual_backup_triggers") or 0),
                "visual_backup_onvif_matches": int(motion.get("visual_backup_onvif_matches") or 0),
                "visual_backup_rate_limited": int(motion.get("visual_backup_rate_limited") or 0),
                "visual_backup_not_ready": int(motion.get("visual_backup_not_ready") or 0),
                "visual_backup_not_promoted": int(motion.get("visual_backup_not_promoted") or 0),
                "visual_backup_uncorrelated_objects": int(
                    motion.get("visual_backup_uncorrelated_objects") or 0
                ),
                "active_followup_candidates": int(
                    motion.get("active_followup_candidates") or 0
                ),
                "active_followup_triggers": int(
                    motion.get("active_followup_triggers") or 0
                ),
                "active_followup_objects": int(
                    motion.get("active_followup_objects") or 0
                ),
                "active_followup_no_object": int(
                    motion.get("active_followup_no_object") or 0
                ),
                "active_followup_deduplicated": int(
                    motion.get("active_followup_deduplicated") or 0
                ),
                "active_followup_rate_limited": int(
                    motion.get("active_followup_rate_limited") or 0
                ),
                "active_followup_episode_limited": int(
                    motion.get("active_followup_episode_limited") or 0
                ),
            },
            "tracking": {
                "active": bool(tracking.get("active")),
                "frames_processed": int(tracking.get("frames_processed") or 0),
                "track_count": int(tracking.get("track_count") or 0),
                "capacity_requests": int(tracking.get("capacity_requests") or 0),
                "capacity_waits": int(tracking.get("capacity_waits") or 0),
                "capacity_timeouts": int(tracking.get("capacity_timeouts") or 0),
                "capacity_wait_seconds_total": float(tracking.get("capacity_wait_seconds_total") or 0.0),
                "capacity_wait_seconds_max": float(tracking.get("capacity_wait_seconds_max") or 0.0),
                "capacity_wait_seconds_last": float(tracking.get("capacity_wait_seconds_last") or 0.0),
                "reid_attempts": int(tracking.get("reid_attempts") or 0),
                "reid_successes": int(tracking.get("reid_successes") or 0),
                "reid_failures": int(tracking.get("reid_failures") or 0),
                "reid_average_ms": float(tracking.get("reid_average_ms") or 0.0),
                "reid_attempts_by_label": dict(tracking.get("reid_attempts_by_label") or {}),
                "reid_attempts_by_reason": dict(tracking.get("reid_attempts_by_reason") or {}),
                "reid_recoveries": int(tracking.get("reid_recoveries") or 0),
                "reid_recoveries_by_label": dict(tracking.get("reid_recoveries_by_label") or {}),
                "reid_avoided_geometry_matches": int(
                    tracking.get("reid_avoided_geometry_matches") or 0
                ),
                "reid_avoided_by_label": dict(tracking.get("reid_avoided_by_label") or {}),
                "prewarm_failures": int(tracking.get("prewarm_failures") or 0),
                "last_prewarm_failure": tracking.get("last_prewarm_failure"),
                "handoff_failures": int(tracking.get("handoff_failures") or 0),
                "last_handoff_failure": tracking.get("last_handoff_failure"),
            },
            "capture": dict(status.get("capture_stats") or {}),
            "lifecycle": dict(status.get("lifecycle") or {}),
            "activity": per_camera_activity.get(
                camera_id,
                {
                    "last_hour": {"events": 0, "object_incidents": 0, "objects": 0, "labels": {}},
                    "last_24h": {"events": 0, "object_incidents": 0, "objects": 0, "labels": {}},
                    "hourly": [],
                },
            ),
            "storage": per_camera_storage.get(
                camera_id,
                {
                    "recording_bytes": 0,
                    "recording_files": 0,
                    "snapshot_bytes": 0,
                    "snapshot_files": 0,
                },
            ),
        }
        payload["performance"] = camera_performance_health(payload)
        return payload

    def telemetry(
        self, manager: AppManager, config: AppConfig, *, hours: int, camera_id: str
    ) -> dict[str, Any]:
        storage_path = manager.storage_dir
        storage_path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(storage_path)
        camera_statuses = manager.statuses()
        detector = manager.detector_status()
        semantic_search = manager.semantic_search_status()
        face_recognition = {}
        faces = getattr(manager, "faces", None)
        if faces is not None:
            try:
                face_recognition = {
                    **faces.stats(),
                    "recognition": faces.recognition_status(),
                }
            except Exception:
                LOGGER.exception("could not collect face-recognition telemetry")
                face_recognition = {"error": "Face telemetry is temporarily unavailable."}
        gpu = self.gpu_status(detector)
        activity = manager.events.telemetry_activity(hours=hours)
        selected_camera_id = str(camera_id or "").strip()[:128]
        persisted = self.persisted_history(manager, selected_camera_id)
        per_camera_activity = activity.get("by_camera", {})
        recorder = getattr(manager, "recorder", None)
        retention_reader = getattr(recorder, "retention_status", None)
        retention = retention_reader() if callable(retention_reader) else {}
        retention_plan = retention.get("plan") if isinstance(retention, dict) else None
        storage_rows = (
            retention_plan.get("per_camera_storage", [])
            if isinstance(retention_plan, dict)
            else []
        )
        per_camera_storage = {
            str(row.get("camera_id") or ""): dict(row)
            for row in storage_rows
            if isinstance(row, dict)
        }
        load_1m, load_5m, load_15m = os.getloadavg()
        memory = self.linux_memory_status()
        process_memory = process_memory_status()
        process_rss_bytes = int(process_memory["rss_bytes"])
        runtime_monitor = getattr(manager, "runtime_monitor", None)
        worker_reader = getattr(runtime_monitor, "worker_memory_status", None)
        worker_memory = (
            worker_reader(detector_status=detector)
            if callable(worker_reader)
            else {"total_rss_bytes": 0, "total_pss_bytes": 0, "workers": {}}
        )
        maintenance_reader = getattr(runtime_monitor, "memory_maintenance_status", None)
        memory_maintenance = maintenance_reader() if callable(maintenance_reader) else {}
        service_memory = self.cgroup_memory_status()
        cpu_count = os.cpu_count() or 1
        generated_at = datetime.now(timezone.utc).isoformat()
        cameras = [
            self._camera_payload(status, per_camera_activity, per_camera_storage)
            for status in camera_statuses
        ]
        detector_runtime = detector.get("runtime") or {}
        history = self.record_history(
            {
                "sampled_at": generated_at,
                "cpu_load_percent": round(min(100.0, (load_1m / cpu_count) * 100.0), 2),
                "memory_used_percent": float(memory.get("used_percent") or 0.0),
                "storage_used_percent": round((usage.used / usage.total) * 100.0, 3)
                if usage.total
                else 0.0,
                "process_rss_bytes": process_rss_bytes,
                "service_application_bytes": service_memory["application_bytes"],
                "service_file_cache_bytes": service_memory["file_cache_bytes"],
                "service_reclaimable_file_cache_bytes": service_memory[
                    "reclaimable_file_cache_bytes"
                ],
                "service_working_set_bytes": service_memory["working_set_bytes"],
                "gpu_utilization_percent": gpu.get("utilization_percent"),
                "inference_ms": detector_runtime.get("average_inference_ms"),
                "detection_fps": detector_runtime.get("detection_fps"),
            },
            time.monotonic(),
        )
        return {
            "generated_at": generated_at,
            "system": {
                "uptime_seconds": round(
                    max(0.0, time.monotonic() - self.process_started_monotonic), 1
                ),
                "cpu_count": cpu_count,
                "load_average": {
                    "one": round(load_1m, 2),
                    "five": round(load_5m, 2),
                    "fifteen": round(load_15m, 2),
                },
                "memory": memory,
                "process_rss_bytes": process_rss_bytes,
                "process_memory": process_memory,
                "worker_memory": worker_memory,
                "memory_maintenance": memory_maintenance,
                "service_memory": service_memory,
                "storage": {
                    "path": str(storage_path),
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                    "used_percent": round((usage.used / usage.total) * 100, 1)
                    if usage.total
                    else 0.0,
                },
                "database": {
                    "path": str(manager.database_dir),
                    "bytes": self.database_bytes(manager.database_dir),
                },
            },
            "detector": detector,
            "semantic_search": semantic_search,
            "face_recognition": face_recognition,
            "gpu": gpu,
            "history": history,
            "runtime_history": persisted["runtime"],
            "tracking_capacity_history": persisted["tracking"],
            "process_memory_history": persisted["memory"],
            "interruptions": persisted["interruptions"],
            "interruption_summary": persisted["interruption_summary"],
            "diagnostics": manager.runtime_monitor.diagnostic_status(),
            "operational_events": manager.telemetry.operational_event_history(hours=24),
            "tracking_capacity": {
                "limit": int(config.detector.tracking.max_active_cameras),
                "burst_limit": int(config.detector.tracking.burst_max_active_cameras),
                "adaptive_burst_enabled": bool(
                    config.detector.tracking.adaptive_burst_enabled
                ),
                "wait_seconds": float(config.detector.tracking.capacity_wait_seconds),
                "active": sum(1 for item in cameras if item["tracking"]["active"]),
                "requests_since_restart": sum(
                    item["tracking"]["capacity_requests"] for item in cameras
                ),
                "waits_since_restart": sum(
                    item["tracking"]["capacity_waits"] for item in cameras
                ),
                "timeouts_since_restart": sum(
                    item["tracking"]["capacity_timeouts"] for item in cameras
                ),
                "limiter": (
                    manager.inference.tracking_limiter.status()
                    if hasattr(manager, "inference")
                    else {}
                ),
            },
            "appearance_backfill": (
                manager.appearance_backfill.status()
                if hasattr(manager, "appearance_backfill")
                else {"enabled": False, "running": False, "counts": {}}
            ),
            "activity": activity,
            "cameras": cameras,
        }


def create_system_telemetry_router(
    dependencies: SystemTelemetryDependencies,
    service: SystemTelemetryService,
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/system/status")
    def system_status() -> dict[str, Any]:
        active_manager = dependencies.get_manager()
        return service.system_status(active_manager)

    @router.get("/api/telemetry")
    def telemetry(hours: int = 24, camera_id: str = "") -> dict[str, Any]:
        active_manager = dependencies.get_manager()
        active_config = getattr(active_manager, "config", None) or dependencies.get_config()
        return service.telemetry(
            active_manager, active_config, hours=hours, camera_id=camera_id
        )

    @router.post("/api/telemetry/diagnostics")
    def start_diagnostics(request: DiagnosticSessionRequest) -> dict[str, Any]:
        active_manager = dependencies.get_manager()
        try:
            return active_manager.runtime_monitor.start_diagnostics(
                scope=request.scope,
                camera_id=request.camera_id,
                duration_seconds=request.duration_seconds,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.delete("/api/telemetry/diagnostics/{session_id}")
    def stop_diagnostics(session_id: str) -> dict[str, bool]:
        active_manager = dependencies.get_manager()
        if not active_manager.runtime_monitor.stop_diagnostics(session_id):
            raise HTTPException(status_code=404, detail="diagnostic session not found")
        return {"stopped": True}

    @router.get("/api/telemetry/diagnostics/{session_id}")
    def export_diagnostics(session_id: str) -> StreamingResponse:
        active_manager = dependencies.get_manager()
        chunks = active_manager.runtime_monitor.export_diagnostics_stream(session_id)
        if chunks is None:
            raise HTTPException(status_code=404, detail="diagnostic session not found")
        return StreamingResponse(
            chunks,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="survng-diagnostics-{session_id}.json"'
                )
            },
        )

    return router
