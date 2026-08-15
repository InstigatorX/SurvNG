"""Bounded, temporary diagnostic telemetry capture."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Iterator

from .telemetry_contract import DIAGNOSTIC_DURATIONS_SECONDS, DIAGNOSTIC_SCOPES
from .security import redact_secret_text
from .telemetry_store import TelemetryStore


def _diagnostic_safe(value: Any) -> Any:
    """Make diagnostic payloads finite, bounded, and credential-safe."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return redact_secret_text(value)[:2000]
    if isinstance(value, Mapping):
        return {
            str(key)[:128]: _diagnostic_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_diagnostic_safe(item) for item in value]
    return redact_secret_text(str(value))[:2000]


class DiagnosticTelemetryController:
    """Retain a small memory prebuffer and persist only active sessions."""

    SAMPLE_INTERVAL_SECONDS = 5.0
    PREBUFFER_SAMPLES = 60

    def __init__(self, store: TelemetryStore) -> None:
        self._store = store
        self._lock = threading.RLock()
        self._ring: deque[tuple[datetime, dict[str, Any]]] = deque(
            maxlen=self.PREBUFFER_SAMPLES
        )
        self._last_sample_monotonic = 0.0
        self._samples_since_budget_check = 0
        self._camera_unavailable_samples: dict[str, int] = {}
        self._camera_anomaly_active: set[str] = set()
        self._detector_failure_count: int | None = None
        self._detector_backlog_samples = 0
        self._storage_anomaly_active = False

    @staticmethod
    def _camera_payload(status: dict[str, Any]) -> dict[str, Any]:
        motion = status.get("motion_qualification") or {}
        return {
            "id": str(status.get("id") or ""),
            "connected": bool(status.get("connected")),
            "expected_enabled": bool(status.get("expected_enabled", True)),
            "last_frame_age_seconds": status.get("last_frame_age_seconds"),
            "main_last_frame_age_seconds": status.get("main_last_frame_age_seconds"),
            "capture": status.get("capture_stats") or {},
            "motion": {
                "analysis_frames_dropped": int(motion.get("analysis_frames_dropped") or 0),
                "analysis_wait_ms_p95": float(motion.get("analysis_wait_ms_p95") or 0.0),
                "analysis_runtime": motion.get("analysis_runtime") or {},
                "event_runtime": motion.get("event_runtime") or {},
            },
            "tracking": status.get("object_tracking") or {},
            "lifecycle": status.get("lifecycle") or {},
        }

    def observe(
        self,
        statuses: list[dict[str, Any]],
        *,
        detector_runtime: dict[str, Any],
        storage_status: dict[str, Any] | None = None,
        camera_startup_complete: bool = True,
        now_monotonic: float | None = None,
        sampled_at: datetime | None = None,
    ) -> None:
        monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        with self._lock:
            if monotonic - self._last_sample_monotonic < self.SAMPLE_INTERVAL_SECONDS:
                return
            self._last_sample_monotonic = monotonic
        current = sampled_at or datetime.now(timezone.utc)
        payload = _diagnostic_safe({
            "detector_runtime": dict(detector_runtime),
            "storage": dict(storage_status or {}),
            "cameras": {
                str(status.get("id")): self._camera_payload(status)
                for status in statuses
                if status.get("id")
            },
        })
        with self._lock:
            self._ring.append((current, payload))
        active = self._store.diagnostic_sessions(active_only=True, now=current)
        for session in active:
            scoped = self._scoped_payload(payload, session)
            self._store.write_diagnostic_samples(
                [str(session["id"])], sampled_at=current, payload=scoped
            )
        if active:
            with self._lock:
                self._samples_since_budget_check += 1
                check_budget = self._samples_since_budget_check >= 12
                if check_budget:
                    self._samples_since_budget_check = 0
            if check_budget:
                self._store.enforce_diagnostic_budget()
        self._detect_anomalies(
            payload,
            current,
            camera_startup_complete=camera_startup_complete,
        )

    @staticmethod
    def _scoped_payload(
        payload: dict[str, Any], session: dict[str, Any]
    ) -> dict[str, Any]:
        scope = str(session.get("scope") or "system")
        if scope == "camera":
            camera_id = str(session.get("camera_id") or "")
            camera = dict(payload.get("cameras") or {}).get(camera_id)
            return {"cameras": {camera_id: camera} if camera is not None else {}}
        if scope == "detector":
            return {"detector_runtime": payload.get("detector_runtime") or {}}
        if scope == "storage":
            return {"storage": payload.get("storage") or {}}
        return payload

    def start(
        self,
        *,
        scope: str,
        duration_seconds: int,
        camera_id: str = "",
        trigger_kind: str = "manual",
        started_at: datetime | None = None,
    ) -> dict[str, Any]:
        if scope not in DIAGNOSTIC_SCOPES:
            raise ValueError("unsupported diagnostic scope")
        if duration_seconds not in DIAGNOSTIC_DURATIONS_SECONDS:
            raise ValueError("unsupported diagnostic duration")
        if scope == "camera" and not camera_id:
            raise ValueError("camera diagnostics require a camera id")
        session = self._store.create_diagnostic_session(
            scope=scope,
            camera_id=camera_id if scope == "camera" else "",
            duration_seconds=duration_seconds,
            trigger_kind=trigger_kind,
            started_at=started_at,
        )
        with self._lock:
            buffered = list(self._ring)
        for sampled_at, payload in buffered:
            self._store.write_diagnostic_samples(
                [str(session["id"])],
                sampled_at=sampled_at,
                payload=self._scoped_payload(payload, session),
            )
        return session

    def _start_automatic_session(
        self, *, kind: str, scope: str, camera_id: str, summary: str, current: datetime
    ) -> None:
        self._store.record_or_coalesce_operational_event(
            occurred_at=current,
            kind=kind,
            scope=scope,
            camera_id=camera_id,
            summary=summary,
        )
        matching = [
            session
            for session in self._store.diagnostic_sessions(active_only=True, now=current)
            if session.get("trigger_kind") == kind
            and session.get("scope") == scope
            and str(session.get("camera_id") or "") == camera_id
        ]
        if not matching:
            self.start(
                scope=scope,
                camera_id=camera_id,
                duration_seconds=900,
                trigger_kind=kind,
                started_at=current,
            )

    def _detect_anomalies(
        self,
        payload: dict[str, Any],
        current: datetime,
        *,
        camera_startup_complete: bool,
    ) -> None:
        cameras = dict(payload.get("cameras") or {})
        present: set[str] = set()
        for camera_id, camera in cameras.items():
            if not isinstance(camera, dict):
                continue
            present.add(str(camera_id))
            if not camera_startup_complete:
                self._camera_unavailable_samples[str(camera_id)] = 0
                continue
            unhealthy = bool(camera.get("expected_enabled", True)) and not bool(
                camera.get("connected")
            )
            count = self._camera_unavailable_samples.get(str(camera_id), 0)
            count = count + 1 if unhealthy else 0
            self._camera_unavailable_samples[str(camera_id)] = count
            if count >= 3 and str(camera_id) not in self._camera_anomaly_active:
                self._camera_anomaly_active.add(str(camera_id))
                self._start_automatic_session(
                    kind="camera_unavailable",
                    scope="camera",
                    camera_id=str(camera_id),
                    summary=f"{camera_id} stopped delivering live video",
                    current=current,
                )
            elif not unhealthy and str(camera_id) in self._camera_anomaly_active:
                self._camera_anomaly_active.remove(str(camera_id))
                self._store.record_or_coalesce_operational_event(
                    occurred_at=current,
                    kind="camera_recovered",
                    scope="camera",
                    camera_id=str(camera_id),
                    summary=f"{camera_id} resumed live video",
                )
        self._camera_unavailable_samples = {
            camera_id: count
            for camera_id, count in self._camera_unavailable_samples.items()
            if camera_id in present
        }
        self._camera_anomaly_active.intersection_update(present)

        detector = dict(payload.get("detector_runtime") or {})
        failures = int(detector.get("failed_inferences") or detector.get("failures") or 0)
        if self._detector_failure_count is not None and failures > self._detector_failure_count:
            self._start_automatic_session(
                kind="detector_failure",
                scope="detector",
                camera_id="",
                summary="Object detector reported a failed inference",
                current=current,
            )
        self._detector_failure_count = failures
        depth = int(detector.get("queue_depth") or 0)
        backlogged = depth > 0
        self._detector_backlog_samples = self._detector_backlog_samples + 1 if backlogged else 0
        if self._detector_backlog_samples == 3:
            self._start_automatic_session(
                kind="detector_backlog",
                scope="detector",
                camera_id="",
                summary="Object detector queue remained backlogged",
                current=current,
            )

        storage = dict(payload.get("storage") or {})
        storage_failed = str(storage.get("state") or "").lower() in {"error", "failed"}
        if storage_failed and not self._storage_anomaly_active:
            self._storage_anomaly_active = True
            self._start_automatic_session(
                kind="retention_failure",
                scope="storage",
                camera_id="",
                summary="Recording retention reported a failure",
                current=current,
            )
        elif not storage_failed:
            self._storage_anomaly_active = False

    def stop(self, session_id: str) -> bool:
        return self._store.stop_diagnostic_session(session_id)

    def status(self) -> dict[str, Any]:
        return {
            "active": self._store.diagnostic_sessions(active_only=True),
            "recent": self._store.diagnostic_sessions(active_only=False)[:20],
            "supported_scopes": sorted(DIAGNOSTIC_SCOPES),
            "supported_durations_seconds": list(DIAGNOSTIC_DURATIONS_SECONDS),
            "prebuffer_seconds": int(
                self.SAMPLE_INTERVAL_SECONDS * self.PREBUFFER_SAMPLES
            ),
        }

    def export(self, session_id: str) -> dict[str, Any] | None:
        return self._store.diagnostic_export(session_id)

    def export_stream(self, session_id: str) -> Iterator[bytes] | None:
        return self._store.diagnostic_export_chunks(session_id)
