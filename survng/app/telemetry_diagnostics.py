"""Bounded, temporary diagnostic telemetry capture."""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from .telemetry_contract import DIAGNOSTIC_DURATIONS_SECONDS, DIAGNOSTIC_SCOPES
from .telemetry_store import TelemetryStore


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
        now_monotonic: float | None = None,
        sampled_at: datetime | None = None,
    ) -> None:
        monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        with self._lock:
            if monotonic - self._last_sample_monotonic < self.SAMPLE_INTERVAL_SECONDS:
                return
            self._last_sample_monotonic = monotonic
        current = sampled_at or datetime.now(timezone.utc)
        payload = {
            "detector_runtime": dict(detector_runtime),
            "storage": dict(storage_status or {}),
            "cameras": {
                str(status.get("id")): self._camera_payload(status)
                for status in statuses
                if status.get("id")
            },
        }
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
        self, *, scope: str, duration_seconds: int, camera_id: str = ""
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

    def stop(self, session_id: str) -> bool:
        return self._store.stop_diagnostic_session(session_id)

    def status(self) -> dict[str, Any]:
        return {
            "active": self._store.diagnostic_sessions(active_only=True),
            "supported_scopes": sorted(DIAGNOSTIC_SCOPES),
            "supported_durations_seconds": list(DIAGNOSTIC_DURATIONS_SECONDS),
            "prebuffer_seconds": int(
                self.SAMPLE_INTERVAL_SECONDS * self.PREBUFFER_SAMPLES
            ),
        }

    def export(self, session_id: str) -> dict[str, Any] | None:
        return self._store.diagnostic_export(session_id)
