from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .durable_payload import durable_json_dumps

LOGGER = logging.getLogger(__name__)


class EmaCandidateSubmitResult(str, Enum):
    QUEUED = "queued"
    COALESCED = "coalesced"
    OVERFLOW_DROPPED = "overflow_dropped"
    STOPPED = "stopped"
    OVERSIZE_DROPPED = "oversize_dropped"


@dataclass(frozen=True)
class _Candidate:
    camera_id: str
    captured_at: float
    bucket: int
    payload_json: str


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def compact_ema_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only evidence required to replay route qualification safely."""
    features = payload.get("features")
    features = features if isinstance(features, dict) else {}
    regions: list[list[float]] = []
    raw_regions = features.get("motion_regions")
    if isinstance(raw_regions, (list, tuple)):
        for raw in raw_regions[-12:]:
            if not isinstance(raw, (list, tuple)) or len(raw) != 4:
                continue
            values = [_finite_float(value, float("nan")) for value in raw]
            if not all(math.isfinite(value) for value in values):
                continue
            regions.append([max(0.0, min(1.0, value)) for value in values])
    compact_features: dict[str, Any] = {"motion_regions": regions}
    for key in ("motion_region_track_id", "primary_motion_source"):
        value = features.get(key)
        if isinstance(value, (str, int, float, bool)) and (
            not isinstance(value, float) or math.isfinite(value)
        ):
            compact_features[key] = value
    return {
        "schema": 1,
        "accepted": bool(payload.get("accepted")),
        "score": _finite_float(payload.get("score")),
        "threshold": _finite_float(payload.get("threshold")),
        "reason": str(payload.get("reason") or "qualified")[:96],
        "frame_count": max(0, int(payload.get("frame_count") or 0)),
        "features": compact_features,
        "telemetry": {},
    }


class EmaRouteCandidateCache:
    """Bounded asynchronous restart cache isolated from security job writes."""

    def __init__(
        self,
        path: Path,
        *,
        legacy_jobs_path: Path | None = None,
        capacity: int = 4096,
        per_camera_capacity: int = 256,
        batch_size: int = 64,
        retention_seconds: float = 600.0,
        coalesce_seconds: float = 0.5,
        maximum_payload_bytes: int = 16 * 1024,
    ) -> None:
        self.path = Path(path)
        self.legacy_jobs_path = Path(legacy_jobs_path) if legacy_jobs_path else None
        self.capacity = max(1, int(capacity))
        self.per_camera_capacity = max(1, int(per_camera_capacity))
        self.batch_size = max(1, int(batch_size))
        self.retention_seconds = max(300.0, float(retention_seconds))
        self.coalesce_seconds = max(0.05, float(coalesce_seconds))
        self.maximum_payload_bytes = max(1024, int(maximum_payload_bytes))
        self._condition = threading.Condition()
        self._pending: dict[str, deque[_Candidate]] = {}
        self._round_robin: deque[str] = deque()
        self._pending_count = 0
        self._accepting = False
        self._draining = False
        self._drain_deadline = 0.0
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._status: dict[str, Any] = {
            "alive": False,
            "degraded": False,
            "queue_depth": 0,
            "queue_high_water": 0,
            "submitted": 0,
            "coalesced": 0,
            "persisted": 0,
            "batches": 0,
            "busy_retries": 0,
            "overflow_drops": 0,
            "oversize_drops": 0,
            "write_error_drops": 0,
            "shutdown_drops": 0,
            "retention_removed": 0,
            "maximum_persist_lag_seconds": 0.0,
            "last_success_at": None,
            "last_error": "",
            "last_error_at": None,
            "per_camera_overflow": {},
        }

    def start(self, timeout: float = 5.0) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._accepting = True
            self._thread = threading.Thread(
                target=self._run,
                name="ema-route-candidate-cache",
                daemon=False,
            )
            self._thread.start()
        if not self._ready.wait(timeout=max(0.1, float(timeout))):
            self.close_admission()
            raise RuntimeError("EMA route candidate cache did not start")
        if self._startup_error is not None:
            raise RuntimeError("EMA route candidate cache failed to start") from self._startup_error

    def submit(
        self,
        camera_id: str,
        captured_at: float,
        payload: dict[str, Any],
    ) -> EmaCandidateSubmitResult:
        captured_at = float(captured_at)
        compact = compact_ema_candidate(payload)
        payload_json = durable_json_dumps(compact, sort_keys=True)
        if len(payload_json.encode("utf-8")) > self.maximum_payload_bytes:
            with self._condition:
                self._status["oversize_drops"] += 1
            return EmaCandidateSubmitResult.OVERSIZE_DROPPED
        candidate = _Candidate(
            camera_id=str(camera_id),
            captured_at=captured_at,
            bucket=int(captured_at / self.coalesce_seconds),
            payload_json=payload_json,
        )
        with self._condition:
            if not self._accepting:
                return EmaCandidateSubmitResult.STOPPED
            queue_for_camera = self._pending.setdefault(candidate.camera_id, deque())
            if queue_for_camera and queue_for_camera[-1].bucket == candidate.bucket:
                queue_for_camera[-1] = candidate
                self._status["coalesced"] += 1
                self._condition.notify()
                return EmaCandidateSubmitResult.COALESCED
            dropped = False
            if len(queue_for_camera) >= self.per_camera_capacity:
                queue_for_camera.popleft()
                self._pending_count -= 1
                self._record_overflow_locked(candidate.camera_id)
                dropped = True
            if self._pending_count >= self.capacity:
                victim_id = max(
                    self._pending,
                    key=lambda key: len(self._pending[key]),
                    default=candidate.camera_id,
                )
                victim = self._pending.get(victim_id)
                if victim:
                    victim.popleft()
                    self._pending_count -= 1
                    self._record_overflow_locked(victim_id)
                    dropped = True
            was_empty = not queue_for_camera
            queue_for_camera.append(candidate)
            self._pending_count += 1
            if was_empty and candidate.camera_id not in self._round_robin:
                self._round_robin.append(candidate.camera_id)
            self._status["submitted"] += 1
            self._update_queue_status_locked()
            self._condition.notify()
            return (
                EmaCandidateSubmitResult.OVERFLOW_DROPPED
                if dropped
                else EmaCandidateSubmitResult.QUEUED
            )

    def between(
        self,
        camera_id: str,
        start_at: float,
        end_at: float,
        *,
        limit: int = 4096,
    ) -> list[tuple[float, dict[str, Any]]]:
        with self._connect(timeout=2.0) as connection:
            rows = connection.execute(
                "select captured_at, payload_json from ema_route_candidates "
                "where camera_id = ? and captured_at >= ? and captured_at <= ? "
                "order by captured_at desc limit ?",
                (
                    str(camera_id),
                    float(start_at),
                    float(end_at),
                    max(1, min(int(limit), 4096)),
                ),
            ).fetchall()
        restored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                restored.append((float(row["captured_at"]), payload))
        return restored

    def close_admission(self) -> None:
        with self._condition:
            self._accepting = False
            self._condition.notify_all()

    def close(self, timeout: float = 2.0) -> None:
        timeout = max(0.0, float(timeout))
        with self._condition:
            self._accepting = False
            self._draining = True
            self._drain_deadline = time.monotonic() + timeout
            self._condition.notify_all()
            thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout + 0.25)
        if thread.is_alive():
            raise RuntimeError("EMA route candidate cache did not stop within deadline")
        with self._condition:
            self._thread = None

    def status(self) -> dict[str, Any]:
        with self._condition:
            result = dict(self._status)
            result["per_camera_overflow"] = dict(
                self._status["per_camera_overflow"]
            )
            result["queue_depth"] = self._pending_count
            oldest = min(
                (queue[0].captured_at for queue in self._pending.values() if queue),
                default=0.0,
            )
            result["oldest_age_seconds"] = (
                max(0.0, time.time() - oldest) if oldest else 0.0
            )
            thread = self._thread
            result["alive"] = bool(thread is not None and thread.is_alive())
            batches = int(result["batches"])
            result["average_batch_size"] = (
                round(int(result["persisted"]) / batches, 2) if batches else 0.0
            )
            return result

    def _connect(self, *, timeout: float) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=timeout)
        connection.row_factory = sqlite3.Row
        connection.execute(f"pragma busy_timeout = {max(1, int(timeout * 1000))}")
        connection.execute("pragma synchronous = full")
        return connection

    def _run(self) -> None:
        try:
            connection = self._connect(timeout=0.025)
            try:
                connection.execute("pragma journal_mode = wal")
                self._initialize(connection)
                self._migrate_legacy(connection)
                with self._condition:
                    self._status["alive"] = True
                    self._ready.set()
                self._writer_loop(connection)
            finally:
                connection.close()
        except BaseException as error:
            with self._condition:
                if not self._ready.is_set():
                    self._startup_error = error
                    self._ready.set()
                else:
                    self._mark_error_locked(error)
                self._accepting = False
        finally:
            with self._condition:
                if self._pending_count:
                    self._status["shutdown_drops"] += self._pending_count
                    self._pending.clear()
                    self._round_robin.clear()
                    self._pending_count = 0
                self._status["queue_depth"] = 0
                self._status["alive"] = False
                self._condition.notify_all()

    def _initialize(self, connection: sqlite3.Connection) -> None:
        with connection:
            connection.execute(
                "create table if not exists ema_route_candidates ("
                "camera_id text not null, captured_at real not null, "
                "payload_json text not null, created_at text not null, "
                "primary key(camera_id, captured_at))"
            )
            connection.execute(
                "create index if not exists idx_ema_route_candidates_window "
                "on ema_route_candidates(camera_id, captured_at)"
            )
            connection.execute(
                "create index if not exists idx_ema_route_candidates_age "
                "on ema_route_candidates(captured_at)"
            )

    def _migrate_legacy(self, connection: sqlite3.Connection) -> None:
        legacy = self.legacy_jobs_path
        if legacy is None or legacy == self.path or not legacy.exists():
            return
        cutoff = time.time() - self.retention_seconds
        with sqlite3.connect(legacy, timeout=2.0) as source:
            exists = source.execute(
                "select 1 from sqlite_master where type='table' "
                "and name='ema_route_candidates'"
            ).fetchone()
            if not exists:
                return
            cursor = source.execute(
                "select camera_id, captured_at, payload_json, created_at "
                "from ema_route_candidates where captured_at >= ? "
                "order by captured_at",
                (cutoff,),
            )
            while True:
                rows = cursor.fetchmany(256)
                if not rows:
                    break
                compact_rows: list[tuple[str, float, str, str]] = []
                for camera_id, captured_at, payload_json, created_at in rows:
                    try:
                        payload = json.loads(str(payload_json))
                        if not isinstance(payload, dict):
                            continue
                        compact_json = durable_json_dumps(
                            compact_ema_candidate(payload), sort_keys=True
                        )
                    except (TypeError, ValueError):
                        continue
                    if len(compact_json.encode("utf-8")) > self.maximum_payload_bytes:
                        continue
                    compact_rows.append(
                        (str(camera_id), float(captured_at), compact_json, str(created_at))
                    )
                if compact_rows:
                    with connection:
                        connection.executemany(
                            "insert or replace into ema_route_candidates "
                            "(camera_id,captured_at,payload_json,created_at) values (?,?,?,?)",
                            compact_rows,
                        )
            source.execute("drop table ema_route_candidates")
            source.commit()

    def _writer_loop(self, connection: sqlite3.Connection) -> None:
        last_prune = 0.0
        while True:
            with self._condition:
                while not self._pending_count and not self._draining:
                    self._condition.wait(timeout=0.1)
                if self._draining and not self._pending_count:
                    return
                if self._draining and time.monotonic() >= self._drain_deadline:
                    return
                batch = self._take_batch_locked()
            if not batch:
                continue
            retry_started = time.monotonic()
            delay = 0.025
            while True:
                try:
                    now, now_iso, removed, last_prune = self._commit_batch(
                        connection, batch, last_prune
                    )
                    with self._condition:
                        recovered = bool(self._status["degraded"])
                        self._status["persisted"] += len(batch)
                        self._status["batches"] += 1
                        self._status["retention_removed"] += removed
                        self._status["last_success_at"] = now_iso
                        self._status["maximum_persist_lag_seconds"] = max(
                            float(self._status["maximum_persist_lag_seconds"]),
                            max(0.0, now - min(item.captured_at for item in batch)),
                        )
                        self._status["degraded"] = False
                        self._status["last_error"] = ""
                    if recovered:
                        LOGGER.info("EMA route candidate cache recovered")
                    break
                except sqlite3.OperationalError as error:
                    busy = "locked" in str(error).lower() or "busy" in str(error).lower()
                    if not busy:
                        self._record_write_error(batch, error)
                        raise
                    with self._condition:
                        self._status["busy_retries"] += 1
                        self._mark_error_locked(error)
                    if time.monotonic() - retry_started >= 2.0:
                        self._requeue_front(batch)
                        break
                    if self._draining and time.monotonic() >= self._drain_deadline:
                        self._requeue_front(batch)
                        return
                    self._wait_for_retry(delay)
                    delay = min(0.5, delay * 2.0)
                except BaseException as error:
                    self._record_write_error(batch, error)
                    raise

    def _commit_batch(
        self,
        connection: sqlite3.Connection,
        batch: list[_Candidate],
        last_prune: float,
    ) -> tuple[float, str, int, float]:
        now_iso = datetime.now(timezone.utc).isoformat()
        with connection:
            connection.executemany(
                "insert or replace into ema_route_candidates "
                "(camera_id,captured_at,payload_json,created_at) values (?,?,?,?)",
                [
                    (item.camera_id, item.captured_at, item.payload_json, now_iso)
                    for item in batch
                ],
            )
            now = time.time()
            if now - last_prune >= 60.0:
                cursor = connection.execute(
                    "delete from ema_route_candidates where captured_at < ?",
                    (now - self.retention_seconds,),
                )
                removed = max(0, int(cursor.rowcount or 0))
                last_prune = now
            else:
                removed = 0
        return now, now_iso, removed, last_prune

    def _wait_for_retry(self, delay: float) -> None:
        with self._condition:
            wait_for = max(0.0, float(delay))
            if self._draining:
                wait_for = min(
                    wait_for,
                    max(0.0, self._drain_deadline - time.monotonic()),
                )
            if wait_for:
                self._condition.wait(timeout=wait_for)

    def _record_write_error(
        self, batch: list[_Candidate], error: BaseException
    ) -> None:
        with self._condition:
            self._status["write_error_drops"] += len(batch)
            self._mark_error_locked(error)

    def _take_batch_locked(self) -> list[_Candidate]:
        batch: list[_Candidate] = []
        while self._round_robin and len(batch) < self.batch_size:
            camera_id = self._round_robin.popleft()
            items = self._pending.get(camera_id)
            if not items:
                self._pending.pop(camera_id, None)
                continue
            batch.append(items.popleft())
            self._pending_count -= 1
            if items:
                self._round_robin.append(camera_id)
            else:
                self._pending.pop(camera_id, None)
        self._update_queue_status_locked()
        return batch

    def _requeue_front(self, batch: list[_Candidate]) -> None:
        with self._condition:
            for item in reversed(batch):
                queue_for_camera = self._pending.setdefault(item.camera_id, deque())
                was_empty = not queue_for_camera
                queue_for_camera.appendleft(item)
                self._pending_count += 1
                if was_empty and item.camera_id not in self._round_robin:
                    self._round_robin.appendleft(item.camera_id)
            self._update_queue_status_locked()
            self._condition.wait(timeout=0.05)

    def _record_overflow_locked(self, camera_id: str) -> None:
        self._status["overflow_drops"] += 1
        per_camera = self._status["per_camera_overflow"]
        per_camera[camera_id] = int(per_camera.get(camera_id, 0)) + 1

    def _update_queue_status_locked(self) -> None:
        self._status["queue_depth"] = self._pending_count
        self._status["queue_high_water"] = max(
            int(self._status["queue_high_water"]), self._pending_count
        )

    def _mark_error_locked(self, error: BaseException) -> None:
        newly_degraded = not bool(self._status["degraded"])
        self._status["degraded"] = True
        self._status["last_error"] = f"{type(error).__name__}: {error}"[:512]
        self._status["last_error_at"] = datetime.now(timezone.utc).isoformat()
        if newly_degraded:
            LOGGER.warning(
                "EMA route candidate cache degraded; restart replay evidence is being retained for retry: %s",
                error,
            )
