from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .appearance_index import AppearanceIndex
from .config import ObjectTrackingConfig
from .incident_utils import event_snapshot_path


LOGGER = logging.getLogger(__name__)


class AppearanceEncoder(Protocol):
    def supports_label(self, label: str) -> bool: ...
    def embed_for_label(self, label: str, crop: np.ndarray) -> np.ndarray: ...
    def model_identity_for_label(self, label: str) -> dict[str, Any] | None: ...


class DeferredAppearanceBackfill:
    """Durably recover ReID evidence when full multi-frame tracking did not."""

    def __init__(
        self,
        database_path: Path,
        storage_dir: Path,
        config: ObjectTrackingConfig,
        event_store: Any,
        index: AppearanceIndex,
        encoder: AppearanceEncoder,
    ) -> None:
        self.database_path = Path(database_path)
        self.storage_dir = Path(storage_dir)
        self.config = config
        self.event_store = event_store
        self.index = index
        self.encoder = encoder
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma busy_timeout = 10000")
        connection.execute("pragma foreign_keys = on")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                create table if not exists appearance_backfill_jobs (
                    event_id integer primary key,
                    camera_id text not null,
                    state text not null,
                    reason text not null default '',
                    attempts integer not null default 0,
                    indexed_count integer not null default 0,
                    available_at real not null,
                    created_at text not null,
                    updated_at text not null,
                    foreign key(event_id) references events(id) on delete cascade
                )
                """
            )
            connection.execute(
                "create index if not exists idx_appearance_backfill_ready on appearance_backfill_jobs(state, available_at)"
            )
            connection.execute(
                """
                delete from appearance_backfill_jobs
                where not exists (
                    select 1 from events where events.id = appearance_backfill_jobs.event_id
                )
                """
            )
            connection.execute(
                "update appearance_backfill_jobs set state = 'queued', reason = 'resumed after restart' where state = 'running'"
            )

    def start(self) -> None:
        if not self.config.deferred_reid_enabled or not self.config.appearance_reid_enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="survng-reid-backfill",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=10.0)
        self._thread = None

    def enqueue(self, event_id: int, camera_id: str, *, delay_seconds: float | None = None) -> bool:
        if not self.config.deferred_reid_enabled or not self.config.appearance_reid_enabled:
            return False
        now = datetime.now(timezone.utc).isoformat()
        available_at = time.time() + (
            self.config.deferred_reid_delay_seconds
            if delay_seconds is None
            else max(0.0, float(delay_seconds))
        )
        with self._connect() as connection:
            connection.execute(
                """
                insert into appearance_backfill_jobs (
                    event_id, camera_id, state, available_at, created_at, updated_at
                ) values (?, ?, 'queued', ?, ?, ?)
                on conflict(event_id) do update set
                    camera_id = excluded.camera_id,
                    state = case
                        when appearance_backfill_jobs.state in ('completed', 'skipped')
                        then appearance_backfill_jobs.state else 'queued' end,
                    available_at = case
                        when appearance_backfill_jobs.state in ('completed', 'skipped')
                        then appearance_backfill_jobs.available_at else excluded.available_at end,
                    updated_at = excluded.updated_at
                """,
                (int(event_id), str(camera_id), available_at, now, now),
            )
        self._wake.set()
        return True

    def _claim(self) -> sqlite3.Row | None:
        with self._connect() as connection:
            connection.execute("begin immediate")
            row = connection.execute(
                """
                select * from appearance_backfill_jobs
                where state = 'queued' and available_at <= ?
                order by available_at, event_id limit 1
                """,
                (time.time(),),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                update appearance_backfill_jobs
                set state = 'running', attempts = attempts + 1, updated_at = ?
                where event_id = ?
                """,
                (datetime.now(timezone.utc).isoformat(), int(row["event_id"])),
            )
            return row

    def _finish(self, event_id: int, state: str, reason: str, indexed_count: int = 0) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                update appearance_backfill_jobs
                set state = ?, reason = ?, indexed_count = ?, updated_at = ?
                where event_id = ?
                """,
                (
                    state,
                    str(reason)[:500],
                    max(0, int(indexed_count)),
                    datetime.now(timezone.utc).isoformat(),
                    int(event_id),
                ),
            )

    def _retry(self, event_id: int, reason: str, attempts: int) -> None:
        if attempts >= 3:
            self._finish(event_id, "failed", reason)
            return
        with self._connect() as connection:
            connection.execute(
                """
                update appearance_backfill_jobs
                set state = 'queued', reason = ?, available_at = ?, updated_at = ?
                where event_id = ?
                """,
                (
                    str(reason)[:500],
                    time.time() + 30.0 * max(1, attempts),
                    datetime.now(timezone.utc).isoformat(),
                    int(event_id),
                ),
            )

    def process_event(self, event_id: int) -> tuple[str, int, str]:
        if self.index.has_event(event_id):
            return ("skipped", 0, "multi-frame appearance evidence already exists")
        event = self.event_store.get(int(event_id))
        if event is None:
            return ("failed", 0, "event no longer exists")
        try:
            snapshot_path = event_snapshot_path(self.storage_dir, event)
        except (FileNotFoundError, PermissionError) as exc:
            return ("failed", 0, str(exc))
        frame = cv2.imread(str(snapshot_path), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return ("failed", 0, "snapshot could not be decoded")
        try:
            objects = json.loads(str(event.get("objects_json") or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return ("failed", 0, "event object metadata is invalid")
        if not isinstance(objects, list):
            return ("failed", 0, "event object metadata is invalid")

        frame_height, frame_width = frame.shape[:2]
        created_at = str(event.get("created_at") or datetime.now(timezone.utc).isoformat())
        records: list[dict[str, Any]] = []
        for position, detected in enumerate(objects):
            if not isinstance(detected, dict) or detected.get("incident_eligible") is False:
                continue
            label = str(detected.get("label") or "").strip().lower()
            if not label or not self.encoder.supports_label(label):
                continue
            box = detected.get("box")
            if not isinstance(box, dict):
                continue
            try:
                source_width = max(1, int(detected.get("detection_frame_width") or frame_width))
                source_height = max(1, int(detected.get("detection_frame_height") or frame_height))
                x1 = int(float(box["x1"]) * frame_width / source_width)
                y1 = int(float(box["y1"]) * frame_height / source_height)
                x2 = int(float(box["x2"]) * frame_width / source_width)
                y2 = int(float(box["y2"]) * frame_height / source_height)
            except (KeyError, TypeError, ValueError):
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)
            pixel_area = max(0, x2 - x1) * max(0, y2 - y1)
            if pixel_area < self.config.deferred_reid_min_crop_pixels:
                continue
            crop = frame[y1:y2, x1:x2]
            identity = self.encoder.model_identity_for_label(label)
            if crop.size == 0 or identity is None:
                continue
            try:
                embedding = self.encoder.embed_for_label(label, crop)
            except Exception as exc:
                LOGGER.warning("deferred ReID failed for event %d %s: %s", event_id, label, exc)
                continue
            quality = min(1.0, max(0.0, float(detected.get("snapshot_quality_score") or 0.5)))
            records.append({
                **identity,
                "embedding": embedding,
                "label": label,
                "track_id": 100_000 + position,
                "observation_count": 1,
                "quality": quality,
                "match_threshold": min(1.0, float(identity["match_threshold"]) + 0.04),
                "first_seen": created_at,
                "last_seen": created_at,
                "created_at": created_at,
                "source": "snapshot_backfill",
            })
        if not records:
            return ("skipped", 0, "no eligible ReID crop in saved snapshot")
        indexed = self.index.append_event(
            int(event_id),
            str(event.get("camera_id") or ""),
            records,
        )
        if indexed <= 0 and self.index.has_event(event_id):
            return ("skipped", 0, "appearance evidence was indexed concurrently")
        return ("completed", indexed, f"indexed {indexed} snapshot appearance vector(s)")

    def _run(self) -> None:
        interval = max(0.5, 60.0 / max(1, self.config.deferred_reid_rate_per_minute))
        while not self._stop.is_set():
            job = self._claim()
            if job is None:
                self._wake.clear()
                self._wake.wait(timeout=1.0)
                continue
            event_id = int(job["event_id"])
            attempts = int(job["attempts"] or 0) + 1
            indexed = 0
            try:
                state, indexed, reason = self.process_event(event_id)
                if state == "failed":
                    self._retry(event_id, reason, attempts)
                else:
                    self._finish(event_id, state, reason, indexed)
            except Exception as exc:
                LOGGER.exception("deferred appearance backfill crashed for event %d", event_id)
                self._retry(event_id, str(exc), attempts)
            if indexed > 0:
                self._stop.wait(interval)

    def status(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "select state, count(*) as count from appearance_backfill_jobs group by state"
            ).fetchall()
            latest = connection.execute(
                "select * from appearance_backfill_jobs order by updated_at desc limit 1"
            ).fetchone()
        counts = {str(row["state"]): int(row["count"] or 0) for row in rows}
        return {
            "enabled": bool(self.config.deferred_reid_enabled and self.config.appearance_reid_enabled),
            "running": bool(self._thread is not None and self._thread.is_alive()),
            "counts": counts,
            "latest": dict(latest) if latest is not None else None,
        }
