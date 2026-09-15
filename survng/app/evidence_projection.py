"""Recover derived incident views from the event database's evidence outbox."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from .semantic_search import semantic_event_searchable

LOGGER = logging.getLogger("uvicorn.error")


class EvidenceProjection:
    def __init__(
        self,
        events: Any,
        semantic_search_provider: Callable[[], Any],
        state_events: Any,
        refresh_notification: Callable[[str, int], None],
        *,
        poll_seconds: float = 1.0,
        retry_seconds: float = 5.0,
        batch_size: int = 64,
    ) -> None:
        self.events = events
        self.semantic_search_provider = semantic_search_provider
        self.state_events = state_events
        self.refresh_notification = refresh_notification
        self.poll_seconds = max(.1, float(poll_seconds))
        self.retry_seconds = max(self.poll_seconds, float(retry_seconds))
        self.batch_size = max(1, min(256, int(batch_size)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._retry_at: OrderedDict[tuple[int, int, int], float] = OrderedDict()
        self._delivery: OrderedDict[int, tuple[tuple, bool, bool]] = OrderedDict()
        self._next_error_log = 0.0
        self._cursor = 0
        self._next_expiry = 0.0

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="survng-evidence-projection", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise RuntimeError("evidence projection did not stop; dependencies must remain open")
        with self._lifecycle_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    close = stop

    def _remember(self, entries: OrderedDict, key: Any, value: Any) -> None:
        entries[key] = value
        entries.move_to_end(key)
        while len(entries) > self.batch_size * 4:
            entries.popitem(last=False)

    def _semantic_current(self, semantic: Any, event: dict, now: float) -> bool:
        if not semantic.config.enabled:
            return True
        if not semantic_event_searchable(event):
            # Removing searchable evidence does not require an available model.
            # The index checks the authoritative event revision in its transaction.
            semantic.index.delete_event(int(event["id"]), expected_event=event)
            return True
        if callable(getattr(semantic, "projection_current", None)) and semantic.projection_current(event):
            return True
        if callable(getattr(semantic, "projection_pending", None)) and semantic.projection_pending(event):
            return False
        key = (int(event["id"]), int(event.get("evidence_revision") or 0), id(semantic))
        if now < self._retry_at.get(key, 0):
            return False
        # Same-revision submissions reuse a pending token. A failed enqueue or
        # failed encoder attempt is retried; neither is a completed projection.
        self._remember(self._retry_at, key, now + self.retry_seconds)
        semantic.queue_event(event)
        return False

    def run_once(self) -> int:
        """Process one bounded batch; safe to invoke directly in focused tests."""
        with self._run_lock:
            now = time.monotonic()
            if now >= self._next_expiry:
                self._next_expiry = now + 30.0
                for name in ("expire_cover_requirements",):
                    expire = getattr(self.events, name, None)
                    if callable(expire):
                        try:
                            expire()
                        except Exception:
                            self._log_failure()
            rows = self.events.pending_evidence_updates(limit=self.batch_size, after_id=self._cursor)
            if not rows and self._cursor:
                self._cursor = 0
                rows = self.events.pending_evidence_updates(limit=self.batch_size, after_id=0)
            if rows:
                self._cursor = max(int(row["id"]) for row in rows)
            grouped: dict[int, list[dict]] = {}
            acknowledged = 0
            for row in rows:
                if row["kind"] == "cover_required":
                    # The durable cover requirement itself is polled by the
                    # evidence workflow; acknowledging this wakeup loses no work.
                    acknowledged += bool(self.events.acknowledge_evidence_update(int(row["id"])))
                elif row["kind"] in {"evidence_updated", "cover_requirement_updated", "incident_metadata_updated"}:
                    grouped.setdefault(int(row["event_id"]), []).append(row)
                else:
                    raise ValueError(f"unsupported evidence outbox kind: {row['kind']}")
            semantic = self.semantic_search_provider()
            for event_id, updates in grouped.items():
                if self._stop.is_set():
                    break
                try:
                    needs_semantic = any(row["kind"] == "evidence_updated" for row in updates)
                    metadata_token = max(
                        (int(row["id"]) for row in updates if row["kind"] == "incident_metadata_updated"), default=0,
                    )
                    publication_needed = not all(row.get("publication_done", False) for row in updates)
                    current, published = self._project_event(
                        event_id, semantic, needs_semantic, metadata_token, publication_needed,
                    )
                    if publication_needed and published:
                        if not self.events.mark_evidence_publication([int(row["id"]) for row in updates]):
                            continue
                    if not current or not published:
                        continue
                    for row in updates:
                        # A later revision has its own entry; never acknowledge
                        # a broad range containing work this batch did not read.
                        acknowledged += bool(self.events.acknowledge_evidence_update(int(row["id"])))
                except Exception:
                    self._log_failure()
            return acknowledged

    def _project_event(
        self, event_id: int, semantic: Any, needs_semantic: bool,
        metadata_token: int = 0, publication_needed: bool = True,
    ) -> tuple[bool, bool]:
        event = self.events.get(event_id)
        if event is None:
            return True, True
        revision = int(event.get("evidence_revision") or 0)
        semantic_current = not needs_semantic
        if needs_semantic:
            try:
                semantic_current = self._semantic_current(semantic, event, time.monotonic())
            except Exception:
                # Image delivery is independent of a failed derived index.
                # Retain the outbox until indexing can catch up.
                self._log_failure()
        latest = self.events.get(event_id)
        if latest is None:
            return True, True
        if int(latest.get("evidence_revision") or 0) != revision:
            return False, False
        if not publication_needed:
            return semantic_current, True
        requirement = latest.get("cover_requirement") or {}
        delivered_identity, notified, published = self._delivery.get(event_id, ((), False, False))
        metadata_token = max(metadata_token, int(delivered_identity[-1]) if delivered_identity else 0)
        identity = (revision, requirement.get("state"), requirement.get("reason"), requirement.get("attempts"), metadata_token)
        if delivered_identity != identity:
            notified = published = False
        camera_id = str(latest.get("camera_id") or "")
        if not notified:
            self.refresh_notification(camera_id, event_id)
            notified = True
            self._remember(self._delivery, event_id, (identity, True, False))
        if not published:
            message = self.state_events.publish("incident", {
                "event_id": event_id, "camera_id": camera_id,
                "evidence_revision": revision, "updated": True,
                "reason": "evidence_updated" if needs_semantic else "incident_metadata_updated",
                "cover_requirement": latest.get("cover_requirement"),
            })
            if message is None:
                return semantic_current, False  # A closed broker has not delivered this revision.
            self._remember(self._delivery, event_id, (identity, notified, True))
        return semantic_current, True

    def _log_failure(self) -> None:
        now = time.monotonic()
        if now >= self._next_error_log:
            self._next_error_log = now + 30.0
            LOGGER.exception("evidence projection failed; durable updates remain pending")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                self._log_failure()
            self._stop.wait(self.poll_seconds)
