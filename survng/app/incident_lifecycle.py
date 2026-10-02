"""Deliver canonical incident revisions from the event database's durable outbox."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from copy import deepcopy

from .incident_payload import canonical_incident_payload

LOGGER = logging.getLogger(__name__)
LEGACY_OUTBOX_PURGE_BUDGET_SECONDS = 0.2
LEGACY_OUTBOX_PURGE_YIELD_SECONDS = 0.02


class CanonicalIncidentLifecycle:
    """One publication owner; membership and lifecycle belong to the scene store.

    A revision is acknowledged only after publication succeeds. A crash between
    delivery and acknowledgement can replay it; consumers deduplicate using the
    canonical incident ID and revision. Closing never completes an incident.
    """

    def __init__(self, events, publish: Callable[[dict], None], *,
                 replay_interval_seconds: float = 1.0) -> None:
        self._events = events
        self._publish = publish
        self._interval = max(0.1, replay_interval_seconds)
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._next_error_log = 0.0

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._running = True
            self.run_once()
            self._thread = threading.Thread(target=self._run, name="survng-incident-publication", daemon=True)
            self._thread.start()

    def close(self) -> None:
        with self._lifecycle_lock:
            self._running = False
            self._stop.set()
            self._wake.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise RuntimeError("incident publication did not stop; dependencies must remain open")
        with self._lifecycle_lock:
            self._thread = None

    def snapshot(self) -> list[dict]:
        """Recovery state in the same form as the lifecycle events it repairs."""
        return [canonical_incident_payload(item) for item in self._events.list_scene_incident_notifications()]

    def get(self, incident_id: str) -> dict | None:
        item = self._events.scene_incident(incident_id=incident_id)
        return canonical_incident_payload(item) if item else None

    def track_incident(self, event: dict, camera_name: str = "", base_path: str = "",
                       allow_new: bool = True) -> None:
        """Compatibility wakeup after a committed event; never regroup evidence."""
        if self._running:
            self._wake.set()

    def run_once(self) -> int:
        """Replay persisted snapshots, preserving their exact revision and scope."""
        delivered = 0
        with self._lock:
            if self._stop.is_set():
                return delivered
            self._events.settle_scene_incidents()
            self._purge_legacy_outbox()
            for entry in self._events.scene_pending_notifications():
                if self._stop.is_set():
                    break
                try:
                    payload = canonical_incident_payload(deepcopy(entry["payload"]))
                    incident_id = str(entry["incident_id"])
                    revision = int(entry["revision"])
                    if payload["incident_id"] != incident_id or int(payload["revision"]) != revision:
                        raise ValueError("incident outbox snapshot does not match its revision")
                    self._publish(payload)
                    self._events.acknowledge_scene_notification(incident_id, revision)
                    delivered += 1
                except Exception:
                    # Preserve publication order when one revision cannot be
                    # delivered. Failed work remains durable for the next pass.
                    self._log_failure()
                    break
        return delivered

    def _purge_legacy_outbox(self) -> None:
        purge = getattr(self._events, "purge_legacy_scene_notifications", None)
        if not callable(purge):
            return
        deadline = time.monotonic() + LEGACY_OUTBOX_PURGE_BUDGET_SECONDS
        # Small slices with a pause between them so writers can take the lock.
        while purge() and time.monotonic() < deadline:
            if self._stop.wait(LEGACY_OUTBOX_PURGE_YIELD_SECONDS):
                return

    def _log_failure(self) -> None:
        now = time.monotonic()
        if now >= self._next_error_log:
            self._next_error_log = now + 30.0
            LOGGER.exception("incident publication failed; durable revision remains pending")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._interval)
            self._wake.clear()
            try:
                self.run_once()
            except Exception:
                self._log_failure()


# Preserve the import name, without preserving a second grouping implementation.
IncidentLifecycle = CanonicalIncidentLifecycle
