"""Application-owned incident notifications, independent of transport availability."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .incident_payload import IncidentPayloadBuilder
from .incident_utils import (
    DEFAULT_INCIDENT_GAP_SECONDS,
    event_epoch,
    stable_incident_id,
)

LOGGER = logging.getLogger(__name__)


class IncidentLifecycle(IncidentPayloadBuilder):
    """Serialize revisions and retain recovery state on the local database volume.

    Shutdown suspends settlement rather than falsely completing active incidents.
    Completed groups remain refreshable for late identity/cover corrections.
    """

    def __init__(self, publish: Callable[[dict], None], path: Path | None = None) -> None:
        self._publish = publish
        self._path = path
        self._lock = threading.RLock()
        self._groups: OrderedDict[str, dict] = OrderedDict()
        self._timers: dict[str, threading.Timer] = {}
        self._running = False
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            if data.get("version") != 1:
                raise ValueError("unsupported incident notification journal version")
            for group in data["groups"]:
                group["events"] = {int(key): value for key, value in group["events"].items()}
                self._groups[group["payload"]["incident_id"]] = group

    def start(self) -> None:
        with self._lock:
            self._running = True
            for key, group in self._groups.items():
                if group["payload"]["state"] != "complete":
                    self._schedule(key, group)

    def close(self) -> None:
        with self._lock:
            self._running = False
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [deepcopy(group["payload"]) for group in self._groups.values()]

    def get(self, incident_id: str) -> dict | None:
        with self._lock:
            group = self._groups.get(incident_id)
            return deepcopy(group["payload"]) if group else None

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            os.chmod(temporary, 0o600)
            json.dump({"version": 1, "groups": list(self._groups.values())}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self._path)

    def _schedule(self, key: str, group: dict) -> None:
        old = self._timers.pop(key, None)
        if old:
            old.cancel()
        timer = threading.Timer(max(0.01, group["settle_at"] - time.time()), self._settle, args=(key,))
        timer.daemon = True
        self._timers[key] = timer
        timer.start()

    def _emit(self, key: str, group: dict, state: str) -> None:
        previous = group.get("payload", {})
        payload = self._incident_payload(group, state)
        revision = int(previous.get("revision", 0)) + 1
        now = datetime.now(timezone.utc).isoformat()
        names = payload["people"] + [
            label for label in payload["classes"]
            if not payload["people"] or label.lower() != "person"
        ]
        subject = ", ".join(names) if names else "Motion"
        summary = f"{subject[:1].upper() + subject[1:]} detected at {payload['camera_name']}."
        representative = group["events"][payload["representative_event_id"]]
        image_available = bool(representative.get("snapshot_path"))
        payload.update(
            incident_id=key, incident_key=key.removeprefix("incident-"),
            schema_version=2, revision=revision, updated_at=now,
            last_activity_at=payload["ended_at"],
            completed_at=(previous.get("completed_at") or now) if state == "complete" else None,
            title=f"{payload['camera_name']} — activity", summary=summary,
            image_available=image_available,
            image_revision=revision,
            snapshot_url=payload["snapshot_url"] if image_available else None,
            initial_event_id=previous.get("initial_event_id") or payload["representative_event_id"],
            initial_image_available=previous.get("initial_image_available", image_available),
            trigger_source=str(representative.get("trigger_source") or ""),
        )
        payload["changed_fields"] = [
            name for name in ("state", "classes", "objects", "zones", "identities", "summary", "representative_event_id")
            if previous.get(name) != payload.get(name)
        ]
        # Event covers can be replaced at the same source URL. Every evidence
        # revision invalidates the image, including same-event cover promotion.
        if image_available:
            payload["changed_fields"].append("image")
        for item in payload["objects"]:
            item["observation_count"] = item["count"]
        group["payload"] = payload
        self._groups[key] = group
        self._groups.move_to_end(key)
        # Never evict an active incident just to meet the completed-history cap.
        completed = [item for item, value in self._groups.items() if value["payload"]["state"] == "complete"]
        for old in completed[:-256]:
            self._groups.pop(old)
        self._save()
        self._publish(deepcopy(payload))

    def track_incident(self, event: dict, camera_name: str, base_path: str = "", allow_new: bool = True) -> None:
        camera_id = str(event.get("camera_id") or "")
        event_id = int(event.get("id") or 0)
        if not camera_id or not event_id or not event.get("created_at"):
            return
        with self._lock:
            if not self._running:
                return
            # Prefer the original group for refinements, even after settlement.
            match = next(((key, value) for key, value in reversed(self._groups.items())
                          if value["camera_id"] == camera_id and event_id in value["events"]), None)
            if match is None and not allow_new:
                return
            if match is None:
                match = next(((key, value) for key, value in reversed(self._groups.items())
                              if value["camera_id"] == camera_id and value["payload"]["state"] != "complete"), None)
                if match and event_epoch(event) - match[1]["last_epoch"] > DEFAULT_INCIDENT_GAP_SECONDS:
                    self._emit(*match, "complete")
                    match = None
            if match is None:
                group = {"camera_id": camera_id, "camera_name": camera_name or camera_id,
                         "base_path": base_path, "events": {}, "last_epoch": event_epoch(event)}
                key = stable_incident_id(camera_id, event_id)
                state = "new"
            else:
                key, group = match
                state = "complete" if group["payload"]["state"] == "complete" else "updated"
            group["events"][event_id] = deepcopy(event)
            group["camera_name"] = camera_name or camera_id
            group["base_path"] = base_path
            group["last_epoch"] = max(group["last_epoch"], event_epoch(event))
            # Refinements do not extend the activity window.
            if allow_new or "settle_at" not in group:
                group["settle_at"] = time.time() + DEFAULT_INCIDENT_GAP_SECONDS
            self._emit(key, group, state)
            if state != "complete":
                self._schedule(key, group)

    def _settle(self, key: str) -> None:
        with self._lock:
            group = self._groups.get(key)
            if not self._running or group is None or group["payload"]["state"] == "complete":
                return
            if group["settle_at"] > time.time():
                self._schedule(key, group)
                return
            self._timers.pop(key, None)
            try:
                self._emit(key, group, "complete")
            except Exception:
                LOGGER.exception("failed to publish incident completion")
                group["settle_at"] = time.time() + 5
                # Permit a retry if persistence or delivery failed.
                group["payload"]["state"] = "updated"
                self._schedule(key, group)
