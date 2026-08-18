from __future__ import annotations

import threading
import time
import weakref
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .security import redact_secret_text

MAX_EVENTS = 1000
MAX_XML_CHARACTERS = 16_384

_TOPIC_CLASSIFICATIONS = {
    "peopledetect": "person",
    "persondetect": "person",
    "humandetect": "person",
    "vehicledetect": "vehicle",
    "dogcatdetect": "animal",
    "petdetect": "animal",
    "animaldetect": "animal",
    "facedetect": "face",
    "motionalarm": "motion",
    "motion": "motion",
    "cellmotion": "motion",
}


@dataclass(frozen=True, slots=True)
class OnvifInspectorEvent:
    seq: int
    received_at: str
    event_at: str
    camera_id: str
    topic: str
    normalized_topic: str
    classification: str | None
    active: bool | None
    recognized: bool
    changed: bool
    simple_items: tuple[tuple[str, str], ...]
    message_xml: str


class OnvifInspector:
    """In-memory, observational ONVIF event tap.

    This object must never participate in camera control or motion admission.
    It only records already-received notifications for diagnostics.
    """

    def __init__(self, max_events: int = MAX_EVENTS) -> None:
        self._events: deque[OnvifInspectorEvent] = deque(maxlen=max(1, max_events))
        self._topic_states: dict[tuple[str, str], bool] = {}
        self._class_states: dict[tuple[str, str], tuple[bool, str]] = {}
        self._listeners: dict[str, weakref.ReferenceType[Any]] = {}
        self._seq = 0
        self._lock = threading.RLock()

    @staticmethod
    def classify_topic(normalized_topic: str) -> str | None:
        topic = str(normalized_topic or "").strip().lower()
        terminal = topic.rsplit("/", 1)[-1] if topic else ""
        if terminal in _TOPIC_CLASSIFICATIONS:
            return _TOPIC_CLASSIFICATIONS[terminal]
        if "motion" in terminal:
            return "motion"
        return None

    def register_listener(self, camera_id: str, listener: Any) -> None:
        with self._lock:
            self._listeners[str(camera_id)] = weakref.ref(listener)

    def record(
        self,
        *,
        camera_id: str,
        topic: str,
        normalized_topic: str,
        active: bool | None,
        simple_items: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
        message_xml: str = "",
        received_at: datetime | None = None,
        event_at: datetime | None = None,
    ) -> OnvifInspectorEvent:
        camera_id = str(camera_id)
        topic = redact_secret_text(str(topic or ""))[:1024]
        normalized_topic = str(normalized_topic or "").strip().lower()[:1024]
        classification = self.classify_topic(normalized_topic)
        received = received_at or datetime.now(timezone.utc)
        if received.tzinfo is None:
            received = received.replace(tzinfo=timezone.utc)
        received = received.astimezone(timezone.utc)

        event_time = ""
        if event_at is not None:
            if event_at.tzinfo is None:
                event_at = event_at.replace(tzinfo=timezone.utc)
            event_time = event_at.astimezone(timezone.utc).isoformat()

        items = tuple(
            (
                redact_secret_text(str(name))[:256],
                redact_secret_text(str(value))[:1024],
            )
            for name, value in simple_items
        )
        safe_xml = redact_secret_text(str(message_xml or ""))[:MAX_XML_CHARACTERS]

        key = (camera_id, normalized_topic)
        with self._lock:
            previous = self._topic_states.get(key)
            changed = active is not None and previous != active
            if active is not None:
                self._topic_states[key] = bool(active)
                if classification is not None:
                    self._class_states[(camera_id, classification)] = (
                        bool(active),
                        received.isoformat(),
                    )

            self._seq += 1
            event = OnvifInspectorEvent(
                seq=self._seq,
                received_at=received.isoformat(),
                event_at=event_time,
                camera_id=camera_id,
                topic=topic,
                normalized_topic=normalized_topic,
                classification=classification,
                active=active,
                recognized=active is not None,
                changed=changed,
                simple_items=items,
                message_xml=safe_xml,
            )
            self._events.append(event)
            return event

    def events_after(
        self,
        after: int = 0,
        *,
        camera: str = "",
        recognized_only: bool = False,
        changes_only: bool = False,
        limit: int = 250,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 1000))
        with self._lock:
            events = list(self._events)
            next_seq = self._seq

        filtered = [
            event
            for event in events
            if event.seq > after
            and (not camera or event.camera_id == camera)
            and (not recognized_only or event.recognized)
            and (not changes_only or event.changed)
        ]
        filtered = filtered[-limit:]
        return {
            "next": next_seq,
            "events": [self._event_dict(event) for event in filtered],
        }

    @staticmethod
    def _event_dict(event: OnvifInspectorEvent) -> dict[str, Any]:
        payload = asdict(event)
        payload["simple_items"] = [
            {"name": name, "value": value}
            for name, value in event.simple_items
        ]
        return payload

    def state_snapshot(self) -> dict[str, Any]:
        with self._lock:
            listener_refs = dict(self._listeners)
            class_states = dict(self._class_states)
            topic_states = dict(self._topic_states)
            seq = self._seq

        cameras: dict[str, Any] = {}
        camera_ids = set(listener_refs)
        camera_ids.update(camera for camera, _ in class_states)
        camera_ids.update(camera for camera, _ in topic_states)

        for camera_id in sorted(camera_ids):
            listener = listener_refs.get(camera_id)
            instance = listener() if listener is not None else None

            classes = {}
            for (candidate_camera, classification), (active, changed_at) in class_states.items():
                if candidate_camera == camera_id:
                    classes[classification] = {
                        "active": active,
                        "changed_at": changed_at,
                    }

            topics = {}
            for (candidate_camera, topic), active in topic_states.items():
                if candidate_camera == camera_id:
                    topics[topic] = active

            listener_state: dict[str, Any] = {
                "registered": instance is not None,
                "running": False,
                "connected": False,
            }
            if instance is not None:
                listener_state.update(
                    {
                        "running": bool(getattr(instance, "running", False)),
                        "connected": bool(getattr(instance, "connected", False)),
                        "last_event_at": str(getattr(instance, "last_event_at", "") or ""),
                        "last_camera_event_at": str(
                            getattr(instance, "last_camera_event_at", "") or ""
                        ),
                        "last_motion_event_at": str(
                            getattr(instance, "last_motion_event_at", "") or ""
                        ),
                        "last_topic": str(getattr(instance, "last_topic", "") or ""),
                        "last_error": str(getattr(instance, "last_error", "") or ""),
                        "last_poll_error": str(
                            getattr(instance, "last_poll_error", "") or ""
                        ),
                        "subscription_termination_time": str(
                            getattr(instance, "subscription_termination_time", "") or ""
                        ),
                        "subscription_lifetime_seconds": getattr(
                            instance, "subscription_lifetime_seconds", None
                        ),
                        "notifications_received": int(
                            getattr(instance, "notifications_received", 0) or 0
                        ),
                        "motion_events_received": int(
                            getattr(instance, "motion_events_received", 0) or 0
                        ),
                        "inactive_motion_events": int(
                            getattr(instance, "inactive_motion_events", 0) or 0
                        ),
                        "unrecognized_notifications": int(
                            getattr(instance, "unrecognized_notifications", 0) or 0
                        ),
                        "renewal_attempts": int(
                            getattr(instance, "renewal_attempts", 0) or 0
                        ),
                        "renewals": int(getattr(instance, "renewals", 0) or 0),
                        "renewal_errors": int(
                            getattr(instance, "renewal_errors", 0) or 0
                        ),
                        "resubscriptions": int(
                            getattr(instance, "resubscriptions", 0) or 0
                        ),
                        "poll_timeouts": int(
                            getattr(instance, "poll_timeouts", 0) or 0
                        ),
                        "poll_errors": int(
                            getattr(instance, "poll_errors", 0) or 0
                        ),
                    }
                )

            cameras[camera_id] = {
                **listener_state,
                "classes": classes,
                "topics": topics,
            }

        return {
            "seq": seq,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cameras": cameras,
        }

    def clear(self) -> dict[str, int]:
        with self._lock:
            count = len(self._events)
            self._events.clear()
            self._topic_states.clear()
            self._class_states.clear()
            return {"cleared": count, "next": self._seq}


ONVIF_INSPECTOR = OnvifInspector()
