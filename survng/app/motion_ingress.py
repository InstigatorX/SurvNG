"""Admission and normalization of external camera motion notifications."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .motion_decisions import priority_motion_topic
from .motion_events import MotionEventCoordinator, MotionTrigger


class MotionEventIngressService:
    """Convert external notifications into evidence and queued typed triggers."""

    def __init__(
        self,
        *,
        camera_id: str,
        events: MotionEventCoordinator,
        accepting: Callable[[], bool],
        detection_enabled: Callable[[], bool],
        configured_mode: Callable[[], str],
        observe_event: Callable[[str, str, datetime, float], None],
        publish_event: Callable[[str, dict[str, Any]], None],
        set_last_motion_at: Callable[[str], None],
        increment_stat: Callable[[str, int], None],
        epoch_now: Callable[[], float] = time.time,
    ) -> None:
        self.camera_id = camera_id
        self.events = events
        self.accepting = accepting
        self.detection_enabled = detection_enabled
        self.configured_mode = configured_mode
        self.observe_event = observe_event
        self.publish_event = publish_event
        self.set_last_motion_at = set_last_motion_at
        self.increment_stat = increment_stat
        self.epoch_now = epoch_now

    def handle(
        self,
        topic: str = "manual",
        message: str = "",
        event_at: datetime | None = None,
    ) -> None:
        if not self.accepting() or not self.detection_enabled():
            return
        received_at = self.epoch_now()
        receipt_time = datetime.fromtimestamp(received_at, timezone.utc)
        self.set_last_motion_at(receipt_time.isoformat())
        normalized_event_at = self._utc(event_at or receipt_time)
        normalized_topic = topic.lower()
        manual = normalized_topic.startswith("manual")

        self.observe_event(topic, message, normalized_event_at, received_at)
        if self.configured_mode() == "adaptive" and not manual:
            # Camera notices remain diagnostic evidence in visual-trigger mode,
            # but cannot create object-detection jobs.
            return
        if priority_motion_topic(topic):
            self.events.remember_priority(received_at)
        if not manual:
            self.events.remember_camera_motion(received_at)

        self.publish_event("motion", {
            "camera_id": self.camera_id,
            "timestamp": normalized_event_at.isoformat(),
            "source": "manual" if manual else "onvif",
        })
        self.enqueue(MotionTrigger(
            topic=topic,
            message=message,
            event_at=normalized_event_at,
            received_at=received_at,
        ))

    def enqueue(
        self,
        trigger: MotionTrigger | Mapping[str, Any],
        *,
        evict_oldest: bool = True,
    ) -> bool:
        return self.events.enqueue(
            trigger,
            evict_oldest=evict_oldest,
            on_trigger=lambda name: self.increment_stat(name, 1),
            on_drop=lambda name: self.increment_stat(name, 1),
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
