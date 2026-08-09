"""Admission and normalization of external camera motion notifications."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol

from .motion_decisions import priority_motion_topic
from .motion_events import MotionEventCoordinator, MotionTrigger
from .domain_events import MotionObserved


class MotionIngressQualification(Protocol):
    def settings(self) -> tuple[str, str, int]: ...
    def observe_event(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        received_at: float,
    ) -> None: ...


class MotionIngressState(Protocol):
    def begin_ingress(self) -> int | None: ...
    def end_ingress(self, generation: int) -> None: ...
    def wait_ingress_idle(self, timeout: float) -> bool: ...
    def ingress_in_flight(self) -> int: ...
    def publish_event(self, event_type: str, payload: dict[str, Any]) -> None: ...
    def set_last_motion_at(self, value: str) -> None: ...
    def increment_stat(self, name: str, amount: int = 1) -> None: ...


class MotionEventIngressService:
    """Convert external notifications into evidence and queued typed triggers."""

    def __init__(
        self,
        *,
        camera_id: str,
        events: MotionEventCoordinator,
        qualification: MotionIngressQualification,
        state: MotionIngressState,
        epoch_now: Callable[[], float] = time.time,
    ) -> None:
        self.camera_id = camera_id
        self.events = events
        self.qualification = qualification
        self.state = state
        self.epoch_now = epoch_now

    def handle(
        self,
        topic: str = "manual",
        message: str = "",
        event_at: datetime | None = None,
    ) -> None:
        generation = self.state.begin_ingress()
        if generation is None:
            return
        try:
            received_at = self.epoch_now()
            receipt_time = datetime.fromtimestamp(received_at, timezone.utc)
            self.state.set_last_motion_at(receipt_time.isoformat())
            normalized_event_at = self._utc(event_at or receipt_time)
            normalized_topic = topic.lower()
            manual = normalized_topic.startswith("manual")

            self.qualification.observe_event(
                topic, message, normalized_event_at, received_at
            )
            if self.qualification.settings()[0] == "adaptive" and not manual:
                # Camera notices remain diagnostic evidence in visual-trigger mode,
                # but cannot create object-detection jobs.
                return
            if priority_motion_topic(topic):
                self.events.remember_priority(received_at)
            if not manual:
                self.events.remember_camera_motion(received_at)

            self.state.publish_event(
                "motion",
                MotionObserved(
                    camera_id=self.camera_id,
                    timestamp=normalized_event_at.isoformat(),
                    source="manual" if manual else "onvif",
                ).to_payload(),
            )
            self.enqueue(MotionTrigger(
                topic=topic,
                message=message,
                event_at=normalized_event_at,
                received_at=received_at,
            ))
        finally:
            self.state.end_ingress(generation)

    def wait_idle(self, timeout: float) -> bool:
        """Wait until every callback admitted for this generation has returned."""
        return self.state.wait_ingress_idle(timeout)

    def in_flight(self) -> int:
        return self.state.ingress_in_flight()

    def enqueue(
        self,
        trigger: MotionTrigger | Mapping[str, Any],
        *,
        evict_oldest: bool = True,
    ) -> bool:
        return self.events.enqueue(
            trigger,
            evict_oldest=evict_oldest,
            on_trigger=lambda name: self.state.increment_stat(name, 1),
            on_drop=lambda name: self.state.increment_stat(name, 1),
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
