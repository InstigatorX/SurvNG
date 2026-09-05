"""Admission and normalization of external camera motion notifications."""

from __future__ import annotations

import time
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol

from .motion_decisions import priority_motion_topic
from .camera_semantics import camera_semantic_reports
from .motion_events import MotionEventCoordinator, MotionEventTiming, MotionTrigger
from .domain_events import MotionObserved
from .ema_v2 import CameraNotice, EpisodeDecisionReason


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
    def lifecycle_generation(self) -> int: ...


class MotionEventClock:
    """Estimate camera clock offset separately from notification delivery delay."""

    def __init__(self, *, history_size: int = 32, warmup_samples: int = 3) -> None:
        self._deltas: deque[float] = deque(maxlen=max(4, history_size))
        self._warmup_samples = max(2, warmup_samples)
        self._lock = threading.Lock()

    def resolve(
        self,
        camera_event_at: datetime | None,
        received_epoch: float,
    ) -> MotionEventTiming:
        received_at = datetime.fromtimestamp(received_epoch, timezone.utc)
        if camera_event_at is None:
            return MotionEventTiming(
                sampling_at=received_at,
                received_at=received_at,
                selection_reason="camera_time_missing",
            )
        camera_at = MotionEventIngressService._utc(camera_event_at)
        delta = received_epoch - camera_at.timestamp()
        with self._lock:
            prior = sorted(self._deltas)
            baseline = prior[max(0, int(len(prior) * 0.1) - 1)] if prior else None
            if baseline is not None and abs(delta - baseline) > 30.0:
                self._deltas.clear()
                baseline = None
                reason = "camera_clock_discontinuity"
            else:
                reason = "clock_model_warming"
            self._deltas.append(delta)
            if baseline is None and abs(delta) <= 5.0:
                sampling_at = min(camera_at, received_at)
                estimated_offset = 0.0
                delivery_delay = max(0.0, delta)
                reason = (
                    "plausible_camera_time"
                    if camera_at <= received_at
                    else "future_camera_time_clamped"
                )
            elif baseline is not None and len(prior) >= self._warmup_samples:
                estimated_offset = baseline
                delivery_delay = max(0.0, delta - baseline)
                corrected_epoch = camera_at.timestamp() + baseline
                if corrected_epoch > received_epoch:
                    corrected_epoch = received_epoch
                    reason = "camera_clock_corrected_clamped"
                else:
                    reason = "camera_clock_corrected"
                sampling_at = datetime.fromtimestamp(corrected_epoch, timezone.utc)
            else:
                sampling_at = received_at
                estimated_offset = None
                delivery_delay = None
        return MotionEventTiming(
            sampling_at=sampling_at,
            received_at=received_at,
            camera_event_at=camera_at,
            camera_to_receive_delta_seconds=round(delta, 3),
            estimated_clock_offset_seconds=(
                round(estimated_offset, 3) if estimated_offset is not None else None
            ),
            estimated_delivery_delay_seconds=(
                round(delivery_delay, 3) if delivery_delay is not None else None
            ),
            selection_reason=reason,
        )

    def status(self) -> dict[str, float | int | None]:
        with self._lock:
            ordered = sorted(self._deltas)
        baseline = (
            ordered[max(0, int(len(ordered) * 0.1) - 1)]
            if ordered
            else None
        )
        return {
            "samples": len(ordered),
            "estimated_clock_offset_seconds": (
                round(baseline, 3) if baseline is not None else None
            ),
        }


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
        model_labels: Callable[[], list[str]] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.events = events
        self.qualification = qualification
        self.state = state
        self.epoch_now = epoch_now
        self.model_labels = model_labels or (lambda: [])
        self.event_clock = MotionEventClock()

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
            normalized_topic = topic.lower()
            manual = normalized_topic.startswith("manual")
            if manual:
                normalized_event_at = self._utc(event_at or receipt_time)
                event_timing = MotionEventTiming(
                    sampling_at=normalized_event_at,
                    received_at=receipt_time,
                    selection_reason="manual_time",
                )
            else:
                event_timing = self.event_clock.resolve(event_at, received_at)
                normalized_event_at = event_timing.sampling_at

            self.qualification.observe_event(
                topic, message, normalized_event_at, received_at
            )
            if self.qualification.settings()[0] == "adaptive" and not manual:
                # Camera notices remain diagnostic evidence in visual-trigger mode,
                # but cannot create object-detection jobs.
                return
            self.state.publish_event(
                "motion",
                MotionObserved(
                    camera_id=self.camera_id,
                    timestamp=normalized_event_at.isoformat(),
                    source="manual" if manual else "onvif",
                ).to_payload() | {"event_timing": event_timing.to_payload()},
            )
            episode = self.events.episode_controller.observe_camera(
                CameraNotice(
                    camera_id=self.camera_id,
                    event_at=normalized_event_at.timestamp(),
                    observed_monotonic=time.monotonic(),
                    topic=topic,
                    message=message,
                    manual=manual,
                ),
                generation=generation,
            )
            if episode.reason is not EpisodeDecisionReason.REQUEST_RESERVED:
                return
            intent = episode.intent
            if intent is None:
                return
            queued = False
            try:
                semantic_reports = camera_semantic_reports(
                    topic, message, self.model_labels()
                )
                queued = self.enqueue(MotionTrigger(
                    topic=topic,
                    message=message,
                    event_at=normalized_event_at,
                    received_at=received_at,
                    event_timing=event_timing,
                    episode_id=intent.episode_id,
                    detection_intent_id=intent.intent_id,
                    lifecycle_generation=intent.generation,
                    camera_semantics=(
                        {"reports": semantic_reports} if semantic_reports else None
                    ),
                ), evict_oldest=False)
            finally:
                self.events.episode_controller.acknowledge_admission(
                    intent.intent_id,
                    admitted=queued,
                    occurred_monotonic=time.monotonic(),
                )
        finally:
            self.state.end_ingress(generation)

    def wait_idle(self, timeout: float) -> bool:
        """Wait until every callback admitted for this generation has returned."""
        return self.state.wait_ingress_idle(timeout)

    def in_flight(self) -> int:
        return self.state.ingress_in_flight()

    def timing_status(self) -> dict[str, float | int | None]:
        return self.event_clock.status()

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
