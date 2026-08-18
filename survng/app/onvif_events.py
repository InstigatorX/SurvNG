from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .config import CameraConfig
from .security import redact_secret_text

LOGGER = logging.getLogger("uvicorn.error")
MOTION_WORDS = ("motion", "cellmotion", "person", "vehicle", "animal", "alarm")
RETRY_INITIAL_SECONDS = 2.0
RETRY_MAX_SECONDS = 60.0
POLL_RETRY_SECONDS = 2.0
PULL_TIMEOUT_SECONDS = 5
TRANSPORT_OPERATION_TIMEOUT_SECONDS = 45
STOP_GRACE_SECONDS = PULL_TIMEOUT_SECONDS + 2
STOP_FORCE_SECONDS = 5
MAX_POLL_FAILURES = 6
TIMEOUT_WORDS = ("timed out", "timeout", "read timed out", "operation timed out")
PULLPOINT_NAMESPACE = "http://www.onvif.org/ver10/events/wsdl/PullPointSubscription"
SUBSCRIPTION_MANAGER_BINDING = "{http://www.onvif.org/ver10/events/wsdl}SubscriptionManagerBinding"
SUBSCRIPTION_DURATION = "PT1H"
SUBSCRIPTION_RENEW_MAX_MARGIN_SECONDS = 300.0
SUBSCRIPTION_RENEW_MIN_MARGIN_SECONDS = 5.0
MAX_EVENT_MESSAGE_CHARACTERS = 16_384
EFFECTIVENESS_WINDOW_SECONDS = 3600.0
EFFECTIVENESS_MINIMUM_OBSERVATIONS = 3
EFFECTIVENESS_DEGRADED_MISMATCH_RATIO = 0.8
UNKNOWN_NOTIFICATION_SAMPLE_LIMIT = 5
EFFECTIVENESS_OBSERVATION_LIMIT = 512
PULLMESSAGES_CAPTURE_LIMIT = 2
REOLINK_MOTION_TOPICS = frozenset({
    "videosource/motionalarm",
    "ruleengine/myruledetector/vehicledetect",
    "ruleengine/myruledetector/dogcatdetect",
    "ruleengine/myruledetector/peopledetect",
    "ruleengine/myruledetector/facedetect",
})


@dataclass(frozen=True, slots=True)
class OnvifStopTicket:
    """Identity and resources owned by one listener stop transaction."""

    generation: int
    thread: threading.Thread | None
    stop_event: threading.Event


@dataclass(frozen=True, slots=True)
class _RawOnvifNotification:
    topic: str
    normalized_topic: str
    message_xml: str
    simple_items: tuple[tuple[str, str], ...]


class _PullMessagesResponseCapture:
    def __init__(self) -> None:
        self._responses: deque[str] = deque(maxlen=PULLMESSAGES_CAPTURE_LIMIT)
        self._lock = threading.Lock()

    def ingress(self, envelope: Any, http_headers: Any, operation: Any):
        if str(getattr(operation, "name", "")) != "PullMessages":
            return envelope, http_headers
        with self._lock:
            self._responses.append(OnvifEventListener._stringify_xml_static(envelope))
        return envelope, http_headers

    def take(self) -> str:
        with self._lock:
            return self._responses.popleft() if self._responses else ""


class OnvifEventListener:
    def __init__(
        self,
        camera: CameraConfig,
        on_motion: Callable[[str, str, datetime | None], None],
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self.camera = camera
        self.on_motion = on_motion
        self._cache_dir = cache_dir or Path("survng/storage")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._generation = 0
        self._pending_stop: OnvifStopTicket | None = None
        self._transport: Any = None
        self._transport_generation: int | None = None
        self._subscription_manager: Any = None
        self._subscription_generation: int | None = None
        self._pullmessages_capture: _PullMessagesResponseCapture | None = None
        self.connected = False
        self.last_event_at = ""
        self.last_camera_event_at = ""
        self.last_motion_event_at = ""
        self.last_error = ""
        self.last_topic = ""
        self.last_connected_at = ""
        self.last_poll_success_at = ""
        self.last_poll_error = ""
        self.last_poll_error_at = ""
        self.retry_attempts = 0
        self.poll_timeouts = 0
        self.poll_errors = 0
        self.resubscriptions = 0
        self.notifications_received = 0
        self.motion_events_received = 0
        self.inactive_motion_events = 0
        self.unrecognized_notifications = 0
        self._unknown_notification_samples: deque[dict[str, Any]] = deque(
            maxlen=UNKNOWN_NOTIFICATION_SAMPLE_LIMIT
        )
        self._effectiveness_lock = threading.Lock()
        self.ema_qualified_observations = 0
        self.ema_onvif_matches = 0
        self.ema_without_onvif = 0
        self.last_ema_observation_at = ""
        self.last_ema_onvif_match_at = ""
        self.last_ema_without_onvif_at = ""
        self._ema_effectiveness_window: deque[tuple[float, bool]] = deque(
            maxlen=EFFECTIVENESS_OBSERVATION_LIMIT
        )
        self.callback_errors = 0
        self.renewal_attempts = 0
        self.renewals = 0
        self.renewal_errors = 0
        self.last_renewed_at = ""
        self.subscription_current_time = ""
        self.subscription_termination_time = ""
        self.subscription_lifetime_seconds: float | None = None
        self._subscription_granted_lifetime_seconds: float | None = None
        self._subscription_expires_monotonic: float | None = None

    @property
    def running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if not self.camera.onvif.enabled:
            return
        stopping_thread: threading.Thread | None = None
        with self._lifecycle_lock:
            if self._pending_stop is not None:
                raise RuntimeError(
                    f"ONVIF listener stop is still pending for {self.camera.id}"
                )
            if self._thread is not None and self._thread.is_alive():
                if not self._stop.is_set():
                    return
                stopping_thread = self._thread
        if stopping_thread is not None:
            if stopping_thread is threading.current_thread():
                raise RuntimeError("ONVIF listener cannot restart from its stopping worker")
            stopping_thread.join(timeout=STOP_FORCE_SECONDS)
            if stopping_thread.is_alive():
                raise RuntimeError(
                    f"previous ONVIF generation is still stopping for {self.camera.id}"
                )
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._generation += 1
            generation = self._generation
            stop_event = threading.Event()
            self._stop = stop_event
            thread = threading.Thread(
                target=self._run,
                args=(generation, stop_event),
                name=f"onvif-{self.camera.id}",
                # Some camera SDK calls can ignore a closed HTTP transport.
                # Cleanup below is bounded, so a broken camera must not pin
                # the whole SurvNG process after shutdown completes.
                daemon=True,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                self._stop.set()
                raise

    def stop(self) -> None:
        ticket = self.request_stop()
        self.wait_stopped(STOP_GRACE_SECONDS + STOP_FORCE_SECONDS, ticket)

    def request_stop(self) -> OnvifStopTicket:
        """Signal the listener without waiting for camera I/O to return."""
        with self._lifecycle_lock:
            if self._pending_stop is not None:
                return self._pending_stop
            ticket = OnvifStopTicket(
                generation=self._generation,
                thread=self._thread,
                stop_event=self._stop,
            )
            self._pending_stop = ticket
            # Bind the stop signal to the snapshotted generation atomically;
            # start() cannot replace this event until wait_stopped finalizes it.
            ticket.stop_event.set()
            return ticket

    def wait_stopped(
        self,
        timeout: float,
        ticket: OnvifStopTicket | None = None,
    ) -> bool:
        """Wait within one caller-owned budget, forcing transport closure once."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lifecycle_lock:
            pending = self._pending_stop
            expected = ticket or pending
            if expected is None:
                return self._thread is None or not self._thread.is_alive()
            if pending is not expected:
                return False
            thread = expected.thread
            generation = expected.generation
        if thread is not None:
            # PullMessages has a short bounded timeout. Let the owning worker
            # return from it and send Unsubscribe while its transport is still
            # usable before resorting to forcibly closing the session.
            thread.join(timeout=min(
                STOP_GRACE_SECONDS,
                max(0.0, deadline - time.monotonic()),
            ))
            if thread.is_alive():
                self._close_transport(generation)
                thread.join(timeout=min(
                    STOP_FORCE_SECONDS,
                    max(0.0, deadline - time.monotonic()),
                ))
                if thread.is_alive():
                    LOGGER.error("ONVIF worker did not stop for %s", self.camera.id)
        with self._lifecycle_lock:
            stopped = thread is None or not thread.is_alive()
            if self._thread is thread and stopped:
                self._thread = None
            if generation == self._generation and stopped:
                self.connected = False
            if self._pending_stop is expected and stopped:
                self._pending_stop = None
        if stopped:
            self._close_transport(generation)
        return stopped

    def _run(
        self,
        generation: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        generation = self._generation if generation is None else generation
        stop_event = self._stop if stop_event is None else stop_event
        try:
            self._run_until_stopped(generation, stop_event)
        finally:
            self._unsubscribe(generation)
            self._close_transport(generation)
            current = threading.current_thread()
            with self._lifecycle_lock:
                if self._thread is current:
                    self._thread = None
                    self.connected = False

    def _run_until_stopped(
        self,
        generation: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        generation = self._generation if generation is None else generation
        stop_event = self._stop if stop_event is None else stop_event
        try:
            from onvif import ONVIFCamera
            from zeep import Transport
            from zeep.cache import SqliteCache
        except ImportError:
            LOGGER.warning("onvif-zeep is not installed; ONVIF events disabled")
            self.last_error = "onvif-zeep is not installed"
            return

        class SingleSubscriptionCamera(ONVIFCamera):
            def update_xaddrs(self) -> None:
                self._defer_pullpoint_subscription = True
                try:
                    super().update_xaddrs()
                finally:
                    self._defer_pullpoint_subscription = False

            def create_events_service(self, from_template: bool = True) -> Any:
                if getattr(self, "_defer_pullpoint_subscription", False):
                    raise RuntimeError("PullPoint subscription is managed by SurvNG")
                return super().create_events_service(from_template)

        retry_delay = RETRY_INITIAL_SECONDS
        while not stop_event.is_set() and self._generation_matches(generation):
            try:
                pullpoint = self._subscribe(
                    SingleSubscriptionCamera,
                    Transport,
                    SqliteCache,
                    generation=generation,
                    stop_event=stop_event,
                )
                if stop_event.is_set() or not self._generation_matches(generation):
                    return
                if self.last_connected_at:
                    self.resubscriptions += 1
                retry_delay = RETRY_INITIAL_SECONDS
                self.retry_attempts = 0
                self.connected = True
                self.last_connected_at = datetime.now(timezone.utc).isoformat()
                self.last_error = ""
            except Exception as exc:
                self.connected = False
                self.retry_attempts += 1
                self.last_error = f"subscription failed: {self._error_text(exc)[:200]}"
                self._unsubscribe(generation)
                self._close_transport(generation)
                LOGGER.warning(
                    "failed to subscribe to ONVIF events for %s, retrying in %.0fs: %s",
                    self.camera.id,
                    retry_delay,
                    self._error_text(exc),
                )
                if stop_event.wait(retry_delay):
                    return
                retry_delay = min(RETRY_MAX_SECONDS, retry_delay * 2)
                continue

            failures = 0
            planned_resubscription = False
            while not stop_event.is_set() and self._generation_matches(generation):
                if self._subscription_renewal_due():
                    if self._renew_subscription(generation, stop_event):
                        continue
                    planned_resubscription = True
                    LOGGER.info(
                        "proactively re-subscribing to ONVIF events for %s",
                        self.camera.id,
                    )
                    break
                try:
                    response = pullpoint.PullMessages(
                        {"Timeout": f"PT{PULL_TIMEOUT_SECONDS}S", "MessageLimit": 10}
                    )
                    if stop_event.is_set() or not self._generation_matches(generation):
                        return
                    raw_notifications = self._raw_pullmessages_notifications(
                        self._pullmessages_capture.take()
                        if self._pullmessages_capture is not None
                        else ""
                    )
                    self._record_subscription_times(response)
                    failures = 0
                    self.connected = True
                    self.last_error = ""
                    self.last_poll_success_at = datetime.now(timezone.utc).isoformat()
                    for index, notification in enumerate(
                        getattr(response, "NotificationMessage", []) or []
                    ):
                        raw_notification = (
                            raw_notifications[index]
                            if index < len(raw_notifications)
                            else None
                        )
                        topic, message = self._extract_event(
                            notification, raw_notification
                        )
                        event_at = self._event_time(notification, message)
                        received_at = datetime.now(timezone.utc)
                        self.notifications_received += 1
                        self.last_event_at = received_at.isoformat()
                        self.last_camera_event_at = event_at.isoformat() if event_at else ""
                        self.last_topic = topic
                        LOGGER.debug("ONVIF event %s: %s", topic, message[:300])
                        motion_state = (
                            self._raw_notification_motion_state(raw_notification)
                            if raw_notification is not None
                            else None
                        )
                        if motion_state is None:
                            motion_state = self._motion_event_state(topic, message)
                        if motion_state is True:
                            self.motion_events_received += 1
                            self.last_motion_event_at = received_at.isoformat()
                            try:
                                self.on_motion(
                                    topic[:1024],
                                    message[:MAX_EVENT_MESSAGE_CHARACTERS],
                                    event_at,
                                )
                            except Exception:
                                self.callback_errors += 1
                                LOGGER.exception(
                                    "ONVIF motion callback failed for %s",
                                    self.camera.id,
                                )
                        elif motion_state is False:
                            self.inactive_motion_events += 1
                        else:
                            self.unrecognized_notifications += 1
                            self._record_unknown_notification(
                                topic,
                                message,
                                received_at,
                            )
                except Exception as exc:
                    if stop_event.is_set() or not self._generation_matches(generation):
                        return
                    error_text = self._error_text(exc)
                    self.last_poll_error = error_text
                    self.last_poll_error_at = datetime.now(timezone.utc).isoformat()
                    if self._is_timeout_error(exc):
                        failures += 1
                        self.poll_timeouts += 1
                        self.connected = True
                        self.last_error = f"poll timeout ({failures}/{MAX_POLL_FAILURES})"
                        LOGGER.debug(
                            "ONVIF event poll timeout for %s (%s/%s): %s",
                            self.camera.id,
                            failures,
                            MAX_POLL_FAILURES,
                            error_text[:200],
                        )
                    else:
                        self.poll_errors += 1
                        self.connected = False
                        self.retry_attempts += 1
                        self.last_error = f"polling failed; re-subscribing: {error_text[:200]}"
                        LOGGER.warning(
                            "ONVIF event polling failed for %s; re-subscribing: %s",
                            self.camera.id,
                            error_text,
                        )
                        break
                    if failures >= MAX_POLL_FAILURES:
                        self.retry_attempts += 1
                        LOGGER.warning(
                            "re-subscribing to ONVIF events for %s after %s polling failures",
                            self.camera.id,
                            failures,
                        )
                        break
                    if stop_event.wait(POLL_RETRY_SECONDS):
                        return

            self._unsubscribe(generation)
            self._close_transport(generation)
            self.connected = False
            if not stop_event.is_set() and self._generation_matches(generation):
                if planned_resubscription:
                    retry_delay = RETRY_INITIAL_SECONDS
                    continue
                if stop_event.wait(retry_delay):
                    return
                retry_delay = min(RETRY_MAX_SECONDS, retry_delay * 2)

    def _subscribe(
        self,
        ONVIFCamera: Any,
        Transport: Any,
        SqliteCache: Any,
        *,
        generation: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> Any:
        generation = self._generation if generation is None else generation
        stop_event = self._stop if stop_event is None else stop_event
        self._close_transport(generation)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_dir / f"onvif-zeep-{self.camera.id}.sqlite3"
        transport = Transport(
            cache=SqliteCache(path=str(cache_path)),
            operation_timeout=TRANSPORT_OPERATION_TIMEOUT_SECONDS,
        )
        with self._lifecycle_lock:
            if generation != self._generation or stop_event.is_set():
                transport.session.close()
                raise RuntimeError("ONVIF subscription generation stopped")
            self._transport = transport
            self._transport_generation = generation
            self._subscription_manager = None
            self._subscription_generation = None
        camera = ONVIFCamera(
            self.camera.onvif.host,
            self.camera.onvif.port,
            self.camera.onvif.username,
            self.camera.onvif.password,
            transport=transport,
        )
        events_service = camera.create_events_service()
        subscription = events_service.CreatePullPointSubscription(
            {"InitialTerminationTime": SUBSCRIPTION_DURATION}
        )
        address = self._subscription_address(subscription)
        if not address:
            raise RuntimeError("ONVIF subscription did not return a PullPoint address")
        camera.xaddrs[PULLPOINT_NAMESPACE] = address
        self._record_subscription_times(subscription)
        try:
            manager = events_service.zeep_client.create_service(
                SUBSCRIPTION_MANAGER_BINDING,
                address,
            )
            with self._lifecycle_lock:
                if generation == self._generation:
                    self._subscription_manager = manager
                    self._subscription_generation = generation
        except Exception:
            LOGGER.debug(
                "ONVIF subscription manager unavailable for %s",
                self.camera.id,
                exc_info=True,
            )
        pullpoint = camera.create_pullpoint_service()
        self._enable_pullmessages_capture(pullpoint)
        if stop_event.is_set() or not self._generation_matches(generation):
            self._unsubscribe(generation)
            self._close_transport(generation)
            raise RuntimeError("ONVIF subscription generation stopped")
        return pullpoint

    @staticmethod
    def _subscription_address(subscription: Any) -> str:
        reference = getattr(subscription, "SubscriptionReference", None)
        address = getattr(reference, "Address", None)
        if isinstance(address, str):
            return address.strip()
        return str(
            getattr(address, "_value_1", "")
            or getattr(address, "value", "")
            or ""
        ).strip()

    def _record_subscription_times(self, response: Any) -> None:
        current = self._parse_event_time(getattr(response, "CurrentTime", None))
        termination = self._parse_event_time(getattr(response, "TerminationTime", None))
        if current is not None:
            self.subscription_current_time = current.isoformat()
        if termination is not None:
            self.subscription_termination_time = termination.isoformat()
        if termination is not None:
            lifetime_reference = current or datetime.now(timezone.utc)
            self.subscription_lifetime_seconds = max(
                0.0, (termination - lifetime_reference).total_seconds()
            )
            self._subscription_expires_monotonic = (
                time.monotonic() + self.subscription_lifetime_seconds
            )
            self._subscription_granted_lifetime_seconds = max(
                self.subscription_lifetime_seconds,
                self._subscription_granted_lifetime_seconds or 0.0,
            )

    def _subscription_renewal_due(self) -> bool:
        if self._subscription_expires_monotonic is None:
            return False
        remaining = max(0.0, self._subscription_expires_monotonic - time.monotonic())
        self.subscription_lifetime_seconds = remaining
        granted = self._subscription_granted_lifetime_seconds
        if granted is None:
            return False
        margin = min(
            SUBSCRIPTION_RENEW_MAX_MARGIN_SECONDS,
            max(SUBSCRIPTION_RENEW_MIN_MARGIN_SECONDS, granted * 0.2),
        )
        return remaining <= margin

    def _renew_subscription(
        self,
        generation: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> bool:
        generation = self._generation if generation is None else generation
        stop_event = self._stop if stop_event is None else stop_event
        with self._lifecycle_lock:
            manager = (
                self._subscription_manager
                if self._subscription_generation in {None, generation}
                else None
            )
        if manager is None:
            return False
        if stop_event.is_set() or not self._generation_matches(generation):
            return False
        self.renewal_attempts += 1
        try:
            # The subscription manager is a raw Zeep service, not the
            # python-onvif wrapper used by the event and pull-point services.
            # Raw Zeep operations require keyword arguments here; a positional
            # dict would be serialized as the literal TerminationTime value.
            response = manager.Renew(TerminationTime=SUBSCRIPTION_DURATION)
            if stop_event.is_set() or not self._generation_matches(generation):
                return False
            self._subscription_granted_lifetime_seconds = None
            self._subscription_expires_monotonic = None
            self.subscription_lifetime_seconds = None
            self._record_subscription_times(response)
            if (
                self.subscription_lifetime_seconds is None
                or self._subscription_expires_monotonic is None
            ):
                raise RuntimeError("ONVIF renewal did not return subscription times")
            self.renewals += 1
            self.last_renewed_at = datetime.now(timezone.utc).isoformat()
            self.last_error = ""
            return True
        except Exception as exc:
            self.renewal_errors += 1
            self.last_error = f"renewal failed; re-subscribing: {self._error_text(exc)[:200]}"
            LOGGER.warning(
                "failed to renew ONVIF subscription for %s: %s",
                self.camera.id,
                self._error_text(exc),
            )
            return False

    def _unsubscribe(self, generation: int | None = None) -> None:
        with self._lifecycle_lock:
            if (
                generation is not None
                and self._subscription_generation not in {None, generation}
            ):
                return
            manager = self._subscription_manager
            self._subscription_manager = None
            self._subscription_generation = None
        if manager is None:
            return
        try:
            manager.Unsubscribe()
            LOGGER.info("released ONVIF subscription for %s", self.camera.id)
        except Exception:
            LOGGER.debug(
                "failed to release ONVIF subscription for %s",
                self.camera.id,
                exc_info=True,
            )

    def _close_transport(self, generation: int | None = None) -> None:
        with self._lifecycle_lock:
            if (
                generation is not None
                and self._transport_generation not in {None, generation}
            ):
                return
            transport = self._transport
            self._transport = None
            self._transport_generation = None
        if transport is None:
            return
        try:
            transport.session.close()
        except Exception:
            LOGGER.debug(
                "failed to close ONVIF transport for %s",
                self.camera.id,
                exc_info=True,
            )

    def _generation_matches(self, generation: int) -> bool:
        with self._lifecycle_lock:
            return generation == self._generation

    def record_ema_observation(
        self,
        matched_onvif: bool,
        observed_at: datetime | float | None = None,
    ) -> None:
        """Record whether one qualified EMA observation matched camera motion.

        This is diagnostic only. It never changes transport state, motion
        admission, or ONVIF subscription behavior.
        """
        if isinstance(observed_at, datetime):
            observed = observed_at
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            observed = observed.astimezone(timezone.utc)
        elif isinstance(observed_at, (int, float)):
            observed = datetime.fromtimestamp(float(observed_at), timezone.utc)
        else:
            observed = datetime.now(timezone.utc)
        value = observed.isoformat()
        observed_monotonic = time.monotonic()
        with self._effectiveness_lock:
            self.ema_qualified_observations += 1
            self.last_ema_observation_at = value
            if matched_onvif:
                self.ema_onvif_matches += 1
                self.last_ema_onvif_match_at = value
            else:
                self.ema_without_onvif += 1
                self.last_ema_without_onvif_at = value
            self._ema_effectiveness_window.append(
                (observed_monotonic, bool(matched_onvif))
            )
            self._prune_effectiveness_window(observed_monotonic)

    def effectiveness_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._effectiveness_lock:
            self._prune_effectiveness_window(now)
            observations = len(self._ema_effectiveness_window)
            matches = sum(1 for _, matched in self._ema_effectiveness_window if matched)
            mismatches = observations - matches
            match_rate = matches / observations if observations else None
            recognized = self.motion_events_received + self.inactive_motion_events
            recognition_rate = (
                recognized / self.notifications_received
                if self.notifications_received
                else None
            )
            active_motion_rate = (
                self.motion_events_received / recognized if recognized else None
            )
            degraded = bool(
                self.camera.onvif.enabled
                and self.connected
                and observations >= EFFECTIVENESS_MINIMUM_OBSERVATIONS
                and mismatches / observations
                >= EFFECTIVENESS_DEGRADED_MISMATCH_RATIO
            )
            if not self.camera.onvif.enabled:
                status = "disabled"
            elif not self.connected:
                status = "transport_unavailable"
            elif observations < EFFECTIVENESS_MINIMUM_OBSERVATIONS:
                status = "insufficient_data"
            elif degraded:
                status = "degraded"
            else:
                status = "effective"
            return {
                "signal_effectiveness_status": status,
                "signal_degraded": degraded,
                "recognized_notifications": recognized,
                "notification_recognition_rate": (
                    round(recognition_rate, 4) if recognition_rate is not None else None
                ),
                "active_motion_rate": (
                    round(active_motion_rate, 4) if active_motion_rate is not None else None
                ),
                "ema_qualified_observations": self.ema_qualified_observations,
                "ema_onvif_matches": self.ema_onvif_matches,
                "ema_without_onvif": self.ema_without_onvif,
                "ema_window_observations": observations,
                "ema_window_onvif_matches": matches,
                "ema_window_without_onvif": mismatches,
                "ema_window_match_rate": (
                    round(match_rate, 4) if match_rate is not None else None
                ),
                "effectiveness_window_seconds": EFFECTIVENESS_WINDOW_SECONDS,
                "last_ema_observation_at": self.last_ema_observation_at,
                "last_ema_onvif_match_at": self.last_ema_onvif_match_at,
                "last_ema_without_onvif_at": self.last_ema_without_onvif_at,
                "unknown_notification_samples": list(
                    self._unknown_notification_samples
                ),
            }

    def _prune_effectiveness_window(self, now: float) -> None:
        cutoff = now - EFFECTIVENESS_WINDOW_SECONDS
        while (
            self._ema_effectiveness_window
            and self._ema_effectiveness_window[0][0] < cutoff
        ):
            self._ema_effectiveness_window.popleft()

    def _record_unknown_notification(
        self,
        topic: str,
        message: str,
        received_at: datetime,
    ) -> None:
        redacted_message = redact_secret_text(message)
        encoded = redacted_message.encode("utf-8", errors="replace")
        sample = {
            "at": received_at.isoformat(),
            "topic": redact_secret_text(topic)[:256],
            "message_characters": len(message),
            "message_fingerprint": hashlib.sha256(encoded).hexdigest()[:16],
        }
        with self._effectiveness_lock:
            self._unknown_notification_samples.append(sample)


    def _is_timeout_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        exception_types = " ".join(
            item.__name__.lower() for item in type(exc).__mro__
        )
        return "timeout" in exception_types or any(word in text for word in TIMEOUT_WORDS)

    @staticmethod
    def _error_text(exc: Exception) -> str:
        detail = str(exc).strip() or repr(exc)
        return redact_secret_text(f"{type(exc).__name__}: {detail}")[:300]

    def _event_time(self, notification: Any, message: str) -> datetime | None:
        for name in ("UtcTime", "utcTime", "timestamp", "Timestamp"):
            value = getattr(notification, name, None)
            parsed = self._parse_event_time(value)
            if parsed is not None:
                return parsed

        match = re.search(
            r"\b(20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\b",
            message,
        )
        if match:
            return self._parse_event_time(match.group(1))
        return None

    def _parse_event_time(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            text = str(value).strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _enable_pullmessages_capture(self, pullpoint: Any) -> None:
        if self._pullmessages_capture is not None:
            return
        plugins = getattr(getattr(pullpoint, "zeep_client", None), "plugins", None)
        if not isinstance(plugins, list):
            return
        capture = _PullMessagesResponseCapture()
        plugins.append(capture)
        self._pullmessages_capture = capture

    @staticmethod
    def _normalized_topic(topic: str) -> str:
        return "/".join(
            segment.rsplit(":", 1)[-1].strip().lower()
            for segment in str(topic or "").strip().split("/")
            if segment.strip()
        )

    @staticmethod
    def _xml_local_name(tag: str) -> str:
        return str(tag).rsplit("}", 1)[-1]

    @staticmethod
    def _stringify_xml_static(element: Any) -> str:
        try:
            from lxml import etree

            return etree.tostring(element, encoding="unicode")
        except Exception:
            try:
                return ElementTree.tostring(element, encoding="unicode")
            except Exception:
                return str(element)

    @classmethod
    def _raw_pullmessages_notifications(
        cls, raw_xml: str
    ) -> list[_RawOnvifNotification]:
        if not raw_xml:
            return []
        try:
            root = ElementTree.fromstring(raw_xml)
        except ElementTree.ParseError:
            return []
        notifications: list[_RawOnvifNotification] = []
        for notification in root.iter():
            if cls._xml_local_name(notification.tag) != "NotificationMessage":
                continue
            topic_element = next(
                (
                    child for child in notification
                    if cls._xml_local_name(child.tag) == "Topic"
                ),
                None,
            )
            message_element = next(
                (
                    child for child in notification
                    if cls._xml_local_name(child.tag) == "Message"
                ),
                None,
            )
            topic = str(topic_element.text or "").strip() if topic_element is not None else ""
            message_xml = (
                cls._stringify_xml_static(message_element)
                if message_element is not None
                else ""
            )
            simple_items = tuple(
                (
                    str(item.attrib.get("Name") or "").strip().lower(),
                    str(item.attrib.get("Value") or "").strip().lower(),
                )
                for item in message_element.iter()
                if cls._xml_local_name(item.tag) == "SimpleItem"
                and str(item.attrib.get("Name") or "").strip()
            ) if message_element is not None else ()
            notifications.append(_RawOnvifNotification(
                topic=topic,
                normalized_topic=cls._normalized_topic(topic),
                message_xml=message_xml,
                simple_items=simple_items,
            ))
        return notifications

    @staticmethod
    def _raw_notification_motion_state(
        notification: _RawOnvifNotification,
    ) -> bool | None:
        if notification.normalized_topic not in REOLINK_MOTION_TOPICS:
            return None
        state_values = [
            value
            for name, value in notification.simple_items
            if name in {"ismotion", "motion", "state"}
        ]
        if state_values:
            active = {"true", "1", "on", "active"}
            inactive = {"false", "0", "off", "inactive"}
            if any(value in active for value in state_values):
                return True
            if all(value in inactive for value in state_values):
                return False
        return True

    def _extract_event(
        self,
        notification: Any,
        raw_notification: _RawOnvifNotification | None = None,
    ) -> tuple[str, str]:
        if raw_notification is not None:
            return raw_notification.topic, raw_notification.message_xml
        topic = str(getattr(notification, "Topic", ""))
        message = self._stringify_message(getattr(notification, "Message", ""))
        return topic, message

    def _motion_event_state(self, topic: str, message: str) -> bool | None:
        searchable = f"{topic} {message}".lower()
        if not any(word in searchable for word in MOTION_WORDS):
            return None
        explicit_states: list[bool] = []
        containers = [
            *re.findall(r"<[^>]+>", searchable),
            *re.findall(r"\{[^{}]{0,500}\}", searchable),
        ]
        for container in containers:
            attributes = {
                name.lower(): value.lower()
                for name, value in re.findall(
                    r"['\"]?\b(name|value)['\"]?\s*[:=]\s*['\"]?([^,'\"}\s/>]+)",
                    container,
                    flags=re.IGNORECASE,
                )
            }
            if attributes.get("name") not in {"ismotion", "motion", "state"}:
                continue
            value = attributes.get("value", "")
            if value in {"true", "1", "on", "active"}:
                explicit_states.append(True)
            elif value in {"false", "0", "off", "inactive"}:
                explicit_states.append(False)
        if explicit_states:
            return any(explicit_states)
        explicit_false = (
            'name="ismotion" value="false"' in searchable
            or "name='ismotion' value='false'" in searchable
            or 'name="state" value="false"' in searchable
            or "name='state' value='false'" in searchable
        )
        explicit_true = (
            'name="ismotion" value="true"' in searchable
            or "name='ismotion' value='true'" in searchable
            or 'name="state" value="true"' in searchable
            or "name='state' value='true'" in searchable
        )
        if explicit_true:
            return True
        if explicit_false:
            return False
        return True

    def _is_motion_event(self, topic: str, message: str) -> bool:
        return self._motion_event_state(topic, message) is True

    def _stringify_message(self, message: Any) -> str:
        if message is None:
            return ""
        if isinstance(message, str):
            return message
        parts: list[str] = [str(message)]
        value = getattr(message, "_value_1", None)
        if value is not None:
            parts.append(self._stringify_xml(value))
        for name in ("Message", "Data", "Source", "Key"):
            value = getattr(message, name, None)
            if value is not None:
                parts.append(str(value))
        return " ".join(part for part in parts if part)

    def _stringify_xml(self, element: Any) -> str:
        return self._stringify_xml_static(element)
