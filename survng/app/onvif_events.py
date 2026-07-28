from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
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
STOP_JOIN_SECONDS = TRANSPORT_OPERATION_TIMEOUT_SECONDS + 5
MAX_POLL_FAILURES = 6
TIMEOUT_WORDS = ("timed out", "timeout", "read timed out", "operation timed out")
PULLPOINT_NAMESPACE = "http://www.onvif.org/ver10/events/wsdl/PullPointSubscription"
SUBSCRIPTION_MANAGER_BINDING = "{http://www.onvif.org/ver10/events/wsdl}SubscriptionManagerBinding"
SUBSCRIPTION_DURATION = "PT1H"
SUBSCRIPTION_RENEW_MAX_MARGIN_SECONDS = 300.0
SUBSCRIPTION_RENEW_MIN_MARGIN_SECONDS = 5.0
MAX_EVENT_MESSAGE_CHARACTERS = 16_384


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
        self._transport: Any = None
        self._subscription_manager: Any = None
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
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            thread = threading.Thread(
                target=self._run,
                name=f"onvif-{self.camera.id}",
                daemon=False,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                self._stop.set()
                raise

    def stop(self) -> None:
        self._stop.set()
        with self._lifecycle_lock:
            thread = self._thread
        if thread is not None:
            # PullMessages has a short bounded timeout. Let the owning worker
            # return from it and send Unsubscribe while its transport is still
            # usable before resorting to forcibly closing the session.
            thread.join(timeout=PULL_TIMEOUT_SECONDS + 5)
            if thread.is_alive():
                self._close_transport()
                thread.join(timeout=STOP_JOIN_SECONDS)
                if thread.is_alive():
                    LOGGER.error("ONVIF worker did not stop for %s", self.camera.id)
        self.connected = False
        with self._lifecycle_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
        if thread is None or not thread.is_alive():
            self._transport = None

    def _run(self) -> None:
        try:
            self._run_until_stopped()
        finally:
            self._unsubscribe()
            self._close_transport()
            self.connected = False
            current = threading.current_thread()
            with self._lifecycle_lock:
                if self._thread is current:
                    self._thread = None

    def _run_until_stopped(self) -> None:
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
        while not self._stop.is_set():
            try:
                pullpoint = self._subscribe(SingleSubscriptionCamera, Transport, SqliteCache)
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
                self._unsubscribe()
                self._close_transport()
                LOGGER.warning(
                    "failed to subscribe to ONVIF events for %s, retrying in %.0fs: %s",
                    self.camera.id,
                    retry_delay,
                    self._error_text(exc),
                )
                if self._stop.wait(retry_delay):
                    return
                retry_delay = min(RETRY_MAX_SECONDS, retry_delay * 2)
                continue

            failures = 0
            planned_resubscription = False
            while not self._stop.is_set():
                if self._subscription_renewal_due():
                    if self._renew_subscription():
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
                    self._record_subscription_times(response)
                    failures = 0
                    self.connected = True
                    self.last_error = ""
                    self.last_poll_success_at = datetime.now(timezone.utc).isoformat()
                    for notification in getattr(response, "NotificationMessage", []) or []:
                        topic, message = self._extract_event(notification)
                        event_at = self._event_time(notification, message)
                        received_at = datetime.now(timezone.utc)
                        self.notifications_received += 1
                        self.last_event_at = received_at.isoformat()
                        self.last_camera_event_at = event_at.isoformat() if event_at else ""
                        self.last_topic = topic
                        LOGGER.debug("ONVIF event %s: %s", topic, message[:300])
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
                except Exception as exc:
                    if self._stop.is_set():
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
                    if self._stop.wait(POLL_RETRY_SECONDS):
                        return

            self._unsubscribe()
            self._close_transport()
            self.connected = False
            if not self._stop.is_set():
                if planned_resubscription:
                    retry_delay = RETRY_INITIAL_SECONDS
                    continue
                if self._stop.wait(retry_delay):
                    return
                retry_delay = min(RETRY_MAX_SECONDS, retry_delay * 2)

    def _subscribe(self, ONVIFCamera: Any, Transport: Any, SqliteCache: Any) -> Any:
        self._close_transport()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_dir / f"onvif-zeep-{self.camera.id}.sqlite3"
        transport = Transport(
            cache=SqliteCache(path=str(cache_path)),
            operation_timeout=TRANSPORT_OPERATION_TIMEOUT_SECONDS,
        )
        self._transport = transport
        self._subscription_manager = None
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
            self._subscription_manager = events_service.zeep_client.create_service(
                SUBSCRIPTION_MANAGER_BINDING,
                address,
            )
        except Exception:
            LOGGER.debug(
                "ONVIF subscription manager unavailable for %s",
                self.camera.id,
                exc_info=True,
            )
        return camera.create_pullpoint_service()

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

    def _renew_subscription(self) -> bool:
        manager = self._subscription_manager
        if manager is None:
            return False
        self.renewal_attempts += 1
        try:
            # The subscription manager is a raw Zeep service, not the
            # python-onvif wrapper used by the event and pull-point services.
            # Raw Zeep operations require keyword arguments here; a positional
            # dict would be serialized as the literal TerminationTime value.
            response = manager.Renew(TerminationTime=SUBSCRIPTION_DURATION)
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

    def _unsubscribe(self) -> None:
        manager = self._subscription_manager
        self._subscription_manager = None
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

    def _close_transport(self) -> None:
        transport = self._transport
        self._transport = None
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

    def _extract_event(self, notification: Any) -> tuple[str, str]:
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
        try:
            from lxml import etree

            return etree.tostring(element, encoding="unicode")
        except Exception:
            try:
                return ElementTree.tostring(element, encoding="unicode")
            except Exception:
                return str(element)
