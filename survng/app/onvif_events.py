from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree

from .config import CameraConfig

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


class OnvifEventListener:
    def __init__(
        self,
        camera: CameraConfig,
        on_motion: Callable[[str, str, datetime | None], None],
    ) -> None:
        self.camera = camera
        self.on_motion = on_motion
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._transport: Any = None
        self._subscription_manager: Any = None
        self.connected = False
        self.last_event_at = ""
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
        self.subscription_current_time = ""
        self.subscription_termination_time = ""
        self.subscription_lifetime_seconds: float | None = None

    def start(self) -> None:
        if not self.camera.onvif.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"onvif-{self.camera.id}",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=PULL_TIMEOUT_SECONDS + 5)
            if thread.is_alive():
                transport = self._transport
                if transport is not None:
                    try:
                        transport.session.close()
                    except Exception:
                        LOGGER.debug("failed to close ONVIF transport for %s", self.camera.id, exc_info=True)
                thread.join(timeout=STOP_JOIN_SECONDS)
                if thread.is_alive():
                    LOGGER.error("ONVIF worker did not stop for %s", self.camera.id)
        self.connected = False
        self._thread = thread if thread is not None and thread.is_alive() else None
        self._transport = None

    def _run(self) -> None:
        try:
            from onvif import ONVIFCamera
            from zeep import Transport
            from zeep.cache import SqliteCache
        except ImportError:
            LOGGER.warning("onvif-zeep is not installed; ONVIF events disabled")
            self.connected = False
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
                self.last_error = f"subscription failed: {str(exc)[:200]}"
                LOGGER.warning(
                    "failed to subscribe to ONVIF events for %s, retrying in %.0fs: %s",
                    self.camera.id,
                    retry_delay,
                    str(exc)[:300],
                )
                if self._stop.wait(retry_delay):
                    return
                retry_delay = min(RETRY_MAX_SECONDS, retry_delay * 2)
                continue

            failures = 0
            while not self._stop.is_set():
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
                        self.last_event_at = (event_at or datetime.now(timezone.utc)).isoformat()
                        self.last_topic = topic
                        LOGGER.debug("ONVIF event %s: %s", topic, message[:300])
                        if self._is_motion_event(topic, message):
                            self.on_motion(topic, message, event_at)
                except Exception as exc:
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
            self.connected = False
            if not self._stop.is_set():
                if self._stop.wait(retry_delay):
                    return
                retry_delay = min(RETRY_MAX_SECONDS, retry_delay * 2)

    def _subscribe(self, ONVIFCamera: Any, Transport: Any, SqliteCache: Any) -> Any:
        cache_path = f"survng/storage/onvif-zeep-{self.camera.id}.sqlite3"
        transport = Transport(
            cache=SqliteCache(path=cache_path),
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
        return str(getattr(address, "_value_1", "") or "").strip()

    def _record_subscription_times(self, response: Any) -> None:
        current = self._parse_event_time(getattr(response, "CurrentTime", None))
        termination = self._parse_event_time(getattr(response, "TerminationTime", None))
        self.subscription_current_time = current.isoformat() if current else ""
        self.subscription_termination_time = termination.isoformat() if termination else ""
        if current is not None and termination is not None:
            self.subscription_lifetime_seconds = max(
                0.0, (termination - current).total_seconds()
            )
        else:
            self.subscription_lifetime_seconds = None

    def _unsubscribe(self) -> None:
        manager = self._subscription_manager
        self._subscription_manager = None
        if manager is None:
            return
        try:
            manager.Unsubscribe()
        except Exception:
            LOGGER.debug(
                "failed to release ONVIF subscription for %s",
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
        return f"{type(exc).__name__}: {detail}"[:300]

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

    def _is_motion_event(self, topic: str, message: str) -> bool:
        searchable = f"{topic} {message}".lower()
        if not any(word in searchable for word in MOTION_WORDS):
            return False
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
