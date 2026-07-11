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

LOGGER = logging.getLogger(__name__)
MOTION_WORDS = ("motion", "cellmotion", "person", "vehicle", "animal", "alarm")
RETRY_INITIAL_SECONDS = 2.0
RETRY_MAX_SECONDS = 60.0
POLL_RETRY_SECONDS = 2.0
PULL_TIMEOUT_SECONDS = 30
TRANSPORT_OPERATION_TIMEOUT_SECONDS = 45
MAX_POLL_FAILURES = 6
TIMEOUT_WORDS = ("timed out", "timeout", "read timed out", "operation timed out")


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
        self.connected = False
        self.last_event_at = ""
        self.last_error = ""
        self.last_topic = ""
        self.last_connected_at = ""
        self.retry_attempts = 0

    def start(self) -> None:
        if not self.camera.onvif.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.connected = False
        self._thread = None

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

        retry_delay = RETRY_INITIAL_SECONDS
        while not self._stop.is_set():
            try:
                pullpoint = self._subscribe(ONVIFCamera, Transport, SqliteCache)
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
                    failures = 0
                    self.connected = True
                    self.last_error = ""
                    for notification in getattr(response, "NotificationMessage", []) or []:
                        topic, message = self._extract_event(notification)
                        event_at = self._event_time(notification, message)
                        self.last_event_at = (event_at or datetime.now(timezone.utc)).isoformat()
                        self.last_topic = topic
                        LOGGER.debug("ONVIF event %s: %s", topic, message[:300])
                        if self._is_motion_event(topic, message):
                            self.on_motion(topic, message, event_at)
                except Exception as exc:
                    error_text = str(exc).strip()
                    if self._is_timeout_error(exc):
                        failures += 1
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
                        failures += 1
                        self.connected = False
                        self.last_error = f"polling failed ({failures}/{MAX_POLL_FAILURES}): {error_text[:200]}"
                        LOGGER.warning("ONVIF event polling failed for %s: %s", self.camera.id, error_text[:300])
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
        camera = ONVIFCamera(
            self.camera.onvif.host,
            self.camera.onvif.port,
            self.camera.onvif.username,
            self.camera.onvif.password,
            transport=transport,
        )
        events_service = camera.create_events_service()
        events_service.CreatePullPointSubscription()
        return camera.create_pullpoint_service()


    def _is_timeout_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(word in text for word in TIMEOUT_WORDS)

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
