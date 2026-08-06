"""Lifecycle coordination for one camera's replaceable tracking session."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Protocol

from .config import CameraConfig
from .object_tracking import (
    CatchupFrameProvider,
    FrameProvider,
    FrameSample,
    ObjectTrackingSession,
    ObjectTrackingSessionFactory,
)

LOGGER = logging.getLogger(__name__)


class TrackingHistory(Protocol):
    def resize(self, sample_fps: float) -> None: ...


class ObjectTrackingLifecycle:
    """Own tracking-session creation, eligibility, replacement, and teardown."""

    def __init__(
        self,
        *,
        camera: CameraConfig,
        factory: ObjectTrackingSessionFactory,
        frame_provider: FrameProvider,
        catchup_frame_provider: CatchupFrameProvider,
        prewarm_frame_provider: Callable[[], FrameSample | None],
        history: Callable[[], TrackingHistory],
        accepting: Callable[[], bool],
        lifecycle_lock: threading.RLock,
    ) -> None:
        self.camera = camera
        self.frame_provider = frame_provider
        self.catchup_frame_provider = catchup_frame_provider
        self.prewarm_frame_provider = prewarm_frame_provider
        self.history = history
        self.accepting = accepting
        self.lifecycle_lock = lifecycle_lock
        self._session = self.create(factory)

    def current(self) -> ObjectTrackingSession:
        with self.lifecycle_lock:
            return self._session

    def bind_for_compatibility(self, session: ObjectTrackingSession) -> None:
        """Support legacy integrations that replace an inactive session directly."""
        with self.lifecycle_lock:
            self._session = session

    def create(
        self,
        factory: ObjectTrackingSessionFactory,
    ) -> ObjectTrackingSession:
        return factory.create(
            camera=self.camera,
            frame_provider=self.frame_provider,
            catchup_frame_provider=self.catchup_frame_provider,
        )

    def prewarm(self) -> FrameSample | None:
        return self.prewarm_frame_provider()

    def sample_fps(self) -> float:
        with self.lifecycle_lock:
            return float(self._session.config.sample_fps)

    def status(self) -> dict[str, object]:
        with self.lifecycle_lock:
            return self._session.status()

    def running(self) -> bool:
        with self.lifecycle_lock:
            return self._session.running()

    def sync_accepting(self) -> None:
        with self.lifecycle_lock:
            self._session.set_accepting(self.accepting())

    def pause(self) -> None:
        with self.lifecycle_lock:
            if not self._session.stop():
                raise RuntimeError(
                    f"object tracking session did not stop for {self.camera.id}"
                )

    def stop(self) -> bool:
        with self.lifecycle_lock:
            return self._session.stop()

    def replace(
        self,
        replacement: ObjectTrackingSession,
    ) -> ObjectTrackingSession:
        with self.lifecycle_lock:
            previous = self._session
            if replacement is previous:
                return previous
            if not previous.stop():
                raise RuntimeError(
                    f"object tracking session did not stop for {self.camera.id}"
                )
            try:
                self._session = replacement
                self.history().resize(replacement.config.sample_fps)
                replacement.set_accepting(self.accepting())
            except BaseException:
                try:
                    replacement.stop()
                except Exception:
                    LOGGER.exception(
                        "replacement object tracking cleanup failed for %s",
                        self.camera.id,
                    )
                finally:
                    self._session = previous
                try:
                    self.history().resize(previous.config.sample_fps)
                except Exception:
                    LOGGER.exception(
                        "previous object tracking history restore failed for %s",
                        self.camera.id,
                    )
                try:
                    previous.set_accepting(self.accepting())
                except Exception:
                    LOGGER.exception(
                        "previous object tracking session restore failed for %s",
                        self.camera.id,
                    )
                raise
            return previous
