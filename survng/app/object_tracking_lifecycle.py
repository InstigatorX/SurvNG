"""Lifecycle coordination for one camera's replaceable tracking session."""

from __future__ import annotations

import logging
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

import numpy as np

from .config import CameraConfig
from .object_tracking import (
    CatchupFrameProvider,
    FrameProvider,
    FrameSample,
    ObjectTrackingSession,
    ObjectTrackingSessionFactory,
    TrackingCoverFrameProvider,
    TrackingSnapshotWriter,
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
        cover_frame_provider: TrackingCoverFrameProvider | None = None,
        snapshot_writer: TrackingSnapshotWriter | None = None,
        scene_job_store: Any | None = None,
        drain_cutoff: Callable[[], float | None] | None = None,
    ) -> None:
        self.camera = camera
        self.drain_cutoff = drain_cutoff or (lambda: None)
        self.frame_provider = frame_provider
        self.catchup_frame_provider = catchup_frame_provider
        self.prewarm_frame_provider = prewarm_frame_provider
        self.history = history
        self.accepting = accepting
        self.lifecycle_lock = lifecycle_lock
        self.cover_frame_provider = cover_frame_provider
        self.snapshot_writer = snapshot_writer
        self.scene_job_store = scene_job_store
        self.scene_context_memory = None
        self.activity_attributor = None
        self._scene_lease_owner = uuid.uuid4().hex
        # Terminal drain owns scene admission until detection is explicitly
        # enabled again. This fences a resume racing the stop/join window.
        self._scene_abort_in_progress = False
        self._scene_abort_waiting = False
        self._scene_abort_failed = False
        self._scene_reenable_pending = False
        self._session = self.create(factory)

    def current(self) -> ObjectTrackingSession:
        return self._session

    def bind_for_compatibility(self, session: ObjectTrackingSession) -> None:
        """Support legacy integrations that replace an inactive session directly."""
        with self.lifecycle_lock:
            if self._session is not session and self._session.running():
                raise RuntimeError(
                    f"cannot replace active object tracking session for {self.camera.id}"
                )
            self._session = session

    def create(
        self,
        factory: ObjectTrackingSessionFactory,
    ) -> ObjectTrackingSession:
        session = factory.create(
            camera=self.camera,
            frame_provider=self.frame_provider,
            catchup_frame_provider=self.catchup_frame_provider,
            cover_frame_provider=self.cover_frame_provider,
            snapshot_writer=self.snapshot_writer,
        )
        session.scene_context_memory = self.scene_context_memory
        session.activity_attributor = self.activity_attributor
        return session

    def prewarm(self) -> FrameSample | None:
        return self.prewarm_frame_provider()

    def sample_fps(self) -> float:
        return float(self._session.config.sample_fps)

    def status(self) -> dict[str, object]:
        return self._session.status()

    def running(self) -> bool:
        return self._session.running()

    def enabled(self) -> bool:
        return bool(self._session.config.enabled)

    def has_trackable_objects(self, objects: list[dict[str, Any]]) -> bool:
        with self.lifecycle_lock:
            if self.scene_job_store is not None:
                return bool(self._session.config.enabled)
            return bool(self._trackable_objects(self._session, objects))

    def start_incident(
        self,
        event_id: int,
        event_at: datetime,
        objects: list[dict[str, Any]],
        initial_frame: np.ndarray | None = None,
    ) -> bool | None:
        """Atomically select and start the current session for one incident."""
        with self.lifecycle_lock:
            session = self._session
            trackable = self._trackable_objects(session, objects)
            if self.scene_job_store is not None and session.config.enabled:
                if self._scene_abort_in_progress:
                    return None
                start, end = (
                    session.window_provider(event_id, event_at)
                    if session.window_provider is not None
                    else (event_at.timestamp(), event_at.timestamp() + session.config.max_session_seconds)
                )
                # The scene quiet boundary outlives an individual tracking
                # compute chunk. Subsequent activity extends this same job.
                end = max(end, event_at.timestamp() + 45.0)
                job = self.scene_job_store.enqueue_scene_tracking(event_id, start, end)
                # No scene membership means the event was not established.
                # None matches the other "nothing to track" path so this is
                # not reported as a declined tracking session.
                if job is None:
                    return None
                self.resume_pending_scene()
                return True
            if not trackable:
                return None
            return session.start(event_id, event_at, trackable, initial_frame)

    def resume_pending_scene(self) -> bool:
        """Resume one durable camera episode on the existing tracking worker."""
        if self.scene_job_store is None:
            return False
        with self.lifecycle_lock:
            if self._scene_abort_in_progress:
                return False
            session = self._session
            live = self.accepting()
            cutoff = None if live else self.drain_cutoff()
            if (not live and cutoff is None) or not session.config.enabled or session.running():
                return False
            if cutoff is not None:
                # Detection is off: only footage recorded before it turned off.
                session.set_accepting(True, recorded_only=True)
            job = self.scene_job_store.claim_scene_tracking(
                self.camera.id, self._scene_lease_owner, end_limit=cutoff,
            )
            if job is None:
                return False
            try:
                resume_tracks = None
                if callable(getattr(self.scene_job_store, "scene_track_resume", None)):
                    resume_tracks = self.scene_job_store.scene_track_resume(int(job["event_id"]))
                started = session.start(
                    int(job["event_id"]), datetime.fromtimestamp(job["event_epoch"], timezone.utc),
                    [], None, recorded_window=(job["start_epoch"], job["end_epoch"]),
                    resume_after=job["cursor_epoch"],
                    scene_analysis_job={"episode_id": job["episode_id"], "lease_owner": self._scene_lease_owner,
                                        "analyzed_through_epoch":job.get("analyzed_epoch"),
                                        "association_after_epoch":max((g["end_epoch"] for g in json.loads(job.get("coverage_gaps_json","[]"))),default=None)},
                    scene_track_resume=resume_tracks,
                )
            except Exception as error:
                self.scene_job_store.release_scene_tracking(job["episode_id"], self._scene_lease_owner, type(error).__name__)
                raise
            if not started:
                self.scene_job_store.release_scene_tracking(job["episode_id"], self._scene_lease_owner, "tracking_start_deferred")
            return started

    def sync_accepting(self) -> None:
        with self.lifecycle_lock:
            accepting = self.accepting()
            if self._scene_abort_in_progress and self._scene_abort_waiting:
                self._scene_reenable_pending = accepting
            if accepting:
                if self._scene_abort_in_progress and self._scene_abort_waiting:
                    return
                if self._scene_abort_failed:
                    raise RuntimeError(
                        f"cannot re-enable scene work after tracking stop failed for {self.camera.id}"
                    )
                self._scene_abort_in_progress = False
            self._session.set_accepting(accepting)
            if accepting and self.scene_job_store is not None:
                self.scene_job_store.clear_scene_tracking_limit(self.camera.id)

    def scene_work_pending(self) -> bool:
        """Recorded scene analysis is running or still queued for this camera."""
        if self.scene_job_store is None:
            return False
        return self._session.running() or self.scene_job_store.scene_tracking_pending(self.camera.id)

    def close_out_scene_work(self, reason: str) -> int:
        if self.scene_job_store is None:
            return 0
        return int(self.scene_job_store.close_scene_tracking(self.camera.id, reason))

    def abort_scene_work(self, reason: str, timeout: float = 10.0) -> int:
        """Stop active compute before terminalizing its durable scene jobs."""
        if self.scene_job_store is None:
            return 0
        with self.lifecycle_lock:
            self._scene_abort_in_progress = True
            self._scene_abort_waiting = True
            self._scene_abort_failed = False
            session = self._session
            session.request_stop()
        # Do not hold the camera lifecycle lock while the worker joins. Session
        # completion may need to publish its final checkpoint through the camera.
        if not session.wait_stopped(max(0.0, float(timeout))):
            with self.lifecycle_lock:
                self._scene_abort_waiting = False
                self._scene_abort_failed = True
            raise RuntimeError(
                f"object tracking session did not stop before closing scene work for {self.camera.id}"
            )
        with self.lifecycle_lock:
            self._scene_abort_waiting = False
            self._scene_abort_failed = False
            accepting = self.accepting()
            self._scene_reenable_pending = False
            if accepting:
                self._scene_abort_in_progress = False
                session.set_accepting(True)
                self.scene_job_store.clear_scene_tracking_limit(self.camera.id)
                return 0
            if self._session is not session:
                raise RuntimeError(
                    f"object tracking session changed while closing scene work for {self.camera.id}"
                )
            return int(self.scene_job_store.close_scene_tracking(self.camera.id, reason))

    def pause(self) -> None:
        with self.lifecycle_lock:
            if not self._session.stop():
                raise RuntimeError(
                    f"object tracking session did not stop for {self.camera.id}"
                )

    def stop(self) -> bool:
        with self.lifecycle_lock:
            return self._session.stop()

    def request_stop(self) -> None:
        with self.lifecycle_lock:
            self._session.request_stop()

    def wait_stopped(self, timeout: float) -> bool:
        with self.lifecycle_lock:
            return self._session.wait_stopped(timeout)

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
                except BaseException:
                    LOGGER.exception(
                        "replacement object tracking cleanup failed for %s",
                        self.camera.id,
                    )
                finally:
                    self._session = previous
                try:
                    self.history().resize(previous.config.sample_fps)
                except BaseException:
                    LOGGER.exception(
                        "previous object tracking history restore failed for %s",
                        self.camera.id,
                    )
                try:
                    previous.set_accepting(self.accepting())
                except BaseException:
                    LOGGER.exception(
                        "previous object tracking session restore failed for %s",
                        self.camera.id,
                    )
                raise
            return previous

    @staticmethod
    def _trackable_objects(
        session: ObjectTrackingSession,
        objects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not session.config.enabled:
            return []
        return [
            item
            for item in objects
            if (
                item.get("label")
                and session.config.tracks_label(item.get("label"))
            )
        ]
