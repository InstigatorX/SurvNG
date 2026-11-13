"""Bounded wake sessions for battery cameras.

SurvNG cannot turn a sleeping camera's radio on. Something else — a hub,
Home Assistant, or the camera's own notify — tells SurvNG the URL is about
to serve video. This owner opens the existing capture and recording paths
for one session, then returns the camera to sleep without reconnecting.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .config import CameraConfig


LOGGER = logging.getLogger(__name__)

WakePhase = str  # asleep | connecting | awake | wake_failed


@dataclass(slots=True)
class _WakeSession:
    generation: int
    phase: WakePhase
    started_at: float
    hard_deadline: float
    open_deadline: float
    idle_deadline: float | None
    saw_frame: bool
    last_frame: float | None


class BatteryWakeService:
    """Own one wake session per battery camera."""

    def __init__(
        self,
        *,
        cameras: Mapping[str, CameraConfig],
        open_capture: Callable[[str], bool],
        release_capture: Callable[[str], None],
        capture_running: Callable[[str], bool],
        frame_monotonic: Callable[[str], float | None],
        notify_motion: Callable[[str], None],
        set_recording_released: Callable[[str, bool], None],
        start_recording: Callable[[CameraConfig], None],
        publish: Callable[[str, str], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        poll_seconds: float = 0.05,
    ) -> None:
        self._cameras = dict(cameras)
        self._open_capture = open_capture
        self._release_capture = release_capture
        self._capture_running = capture_running
        self._frame_monotonic = frame_monotonic
        self._notify_motion = notify_motion
        self._set_recording_released = set_recording_released
        self._start_recording = start_recording
        self._publish = publish
        self._monotonic = monotonic
        self._poll_seconds = max(0.01, float(poll_seconds))
        self._lock = threading.Lock()
        self._camera_locks = {
            camera_id: threading.Lock() for camera_id in self._cameras
        }
        self._sessions: dict[str, _WakeSession] = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def replace_cameras(self, cameras: Mapping[str, CameraConfig]) -> None:
        with self._lock:
            self._cameras = dict(cameras)
            for camera_id in cameras:
                self._camera_locks.setdefault(camera_id, threading.Lock())

    def start(self) -> None:
        self._stop.clear()
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="battery-wake",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        for camera_id in list(self._cameras):
            self.cancel(camera_id)

    def status(self, camera_id: str) -> dict[str, object]:
        now = self._monotonic()
        with self._lock:
            session = self._sessions.get(camera_id)
            if session is None:
                return {"phase": "asleep", "overdue": False, "saw_frame": False}
            overdue = (
                session.phase in {"connecting", "awake"}
                and now > session.hard_deadline + 1.0
            )
            return {
                "phase": session.phase,
                "overdue": overdue,
                "saw_frame": session.saw_frame,
            }

    def arm(self, camera_id: str) -> bool:
        """Start a wake, or refresh the current one without a second process."""
        camera = self._cameras.get(camera_id)
        if camera is None or not camera.battery_powered():
            return False
        if self._stop.is_set():
            return False
        camera_lock = self._camera_locks[camera_id]
        with camera_lock:
            now = self._monotonic()
            with self._lock:
                session = self._sessions.get(camera_id)
                if session is not None and session.phase in {"connecting", "awake"}:
                    if session.phase == "awake":
                        session.idle_deadline = now + camera.wake_idle_seconds
                    else:
                        session.open_deadline = now + camera.wake_open_timeout_seconds
                    return True
                generation = 1 if session is None else session.generation + 1
                session = _WakeSession(
                    generation=generation,
                    phase="connecting",
                    started_at=now,
                    hard_deadline=now + camera.wake_max_seconds,
                    open_deadline=now + camera.wake_open_timeout_seconds,
                    idle_deadline=None,
                    saw_frame=False,
                    last_frame=None,
                )
                self._sessions[camera_id] = session
            self._announce(camera_id, "connecting")
            opened = False
            try:
                opened = bool(self._open_capture(camera_id))
            except Exception:
                LOGGER.exception("battery wake capture failed for %s", camera_id)
            if not opened:
                self._close_locked_owner(
                    camera_id,
                    generation,
                    "wake_failed",
                    camera_lock_held=True,
                )
                return True
            if camera.record or (camera.record_sub and camera.live_stream_url):
                try:
                    self._set_recording_released(camera_id, True)
                    self._start_recording(camera)
                except Exception:
                    LOGGER.exception("battery wake recording failed for %s", camera_id)
                    self._close_locked_owner(
                        camera_id,
                        generation,
                        "wake_failed",
                        camera_lock_held=True,
                    )
                    return True
            self._wake.set()
            return True

    def cancel(self, camera_id: str) -> None:
        """End a session because the camera was disabled or the process is stopping."""
        camera_lock = self._camera_locks.get(camera_id)
        if camera_lock is None:
            return
        with camera_lock:
            with self._lock:
                session = self._sessions.get(camera_id)
                if session is None or session.phase not in {"connecting", "awake"}:
                    return
                generation = session.generation
            self._close_locked_owner(
                camera_id,
                generation,
                "asleep",
                camera_lock_held=True,
            )

    def _close_locked_owner(
        self,
        camera_id: str,
        generation: int,
        phase: WakePhase,
        *,
        camera_lock_held: bool,
    ) -> None:
        if not camera_lock_held:
            raise RuntimeError("wake session close requires the camera lock")
        with self._lock:
            session = self._sessions.get(camera_id)
            if (
                session is None
                or session.generation != generation
                or session.phase not in {"connecting", "awake"}
            ):
                return
            session.phase = phase
        try:
            self._release_capture(camera_id)
        except Exception:
            LOGGER.exception("battery wake capture release failed for %s", camera_id)
        try:
            self._set_recording_released(camera_id, False)
        except Exception:
            LOGGER.exception("battery wake recording release failed for %s", camera_id)
        self._announce(camera_id, phase)

    def _announce(self, camera_id: str, phase: str) -> None:
        if self._publish is None:
            return
        try:
            self._publish(camera_id, phase)
        except Exception:
            LOGGER.exception("battery wake state publish failed for %s", camera_id)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._poll()
            self._wake.wait(self._poll_seconds)
            self._wake.clear()

    def _poll(self) -> None:
        now = self._monotonic()
        with self._lock:
            active = [
                (camera_id, session.generation, session.phase)
                for camera_id, session in self._sessions.items()
                if session.phase in {"connecting", "awake"}
            ]
        for camera_id, generation, _phase in active:
            if self._stop.is_set():
                return
            camera = self._cameras.get(camera_id)
            camera_lock = self._camera_locks.get(camera_id)
            if camera is None or camera_lock is None:
                continue
            try:
                running = bool(self._capture_running(camera_id))
                frame_at = self._frame_monotonic(camera_id)
            except Exception:
                LOGGER.exception("battery wake poll failed for %s", camera_id)
                continue
            close_phase: WakePhase | None = None
            announce: str | None = None
            with camera_lock:
                with self._lock:
                    session = self._sessions.get(camera_id)
                    if (
                        session is None
                        or session.generation != generation
                        or session.phase not in {"connecting", "awake"}
                    ):
                        continue
                    if frame_at is not None and (
                        session.last_frame is None or frame_at > session.last_frame
                    ):
                        first_frame = not session.saw_frame
                        session.last_frame = frame_at
                        session.saw_frame = True
                        session.phase = "awake"
                        session.idle_deadline = now + camera.wake_idle_seconds
                        if first_frame:
                            announce = "awake"
                    if now >= session.hard_deadline:
                        close_phase = "asleep" if session.saw_frame else "wake_failed"
                    elif session.phase == "connecting" and now >= session.open_deadline:
                        close_phase = "wake_failed"
                    elif (
                        session.phase == "awake"
                        and session.idle_deadline is not None
                        and now >= session.idle_deadline
                    ):
                        close_phase = "asleep"
                    elif session.phase == "awake" and not running:
                        close_phase = "asleep"
                    elif (
                        session.phase == "connecting"
                        and not running
                        and now > session.started_at + 0.25
                    ):
                        close_phase = "wake_failed"
                if announce == "awake":
                    try:
                        self._notify_motion(camera_id)
                    except Exception:
                        LOGGER.exception(
                            "battery wake motion notice failed for %s",
                            camera_id,
                        )
                    self._announce(camera_id, "awake")
                if close_phase is not None:
                    self._close_locked_owner(
                        camera_id,
                        generation,
                        close_phase,
                        camera_lock_held=True,
                    )
