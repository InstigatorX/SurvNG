from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .security import redact_secret_text


LOGGER = logging.getLogger("uvicorn.error")


@dataclass(frozen=True, slots=True)
class CameraStartupTask:
    """One camera's bounded startup workflow.

    The coordinator owns ordering and concurrency. The callbacks keep camera,
    capture, and recorder implementations outside this service.
    """

    camera_id: str
    is_enabled: Callable[[], bool]
    start_camera: Callable[[], None]
    capture_ready: Callable[[], bool]
    start_recorders: Callable[[], None]
    publish_state: Callable[[], None]


@dataclass(slots=True)
class _CameraStartupState:
    phase: str = "queued"
    queued_at: str = ""
    started_at: str = ""
    first_frame_at: str = ""
    completed_at: str = ""
    wait_seconds: float = 0.0
    total_seconds: float = 0.0
    error: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "first_frame_at": self.first_frame_at,
            "completed_at": self.completed_at,
            "wait_seconds": round(self.wait_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "error": self.error,
        }


class CameraStartupCoordinator:
    """Admit camera connection workflows with bounded concurrency.

    A camera that does not produce a frame within the readiness window is
    degraded, not fatal. Its capture service keeps reconnecting independently,
    while the coordinator releases the slot so other cameras can start.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 2,
        readiness_timeout_seconds: float = 5.0,
        recorder_settle_seconds: float = 0.5,
        poll_interval_seconds: float = 0.1,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self.readiness_timeout_seconds = max(0.0, float(readiness_timeout_seconds))
        self.recorder_settle_seconds = max(0.0, float(recorder_settle_seconds))
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._completed = threading.Event()
        self._thread: threading.Thread | None = None
        self._states: dict[str, _CameraStartupState] = {}
        self._admission_complete = False
        self._started_monotonic: float | None = None
        self._completed_monotonic: float | None = None
        self._on_complete: Callable[[], None] | None = None

    def start(
        self,
        tasks: Sequence[CameraStartupTask],
        *,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        task_snapshot = tuple(tasks)
        complete_inline = False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("camera startup is already active")
            self._cancel.clear()
            self._completed.clear()
            self._admission_complete = False
            queued_at = self._now_iso()
            self._states = {
                task.camera_id: _CameraStartupState(queued_at=queued_at)
                for task in task_snapshot
            }
            self._started_monotonic = self._monotonic_clock()
            self._completed_monotonic = None
            self._on_complete = on_complete
            if not task_snapshot:
                self._completed_monotonic = self._started_monotonic
                self._admission_complete = True
                self._thread = None
                self._completed.set()
                complete_inline = True
            else:
                thread = threading.Thread(
                    target=self._run,
                    args=(task_snapshot,),
                    name="camera-startup-coordinator",
                    daemon=False,
                )
                self._thread = thread
                thread.start()
        if complete_inline and on_complete is not None:
            try:
                on_complete()
            except Exception as exc:
                LOGGER.error(
                    "camera startup completion callback failed: %s",
                    redact_secret_text(exc)[:240],
                )

    def cancel(self, *, timeout: float = 10.0) -> bool:
        thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        self._cancel.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        return not thread.is_alive()

    def wait(self, timeout: float | None = None) -> bool:
        return self._completed.wait(timeout)

    def status(self) -> dict[str, Any]:
        with self._lock:
            states = {
                camera_id: state.snapshot()
                for camera_id, state in self._states.items()
            }
            started = self._started_monotonic
            completed = self._completed_monotonic
            active = (
                self._thread is not None
                and self._thread.is_alive()
                and not self._admission_complete
            )
            complete = self._admission_complete
        phases: dict[str, int] = {}
        for state in states.values():
            phase = str(state["phase"])
            phases[phase] = phases.get(phase, 0) + 1
        elapsed = 0.0
        if started is not None:
            elapsed = (completed or self._monotonic_clock()) - started
        return {
            "active": active,
            "complete": complete,
            "cancelled": self._cancel.is_set(),
            "max_concurrency": self.max_concurrency,
            "readiness_timeout_seconds": self.readiness_timeout_seconds,
            "elapsed_seconds": round(max(0.0, elapsed), 3),
            "counts": phases,
            "cameras": states,
        }

    def _run(self, tasks: tuple[CameraStartupTask, ...]) -> None:
        work: queue.Queue[CameraStartupTask] = queue.Queue()
        for task in tasks:
            work.put(task)
        workers = [
            threading.Thread(
                target=self._run_worker,
                args=(work,),
                name=f"camera-startup-{index + 1}",
                daemon=False,
            )
            for index in range(min(self.max_concurrency, len(tasks)))
        ]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            self._mark_unstarted_cancelled()
        finally:
            with self._lock:
                self._completed_monotonic = self._monotonic_clock()
                self._admission_complete = True
        try:
            callback = self._on_complete
            if callback is not None and not self._cancel.is_set():
                try:
                    callback()
                except Exception as exc:
                    LOGGER.error(
                        "camera startup completion callback failed: %s",
                        redact_secret_text(exc)[:240],
                    )
        finally:
            self._completed.set()

    def _run_worker(self, work: queue.Queue[CameraStartupTask]) -> None:
        while not self._cancel.is_set():
            try:
                task = work.get_nowait()
            except queue.Empty:
                return
            try:
                self._run_task(task)
            finally:
                work.task_done()

    def _run_task(self, task: CameraStartupTask) -> None:
        started = self._monotonic_clock()
        self._update(task.camera_id, phase="starting", started_at=self._now_iso())
        ready = False
        try:
            if not task.is_enabled():
                self._publish_state(task)
                self._finish(task.camera_id, started, phase="skipped")
                return
            task.start_camera()
            if self._cancel.is_set():
                self._finish(task.camera_id, started, phase="cancelled")
                return
            if not task.is_enabled():
                self._publish_state(task)
                self._finish(task.camera_id, started, phase="skipped")
                return
            self._update(task.camera_id, phase="starting_recorders")
            task.start_recorders()
            recorders_started = self._monotonic_clock()
            self._update(task.camera_id, phase="waiting_for_frame")
            deadline = started + self.readiness_timeout_seconds
            while not self._cancel.is_set():
                if not task.is_enabled():
                    self._publish_state(task)
                    self._finish(task.camera_id, started, phase="skipped")
                    return
                if task.capture_ready():
                    ready = True
                    self._update(
                        task.camera_id,
                        first_frame_at=self._now_iso(),
                        wait_seconds=self._monotonic_clock() - started,
                    )
                    break
                remaining = deadline - self._monotonic_clock()
                if remaining <= 0:
                    break
                self._cancel.wait(min(self.poll_interval_seconds, remaining))
            if not ready:
                self._update(
                    task.camera_id,
                    wait_seconds=self._monotonic_clock() - started,
                )
            if self._cancel.is_set():
                self._finish(task.camera_id, started, phase="cancelled")
                return
            settle_remaining = self.recorder_settle_seconds - (
                self._monotonic_clock() - recorders_started
            )
            if settle_remaining > 0:
                self._cancel.wait(settle_remaining)
            if self._cancel.is_set():
                self._finish(task.camera_id, started, phase="cancelled")
                return
            self._publish_state(task)
            self._finish(
                task.camera_id,
                started,
                phase="ready" if ready else "degraded",
            )
            LOGGER.info(
                "camera startup camera=%s phase=%s total=%.2fs",
                task.camera_id,
                "ready" if ready else "degraded",
                self._monotonic_clock() - started,
            )
        except Exception as exc:
            error = redact_secret_text(exc)[:240]
            self._finish(
                task.camera_id,
                started,
                phase="failed",
                error=error,
            )
            self._publish_state(task)
            LOGGER.error("camera startup failed for %s: %s", task.camera_id, error)

    @staticmethod
    def _publish_state(task: CameraStartupTask) -> None:
        try:
            task.publish_state()
        except Exception as exc:
            LOGGER.error(
                "camera startup state publish failed for %s: %s",
                task.camera_id,
                redact_secret_text(exc)[:240],
            )

    def _finish(
        self,
        camera_id: str,
        started: float,
        *,
        phase: str,
        wait_seconds: float | None = None,
        error: str = "",
    ) -> None:
        values: dict[str, Any] = {
            "phase": phase,
            "completed_at": self._now_iso(),
            "total_seconds": self._monotonic_clock() - started,
            "error": error,
        }
        if wait_seconds is not None:
            values["wait_seconds"] = wait_seconds
        self._update(camera_id, **values)

    def _mark_unstarted_cancelled(self) -> None:
        if not self._cancel.is_set():
            return
        now = self._now_iso()
        with self._lock:
            for state in self._states.values():
                if state.phase == "queued":
                    state.phase = "cancelled"
                    state.completed_at = now

    def _update(self, camera_id: str, **values: Any) -> None:
        with self._lock:
            state = self._states[camera_id]
            for name, value in values.items():
                setattr(state, name, value)

    def _now_iso(self) -> str:
        return self._wall_clock().isoformat()
