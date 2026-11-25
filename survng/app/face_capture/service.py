"""Background face capture: one recorded person event at a time, after the fact.

Runs only while ``detector.face_capture_enabled`` is on. It picks recent
recorded person events that have not been captured (newest first), waits until
their recordings have settled, and runs the two-pass capture with the isolated
inference workers at enrichment priority, so incident inference sheds it. A
failed event is retried a few times, then left. Faces of events that retention
has removed are forgotten periodically.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .events import person_events, plan_offsets, stored_people
from .pipeline import CaptureCancelled, CaptureSettings, capture_event, main_stream_source, sub_stream_source
from .store import CAPTURE_VERSION, ClosingConnection, FaceCaptureStore

LOGGER = logging.getLogger(__name__)

IDLE_SECONDS = 30.0
BETWEEN_EVENTS_SECONDS = 1.0
# Recordings of the event window are finalised and indexed by then.
SETTLE_SECONDS = 90.0
# A recording segment is indexed only once it closes: the event window (up to
# 30 s) plus one segment plus indexing slack must have passed.
SETTLE_SLACK_SECONDS = 60.0
MAX_ATTEMPTS = 3
CANDIDATE_BATCH = 200
PRUNE_INTERVAL_SECONDS = 6 * 3600.0
ERROR_LOG_INTERVAL_SECONDS = 300.0
# A deferred event (inference shed, worker timed out or unavailable) waits
# this long, doubling per deferral, so one event cannot hold the queue.
DEFER_BASE_SECONDS = IDLE_SECONDS
DEFER_MAX_SECONDS = 3600.0
MAX_TRACKED_DEFERRALS = 10_000


class FaceCaptureService:
    def __init__(
        self,
        *,
        config: Callable[[], Any],
        events_db_path: Path,
        inference: Callable[[], Any],
        recording_rows: Callable[..., list[dict[str, Any]]],
    ) -> None:
        self._config = config
        self._events_db_path = Path(events_db_path)
        self._store: FaceCaptureStore | None = None
        self._inference = inference
        self._recording_rows = recording_rows
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_prune = 0.0
        self._last_error_log = 0.0
        # event id -> (deferrals, monotonic time it may be tried again)
        self._deferred: dict[int, tuple[int, float]] = {}
        self.processed = 0
        self.failed = 0
        self.last_event_id: int | None = None
        self.last_error = ""

    # Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            if not self._stop.is_set():
                return
            # A previous stop timed out: let that pass finish first, so two
            # threads never capture into the same store.
            self._thread.join(30.0)
            if self._thread.is_alive():
                LOGGER.warning("face capture did not stop in time; not starting another")
                return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="face-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                return  # still finishing; start() waits for it
        self._thread = None

    def wake(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            worked = False
            try:
                worked = self.run_once()
            except Exception as exc:  # the loop must survive one bad event
                self._log_error(f"face capture pass failed: {exc}")
            self._wake.wait(BETWEEN_EVENTS_SECONDS if worked else IDLE_SECONDS)
            self._wake.clear()

    def _log_error(self, message: str) -> None:
        self.last_error = message
        now = time.monotonic()
        if now - self._last_error_log >= ERROR_LOG_INTERVAL_SECONDS:
            self._last_error_log = now
            LOGGER.warning("%s", message)

    # Work ----------------------------------------------------------------

    @property
    def store(self) -> FaceCaptureStore:
        """Created on first use, so a disabled capture leaves nothing on disk."""
        if self._store is None:
            self._store = FaceCaptureStore(self._events_db_path.parent)
        return self._store

    def enabled(self) -> bool:
        return bool(self._config().detector.face_capture_enabled)

    def next_event(self, now: datetime | None = None) -> tuple[int, str, float, dict[str, Any]] | None:
        config = self._config().detector
        now = now or datetime.now(timezone.utc)
        since = (now - timedelta(hours=float(config.face_capture_backfill_hours))).isoformat()
        segment = float(getattr(self._config(), "recording_segment_seconds", 0.0) or 0.0)
        settle = max(SETTLE_SECONDS, segment + SETTLE_SLACK_SECONDS)
        until = (now - timedelta(seconds=settle)).isoformat()
        waiting = time.monotonic()
        before_id: int | None = None
        # Page back through the backfill window, so events older than the
        # newest batch are still reached once that batch is captured.
        while True:
            with self._events_connection() as connection:
                candidates = person_events(
                    connection, since=since, until=until, limit=CANDIDATE_BATCH, before_id=before_id,
                )
            seen = self.store.attempts(event_id for event_id, *_ in candidates)
            for candidate in candidates:
                deferred = self._deferred.get(candidate[0])
                if deferred is not None and deferred[1] > waiting:
                    continue
                previous = seen.get(candidate[0])
                if previous is None:
                    return candidate
                state, attempts, version = previous
                if state == "failed" and attempts < MAX_ATTEMPTS:
                    return candidate
                if version < CAPTURE_VERSION and state == "done":
                    return candidate
            if len(candidates) < CANDIDATE_BATCH:
                return None
            before_id = candidates[-1][0]

    def run_once(self, now: datetime | None = None) -> bool:
        """Capture one event if one is due; True when an event was captured.

        A deferred or failed capture returns False, so the loop waits the
        idle interval before the next pass (and a failure's next attempt).
        """
        if not self.enabled():
            return False
        self._prune_if_due()
        candidate = self.next_event(now)
        if candidate is None:
            return False
        failed = self.failed
        # A capture deferred for incident work waits the idle interval.
        return self.capture(*candidate) is not None and self.failed == failed

    def capture(self, event_id: int, camera_id: str, epoch: float, qualification: dict[str, Any]) -> int | None:
        """Capture one event; the number of faces kept, or None when deferred."""
        from ..evidence_work import EvidenceWorkPreempted, cancellable_evidence_work
        from ..inference_runtime.types import InferenceUnavailable
        from ..recording_media import mp4_video_dimensions

        app_config = self._config()
        config = app_config.detector
        inference = self._inference()
        started = time.monotonic()
        self.last_event_id = event_id

        def rows(source: str, start: float, end: float) -> list[dict[str, Any]]:
            return list(self._recording_rows(camera_id, start, end, source, discover_missing=False))

        ffmpeg = str(app_config.ffmpeg_path or "ffmpeg")
        ffprobe = str(Path(ffmpeg).with_name("ffprobe")) if "/" in ffmpeg else "ffprobe"
        hardware = str(app_config.hardware_acceleration or "")
        try:
            # Models that did not load defer every event instead of using up
            # each one's attempts.
            unavailable = getattr(inference, "face_capture_unavailable", None)
            reason = unavailable() if callable(unavailable) else ""
            if reason:
                raise InferenceUnavailable(reason)
            # Inside the try: an event whose lookups fail is recorded as a
            # failed attempt instead of being picked again on every pass.
            main_rows = rows("main", epoch - 1.0, epoch + 1.0)
            main_size = mp4_video_dimensions(Path(main_rows[0]["path"])) if main_rows else None
            with self._events_connection() as connection:
                stored = stored_people(connection, event_id, epoch, main_size)
            with cancellable_evidence_work(self._stop.is_set, optional=True, source="face_capture"):
                result = capture_event(
                    plan_offsets=plan_offsets(config, qualification, epoch, camera_id),
                    main_frames=main_stream_source(rows, epoch, ffmpeg, ffprobe, hardware),
                    sub_frames=sub_stream_source(rows, epoch, ffmpeg),
                    find_faces=lambda crop: inference.capture_faces(crop, embed=False),
                    main_size=main_size,
                    stored_people=stored,
                    detect_people=lambda frame: _people(inference, config, frame),
                    capture_faces=lambda crop: inference.capture_faces(crop),
                    settings=CaptureSettings.from_config(config),
                    cancelled=self._stop.is_set,
                )
        except CaptureCancelled:
            return None
        except (EvidenceWorkPreempted, InferenceUnavailable):
            if not self._stop.is_set():
                self._defer(event_id)
            return None
        except Exception as exc:
            if self._stop.is_set():
                return None
            return self._fail(event_id, camera_id, epoch, str(exc))
        self._deferred.pop(event_id, None)
        if not result.faces and any(error.startswith("main stream") for error in result.errors):
            # Main-stream decoding or the face models failed and nothing was
            # kept: an attempt to retry, not an event with no faces.
            return self._fail(event_id, camera_id, epoch, "; ".join(result.errors))
        if not result.faces and not result.main_frames:
            # No main-stream recording of the window (not indexed yet, or
            # missing): an attempt to retry, not an event with no faces.
            return self._fail(event_id, camera_id, epoch, "no main-stream frames in the event window")
        try:
            kept = self.store.record(
                event_id=event_id, camera_id=camera_id, event_epoch=epoch, result=result,
                embedding_model=Path(config.face_capture_embedding_model_path).stem,
                seconds=time.monotonic() - started,
            )
        except (OSError, sqlite3.Error) as exc:
            return self._fail(event_id, camera_id, epoch, f"storing faces failed: {exc}")
        self.processed += 1
        return kept

    def _fail(self, event_id: int, camera_id: str, epoch: float, error: str) -> int:
        self._deferred.pop(event_id, None)
        self.failed += 1
        self.store.record_failure(event_id=event_id, camera_id=camera_id, event_epoch=epoch, error=error)
        self._log_error(f"face capture failed for event {event_id} on {camera_id}: {error}")
        return 0

    def _defer(self, event_id: int) -> None:
        if len(self._deferred) >= MAX_TRACKED_DEFERRALS:
            self._deferred.clear()
        count = self._deferred.get(event_id, (0, 0.0))[0]
        delay = min(DEFER_MAX_SECONDS, DEFER_BASE_SECONDS * (2 ** min(count, 16)))
        self._deferred[event_id] = (count + 1, time.monotonic() + delay)

    def _events_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(
            f"file:{self._events_db_path}?mode=ro", uri=True, timeout=10.0, factory=ClosingConnection,
        )

    def _prune_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_prune < PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        captured = self.store.captured_event_ids()
        if not captured:
            return
        existing: set[int] = set()
        with self._events_connection() as connection:
            for start in range(0, len(captured), 500):
                chunk = captured[start:start + 500]
                existing.update(int(row[0]) for row in connection.execute(
                    f"select id from events where id in ({','.join('?' * len(chunk))})", chunk))
        gone = [event_id for event_id in captured if event_id not in existing]
        if gone:
            removed = self.store.forget_events(gone)
            LOGGER.info("face capture forgot %d events removed from the event store", removed)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled(),
            "running": self._thread is not None and self._thread.is_alive(),
            "processed": self.processed,
            "failed": self.failed,
            "last_event_id": self.last_event_id,
            "last_error": self.last_error,
            **(self._store.summary() if self._store is not None else {"events": {}, "faces": 0}),
        }


def _people(inference: Any, config: Any, frame: Any) -> list[tuple[float, float, float, float]]:
    from ..inference_runtime.types import InferenceUnavailable
    from ..motion_pipeline.object_detection import _box

    objects = inference.detect_enrichment(
        frame, confidence_threshold=float(config.event_candidate_confidence_threshold),
    )
    for item in objects:
        if isinstance(item, dict) and item.get("status") in {"detector_unavailable", "inference_deferred"}:
            # No answer is not "no people": defer the capture rather than
            # recording the event as done without faces.
            raise InferenceUnavailable(str(item.get("error") or "person detection unavailable"))
    return [
        box for item in objects
        if isinstance(item, dict) and str(item.get("label") or "").lower() == "person"
        and (box := _box(item)) is not None
    ]
