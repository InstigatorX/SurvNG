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
import urllib.parse
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .events import person_events, plan_offsets, stored_people
from .pipeline import (
    MAX_WINDOW_SECONDS,
    CaptureCancelled,
    CaptureSettings,
    capture_event,
    rows_for_window,
    event_window,
    main_stream_source,
    sub_stream_source,
)
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
# After a full scan finds nothing to do, passes only look at newer events,
# with a full rescan this often (retries, capture version changes).
FULL_SCAN_SECONDS = 3600.0
PRUNE_INTERVAL_SECONDS = 6 * 3600.0
ERROR_LOG_INTERVAL_SECONDS = 300.0
# A deferred event (inference shed, worker timed out or unavailable) waits
# this long, doubling per deferral, so one event cannot hold the queue.
DEFER_BASE_SECONDS = IDLE_SECONDS
DEFER_MAX_SECONDS = 3600.0
MAX_TRACKED_DEFERRALS = 10_000
# An event whose inference keeps failing (a face-worker timeout restarts the
# worker recognition shares) counts as a failed attempt after this many
# deferrals in a row, about an hour of backoff. Shedding for incident work,
# preemption and capture models that did not load are not counted.
MAX_CONSECUTIVE_DEFERRALS = 8
# A failed event waits this long before its next attempt, so a passing cause
# (a segment not indexed yet, a decoder timing out under load) does not use
# up every attempt within a minute.
FAILURE_RETRY_SECONDS = 600.0


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
        # event id -> (deferrals, monotonic time it may be tried again); failed
        # events are held here too, with no deferrals counted.
        self._deferred: dict[int, tuple[int, float]] = {}
        # Every event at or below this id is finished for good (done or out
        # of attempts) as of the last full scan.
        self._scan_floor = 0
        self._full_scan_due = 0.0
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
        # The window is anchored to the event epoch, which can be later than
        # created_at: its recordings must have settled too.
        settled_epoch = now.timestamp() - settle - MAX_WINDOW_SECONDS
        waiting = time.monotonic()
        # Passes stay full scans until one reaches the end of the window: a
        # full scan that returns a newer candidate first has not yet reached
        # older events (e.g. confirmed after a newer one was scanned).
        floor = self._scan_floor if waiting < self._full_scan_due else 0
        pending: int | None = None  # lowest id that may still need work
        newest = 0
        before_id: int | None = None
        # Page back through the backfill window, so events older than the
        # newest batch are still reached once that batch is captured.
        while True:
            with self._events_connection() as connection:
                # The floor is applied in SQL too, so an incremental pass does
                # not read and parse every event of the backfill window.
                candidates = person_events(
                    connection, since=since, until=until, limit=CANDIDATE_BATCH, before_id=before_id,
                    after_id=floor or None,
                )
            seen = self.store.attempts(event_id for event_id, *_ in candidates)
            reached_floor = False
            for candidate in candidates:
                if candidate[0] <= floor:
                    reached_floor = True
                    break
                newest = max(newest, candidate[0])
                verdict = self._judge(candidate, seen.get(candidate[0]), waiting, settled_epoch)
                if verdict == "take":
                    return candidate
                if verdict == "wait":
                    pending = candidate[0] if pending is None else min(pending, candidate[0])
            if reached_floor or len(candidates) < CANDIDATE_BATCH:
                self._scan_floor = pending - 1 if pending is not None else max(floor, newest)
                if floor == 0:
                    self._full_scan_due = waiting + FULL_SCAN_SECONDS
                return None
            before_id = candidates[-1][0]

    def _judge(
        self, candidate: tuple[int, str, float, dict[str, Any]], previous: tuple[str, int, int] | None,
        waiting: float, settled_epoch: float,
    ) -> str:
        """take (capture it now), wait (may need work later) or skip (finished)."""
        deferred = self._deferred.get(candidate[0])
        if (deferred is not None and deferred[1] > waiting) or candidate[2] > settled_epoch:
            return "wait"
        if previous is None:
            return "take"
        state, attempts, version = previous
        if version < CAPTURE_VERSION:
            return "take"  # attempts restart with the new version
        if state == "failed" and attempts < MAX_ATTEMPTS:
            if deferred is None:
                # No hold in memory (this service started after the failure,
                # e.g. a manager reload): space the retry from the stored
                # failure time instead of retrying at once.
                failed_at = self.store.failed_at(candidate[0])
                wait = 0.0 if failed_at is None else failed_at + FAILURE_RETRY_SECONDS - time.time()
                if wait > 0:
                    self._hold(candidate[0], 0, wait)
                    return "wait"
            return "take"
        return "skip"

    def _settle(self, now: datetime) -> tuple[float, float]:
        """(latest created_at epoch, latest event epoch) whose recordings have settled."""
        segment = float(getattr(self._config(), "recording_segment_seconds", 0.0) or 0.0)
        settle = max(SETTLE_SECONDS, segment + SETTLE_SLACK_SECONDS)
        return now.timestamp() - settle, now.timestamp() - settle - MAX_WINDOW_SECONDS

    def seed_events(
        self, start_epoch: float, end_epoch: float, cameras: list[str] | None,
    ) -> Iterator[tuple[int, str, float, dict[str, Any]]]:
        """Recorded person events created in a range, newest first, paged."""
        since = datetime.fromtimestamp(start_epoch, timezone.utc).isoformat()
        until = datetime.fromtimestamp(end_epoch, timezone.utc).isoformat()
        before_id: int | None = None
        while True:
            with self._events_connection() as connection:
                page = person_events(
                    connection, since=since, until=until, cameras=cameras or None,
                    limit=CANDIDATE_BATCH, before_id=before_id,
                )
            yield from page
            if len(page) < CANDIDATE_BATCH:
                return
            before_id = page[-1][0]

    def count_seed_events(self, start_epoch: float, end_epoch: float, cameras: list[str] | None) -> int:
        """How many person events in the range still need capturing."""
        total = 0
        batch: list[tuple[int, str, float, dict[str, Any]]] = []

        def flush() -> int:
            seen = self.store.attempts(c[0] for c in batch)
            return sum(1 for c in batch if self._judge(c, seen.get(c[0]), float("-inf"), float("inf")) != "skip")

        for candidate in self.seed_events(start_epoch, end_epoch, cameras):
            batch.append(candidate)
            if len(batch) >= CANDIDATE_BATCH:
                total += flush()
                batch = []
        return total + (flush() if batch else 0)

    def next_seed_event(
        self, seed: dict[str, Any], now: datetime | None = None,
    ) -> tuple[tuple[int, str, float, dict[str, Any]] | None, bool]:
        """(the seed's next event to capture, whether any of its events may still need work)."""
        now = now or datetime.now(timezone.utc)
        latest_created, settled_epoch = self._settle(now)
        end = min(float(seed["end_epoch"]), latest_created)
        waiting = time.monotonic()
        pending = False
        batch: list[tuple[int, str, float, dict[str, Any]]] = []

        def decide() -> tuple[int, str, float, dict[str, Any]] | None:
            nonlocal pending
            seen = self.store.attempts(c[0] for c in batch)
            for candidate in batch:
                verdict = self._judge(candidate, seen.get(candidate[0]), waiting, settled_epoch)
                if verdict == "take":
                    return candidate
                pending = pending or verdict == "wait"
            return None

        for candidate in self.seed_events(float(seed["start_epoch"]), end, seed.get("cameras") or None):
            batch.append(candidate)
            if len(batch) >= CANDIDATE_BATCH:
                if (chosen := decide()) is not None:
                    return chosen, True
                batch = []
        if batch and (chosen := decide()) is not None:
            return chosen, True
        return None, pending or float(seed["end_epoch"]) > latest_created

    def run_once(self, now: datetime | None = None) -> bool:
        """Capture one event if one is due; True when an event was captured.

        A deferred or failed capture returns False, so the loop waits the
        idle interval before the next pass (and a failure's next attempt).
        """
        if not self.enabled():
            return False
        self._prune_if_due()
        candidate = self.next_event(now)
        if candidate is not None:
            failed = self.failed
            # A capture deferred for incident work waits the idle interval.
            return self.capture(*candidate) is not None and self.failed == failed
        # Nothing new waits: work through a requested seed of past events.
        seed = self.store.active_seed()
        if seed is None or seed["state"] != "running":
            return False
        candidate, more = self.next_seed_event(seed, now)
        if candidate is None:
            if not more:
                self.store.set_seed_state(seed["id"], "done")
            return False
        failed = self.failed
        kept = self.capture(*candidate)
        after = self.store.attempts([candidate[0]]).get(candidate[0])
        if after is not None and after[2] == CAPTURE_VERSION:
            if after[0] == "done" and kept is not None:
                self.store.seed_progress(seed["id"], faces=kept)
            elif after[0] == "failed" and after[1] >= MAX_ATTEMPTS and self.failed != failed:
                self.store.seed_progress(seed["id"], failed=True)
        return kept is not None and self.failed == failed

    def capture(self, event_id: int, camera_id: str, epoch: float, qualification: dict[str, Any]) -> int | None:
        """Capture one event; the number of faces kept, or None when deferred."""
        from ..evidence_work import EvidenceWorkPreempted, cancellable_evidence_work
        from ..inference_runtime.types import InferenceShed, InferenceUnavailable
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
        # Models that did not load hold up every event alike: nothing is held
        # or counted against this one, and the next pass checks again.
        unavailable = getattr(inference, "face_capture_unavailable", None)
        reason = unavailable() if callable(unavailable) else ""
        if reason:
            self._log_error(f"face capture waiting: {reason}")
            return None
        try:
            # Inside the try: an event whose lookups fail is recorded as a
            # failed attempt instead of being picked again on every pass.
            plan = plan_offsets(config, qualification, epoch, camera_id)
            settings = CaptureSettings.from_config(config)
            # The main stream's size from any segment of the window, so the
            # sub-stream and stored boxes are in main-stream pixels even when
            # no segment covers the epoch itself.
            low, high = event_window(plan, settings.margin_seconds)
            # One recording-index lookup per source for the whole event; it
            # reaches past the window by the moment radius, as pass two does.
            reach = settings.moment_radius_seconds + 1.0
            window_rows = rows_for_window(rows, epoch + low - reach, epoch + high + reach)
            main_rows = window_rows("main", epoch + low, epoch + high)
            main_size = next(
                (size for row in main_rows if (size := mp4_video_dimensions(Path(row["path"])))), None,
            )
            with self._events_connection() as connection:
                stored = stored_people(connection, event_id, epoch, main_size)
            with cancellable_evidence_work(self._stop.is_set, optional=True, source="face_capture"):
                result = capture_event(
                    plan_offsets=plan,
                    main_frames=main_stream_source(window_rows, epoch, ffmpeg, ffprobe, hardware),
                    sub_frames=sub_stream_source(window_rows, epoch, ffmpeg),
                    find_faces=lambda crop: inference.capture_faces(crop, embed=False),
                    main_size=main_size,
                    stored_people=stored,
                    detect_people=lambda frame: _people(inference, config, frame),
                    capture_faces=lambda crop: inference.capture_faces(crop),
                    settings=settings,
                    cancelled=self._stop.is_set,
                )
        except CaptureCancelled:
            return None
        except (EvidenceWorkPreempted, InferenceShed):
            # Incident work came first: retried soon, never counted.
            if not self._stop.is_set():
                self._defer(event_id, counted=False)
            return None
        except InferenceUnavailable as exc:
            if self._stop.is_set():
                return None
            if self._deferred.get(event_id, (0, 0.0))[0] + 1 >= MAX_CONSECUTIVE_DEFERRALS:
                self._fail(event_id, camera_id, epoch, f"deferred {MAX_CONSECUTIVE_DEFERRALS} times: {exc}")
                return None
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
        except Exception as exc:  # cv2 and encoding errors too: an attempt, not a crashed pass
            return self._fail(event_id, camera_id, epoch, f"storing faces failed: {exc}")
        self.processed += 1
        return kept

    def _fail(self, event_id: int, camera_id: str, epoch: float, error: str) -> int:
        self._hold(event_id, 0, FAILURE_RETRY_SECONDS)
        self.failed += 1
        self.store.record_failure(event_id=event_id, camera_id=camera_id, event_epoch=epoch, error=error)
        self._log_error(f"face capture failed for event {event_id} on {camera_id}: {error}")
        return 0

    def _defer(self, event_id: int, *, counted: bool = True) -> None:
        count = self._deferred.get(event_id, (0, 0.0))[0]
        delay = min(DEFER_MAX_SECONDS, DEFER_BASE_SECONDS * (2 ** min(count, 16)))
        self._hold(event_id, count + 1 if counted else count, delay)

    def _hold(self, event_id: int, count: int, delay: float) -> None:
        if event_id not in self._deferred and len(self._deferred) >= MAX_TRACKED_DEFERRALS:
            self._deferred.clear()
        self._deferred[event_id] = (count, time.monotonic() + delay)

    def _events_connection(self) -> sqlite3.Connection:
        # Quoted: a '?', '#' or '%' in the path would otherwise be read as URI syntax.
        return sqlite3.connect(
            f"file:{urllib.parse.quote(str(self._events_db_path))}?mode=ro",
            uri=True, timeout=10.0, factory=ClosingConnection,
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
            "seeds": self._store.seeds(10) if self._store is not None else [],
            **(self._store.summary() if self._store is not None else {"events": {}, "faces": 0}),
        }


def _people(inference: Any, config: Any, frame: Any) -> list[tuple[float, float, float, float]]:
    from ..inference_runtime.types import InferenceShed, InferenceUnavailable
    from ..motion_pipeline.object_detection import _box

    objects = inference.detect_enrichment(
        frame, confidence_threshold=float(config.event_candidate_confidence_threshold),
    )
    for item in objects:
        if isinstance(item, dict) and item.get("status") == "inference_deferred":
            # Shed for incident work: no answer is not "no people".
            raise InferenceShed(str(item.get("error") or "person detection deferred"))
        if isinstance(item, dict) and item.get("status") == "detector_unavailable":
            raise InferenceUnavailable(str(item.get("error") or "person detection unavailable"))
    return [
        box for item in objects
        if isinstance(item, dict) and str(item.get("label") or "").lower() == "person"
        and (box := _box(item)) is not None
    ]
