"""Bounded, user-requested recording review, independent of incident state.

Results describe sparse samples, never exhaustive coverage or unique objects.
The analyzer must bound individual frame reads and honor its stop event.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path


LOGGER = logging.getLogger(__name__)
POLICY_VERSION = "recording-review-sparse-v1"
PLANNED_SAMPLES = 12
MAX_PENDING = 8
MAX_RETAINED = 1000
REQUEST_TTL_SECONDS = 60.0
PROCESSING_BUDGET_SECONDS = 180.0
MAX_OBJECTS_PER_SAMPLE = 100


class RecordingReviewService:
    def __init__(self, database_dir, *, manifest_provider, analysis_identity, analyze):
        self.database_path = Path(database_dir) / "recording_review.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_provider = manifest_provider
        self.analysis_identity = analysis_identity
        self.analyze = analyze
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._worker_lock_file = None
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, manifest TEXT NOT NULL,
                state TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                demand_until REAL NOT NULL)""")
            connection.execute("CREATE INDEX IF NOT EXISTS review_queue ON reviews(state, created_at)")

    @contextmanager
    def _connect(self, *, readonly=False):
        connection = sqlite3.connect(
            self.database_path.absolute().as_uri() + ("?mode=ro" if readonly else ""),
            uri=True, timeout=5,
        )
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _identity(self, camera_id, source, epoch):
        try:
            timestamp = float(epoch)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("a finite timestamp is required") from None
        if not camera_id or source not in {"main", "live"} or not math.isfinite(timestamp):
            raise ValueError("a camera, main/live source, and finite timestamp are required")
        start = math.floor(timestamp / 60) * 60
        end = start + 60
        if start < 0 or end > time.time():
            raise ValueError("select a completed recording minute")
        manifest = list(self.manifest_provider(camera_id, source, start, end))
        if len(manifest) > 128:
            raise ValueError("recording review manifest exceeds the segment limit")
        # Sort canonically so catalog ordering alone cannot invalidate results.
        manifest = sorted(manifest, key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False))
        revision = self._analysis_revision()
        identity = json.dumps([POLICY_VERSION, camera_id, source, start, end, revision, manifest],
                              sort_keys=True, separators=(",", ":"), allow_nan=False)
        request_id = hashlib.sha256(identity.encode()).hexdigest()
        payload = {
            "request_id": request_id, "camera_id": camera_id, "source": source,
            "start_epoch": start, "end_epoch": end, "analysis_revision": revision,
            "state": "unreviewed" if manifest else "unavailable",
            "has_recordings": bool(manifest), "sample_count": 0,
            "planned_samples": PLANNED_SAMPLES, "observations": [],
            "sampled_timestamps": [], "missing_timestamps": [],
        }
        if not manifest:
            payload["message"] = "No indexed recordings are available for this minute."
        return payload, manifest

    def _analysis_revision(self):
        return hashlib.sha256(str(self.analysis_identity()).encode()).hexdigest()[:16]

    @staticmethod
    def _interrupted(payload, message):
        payload = dict(payload)
        payload["state"] = "partial" if payload["sample_count"] else "failed"
        payload["message"] = message
        return payload

    def status(self, camera_id, source, epoch):
        """Read only: no acquisition, requests, pruning, or lease renewal."""
        payload, _ = self._identity(camera_id, source, epoch)
        with self._connect(readonly=True) as connection:
            row = connection.execute("SELECT payload,state,demand_until FROM reviews WHERE id=?",
                                     (payload["request_id"],)).fetchone()
        if row is None:
            return payload
        result = json.loads(row["payload"])
        if row["state"] == "queued" and row["demand_until"] <= time.time():
            result = self._interrupted(result, "Review request expired; request it again to retry.")
        return result

    def request(self, camera_id, source, epoch, *, retry=False, expected_request_id=None):
        payload, manifest = self._identity(camera_id, source, epoch)
        if expected_request_id is not None and expected_request_id != payload["request_id"]:
            # Heartbeats renew existing consent, never authorize changed work.
            return self.status(camera_id, source, epoch)
        if not manifest:
            return payload
        now = time.time()
        with self._lock, self._connect() as connection:
            if self._stop.is_set():
                raise RuntimeError("Recording review is stopping; no new requests can be accepted.")
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM reviews WHERE id=?", (payload["request_id"],)).fetchone()
            if row is not None:
                result = json.loads(row["payload"])
                if row["state"] in {"queued", "analyzing"}:
                    connection.execute("UPDATE reviews SET demand_until=?,updated_at=? WHERE id=?",
                                       (now + REQUEST_TTL_SECONDS, now, payload["request_id"]))
                    self._wake.set()
                    return result
                if not retry or row["state"] not in {"failed", "partial", "unavailable"}:
                    return result
            self._expire_queued(connection, now)
            pending = connection.execute("SELECT count(*) FROM reviews WHERE state IN ('queued','analyzing')").fetchone()[0]
            if pending >= MAX_PENDING:
                raise RuntimeError("Recording review queue is full; try again after a review finishes.")
            payload["state"] = "queued"
            connection.execute("""INSERT INTO reviews VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,manifest=excluded.manifest,
                state=excluded.state,created_at=excluded.created_at,updated_at=excluded.updated_at,
                demand_until=excluded.demand_until""",
                               (payload["request_id"], json.dumps(payload), json.dumps(manifest), "queued",
                                now, now, now + REQUEST_TTL_SECONDS))
            connection.execute("""DELETE FROM reviews WHERE id IN (
                SELECT id FROM reviews WHERE state NOT IN ('queued','analyzing')
                ORDER BY updated_at DESC LIMIT -1 OFFSET ?)""", (MAX_RETAINED - MAX_PENDING,))
        self._wake.set()
        return payload

    def _expire_queued(self, connection, now):
        rows = connection.execute("SELECT id,payload FROM reviews WHERE state='queued' AND demand_until<=?", (now,)).fetchall()
        for row in rows:
            payload = self._interrupted(json.loads(row["payload"]), "Review request expired before it started.")
            self._save(connection, payload, now)

    @staticmethod
    def _save(connection, payload, now):
        connection.execute("UPDATE reviews SET payload=?,state=?,updated_at=? WHERE id=?",
                           (json.dumps(payload, allow_nan=False), payload["state"], now, payload["request_id"]))

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            lock_file = self.database_path.with_suffix(".lock").open("a")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lock_file.close()
                raise RuntimeError("A recording review worker already owns this database.") from None
            self._worker_lock_file = lock_file
            # Crash/shutdown recovery never silently resumes expensive work.
            try:
                with self._connect() as connection:
                    for row in connection.execute("SELECT payload FROM reviews WHERE state IN ('queued','analyzing')").fetchall():
                        payload = self._interrupted(json.loads(row["payload"]), "Review interrupted by service restart.")
                        self._save(connection, payload, time.time())
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, name="recording-review", daemon=True)
                self._thread.start()
            except Exception:
                lock_file.close()
                self._worker_lock_file = None
                raise

    def stop(self):
        with self._lock:
            self._stop.set()
        if self._worker_lock_file is None:
            return
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=10)
            if thread.is_alive():
                LOGGER.error("Recording review analyzer did not stop within its shutdown grace period")
                raise RuntimeError("Recording review analyzer is still running; shared resources must remain available.")
        with self._lock, self._connect() as connection:
            for row in connection.execute("SELECT payload FROM reviews WHERE state IN ('queued','analyzing')").fetchall():
                self._save(connection, self._interrupted(json.loads(row["payload"]), "Review interrupted by shutdown."), time.time())
        if self._worker_lock_file is not None:
            self._worker_lock_file.close()
            self._worker_lock_file = None

    def _run(self):
        last_failure_log = -float("inf")
        while not self._stop.is_set():
            self._wake.clear()
            try:
                with self._lock, self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    # This thread owns the sole worker lease and no analyzer is
                    # active between iterations. A leftover running row means
                    # final persistence failed; recover without replaying video.
                    for abandoned in connection.execute("SELECT payload FROM reviews WHERE state='analyzing'").fetchall():
                        recovered = self._interrupted(json.loads(abandoned["payload"]),
                                                      "Review interrupted while saving results; completed samples are preserved.")
                        self._save(connection, recovered, time.time())
                    self._expire_queued(connection, time.time())
                    row = connection.execute("SELECT * FROM reviews WHERE state='queued' ORDER BY created_at LIMIT 1").fetchone()
                    if row is not None:
                        payload = json.loads(row["payload"])
                        payload["state"] = "analyzing"
                        self._save(connection, payload, time.time())
                if row is None:
                    self._wake.wait()
                    continue
                self._process(payload, json.loads(row["manifest"]))
            except Exception as error:
                # Do not expose media paths, stream URLs, or detector internals.
                if time.monotonic() - last_failure_log >= 60:
                    LOGGER.error("Recording review worker failed (%s)", type(error).__name__)
                    last_failure_log = time.monotonic()
                self._wake.wait(1)

    def _process(self, payload, manifest):
        began = time.monotonic()
        targets = [payload["start_epoch"] + 2.5 + 5 * index for index in range(PLANNED_SAMPLES)]
        seen = set()
        iterator = None
        interrupted = False
        policy_changed = False
        failed_samples = 0
        deferred_samples = 0
        try:
            current, _ = self._identity(payload["camera_id"], payload["source"], payload["start_epoch"])
            if current["request_id"] != payload["request_id"]:
                payload["state"] = "failed"
                payload["message"] = "Recording availability or analysis settings changed; request a fresh review."
                return
            job = {**payload, "manifest": manifest, "sample_targets": targets}
            iterator = iter(self.analyze(job, self._stop))
            while len(seen) < PLANNED_SAMPLES:
                with self._connect(readonly=True) as connection:
                    demand = connection.execute("SELECT demand_until FROM reviews WHERE id=?", (payload["request_id"],)).fetchone()
                if self._stop.is_set() or time.monotonic() - began >= PROCESSING_BUDGET_SECONDS or demand[0] <= time.time():
                    interrupted = True
                    break
                if self._analysis_revision() != payload["analysis_revision"]:
                    policy_changed = True
                    break
                try:
                    sample = next(iterator)
                except StopIteration:
                    interrupted = self._stop.is_set() or time.monotonic() - began >= PROCESSING_BUDGET_SECONDS
                    break
                # A detector/config reload during this frame makes its provenance
                # uncertain. Never append it to results from the previous policy.
                if self._analysis_revision() != payload["analysis_revision"]:
                    policy_changed = True
                    break
                timestamp = float(sample["timestamp"])
                if timestamp not in targets or timestamp in seen:
                    raise ValueError("analyzer returned an invalid or duplicate sample target")
                seen.add(timestamp)
                if sample.get("status") not in {"sampled", "unavailable", "deferred", "failed"}:
                    raise ValueError("analyzer returned an invalid sample status")
                failed_samples += sample.get("status") == "failed"
                deferred_samples += sample.get("status") == "deferred"
                if sample.get("status") == "sampled":
                    payload["sampled_timestamps"].append(timestamp)
                    for raw in list(sample.get("objects") or [])[:MAX_OBJECTS_PER_SAMPLE]:
                        label = str(raw.get("label") or "")[:80]
                        confidence = float(raw.get("confidence", 0))
                        if not label or not math.isfinite(confidence):
                            continue
                        observation = {"timestamp": timestamp, "label": label, "confidence": max(0.0, min(1.0, confidence))}
                        box = raw.get("box")
                        if isinstance(box, dict):
                            safe_box = {key: float(box[key]) for key in ("x1", "y1", "x2", "y2") if key in box}
                            if len(safe_box) == 4 and all(math.isfinite(value) for value in safe_box.values()):
                                observation["box"] = safe_box
                        payload["observations"].append(observation)
                else:
                    payload["missing_timestamps"].append(timestamp)
                payload["sample_count"] = len(payload["sampled_timestamps"])
                with self._connect() as connection:
                    demand = connection.execute("SELECT demand_until FROM reviews WHERE id=?", (payload["request_id"],)).fetchone()
                    interrupted = (self._stop.is_set() or time.monotonic() - began >= PROCESSING_BUDGET_SECONDS
                                   or demand[0] <= time.time())
                    self._save(connection, payload, time.time())
                if interrupted:
                    break
            payload["missing_timestamps"] = sorted(set(targets) - set(payload["sampled_timestamps"]))
            if policy_changed:
                payload = self._interrupted(payload, "Analysis settings changed during review; request a fresh review.")
            elif interrupted:
                payload = self._interrupted(payload, "Review stopped because demand expired, the budget was reached, or the service stopped.")
            else:
                payload["state"] = "sampled" if payload["sample_count"] == PLANNED_SAMPLES else "partial" if payload["sample_count"] else "failed" if failed_samples or deferred_samples else "unavailable"
                payload["message"] = ("Some samples could not run while resources were busy; retry explicitly."
                                      if deferred_samples else "Sparse samples only; unsampled activity may be missed.")
        except Exception as error:
            LOGGER.warning("Recording review %s failed (%s)", payload["request_id"][:12], type(error).__name__)
            payload["state"] = "partial" if payload["sample_count"] else "failed"
            payload["message"] = "Review could not finish; any completed samples are preserved."
            payload["missing_timestamps"] = sorted(set(targets) - set(payload["sampled_timestamps"]))
        finally:
            if iterator is not None and callable(getattr(iterator, "close", None)):
                try:
                    iterator.close()
                except Exception as error:
                    LOGGER.warning("Recording review analyzer cleanup failed (%s)", type(error).__name__)
            with self._connect() as connection:
                self._save(connection, payload, time.time())
