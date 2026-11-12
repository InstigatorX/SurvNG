"""Age out database rows that no incident or retained evidence depends on.

Live scene discovery stores a sample and an immutable activity decision for
every analysed frame, including empty ones, and motion qualification stores
an audit per trigger. Recording and snapshot retention expire media but never
these rows, so the database grew without bound.

Rules, each rechecked inside the deleting transaction:

* Event-linked rows are never removed here; they follow their event.
* Unreferenced ``pending`` decisions (almost all empty discovery frames) age
  out after ``empty_discovery_hours``; unreferenced unsupported/incomplete
  decisions after ``activity_days``. Supported decisions are kept: admission
  may still consume them.
* Acquired samples age out on the same two schedules (empty live-discovery
  frames vs. everything else) once nothing links them: an event, an episode,
  a candidate job, or a retained decision.
* Acquired observations age out after ``activity_days`` once no sample and no
  scene observation links them.
* Finished candidate jobs that never admitted an event age out after
  ``activity_days``.
* Motion audits not linked to an incident age out after
  ``motion_audit_days``; their images (in a motion-sample or snapshot
  directory) are deleted with them when no other row names the file in
  either its relative or absolute spelling.

Acquisition images are registered in ``scene_snapshot_assets`` and keep
following snapshot retention, so removing their rows never orphans a file.
Scans read one page at a time and deletes commit in small batches, so memory
stays bounded and live writers interleave.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .config import DatabaseRetentionConfig
from .incident_utils import event_snapshot_path
from .volume_identity import VolumeGate, VolumeIdentityStore, identity_db_path

LOGGER = logging.getLogger(__name__)

STARTUP_DELAY_SECONDS = 600.0
BATCH_ROWS = 500
SCAN_ROWS = 5000
BATCH_PAUSE_SECONDS = 0.05
# A decision is recorded within minutes of its samples; this margin bounds
# which surviving decisions can still name a sample old enough to prune.
DECISION_SAMPLE_MARGIN_SECONDS = 86400.0
STEPS = (
    "candidate_jobs",
    "activity_decisions",
    "acquired_samples",
    "acquired_observations",
    "motion_audits",
)
# Columns whose value keeps an image file alive. Audit images live in the
# motion-sample directory or, for most triggers, the snapshot directory, so
# every table that can name a snapshot must be listed here.
IMAGE_OWNERS = (
    ("motion_audits", "snapshot_path"),
    ("events", "snapshot_path"),
    ("incidents", "snapshot_path"),
    ("incident_observations", "snapshot_path"),
    ("event_source_observations", "snapshot_path"),
    ("face_observations", "snapshot_path"),
    ("scene_observations", "snapshot_path"),
    ("scene_evidence_images", "snapshot_path"),
    ("acquired_samples", "snapshot_path"),
    ("acquired_observations", "snapshot_path"),
    ("scene_snapshot_assets", "snapshot_path"),
    ("semantic_embeddings", "image_path"),
    ("semantic_projection_receipts", "image_path"),
)
# Directories an audit image may be removed from.
AUDIT_IMAGE_ROLES = ("motion_audits", "snapshots")
AUDIT_IMAGE_DIRECTORIES = ("motion_samples", "snapshots")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for offset in range(0, len(items), size):
        yield items[offset:offset + size]


def _marks(items: Sequence[Any]) -> str:
    return ",".join("?" for _ in items)


def _path_spellings(storage_dir: Path, raw: str) -> set[str]:
    """Every way a row may name this file: as stored, relative, absolute."""
    spellings = {raw}
    for root in dict.fromkeys((storage_dir, storage_dir.expanduser().resolve(strict=False))):
        path = Path(raw)
        absolute = path if path.is_absolute() else root / path
        spellings.add(str(absolute))
        try:
            spellings.add(str(absolute.relative_to(root)))
        except ValueError:
            pass
    return spellings


def _audit_image_location(storage_dir: Path, path: Path, media_storage: Any) -> bool:
    """True when ``path`` is in a directory audit images may be removed from."""
    if media_storage is None:
        resolved = path.resolve(strict=False)
        root = storage_dir.resolve(strict=False)
        for directory in AUDIT_IMAGE_DIRECTORIES:
            try:
                resolved.relative_to(root / directory)
            except ValueError:
                continue
            return True
        return False
    return any(
        media_storage.location_id_for(path, role=role) is not None
        for role in AUDIT_IMAGE_ROLES
    )


class _Stopped(Exception):
    """Shutdown requested mid-run; committed batches stay committed."""


class DatabaseRetention:
    """Daily background pruning of unreferenced working data."""

    def __init__(
        self,
        events: Any,
        config: Callable[[], DatabaseRetentionConfig],
        *,
        clock: Callable[[], float] = time.time,
        pause_seconds: float = BATCH_PAUSE_SECONDS,
        startup_delay_seconds: float = STARTUP_DELAY_SECONDS,
    ) -> None:
        self._events = events
        self._config = config
        self._clock = clock
        self._pause_seconds = pause_seconds
        self._startup_delay_seconds = startup_delay_seconds
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_lock = threading.Lock()
        self._running = False
        self._status: dict[str, Any] = {"runs": 0, "last_run": None, "next_run_at": None}

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="survng-database-retention", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=30.0)
        self._thread = None

    def reschedule(self) -> None:
        """Run again after the startup delay, e.g. after a policy change."""
        self._wake.set()

    def status(self) -> dict[str, Any]:
        config = self._config()
        with self._status_lock:
            return {
                "enabled": bool(config.enabled),
                "dry_run": bool(config.dry_run),
                "running": self._running,
                "policy": {
                    "empty_discovery_hours": float(config.empty_discovery_hours),
                    "activity_days": int(config.activity_days),
                    "motion_audit_days": int(config.motion_audit_days),
                    "interval_hours": float(config.interval_hours),
                },
                **dict(self._status),
            }

    def _loop(self) -> None:
        delay = self._startup_delay_seconds
        while True:
            with self._status_lock:
                self._status["next_run_at"] = _iso(self._clock() + delay)
            woken = self._wake.wait(delay)
            self._wake.clear()
            if self._stop.is_set():
                return
            if woken:
                delay = self._startup_delay_seconds
                continue
            config = self._config()
            if config.enabled:
                try:
                    self.run_once(honor_config=True)
                except Exception as error:  # keep the loop alive; report it
                    LOGGER.exception("database retention run failed")
                    with self._status_lock:
                        self._running = False
                        self._status["last_error"] = type(error).__name__
            delay = max(3600.0, float(config.interval_hours) * 3600.0)

    # ------------------------------------------------------------------ run
    def run_once(self, *, dry_run: bool | None = None, honor_config: bool = False) -> dict[str, Any]:
        config = self._config()
        dry = bool(config.dry_run if dry_run is None else dry_run)
        self._honor_config = honor_config
        now = self._clock()
        cutoffs = {
            "empty": now - float(config.empty_discovery_hours) * 3600.0,
            "activity": now - float(config.activity_days) * 86400.0,
            "audit": now - float(config.motion_audit_days) * 86400.0,
        }
        started = time.monotonic()
        with self._status_lock:
            self._running = True
        result: dict[str, Any] = {
            "started_at": _iso(now),
            "dry_run": dry,
            "steps": {name: {"eligible": 0, "deleted": 0} for name in STEPS},
            "motion_sample_files_deleted": 0,
            "motion_sample_files_missing": 0,
            "motion_sample_files_failed": 0,
        }
        steps = result["steps"]
        try:
            self._ensure_deletion_work()
            if not dry:
                self._drain_motion_deletions(result)
            self._candidate_jobs(steps["candidate_jobs"], cutoffs, dry)
            referenced = self._referenced_decisions()
            self._activity_decisions(steps["activity_decisions"], cutoffs, referenced, dry)
            kept = self._retained_decision_samples(cutoffs, referenced)
            self._acquired_samples(steps["acquired_samples"], cutoffs, kept, dry)
            self._acquired_observations(steps["acquired_observations"], cutoffs, dry)
            self._motion_audits(result, cutoffs, dry)
        except _Stopped:
            result["interrupted"] = True
        finally:
            result["duration_seconds"] = round(time.monotonic() - started, 1)
            with self._status_lock:
                self._running = False
                self._status["runs"] = int(self._status.get("runs") or 0) + 1
                self._status["last_run"] = result
                self._status.pop("last_error", None)
        LOGGER.info(
            "database retention %s: %s", "dry run" if dry else "run", result["steps"],
        )
        return result

    # ---------------------------------------------------------------- rules
    @staticmethod
    def _decision_expires(verdict: str, created: float, cutoffs: dict[str, float]) -> bool:
        if verdict == "pending":
            return created < cutoffs["empty"]
        if verdict in {"unsupported", "incomplete"}:
            return created < cutoffs["activity"]
        return False

    # ---------------------------------------------------------------- steps
    def _candidate_jobs(self, step: dict[str, int], cutoffs: dict[str, float], dry: bool) -> None:
        due = (
            "state in ('complete','incomplete') and updated_at < ? and not exists("
            "select 1 from scene_candidate_admissions a where a.candidate_id=scene_candidate_jobs.id)"
        )
        cutoff = cutoffs["activity"]

        def delete(conn: sqlite3.Connection, batch: Sequence[str]) -> int:
            final = [str(row[0]) for row in conn.execute(
                f"select id from scene_candidate_jobs where id in ({_marks(batch)}) and {due}",
                (*batch, cutoff),
            )]
            if not final:
                return 0
            conn.execute(
                f"delete from scene_candidate_seeds where candidate_id in ({_marks(final)})", final,
            )
            return conn.execute(
                f"delete from scene_candidate_jobs where id in ({_marks(final)})", final,
            ).rowcount

        # Small table (tens of thousands): one keyed page loop is enough.
        last_id = ""
        while True:
            self._check_stop()
            with self._events._connect() as conn:
                rows = conn.execute(
                    f"select id from scene_candidate_jobs where id > ? and {due} order by id limit ?",
                    (last_id, cutoff, SCAN_ROWS),
                ).fetchall()
            if not rows:
                return
            last_id = str(rows[-1][0])
            ids = [str(row[0]) for row in rows]
            step["eligible"] += len(ids)
            if not dry:
                for batch in _chunks(ids, BATCH_ROWS):
                    self._write(step, lambda conn, batch=batch: delete(conn, batch))

    def _referenced_decisions(self) -> set[str]:
        """Decisions an event or candidate names (tables without that index)."""
        with self._events._connect() as conn:
            return {str(row[0]) for row in conn.execute(
                "select decision_id from scene_event_establishment "
                "union select decision_id from scene_candidate_admissions "
                "union select decision_id from scene_candidate_jobs where decision_id is not null"
            )}

    def _activity_decisions(
        self, step: dict[str, int], cutoffs: dict[str, float], referenced: set[str], dry: bool,
    ) -> None:
        recheck = (
            "verdict in ('pending','unsupported','incomplete') and not exists("
            "select 1 from scene_activity_admissions a where a.decision_id=scene_activity_decisions.id)"
        )
        newest = max(cutoffs["empty"], cutoffs["activity"])
        last_rowid = 0
        # Decisions are appended in time order: page by rowid and stop at the
        # newest cutoff. Establishment and candidate admissions also hold
        # foreign keys, so a reference gained since the scan aborts the batch.
        while True:
            self._check_stop()
            with self._events._connect() as conn:
                rows = conn.execute(
                    "select rowid, id, verdict, created_at from scene_activity_decisions "
                    "where rowid > ? order by rowid limit ?",
                    (last_rowid, SCAN_ROWS),
                ).fetchall()
            if not rows:
                return
            last_rowid = int(rows[-1][0])
            ids = [
                str(row[1]) for row in rows
                if self._decision_expires(str(row[2]), float(row[3]), cutoffs)
                and str(row[1]) not in referenced
            ]
            step["eligible"] += len(ids)
            if not dry:
                for batch in _chunks(ids, BATCH_ROWS):
                    self._write(step, lambda conn, batch=batch: conn.execute(
                        f"delete from scene_activity_decisions where id in ({_marks(batch)}) "
                        f"and {recheck}",
                        batch,
                    ).rowcount)
            if float(rows[-1][3]) >= newest:
                return

    def _retained_decision_samples(
        self, cutoffs: dict[str, float], referenced: set[str],
    ) -> set[str]:
        """Samples named (as JSON, without a FK) by decisions that survive.

        Only decisions that can name a sample old enough to prune matter:
        supported and referenced ones (kept indefinitely) and surviving ones
        recorded near or after the oldest sample cutoff.
        """
        horizon = min(cutoffs["empty"], cutoffs["activity"]) - DECISION_SAMPLE_MARGIN_SECONDS
        kept: set[str] = set()
        last_rowid = 0
        while True:
            self._check_stop()
            with self._events._connect() as conn:
                rows = conn.execute(
                    "select rowid, id, verdict, created_at, sample_ids_json "
                    "from scene_activity_decisions where rowid > ? order by rowid limit ?",
                    (last_rowid, SCAN_ROWS),
                ).fetchall()
            if not rows:
                return kept
            last_rowid = int(rows[-1][0])
            for row in rows:
                decision_id, verdict, created = str(row[1]), str(row[2]), float(row[3])
                relevant = (
                    verdict == "supported" or decision_id in referenced or created >= horizon
                )
                if not relevant or (
                    decision_id not in referenced
                    and self._decision_expires(verdict, created, cutoffs)
                ):
                    continue
                try:
                    kept.update(str(value) for value in json.loads(row[4] or "[]"))
                except (TypeError, ValueError):
                    continue

    def _acquired_samples(
        self, step: dict[str, int], cutoffs: dict[str, float], kept: set[str], dry: bool,
    ) -> None:
        unlinked = (
            "not exists(select 1 from acquired_sample_events e where e.sample_id=s.id) "
            "and not exists(select 1 from acquired_sample_episodes p where p.sample_id=s.id) "
            "and not exists(select 1 from scene_candidate_seeds c where c.sample_id=s.id) "
            "and not exists(select 1 from scene_activity_decisions d, json_each(d.sample_ids_json) j "
            "where json_valid(d.sample_ids_json) and j.value = s.id)"
        )
        empty = "not exists(select 1 from acquired_sample_observations o where o.sample_id=s.id)"
        # Empty live-discovery frames expire on the short schedule, everything
        # else on the activity schedule.
        due = (
            f"((s.captured_epoch < ? and s.source='live_discovery' and s.status='complete' "
            f"and {empty}) or s.captured_epoch < ?)"
        )
        due_args = (cutoffs["empty"], cutoffs["activity"])
        newest = max(cutoffs["empty"], cutoffs["activity"])

        def delete(conn: sqlite3.Connection, batch: Sequence[str]) -> int:
            final = [str(row[0]) for row in conn.execute(
                f"select s.id from acquired_samples s where s.id in ({_marks(batch)}) "
                f"and {due} and {unlinked}",
                (*batch, *due_args),
            )]
            if not final:
                return 0
            marks = _marks(final)
            conn.execute(f"delete from scene_activity_measurements where sample_id in ({marks})", final)
            conn.execute(f"delete from acquired_sample_observations where sample_id in ({marks})", final)
            return conn.execute(f"delete from acquired_samples where id in ({marks})", final).rowcount

        with self._events._connect() as conn:
            cameras = [str(row[0]) for row in conn.execute(
                "select distinct camera_id from acquired_samples"
            )]
        for camera_id in cameras:
            last_epoch, last_id = -1.0, ""
            while True:
                self._check_stop()
                with self._events._connect() as conn:
                    rows = conn.execute(
                        f"select s.id, s.captured_epoch, ({due} and {unlinked}) as prunable "
                        "from acquired_samples s where s.camera_id=? and s.captured_epoch < ? "
                        "and (s.captured_epoch > ? or (s.captured_epoch = ? and s.id > ?)) "
                        "order by s.captured_epoch, s.id limit ?",
                        (*due_args, camera_id, newest, last_epoch, last_epoch, last_id, SCAN_ROWS),
                    ).fetchall()
                if not rows:
                    break
                last_epoch, last_id = float(rows[-1][1]), str(rows[-1][0])
                ids = [str(row[0]) for row in rows if row[2] and str(row[0]) not in kept]
                step["eligible"] += len(ids)
                if not dry:
                    for batch in _chunks(ids, BATCH_ROWS):
                        self._write(step, lambda conn, batch=batch: delete(conn, batch))

    def _acquired_observations(
        self, step: dict[str, int], cutoffs: dict[str, float], dry: bool,
    ) -> None:
        unlinked = (
            "not exists(select 1 from acquired_sample_observations a where a.observation_id=o.id) "
            "and not exists(select 1 from scene_observations s where s.source_observation_id=o.id)"
        )
        cutoff = cutoffs["activity"]
        with self._events._connect() as conn:
            cameras = [str(row[0]) for row in conn.execute(
                "select distinct camera_id from acquired_observations"
            )]
        # In a dry run, samples are not removed, so observations they still
        # link are not counted here; a real run frees those in the same pass.
        for camera_id in cameras:
            last_epoch, last_id = -1.0, ""
            while True:
                self._check_stop()
                with self._events._connect() as conn:
                    rows = conn.execute(
                        f"select o.id, o.captured_epoch, ({unlinked}) as prunable "
                        "from acquired_observations o where o.camera_id=? and o.captured_epoch < ? "
                        "and (o.captured_epoch > ? or (o.captured_epoch = ? and o.id > ?)) "
                        "order by o.captured_epoch, o.id limit ?",
                        (camera_id, cutoff, last_epoch, last_epoch, last_id, SCAN_ROWS),
                    ).fetchall()
                if not rows:
                    break
                last_epoch, last_id = float(rows[-1][1]), str(rows[-1][0])
                ids = [str(row[0]) for row in rows if row[2]]
                step["eligible"] += len(ids)
                if not dry:
                    for batch in _chunks(ids, BATCH_ROWS):
                        self._write(step, lambda conn, batch=batch: conn.execute(
                            f"delete from acquired_observations as o where o.id in ({_marks(batch)}) "
                            f"and o.captured_epoch < ? and {unlinked}",
                            (*batch, cutoff),
                        ).rowcount)

    def _motion_audits(self, result: dict[str, Any], cutoffs: dict[str, float], dry: bool) -> None:
        step = result["steps"]["motion_audits"]
        cutoff = _iso(cutoffs["audit"])
        unlinked = "coalesce(event_id,0)=0 and coalesce(related_event_id,0)=0"
        last_created, last_id = "", 0
        while True:
            self._check_stop()
            with self._events._connect() as conn:
                rows = conn.execute(
                    f"select id, created_at from motion_audits where created_at < ? and {unlinked} "
                    "and (created_at > ? or (created_at = ? and id > ?)) "
                    "order by created_at, id limit ?",
                    (cutoff, last_created, last_created, last_id, SCAN_ROWS),
                ).fetchall()
            if not rows:
                return
            last_created, last_id = str(rows[-1][1]), int(rows[-1][0])
            ids = [int(row[0]) for row in rows]
            step["eligible"] += len(ids)
            if dry:
                continue
            for batch in _chunks(ids, BATCH_ROWS):
                paths: list[str] = []

                def delete(conn: sqlite3.Connection, batch=batch, paths=paths) -> int:
                    marks = _marks(batch)
                    owned = [str(row[0]) for row in conn.execute(
                        f"select snapshot_path from motion_audits where id in ({marks}) "
                        f"and created_at < ? and {unlinked} and snapshot_path != ''",
                        (*batch, cutoff),
                    )]
                    paths.extend(owned)
                    claimed_at = _iso(self._clock())
                    conn.execute(
                        "create table if not exists media_deletion_work ("
                        "path text primary key, role text not null, created_at text not null)"
                    )
                    for raw in dict.fromkeys(owned):
                        conn.execute(
                            "insert or ignore into media_deletion_work (path, role, created_at) "
                            "values (?, 'motion_audit', ?)",
                            (raw, claimed_at),
                        )
                        conn.execute(
                            "insert or ignore into media_deletion_claims (path, role, claimed_at) "
                            "values (?, 'motion_audit', ?)",
                            (raw, claimed_at),
                        )
                    return conn.execute(
                        f"delete from motion_audits where id in ({marks}) "
                        f"and created_at < ? and {unlinked}",
                        (*batch, cutoff),
                    ).rowcount

                self._write(step, delete)
                self._drain_motion_deletions(result)

    # -------------------------------------------------------------- helpers
    def _ensure_deletion_work(self) -> None:
        with self._events._connect() as conn:
            conn.execute(
                "create table if not exists media_deletion_work ("
                "path text primary key, role text not null, created_at text not null)"
            )

    def _motion_verdict(self, path: Path) -> str:
        events = self._events
        media_storage = getattr(events, "media_storage", None)
        roots = list(media_storage._roots.values()) if media_storage is not None else []
        store = VolumeIdentityStore(identity_db_path(Path(events.db_path), roots))
        fallback = None if media_storage is not None else Path(events.storage_dir)
        return VolumeGate(store, media_storage=media_storage, fallback_root=fallback).verdict(path)

    def _release_motion_work(self, raw: str) -> None:
        with self._events._lock, self._events._connect() as conn:
            conn.execute("begin immediate")
            conn.execute(
                "delete from media_deletion_work where path = ? and role = 'motion_audit'",
                (raw,),
            )
            conn.execute(
                "delete from media_deletion_claims where path = ? and role = 'motion_audit'",
                (raw,),
            )

    def _drain_motion_deletions(self, result: dict[str, Any]) -> None:
        """Unlink files whose audit row is already gone, while the claim is held.

        The claim is inserted in the same transaction as the row delete. Writers
        that consult media_deletion_claims cannot adopt the path between the
        owner check and unlink. Unavailable storage leaves both rows in place.
        """
        self._ensure_deletion_work()
        with self._events._connect() as conn:
            queued = [
                str(row[0])
                for row in conn.execute(
                    "select path from media_deletion_work where role = 'motion_audit'"
                )
            ]
        if not queued:
            return
        storage_dir = Path(self._events.storage_dir)
        media_storage = getattr(self._events, "media_storage", None)
        for raw in queued:
            spellings = _path_spellings(storage_dir, raw)
            if self._named_image_paths(spellings):
                self._release_motion_work(raw)
                continue
            path: Path | None = None
            try:
                path = event_snapshot_path(storage_dir, {"snapshot_path": raw}, media_storage)
                if not _audit_image_location(storage_dir, path, media_storage):
                    result["motion_sample_files_failed"] += 1
                    continue
                verdict = self._motion_verdict(path)
                if verdict not in {"present", "confirmed_absent"}:
                    result["motion_sample_files_failed"] += 1
                    continue
                if self._named_image_paths(spellings):
                    self._release_motion_work(raw)
                    continue
                if verdict == "present":
                    path.unlink()
                self._release_motion_work(raw)
                if verdict == "present":
                    result["motion_sample_files_deleted"] += 1
                else:
                    result["motion_sample_files_missing"] += 1
            except FileNotFoundError:
                if path is not None and self._motion_verdict(path) == "confirmed_absent":
                    self._release_motion_work(raw)
                    result["motion_sample_files_missing"] += 1
                else:
                    result["motion_sample_files_failed"] += 1
            except (PermissionError, OSError, RuntimeError, ValueError):
                result["motion_sample_files_failed"] += 1

    def _named_image_paths(self, values: set[str]) -> set[str]:
        """The subset of ``values`` that any image-owning row still names."""
        named: set[str] = set()
        ordered = sorted(values)
        with self._events._connect() as conn:
            for table, column in IMAGE_OWNERS:
                columns = {str(row[1]) for row in conn.execute(f"pragma table_info('{table}')")}
                if column not in columns:
                    continue
                # ``!= ''`` lets partial snapshot indexes serve the lookup.
                for chunk in _chunks(ordered, BATCH_ROWS):
                    named.update(str(row[0]) for row in conn.execute(
                        f"select {column} from {table} where {column} != '' "
                        f"and {column} in ({_marks(chunk)})",
                        tuple(chunk),
                    ))
        return named

    def _write(self, step: dict[str, int], operation: Callable[[sqlite3.Connection], int]) -> None:
        self._check_stop()
        try:
            with self._events._lock, self._events._connect() as conn:
                conn.execute("begin immediate")
                step["deleted"] += max(0, int(operation(conn) or 0))
        except sqlite3.IntegrityError:
            # A row gained a foreign-key reference after it was selected;
            # skip this batch, the next run reconsiders it.
            step["skipped_batches"] = int(step.get("skipped_batches", 0)) + 1
        if self._pause_seconds:
            self._stop.wait(self._pause_seconds)

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise _Stopped()
        if getattr(self, "_honor_config", False) and not self._config().enabled:
            raise _Stopped()
