from __future__ import annotations

import sqlite3
import threading
import uuid
from pathlib import Path
from queue import Queue
from typing import Any, Callable

from ..face_recognition import OpenVinoFaceRecognizer
from ..inference import INFERENCE_REQUEST_TIMEOUT_SECONDS
from ..incident_utils import event_snapshot_path, portable_media_path
from ..media_storage import MediaStorageRegistry
from ..main_database import connect_main_database
from .benchmarks import FaceStoreBenchmarkMixin
from .ingest import FaceStoreIngestMixin
from .people import FaceStorePeopleMixin
from .quality import (
    FACE_OUTCOME_EMBEDDED,
    FACE_OUTCOME_FAILED,
    FACE_OUTCOME_PENDING,
    FACE_OUTCOME_TOO_SMALL,
    FACE_QUALITY_VERSION,
    LOGGER,
)
from .queries import FaceStoreQueryMixin
from .recognition import FaceStoreRecognitionMixin
from .unknown import FaceStoreUnknownMixin


_PEOPLE_DIRECTORY_REVISION_KEY = "face_people_directory_revision_v1"
_UNKNOWN_CLUSTERS_REVISION_KEY = "face_unknown_clusters_revision_v1"


class FaceStore(
    FaceStoreIngestMixin,
    FaceStoreRecognitionMixin,
    FaceStoreUnknownMixin,
    FaceStoreQueryMixin,
    FaceStoreBenchmarkMixin,
    FaceStorePeopleMixin,
):
    def __init__(
        self,
        storage_dir: Path,
        max_observations: int = 1000,
        recognizer: OpenVinoFaceRecognizer | None = None,
        start_recognition: bool = True,
        database_dir: Path | None = None,
        media_storage: MediaStorageRegistry | None = None,
        database_write_lock: threading.RLock | None = None,
    ) -> None:
        self.storage_dir = storage_dir.resolve()
        self.media_storage = media_storage
        resolved_database_dir = (database_dir or self.storage_dir).resolve()
        resolved_database_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = resolved_database_dir / "survng.sqlite3"
        self.max_observations = max(100, int(max_observations))
        self.recognizer = recognizer
        self._lock = threading.Lock()
        self._database_write_lock = database_write_lock or threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._recognition_queue: Queue[int | None] = Queue(maxsize=self.max_observations + 1)
        self._recognition_pending: set[int] = set()
        self._recognition_pending_lock = threading.Lock()
        self._gallery_lock = threading.RLock()
        self._gallery_generation = 0
        self._gallery_cache_key: tuple[str, tuple[int, ...], int, int] | None = None
        self._gallery_cache: list[dict[str, Any]] = []
        self._directory_revision_instance = uuid.uuid4().hex
        self._identity_event_publisher: Callable[[dict[str, Any]], None] | None = None
        self._recognition_refill_needed = threading.Event()
        self._match_refresh_needed = threading.Event()
        self._recognition_stop = threading.Event()
        self._recognition_thread: threading.Thread | None = None
        self._init_db()
        if start_recognition:
            self.start()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._recognition_thread is not None and self._recognition_thread.is_alive():
                return
            if self.recognizer is None or not self.recognizer.enabled:
                return
            self._recognition_stop.clear()
            self._recognition_thread = threading.Thread(
                target=self._recognition_loop,
                name="survng-face-recognition",
                daemon=False,
            )
            self._recognition_thread.start()
        try:
            self._queue_pending_recognition()
        except Exception:
            LOGGER.exception("Could not restore pending face recognition work")

    def reconfigure_max_observations(self, max_observations: int) -> int:
        """Apply a new history limit and prune transactionally without stopping recognition."""
        next_limit = max(100, int(max_observations))
        pruned_paths: list[Path] = []
        with self._lock, self._connect() as connection:
            previous_limit = self.max_observations
            self.max_observations = next_limit
            try:
                removed = self._prune_locked(connection, pruned_paths)
            except BaseException:
                self.max_observations = previous_limit
                raise
        self._delete_face_snapshots(pruned_paths, "pruned")
        return removed

    def _connect(self) -> sqlite3.Connection:
        connection = connect_main_database(
            self.db_path, timeout=10, write_lock=self._database_write_lock
        )
        connection.row_factory = sqlite3.Row
        connection.execute("pragma busy_timeout = 10000")
        connection.execute("pragma foreign_keys = on")
        return connection

    def _init_db(self) -> None:
        pruned_paths: list[Path] = []
        with self._connect() as connection:
            connection.execute("pragma journal_mode = wal")
            connection.execute("pragma synchronous = normal")
            connection.execute(
                "create table if not exists media_deletion_claims ("
                "path text primary key, role text not null, claimed_at text not null)"
            )
            connection.execute(
                "create table if not exists survng_metadata (key text primary key, value text not null)"
            )
            connection.executescript(
                """
                create table if not exists face_people (
                    id integer primary key autoincrement,
                    name text not null,
                    notes text not null default '',
                    created_at text not null,
                    updated_at text not null
                );
                create table if not exists face_observations (
                    id integer primary key autoincrement,
                    event_id integer not null,
                    object_index integer not null,
                    person_id integer,
                    camera_id text not null,
                    snapshot_path text not null,
                    box_json text not null,
                    confidence real not null default 0,
                    observed_at text not null,
                    match_confidence real,
                    review_status text not null default 'unknown',
                    created_at text not null,
                    unique(event_id, object_index),
                    foreign key(person_id) references face_people(id) on delete set null
                );
                create index if not exists idx_face_observations_observed_at
                    on face_observations(observed_at desc);
                create index if not exists idx_face_observations_person
                    on face_observations(person_id, observed_at desc);
                create table if not exists face_unknown_members (
                    observation_id integer primary key,
                    cluster_id integer not null,
                    updated_at text not null,
                    model_fingerprint text not null default '',
                    policy_threshold real not null default 0,
                    generation integer not null default 0,
                    foreign key(observation_id) references face_observations(id) on delete cascade
                );
                create index if not exists idx_face_unknown_members_cluster
                    on face_unknown_members(cluster_id, observation_id);
                create table if not exists face_rejections (
                    observation_id integer not null,
                    person_id integer not null,
                    created_at text not null,
                    primary key(observation_id, person_id),
                    foreign key(observation_id) references face_observations(id) on delete cascade,
                    foreign key(person_id) references face_people(id) on delete cascade
                );
                """
            )
            columns = {str(row[1]) for row in connection.execute("pragma table_info(face_observations)")}
            unknown_member_columns = {
                str(row[1])
                for row in connection.execute("pragma table_info(face_unknown_members)")
            }
            unknown_member_migrations = {
                "model_fingerprint": "alter table face_unknown_members add column model_fingerprint text not null default ''",
                "policy_threshold": "alter table face_unknown_members add column policy_threshold real not null default 0",
                "generation": "alter table face_unknown_members add column generation integer not null default 0",
            }
            for name, statement in unknown_member_migrations.items():
                if name not in unknown_member_columns:
                    connection.execute(statement)
            migrations = {
                "embedding_blob": "alter table face_observations add column embedding_blob blob",
                "embedding_model": "alter table face_observations add column embedding_model text not null default ''",
                "candidate_person_id": "alter table face_observations add column candidate_person_id integer",
                "candidate_confidence": "alter table face_observations add column candidate_confidence real",
                "rejected_person_id": "alter table face_observations add column rejected_person_id integer",
                "recognition_error": "alter table face_observations add column recognition_error text not null default ''",
                "recognized_at": "alter table face_observations add column recognized_at text not null default ''",
                "recognition_pending": "alter table face_observations add column recognition_pending integer not null default 1",
                "recognition_outcome": "alter table face_observations add column recognition_outcome text not null default 'pending'",
                "quality_score": "alter table face_observations add column quality_score real",
                "quality_json": "alter table face_observations add column quality_json text not null default '{}'",
                "quality_version": "alter table face_observations add column quality_version integer not null default 0",
                "reference_pinned": "alter table face_observations add column reference_pinned integer not null default 0",
                "reference_auto_pinned": "alter table face_observations add column reference_auto_pinned integer not null default 0",
                "auto_identified": "alter table face_observations add column auto_identified integer not null default 0",
                "match_details_json": "alter table face_observations add column match_details_json text not null default '{}'",
                "candidate_track_id": "alter table face_observations add column candidate_track_id text not null default ''",
                "candidate_rank": "alter table face_observations add column candidate_rank integer not null default 1",
                "candidate_offset_seconds": "alter table face_observations add column candidate_offset_seconds real not null default 0",
                "canonical": "alter table face_observations add column canonical integer not null default 1",
                "duplicate_of_observation_id": "alter table face_observations add column duplicate_of_observation_id integer",
                "consensus_json": "alter table face_observations add column consensus_json text not null default '{}'",
            }
            for name, statement in migrations.items():
                if name not in columns:
                    connection.execute(statement)
            connection.execute(
                "create index if not exists idx_face_duplicate_of "
                "on face_observations(duplicate_of_observation_id)"
            )
            connection.execute(
                """
                update face_observations
                set recognition_outcome = case
                    when recognition_pending = 1 then ?
                    when recognition_error like 'Face is smaller than %' then ?
                    when recognition_error != '' then ?
                    when embedding_blob is not null then ?
                    else ?
                end
                where recognition_outcome = ''
                    or recognition_outcome not in (?, ?, ?, ?)
                    or (recognition_outcome = ? and recognition_pending = 0)
                """,
                (
                    FACE_OUTCOME_PENDING,
                    FACE_OUTCOME_TOO_SMALL,
                    FACE_OUTCOME_FAILED,
                    FACE_OUTCOME_EMBEDDED,
                    FACE_OUTCOME_FAILED,
                    FACE_OUTCOME_PENDING,
                    FACE_OUTCOME_EMBEDDED,
                    FACE_OUTCOME_TOO_SMALL,
                    FACE_OUTCOME_FAILED,
                    FACE_OUTCOME_PENDING,
                ),
            )
            connection.execute(
                """
                create unique index if not exists idx_face_candidate_identity
                on face_observations(event_id, candidate_track_id, candidate_offset_seconds)
                where candidate_track_id != ''
                """
            )
            connection.execute(
                """
                update face_observations
                set recognition_pending = 1, recognition_outcome = ?
                where quality_version != ? and embedding_blob is not null
                    and recognition_error = ''
                """,
                (FACE_OUTCOME_PENDING, FACE_QUALITY_VERSION),
            )
            connection.execute(
                """
                insert or ignore into face_rejections (observation_id, person_id, created_at)
                select o.id, o.rejected_person_id, coalesce(nullif(o.recognized_at, ''), o.created_at)
                from face_observations o
                join face_people p on p.id = o.rejected_person_id
                where o.rejected_person_id is not null
                """
            )
            connection.execute(
                """
                update face_observations
                set candidate_person_id = null, candidate_confidence = null
                where candidate_person_id is not null
                    and not exists (
                        select 1 from face_people p where p.id = candidate_person_id
                    )
                """
            )
            connection.execute(
                """
                update face_observations
                set rejected_person_id = null
                where rejected_person_id is not null
                    and not exists (
                        select 1 from face_people p where p.id = rejected_person_id
                    )
                """
            )
            connection.execute(
                """
                update face_observations
                set review_status = 'unknown', match_confidence = null
                where person_id is null and review_status = 'confirmed'
                """
            )
            storage_root = str(self.storage_dir.resolve())
            metadata = connection.execute(
                "select value from survng_metadata where key = 'face_storage_root'"
            ).fetchone()
            if metadata is None or str(metadata[0]) != storage_root:
                rows = connection.execute("select id, snapshot_path from face_observations").fetchall()
                updates = []
                for row in rows:
                    raw_path = str(row["snapshot_path"] or "")
                    portable_path = portable_media_path(self.storage_dir, raw_path)
                    if portable_path != raw_path:
                        updates.append((portable_path, int(row["id"])))
                if updates:
                    connection.executemany(
                        "update face_observations set snapshot_path = ? where id = ?",
                        updates,
                    )
                connection.execute(
                    """
                    insert into survng_metadata (key, value) values ('face_storage_root', ?)
                    on conflict(key) do update set value = excluded.value
                    """,
                    (storage_root,),
                )
            self._prune_locked(connection, pruned_paths)
            for key in (
                _PEOPLE_DIRECTORY_REVISION_KEY,
                _UNKNOWN_CLUSTERS_REVISION_KEY,
            ):
                connection.execute(
                    "insert or ignore into survng_metadata (key, value) values (?, '0')",
                    (key,),
                )
            connection.executescript(
                f"""
                create trigger if not exists face_people_directory_insert_revision
                after insert on face_people begin
                    update survng_metadata set value = cast(value as integer) + 1
                    where key = '{_PEOPLE_DIRECTORY_REVISION_KEY}';
                end;
                create trigger if not exists face_people_directory_update_revision
                after update on face_people begin
                    update survng_metadata set value = cast(value as integer) + 1
                    where key = '{_PEOPLE_DIRECTORY_REVISION_KEY}';
                end;
                create trigger if not exists face_people_directory_delete_revision
                after delete on face_people begin
                    update survng_metadata set value = cast(value as integer) + 1
                    where key = '{_PEOPLE_DIRECTORY_REVISION_KEY}';
                end;

                create trigger if not exists face_observation_insert_directory_revisions
                after insert on face_observations begin
                    update survng_metadata set value = cast(value as integer) + 1
                    where key in (
                        '{_PEOPLE_DIRECTORY_REVISION_KEY}',
                        '{_UNKNOWN_CLUSTERS_REVISION_KEY}'
                    );
                end;
                create trigger if not exists face_observation_update_directory_revisions
                after update on face_observations begin
                    update survng_metadata set value = cast(value as integer) + 1
                    where key in (
                        '{_PEOPLE_DIRECTORY_REVISION_KEY}',
                        '{_UNKNOWN_CLUSTERS_REVISION_KEY}'
                    );
                end;
                create trigger if not exists face_observation_delete_directory_revisions
                after delete on face_observations begin
                    update survng_metadata set value = cast(value as integer) + 1
                    where key in (
                        '{_PEOPLE_DIRECTORY_REVISION_KEY}',
                        '{_UNKNOWN_CLUSTERS_REVISION_KEY}'
                    );
                end;

                create trigger if not exists face_unknown_member_insert_revision
                after insert on face_unknown_members begin
                    update survng_metadata set value = cast(value as integer) + 1
                    where key = '{_UNKNOWN_CLUSTERS_REVISION_KEY}';
                end;
                create trigger if not exists face_unknown_member_update_revision
                after update on face_unknown_members begin
                    update survng_metadata set value = cast(value as integer) + 1
                    where key = '{_UNKNOWN_CLUSTERS_REVISION_KEY}';
                end;
                create trigger if not exists face_unknown_member_delete_revision
                after delete on face_unknown_members begin
                    update survng_metadata set value = cast(value as integer) + 1
                    where key = '{_UNKNOWN_CLUSTERS_REVISION_KEY}';
                end;
                """
            )
        self._delete_face_snapshots(pruned_paths, "pruned")

    def _directory_revision(self, key: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "select value from survng_metadata where key = ?",
                (key,),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def people_directory_revision(self) -> str:
        status_reader = getattr(self.recognizer, "status", None)
        status = status_reader() if callable(status_reader) else {}
        fingerprint = str(status.get("model_fingerprint") or "")
        revision = self._directory_revision(_PEOPLE_DIRECTORY_REVISION_KEY)
        return f"{self._directory_revision_instance}:people:{revision}:{fingerprint}"

    def unknown_clusters_revision(self) -> str:
        fingerprint, threshold = self._unknown_cluster_policy()
        revision = self._directory_revision(_UNKNOWN_CLUSTERS_REVISION_KEY)
        return (
            f"{self._directory_revision_instance}:unknown:{revision}:"
            f"{fingerprint}:{threshold:.6f}"
        )

    def close(self) -> None:
        with self._lifecycle_lock:
            self._recognition_stop.set()
            thread = self._recognition_thread
            if thread is not None:
                try:
                    self._recognition_queue.put_nowait(None)
                except Full:
                    pass
        if thread is not None:
            thread.join(timeout=INFERENCE_REQUEST_TIMEOUT_SECONDS + 5.0)
            if thread.is_alive():
                LOGGER.error("face recognition worker did not stop")
                raise RuntimeError("face recognition worker did not stop")
            with self._lifecycle_lock:
                if self._recognition_thread is thread:
                    self._recognition_thread = None

    def _delete_face_snapshots(self, paths: list[Path], reason: str) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.debug("could not remove %s face candidate %s", reason, path)

    def _prune_locked(
        self,
        connection: sqlite3.Connection,
        deferred_paths: list[Path] | None = None,
    ) -> int:
        total = int(connection.execute(
            "select count(*) from face_observations where canonical = 1"
        ).fetchone()[0])
        excess = total - self.max_observations
        if excess <= 0:
            return 0
        selected = connection.execute(
            """
            select id, event_id, candidate_track_id from face_observations
            where canonical = 1 and reference_pinned = 0 and id not in (
                select id from (
                    select id, row_number() over (
                        partition by person_id order by observed_at desc, id desc
                    ) as position
                    from face_observations
                    where person_id is not null and canonical = 1
                ) where position = 1
            )
            order by observed_at asc, id asc
            limit ?
            """,
            (excess,),
        ).fetchall()
        remove_ids: set[int] = set()
        candidate_paths: list[str] = []
        for row in selected:
            track_id = str(row["candidate_track_id"] or "")
            if not track_id:
                remove_ids.add(int(row["id"]))
                continue
            group = connection.execute(
                """
                select id, snapshot_path, reference_pinned
                from face_observations
                where event_id = ? and candidate_track_id = ?
                """,
                (int(row["event_id"]), track_id),
            ).fetchall()
            if any(bool(item["reference_pinned"]) for item in group):
                continue
            remove_ids.update(int(item["id"]) for item in group)
            candidate_paths.extend(str(item["snapshot_path"] or "") for item in group)
        if remove_ids:
            connection.executemany("delete from face_observations where id = ?", ((item,) for item in remove_ids))
            # Commit the authoritative database removal before deleting the
            # uniquely owned crop files. A later SQLite commit failure must
            # never leave retained rows pointing at already deleted images.
            connection.commit()
            self._invalidate_reference_gallery()
            for raw_path in candidate_paths:
                try:
                    path = event_snapshot_path(
                        self.storage_dir,
                        {"snapshot_path": raw_path},
                        self.media_storage,
                    )
                except (FileNotFoundError, PermissionError, OSError, RuntimeError):
                    LOGGER.debug("could not remove pruned face candidate %s", raw_path)
                    continue
                if deferred_paths is not None:
                    deferred_paths.append(path)
                else:
                    self._delete_face_snapshots([path], "pruned")
        return len(remove_ids)
