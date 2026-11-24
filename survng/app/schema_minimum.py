"""The oldest databases this build opens: ones SurvNG v1.2.0 has already opened.

Each map lists, per table, the columns v1.2.0 added to older databases. Later
builds no longer add them, so an existing table without them predates the
minimum (see docs/current-only.md). Stores check their own tables before their
first DDL; the manager checks every database before it builds any store.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from .main_database import require_minimum_schema

SchemaMinimum = dict[str, tuple[str, ...]]

# survng.sqlite3, by owning store.
EVENT_SCHEMA_MINIMUM: SchemaMinimum = {
    "events": ("snapshot_size_bytes",),
    "motion_audits": ("decision_id", "related_event_id", "category"),
    "event_evidence_outbox": ("publication_done",),
    "scene_objects": ("facet_label",),
}
FACE_SCHEMA_MINIMUM: SchemaMinimum = {
    "face_unknown_members": ("model_fingerprint", "policy_threshold", "generation"),
}
SEMANTIC_SCHEMA_MINIMUM: SchemaMinimum = {"semantic_embeddings": ("evidence_revision",)}
APPEARANCE_SCHEMA_MINIMUM: SchemaMinimum = {"appearance_embeddings": ("source",)}
# Every store in survng.sqlite3. The manager and the event store, which writes
# the shared file first, check all of it; the face, semantic and appearance
# stores check only their own tables.
MAIN_SCHEMA_MINIMUM: SchemaMinimum = {
    **EVENT_SCHEMA_MINIMUM,
    **FACE_SCHEMA_MINIMUM,
    **SEMANTIC_SCHEMA_MINIMUM,
    **APPEARANCE_SCHEMA_MINIMUM,
}
# detection-jobs.sqlite3, exports.sqlite3 and recordings.sqlite3.
JOBS_SCHEMA_MINIMUM: SchemaMinimum = {
    "detection_jobs": ("lease_owner",),
    "motion_trigger_jobs": ("lease_owner",),
}
EXPORTS_SCHEMA_MINIMUM: SchemaMinimum = {"media_exports": ("protected", "label", "origin")}
RECORDINGS_SCHEMA_MINIMUM: SchemaMinimum = {
    "recordings": (
        "playable", "health_error", "validated", "stream_fingerprint",
        "fingerprint_checked", "playback_duration_seconds", "location_id",
    ),
}


def require_minimum_databases(database_dir: Path, recording_index_dir: Path) -> None:
    """Check every existing database, read-only, before any store opens one.

    A missing file is a new installation and passes. Nothing is created or
    changed here, so a rejected installation is left exactly as it was.
    """
    for path, minimum in (
        (database_dir / "survng.sqlite3", MAIN_SCHEMA_MINIMUM),
        (database_dir / "detection-jobs.sqlite3", JOBS_SCHEMA_MINIMUM),
        (database_dir / "exports.sqlite3", EXPORTS_SCHEMA_MINIMUM),
        (recording_index_dir / "recordings.sqlite3", RECORDINGS_SCHEMA_MINIMUM),
    ):
        require_minimum_file(path, minimum)


def require_minimum_file(path: Path, minimum: SchemaMinimum) -> None:
    """Check one database file read-only; a missing file is new and passes.

    A cleanly closed WAL database is opened immutable, so SQLite creates no
    -wal or -shm file beside it. After a crash the -wal file holds committed
    pages, so the file is opened read-only instead, which can add -shm.
    """
    if not path.is_file():
        return
    wal = path.with_name(path.name + "-wal")
    mode = "mode=ro" if wal.exists() else "immutable=1"
    uri = f"{path.resolve().as_uri()}?{mode}"
    with closing(sqlite3.connect(uri, uri=True, timeout=10.0)) as connection:
        require_minimum_schema(connection, minimum)
