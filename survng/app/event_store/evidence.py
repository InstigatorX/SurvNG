"""Event-owned evidence requirements and atomic revision/outbox commits.

This ledger deliberately lives with events, not leased security inference jobs.
Legacy events receive a revision only; migration never schedules historic work.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any


class EventSnapshotChangedError(RuntimeError):
    """Evidence no longer describes the event's current revision or snapshot."""


def _objects(raw: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


class EventStoreEvidenceMixin:
    def _init_evidence_db(self) -> None:
        with self._lock, self._connect() as conn:
            columns = {row["name"] for row in conn.execute("pragma table_info(events)")}
            if "evidence_revision" not in columns:
                conn.execute("alter table events add column evidence_revision integer not null default 0")
            conn.executescript("""
                create table if not exists event_cover_requirements (
                    event_id integer primary key references events(id) on delete cascade,
                    policy_version integer not null default 1,
                    state text not null check(state in ('pending','satisfied','exhausted')),
                    deadline_epoch real not null, available_at_epoch real not null,
                    attempts integer not null default 0, reason text not null default '',
                    lease_owner text not null default '', lease_expires_at_epoch real,
                    payload_json text not null, created_at real not null, updated_at real not null
                );
                create index if not exists idx_cover_requirements_due
                    on event_cover_requirements(state, available_at_epoch);
                create table if not exists event_evidence_outbox (
                    id integer primary key autoincrement,
                    event_id integer not null references events(id) on delete cascade,
                    evidence_revision integer not null, kind text not null,
                    payload_json text not null, created_at real not null,
                    publication_done integer not null default 0
                );
                create unique index if not exists idx_event_evidence_outbox_revision
                    on event_evidence_outbox(event_id,evidence_revision,kind)
                    where kind in ('cover_required','evidence_updated');
                create table if not exists event_evidence_attempts (
                    id integer primary key autoincrement,
                    event_id integer not null references events(id) on delete cascade,
                    summary_json text not null, created_at real not null
                );
                create index if not exists idx_event_evidence_attempts on event_evidence_attempts(event_id,id);
            """)
            outbox_columns = {row["name"] for row in conn.execute("pragma table_info(event_evidence_outbox)")}
            if "publication_done" not in outbox_columns:
                conn.execute("alter table event_evidence_outbox add column publication_done integer not null default 0")

    @staticmethod
    def _check_evidence_revision(row: sqlite3.Row, expected_revision: int | None) -> None:
        if expected_revision is not None and int(row["evidence_revision"]) != expected_revision:
            raise EventSnapshotChangedError("event evidence revision changed")


    def _evidence_outbox(self, conn, row, kind, **extra) -> None:
        payload = {"event_id": int(row["id"]), "camera_id": row["camera_id"],
                   "evidence_revision": int(row["evidence_revision"]), **extra}
        conn.execute(
            "insert or ignore into event_evidence_outbox "
            "(event_id,evidence_revision,kind,payload_json,created_at) values(?,?,?,?,?)",
            (row["id"], row["evidence_revision"], kind, json.dumps(payload), time.time()),
        )

    def _admit_event_evidence(self, conn, event_id: int) -> None:
        conn.execute("update events set evidence_revision=1 where id=?", (event_id,))
        row = conn.execute("select * from events where id=?", (event_id,)).fetchone()
        objects = _objects(row["objects_json"])
        # Initial callbacks are best effort too. The projection obligation is
        # durable even if the process stops immediately after event admission.
        self._evidence_outbox(conn, row, "evidence_updated", reason="incident_admitted")
        if not any(item.get("label") and item.get("provisional_detection") is True
                   and item.get("incident_eligible") is not False for item in objects):
            return
        now = time.time()
        qualification = next((item.get("motion_qualification", {}) for item in objects
                              if item.get("status") == "motion_qualification"), {})
        qualification = dict(qualification) if isinstance(qualification, dict) else {}
        telemetry = qualification.get("telemetry")
        if isinstance(telemetry, dict):
            qualification["telemetry"] = {
                "schema_version": telemetry.get("schema_version"),
                "origins": telemetry.get("origins", {}),
                "compacted_for_cover_requirement": True,
            }
        payload = {"event_id": event_id, "camera_id": row["camera_id"],
                   "topic": row["topic"], "message": row["message"],
                   "event_at": row["created_at"], "qualification": qualification,
                   "objects": [item for item in objects if item.get("label")], "snapshot_path": row["snapshot_path"],
                   "recording_path": row["recording_path"], "evidence_revision": 1}
        conn.execute(
            "insert into event_cover_requirements "
            "(event_id,state,deadline_epoch,available_at_epoch,payload_json,created_at,updated_at) "
            "values(?,'pending',?,?,?,?,?)",
            (event_id, now + 300, now + 15, json.dumps(payload), now, now),
        )
        self._evidence_outbox(conn, row, "cover_required", policy_version=1, deadline_epoch=now + 300)

    def _finish_evidence_commit(self, conn, event_id, before, *, reason, cover_satisfied=False):
        """Finish a writer's transaction; bytes/annotations/revision advance together."""
        row = conn.execute("select * from events where id=?", (event_id,)).fetchone()
        if row is None:
            return None
        before_objects = _objects(before["objects_json"])
        after_objects = _objects(row["objects_json"])
        changed = (before["snapshot_path"] or "") != (row["snapshot_path"] or "") or (
            self._presentation_evidence(before_objects) != self._presentation_evidence(after_objects)
        )
        if changed:
            conn.execute("update events set evidence_revision=evidence_revision+1 where id=?", (event_id,))
            row = conn.execute("select * from events where id=?", (event_id,)).fetchone()
            self._evidence_outbox(conn, row, "evidence_updated", reason=reason)
        elif reason != "tracking_updated" and (
            before_objects != after_objects or before["recording_path"] != row["recording_path"]
        ):
            # Tracking progress already has its own callback. Metadata updates
            # reach clients without re-encoding unchanged image embeddings.
            self._evidence_outbox(conn, row, "incident_metadata_updated", reason=reason)
        if cover_satisfied:
            settled = conn.execute("update event_cover_requirements set state='satisfied', reason=?, "
                         "lease_owner='',lease_expires_at_epoch=null,updated_at=? where event_id=? "
                         "and state='pending'", (reason, time.time(), event_id)).rowcount
            if settled:
                self._requirement_outbox(conn, event_id)
        return row

    @staticmethod
    def _decode_requirement(row):
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def cover_requirement(self, event_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            return self._decode_requirement(conn.execute(
                "select * from event_cover_requirements where event_id=?", (event_id,)).fetchone())

    def pending_cover_requirements(self, camera_id: str = "", *, limit: int = 100):
        with self._connect() as conn:
            rows = conn.execute("select r.*,e.camera_id from event_cover_requirements r join events e "
                                "on e.id=r.event_id where r.state='pending' and r.available_at_epoch<=? "
                                "and (?='' or e.camera_id=?) order by r.available_at_epoch limit ?",
                                (time.time(), camera_id, camera_id, max(1, min(limit, 1000)))).fetchall()
        return [self._decode_requirement(row) for row in rows]

    def claim_cover_requirement(self, camera_id: str, *, lease_owner: str,
                                lease_seconds: float = 60, maximum_attempts: int = 3):
        if not lease_owner:
            raise ValueError("cover requirement lease owner is required")
        now = time.time()
        # Idle per-camera refiners must not reserve the shared main-DB writer.
        with self._connect() as conn:
            due = conn.execute(
                "select 1 from event_cover_requirements r join events e on e.id=r.event_id "
                "where e.camera_id=? and r.state='pending' "
                "and (r.available_at_epoch<=? or r.deadline_epoch<=? or r.attempts>=?) "
                "and (r.lease_expires_at_epoch is null or r.lease_expires_at_epoch<=?) limit 1",
                (camera_id, now, now, maximum_attempts, now),
            ).fetchone()
        if due is None:
            return None
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            expired = conn.execute("update event_cover_requirements set state='exhausted',reason=case when "
                         "deadline_epoch<=? then 'deadline_expired' else 'attempts_exhausted' end, "
                         "lease_owner='',lease_expires_at_epoch=null,updated_at=? "
                         "where state='pending' and (deadline_epoch<=? or attempts>=?) "
                         "and (lease_expires_at_epoch is null or lease_expires_at_epoch<=?) "
                         "and event_id in(select id from events where camera_id=?) returning event_id",
                         (now, now, now, maximum_attempts, now, camera_id)).fetchall()
            for expired_row in expired:
                self._requirement_outbox(conn, int(expired_row["event_id"]))
            row = conn.execute("select r.* from event_cover_requirements r join events e on e.id=r.event_id "
                               "where e.camera_id=? and r.state='pending' and r.available_at_epoch<=? "
                               "and (r.lease_expires_at_epoch is null or r.lease_expires_at_epoch<=?) "
                               "order by r.available_at_epoch,r.event_id limit 1", (camera_id, now, now)).fetchone()
            if row is None:
                return None
            conn.execute("update event_cover_requirements set attempts=attempts+1,lease_owner=?, "
                         "lease_expires_at_epoch=?,updated_at=? where event_id=?",
                         (lease_owner, now + max(1, lease_seconds), now, row["event_id"]))
            result = self._decode_requirement(conn.execute("select * from event_cover_requirements where event_id=?",
                                                           (row["event_id"],)).fetchone())
            result["latest_event"] = dict(conn.execute("select * from events where id=?", (row["event_id"],)).fetchone())
            return result

    def finish_cover_attempt(self, event_id: int, *, lease_owner: str, reason: str,
                             retry_delay_seconds: float = 15, maximum_attempts: int = 3) -> bool:
        now = time.time()
        with self._lock, self._connect() as conn:
            cursor = conn.execute("update event_cover_requirements set state=case when attempts>=? or deadline_epoch<=? "
                                  "then 'exhausted' else 'pending' end,reason=?,available_at_epoch=?,lease_owner='', "
                                  "lease_expires_at_epoch=null,updated_at=? where event_id=? and state='pending' and lease_owner=?",
                                  (maximum_attempts, now, reason[:200], now + max(0, retry_delay_seconds), now, event_id, lease_owner))
            changed = cursor.rowcount == 1
            if changed:
                self._requirement_outbox(conn, event_id)
            return changed

    def defer_cover_requirement(self, event_id: int, *, lease_owner: str, reason: str) -> bool:
        """Yield a live lease without charging a failed attempt or extending its deadline."""
        now = time.time()
        with self._lock, self._connect() as conn:
            changed = conn.execute(
                "update event_cover_requirements set attempts=max(0,attempts-1), "
                "available_at_epoch=?,reason=?,lease_owner='',lease_expires_at_epoch=null,updated_at=? "
                "where event_id=? and state='pending' and lease_owner=? and lease_expires_at_epoch>?",
                (now + 5, reason, now, event_id, lease_owner, now),
            ).rowcount == 1
            if changed:
                self._requirement_outbox(conn, event_id)
            return changed

    def settle_cover_requirement(self, event_id: int, *, state: str, reason: str, policy_version: int = 1) -> bool:
        if state not in {"satisfied", "exhausted"}:
            raise ValueError("cover requirement settlement must be terminal")
        with self._lock, self._connect() as conn:
            changed = conn.execute("update event_cover_requirements set state=?,reason=?,lease_owner='', "
                                "lease_expires_at_epoch=null,updated_at=? where event_id=? and policy_version=? and state='pending'",
                                (state, reason[:200], time.time(), event_id, policy_version)).rowcount == 1
            if changed:
                self._requirement_outbox(conn, event_id)
            return changed

    def pending_evidence_updates(self, *, limit: int = 100, after_id: int = 0):
        with self._connect() as conn:
            rows = conn.execute("select * from event_evidence_outbox where id>? order by id limit ?",
                                (max(0, int(after_id)), max(1, min(limit, 1000)))).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def acknowledge_evidence_update(self, outbox_id: int) -> bool:
        with self._lock, self._connect() as conn:
            return conn.execute("delete from event_evidence_outbox where id=?", (outbox_id,)).rowcount == 1

    def record_evidence_attempt(self, event_id: int, summary: dict[str, Any]) -> None:
        """Retain the last three diagnostic attempts without rewriting admission."""
        raw = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(raw.encode()) > 131072:
            raise ValueError("evidence attempt summary exceeds 128 KiB")
        with self._lock, self._connect() as conn:
            conn.execute("insert into event_evidence_attempts(event_id,summary_json,created_at) values(?,?,?)",
                         (event_id, raw, time.time()))
            conn.execute("delete from event_evidence_attempts where event_id=? and id not in "
                         "(select id from event_evidence_attempts where event_id=? order by id desc limit 3)",
                         (event_id, event_id))

    def evidence_attempts(self, event_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [json.loads(row[0]) for row in conn.execute(
                "select summary_json from event_evidence_attempts where event_id=? order by id", (event_id,))]


    def _check_cover_requirement_lease(self, conn, event_id: int, lease_owner: str | None) -> None:
        if lease_owner is None:
            return
        row = conn.execute("select 1 from event_cover_requirements where event_id=? and state='pending' "
                           "and lease_owner=? and lease_expires_at_epoch>? and deadline_epoch>?",
                           (event_id, lease_owner, time.time(), time.time())).fetchone()
        if row is None:
            raise EventSnapshotChangedError("cover requirement lease changed or expired")

    def _event_views(self, rows) -> list[dict[str, Any]]:
        """Expose bounded operational state, never frozen private job payloads."""
        result = [dict(row) for row in rows]
        by_id = {int(row["id"]): row for row in result}
        ids = list(by_id)
        if not ids:
            return result
        with self._connect() as conn:
            for offset in range(0, len(ids), 500):
                chunk = ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                for row in conn.execute(
                    "select event_id,policy_version,state,deadline_epoch,attempts,reason from event_cover_requirements "
                    f"where event_id in({placeholders})", chunk
                ):
                    view = dict(row)
                    event_id = int(view.pop("event_id"))
                    by_id[event_id]["cover_requirement"] = view
        return result

    def _requirement_outbox(self, conn, event_id: int) -> None:
        row = conn.execute("select * from events where id=?", (event_id,)).fetchone()
        requirement = conn.execute("select state,reason,attempts,deadline_epoch,policy_version "
                                   "from event_cover_requirements where event_id=?", (event_id,)).fetchone()
        if row is not None and requirement is not None:
            self._evidence_outbox(conn, row, "cover_requirement_updated", cover_requirement=dict(requirement))

    @staticmethod
    def _presentation_evidence(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fields = {"label", "box", "mask_polygon", "confidence", "incident_eligible",
                  "snapshot_visible", "detection_frame_width", "detection_frame_height",
                  "snapshot_detection_confidence", "face_match", "identity"}
        return [{key: value for key, value in item.items() if key in fields}
                for item in objects if item.get("label") and not item.get("status")]

    @staticmethod
    def _cover_frame_area(objects: list[dict[str, Any]]) -> int:
        areas = []
        for item in objects:
            if not item.get("label") or item.get("snapshot_visible") is False:
                continue
            try:
                width = int(item.get("detection_frame_width") or 0)
                height = int(item.get("detection_frame_height") or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if width > 0 and height > 0:
                areas.append(width * height)
        return max(areas, default=0)

    def mark_evidence_publication(self, outbox_ids: list[int]) -> bool:
        ids = sorted({int(value) for value in outbox_ids})
        if not ids:
            return False
        changed = 0
        with self._lock, self._connect() as conn:
            for offset in range(0, len(ids), 500):
                chunk = ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                changed += conn.execute(
                    "update event_evidence_outbox set publication_done=1 "
                    f"where id in({placeholders})", chunk,
                ).rowcount
        return changed == len(ids)

    def expire_cover_requirements(self, *, limit: int = 100) -> int:
        """Settle expired obligations even for cameras with no running worker."""
        now = time.time()
        with self._connect() as conn:
            due = conn.execute("select 1 from event_cover_requirements where state='pending' and deadline_epoch<=? "
                               "and (lease_expires_at_epoch is null or lease_expires_at_epoch<=?) limit 1",
                               (now, now)).fetchone()
        if due is None:
            return 0
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            rows = conn.execute("select event_id from event_cover_requirements where state='pending' and deadline_epoch<=? "
                                "and (lease_expires_at_epoch is null or lease_expires_at_epoch<=?) "
                                "order by deadline_epoch,event_id limit ?",
                                (now, now, max(1, min(int(limit), 1000)))).fetchall()
            for row in rows:
                event_id = int(row["event_id"])
                conn.execute("update event_cover_requirements set state='exhausted',reason='deadline_expired', "
                             "lease_owner='',lease_expires_at_epoch=null,updated_at=? where event_id=?", (now, event_id))
                self._requirement_outbox(conn, event_id)
        return len(rows)
