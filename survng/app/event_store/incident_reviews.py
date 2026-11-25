"""Operator review verdicts for incidents.

A verdict records that a person looked at an incident and what they decided.
It is not evidence: it never changes observations, establishment, corrections
or alert history. "Nothing here" is feedback that the incident was not worth
attention; the incident stays listed, and corrections remain the way to fix
what the evidence says.

"Changed since review" compares an evidence mark (retained observations and
operator corrections), not the incident revision: settling, snapshot expiry
and other presentation changes advance the revision without new evidence.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

REVIEW_VERDICTS = ("reviewed", "keep", "nothing_here")
REVIEW_QUEUE_LIMIT = 1000


def _evidence_mark_sql(incident_column):
    """SQL for an incident's evidence mark: observation count and newest capture, plus corrections."""
    return (
        "(select count(*)||':'||coalesce(max(o.captured_epoch),0) from scene_observations o "
        f"join scene_episodes p on p.id=o.episode_id where p.incident_id={incident_column})"
        "||':'||"
        f"(select count(*) from scene_corrections c where c.incident_id={incident_column})"
    )


def _iso(epoch):
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()


class EventStoreIncidentReviewMixin:
    def _init_incident_review_db(self):
        with self._lock, self._connect() as conn:
            conn.execute(
                "create table if not exists scene_incident_reviews ("
                " incident_id text primary key,"
                " verdict text not null check (verdict in ('reviewed','keep','nothing_here')),"
                " incident_revision integer not null,"
                " evidence_mark text not null,"
                " updated_epoch real not null)"
            )

    @staticmethod
    def _carry_incident_review(conn, source, target):
        """Keep a merged-away incident's verdict on the survivor when it has none.

        The carried mark no longer matches the merged evidence, so the survivor
        reads as changed since review. Called inside the merge transaction.
        """
        conn.execute(
            "insert or ignore into scene_incident_reviews(incident_id,verdict,incident_revision,evidence_mark,updated_epoch) "
            "select ?,verdict,incident_revision,evidence_mark,updated_epoch from scene_incident_reviews where incident_id=?",
            (target, source),
        )
        conn.execute("delete from scene_incident_reviews where incident_id=?", (source,))

    @staticmethod
    def _incident_review_payload(incident_id, row, revision, mark):
        if row is None:
            return {"incident_id": incident_id, "verdict": None, "revision": revision,
                    "reviewed_revision": None, "updated_at": None, "changed_since_review": False}
        return {"incident_id": incident_id, "verdict": row["verdict"], "revision": revision,
                "reviewed_revision": row["incident_revision"],
                "updated_at": _iso(row["updated_epoch"]),
                "changed_since_review": row["evidence_mark"] != mark}

    def _incident_for_review(self, conn, incident_id):
        resolved = self._scene_resolve(conn, incident_id)
        incident = conn.execute(
            "select i.id,i.revision," + _evidence_mark_sql("i.id") + " as mark from scene_incidents i where i.id=?",
            (resolved,)).fetchone() if resolved else None
        if incident is None:
            raise LookupError("incident not found")
        return incident

    def incident_review(self, incident_id):
        with self._connect() as conn:
            incident = self._incident_for_review(conn, incident_id)
            row = conn.execute("select * from scene_incident_reviews where incident_id=?", (incident["id"],)).fetchone()
            return self._incident_review_payload(incident["id"], row, incident["revision"], incident["mark"])

    def set_incident_review(self, incident_id, verdict, *, now=None):
        """Record (or with ``verdict=None`` clear) a verdict on the current incident."""
        if verdict is not None and verdict not in REVIEW_VERDICTS:
            raise ValueError(f"verdict must be one of {', '.join(REVIEW_VERDICTS)} or null")
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            incident = self._incident_for_review(conn, incident_id)
            if verdict is None:
                conn.execute("delete from scene_incident_reviews where incident_id=?", (incident["id"],))
            else:
                conn.execute(
                    "insert into scene_incident_reviews(incident_id,verdict,incident_revision,evidence_mark,updated_epoch) "
                    "values(?,?,?,?,?) on conflict(incident_id) do update set verdict=excluded.verdict,"
                    "incident_revision=excluded.incident_revision,evidence_mark=excluded.evidence_mark,"
                    "updated_epoch=excluded.updated_epoch",
                    (incident["id"], verdict, incident["revision"], incident["mark"], time.time() if now is None else now),
                )
            row = conn.execute("select * from scene_incident_reviews where incident_id=?", (incident["id"],)).fetchone()
            return self._incident_review_payload(incident["id"], row, incident["revision"], incident["mark"])

    def incident_review_queue(self, *, start_epoch, end_epoch, camera_id="", event_type="object", object_label="", zone=""):
        """Listed incidents in a window, newest first (the incident list order), with verdicts."""
        where, args = self._scene_query_where(start_epoch=start_epoch, end_epoch=end_epoch, camera_id=camera_id,
                                              event_type=event_type, object_label=object_label, zone=zone)
        with self._connect() as conn:
            # The evidence mark is computed only for incidents that have a verdict.
            rows = conn.execute(
                "select i.id,i.start_epoch,i.end_epoch,r.verdict,r.evidence_mark,"
                "case when r.verdict is null then null else " + _evidence_mark_sql("i.id") + " end as mark "
                "from scene_incidents i left join scene_incident_reviews r on r.incident_id=i.id where " + where +
                " order by i.start_epoch desc,i.id limit ?", (*args, REVIEW_QUEUE_LIMIT + 1)).fetchall()
        items = [{"incident_id": row["id"], "start_at": _iso(row["start_epoch"]), "end_at": _iso(row["end_epoch"]),
                  "verdict": row["verdict"],
                  "changed_since_review": row["verdict"] is not None and row["evidence_mark"] != row["mark"]}
                 for row in rows[:REVIEW_QUEUE_LIMIT]]
        reviewed = sum(1 for item in items if item["verdict"] in {"reviewed", "nothing_here"})
        kept = sum(1 for item in items if item["verdict"] == "keep")
        return {"items": items, "total": len(items), "truncated": len(rows) > REVIEW_QUEUE_LIMIT,
                "reviewed": reviewed, "kept": kept, "remaining": sum(1 for item in items if item["verdict"] is None)}
