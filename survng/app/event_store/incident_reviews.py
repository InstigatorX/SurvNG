"""Operator review verdicts for incidents.

A verdict records that a person looked at an incident and what they decided.
It is not evidence: it never changes observations, establishment, corrections
or alert history. "Nothing here" is feedback that the incident was not worth
attention; the incident stays listed, and corrections remain the way to fix
what the evidence says.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

REVIEW_VERDICTS = ("reviewed", "keep", "nothing_here")
REVIEW_QUEUE_LIMIT = 1000


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
                               " updated_epoch real not null)"
            )

    @staticmethod
    def _incident_review_payload(incident_id, row, revision):
        if row is None:
            return {"incident_id": incident_id, "verdict": None, "revision": revision,
                    "reviewed_revision": None, "updated_at": None, "changed_since_review": False}
        return {"incident_id": incident_id, "verdict": row["verdict"], "revision": revision,
                "reviewed_revision": row["incident_revision"],
                "updated_at": _iso(row["updated_epoch"]),
                "changed_since_review": revision is not None and row["incident_revision"] != revision}

    def incident_review(self, incident_id):
        with self._connect() as conn:
            resolved = self._scene_resolve(conn, incident_id)
            incident = conn.execute("select id,revision from scene_incidents where id=?", (resolved,)).fetchone() if resolved else None
            if incident is None:
                raise LookupError("incident not found")
            row = conn.execute("select * from scene_incident_reviews where incident_id=?", (incident["id"],)).fetchone()
            return self._incident_review_payload(incident["id"], row, incident["revision"])

    def set_incident_review(self, incident_id, verdict, *, now=None):
        """Record (or with ``verdict=None`` clear) a verdict on the current incident."""
        if verdict is not None and verdict not in REVIEW_VERDICTS:
            raise ValueError(f"verdict must be one of {', '.join(REVIEW_VERDICTS)} or null")
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            resolved = self._scene_resolve(conn, incident_id)
            incident = conn.execute("select id,revision from scene_incidents where id=?", (resolved,)).fetchone() if resolved else None
            if incident is None:
                raise LookupError("incident not found")
            if verdict is None:
                conn.execute("delete from scene_incident_reviews where incident_id=?", (incident["id"],))
            else:
                conn.execute(
                    "insert into scene_incident_reviews(incident_id,verdict,incident_revision,updated_epoch) "
                    "values(?,?,?,?) on conflict(incident_id) do update set verdict=excluded.verdict,"
                    "incident_revision=excluded.incident_revision,"
                    "updated_epoch=excluded.updated_epoch",
                    (incident["id"], verdict, incident["revision"], time.time() if now is None else now),
                )
            row = conn.execute("select * from scene_incident_reviews where incident_id=?", (incident["id"],)).fetchone()
            return self._incident_review_payload(incident["id"], row, incident["revision"])

    def incident_review_queue(self, *, start_epoch, end_epoch, camera_id="", event_type="all", object_label="", zone=""):
        """Listed incidents in a window, newest first (the incident list order), with verdicts."""
        where, args = self._scene_query_where(start_epoch=start_epoch, end_epoch=end_epoch, camera_id=camera_id,
                                              event_type=event_type, object_label=object_label, zone=zone)
        with self._connect() as conn:
            rows = conn.execute(
                "select i.id,i.revision,i.start_epoch,i.end_epoch,r.verdict,r.incident_revision "
                "from scene_incidents i left join scene_incident_reviews r on r.incident_id=i.id where " + where +
                " order by i.start_epoch desc,i.id limit ?", (*args, REVIEW_QUEUE_LIMIT + 1)).fetchall()
        items = [{"incident_id": row["id"], "start_at": _iso(row["start_epoch"]), "end_at": _iso(row["end_epoch"]),
                  "verdict": row["verdict"],
                  "changed_since_review": row["verdict"] is not None and row["incident_revision"] != row["revision"]}
                 for row in rows[:REVIEW_QUEUE_LIMIT]]
        reviewed = sum(1 for item in items if item["verdict"] in {"reviewed", "nothing_here"})
        kept = sum(1 for item in items if item["verdict"] == "keep")
        return {"items": items, "total": len(items), "truncated": len(rows) > REVIEW_QUEUE_LIMIT,
                "reviewed": reviewed, "kept": kept, "remaining": sum(1 for item in items if item["verdict"] is None)}
