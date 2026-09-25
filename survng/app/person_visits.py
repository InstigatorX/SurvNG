"""Reversible visits projected from authoritative person and face evidence.

The projection is rebuilt on a bounded read, so corrected/deleted faces and
replaced tracking evidence cannot leave a stale name on a visit. Only operator
decisions are stored here; inferred sightings never become face references.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .main_database import connect_main_database


def epoch(value: object) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        result = parsed.timestamp()
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def iso(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def route_between(left: dict, right: dict, routes: list) -> str | None:
    if left["end"] > right["start"]:
        return None
    gap = right["start"] - left["end"]
    for route in routes:
        if not route.enabled or not route.min_seconds <= gap <= route.max_seconds:
            continue
        cameras = (left["camera_id"], right["camera_id"])
        if cameras == (route.from_camera, route.to_camera) or (
            route.bidirectional and cameras == (route.to_camera, route.from_camera)
        ):
            return route.name or f"{cameras[0]} → {cameras[1]}"
    return None


def similarity(left: dict, right: dict) -> float | None:
    if left.get("vector") is None or right.get("vector") is None:
        return None
    if (left["model"], left["vector"].size) != (right["model"], right["vector"].size):
        return None
    return float(np.clip(left["vector"] @ right["vector"], -1, 1))


def _pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def project_visits(sightings: list[dict], decisions: dict, config: Any) -> dict:
    """Pure, deterministic association; no names or embeddings are mutated."""
    nodes = {item["id"]: item for item in sightings}
    groups = {key: {key} for key in nodes}
    roots = {key: key for key in nodes}
    accepted: list[dict] = []
    candidates: list[dict] = []
    rejected = {pair for pair, value in decisions.items() if value == "reject"}

    def compatible(a: set[str], b: set[str], *, automatic: bool) -> bool:
        combined = a | b
        names = {nodes[key]["person_id"] for key in combined if nodes[key].get("person_id")}
        if len(names) > 1 or any(nodes[key].get("identity_conflict") for key in combined):
            return False
        if max(nodes[key]["end"] for key in combined) - min(nodes[key]["start"] for key in combined) > config.visit_max_seconds:
            return False
        for x in a:
            for y in b:
                l, r = nodes[x], nodes[y]
                if _pair(x, y) in rejected:
                    return False
                # Separate people in the same incident must never collapse.
                same_event_owner = not automatic and l["event_id"] == r["event_id"] and ((l.get("track_id") is None) != (r.get("track_id") is None))
                if l["event_id"] == r["event_id"] and not same_event_owner:
                    return False
                if not same_event_owner and l["camera_id"] == r["camera_id"] and max(l["start"], r["start"]) <= min(l["end"], r["end"]):
                    return False
                if automatic:
                    score = similarity(l, r)
                    # Complete-link checking prevents gradual appearance drift.
                    if score is None or score < max(config.visit_match_threshold, l.get("threshold", 0), r.get("threshold", 0)):
                        return False
        return True

    def merge(edge: dict, *, automatic: bool) -> bool:
        a, b = roots[edge["left"]], roots[edge["right"]]
        if a == b:
            return True
        if not compatible(groups[a], groups[b], automatic=automatic):
            return False
        keep, drop = sorted((a, b))
        groups[keep] |= groups.pop(drop)
        for key in groups[keep]:
            roots[key] = keep
        accepted.append(edge)
        return True

    for (left, right), decision in sorted(decisions.items()):
        if decision == "accept" and left in nodes and right in nodes:
            edge = {"left": left, "right": right, "source": "operator", "reason": "You confirmed these sightings belong to the same visit"}
            if not merge(edge, automatic=False):
                candidates.append({**edge, "blocked": True, "reason": "New evidence conflicts with this previously confirmed link"})

    ordered = sorted(sightings, key=lambda item: (item["start"], item["id"]))
    possible: list[dict] = []
    competitors: dict[tuple[str, str], list[float]] = {}
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            if right["start"] - left["start"] > config.visit_max_seconds:
                break
            pair = _pair(left["id"], right["id"])
            if pair in decisions or left["camera_id"] == right["camera_id"]:
                continue
            route = route_between(left, right, config.camera_transition_routes)
            if route is None or not compatible({left["id"]}, {right["id"]}, automatic=False):
                continue
            if min(left.get("quality", 0), right.get("quality", 0)) < config.visit_min_quality:
                continue
            if min(left.get("observation_count", 0), right.get("observation_count", 0)) < 2:
                continue
            score = similarity(left, right)
            if score is None:
                continue
            competitors.setdefault((left["id"], right["camera_id"]), []).append(score)
            competitors.setdefault((right["id"], left["camera_id"]), []).append(score)
            if score < max(config.visit_match_threshold, left["threshold"], right["threshold"]):
                continue
            possible.append({"left": left["id"], "right": right["id"], "source": "appearance_route", "similarity": round(score, 4), "route": route, "gap_seconds": round(right["start"] - left["end"], 3), "reason": "Compatible body appearance and camera transition", "_score": score})
    for edge in sorted(possible, key=lambda item: (-item["_score"], item["left"], item["right"])):
        left, right = nodes[edge["left"]], nodes[edge["right"]]
        score = edge.pop("_score")
        margins = []
        for key in ((left["id"], right["camera_id"]), (right["id"], left["camera_id"])):
            others = sorted(competitors[key], reverse=True)
            margins.append(score - others[1] if len(others) > 1 else 2.0)
        ambiguous = min(margins) < config.visit_top_two_margin
        edge["ambiguous"] = ambiguous
        if roots[edge["left"]] == roots[edge["right"]]:
            continue
        if config.visit_auto_link_enabled and not ambiguous and merge(edge, automatic=True):
            continue
        candidates.append(edge)

    visits = []
    for key, members in groups.items():
        entries = sorted((nodes[item] for item in members), key=lambda item: (item["start"], item["id"]))
        anchors = [item for item in entries if item.get("person_id")]
        person = anchors[0] if anchors else None
        conflict = any(item.get("identity_conflict") for item in entries)
        public = []
        for item in entries:
            public.append({k: v for k, v in item.items() if k not in {"vector", "model", "start", "end", "threshold"}} | {
                "first_seen": iso(item["start"]), "last_seen": iso(item["end"]),
                "identity_status": "conflict" if item.get("identity_conflict") else item.get("anchor_status") if item.get("person_id") else "linked" if person else "unresolved",
                "linked_person_id": person["person_id"] if person and not item.get("person_id") else None,
            })
        visits.append({"id": key, "person_id": person["person_id"] if person else None, "person_name": person["person_name"] if person else "", "status": "conflict" if conflict else "identified" if person else "unresolved", "first_seen": iso(entries[0]["start"]), "last_seen": iso(max(item["end"] for item in entries)), "sightings": public, "links": [edge for edge in accepted if edge["left"] in members and edge["right"] in members], "anchor_ids": [item["id"] for item in anchors]})
    return {"visits": sorted(visits, key=lambda item: (item["last_seen"], item["id"]), reverse=True), "suggestions": candidates[:200], "suggestions_truncated": len(candidates) > 200}


class PersonVisitStore:
    MAX_SIGHTINGS = 400

    def __init__(self, database_path, database_write_lock=None):
        self.database_path = database_path
        self.database_write_lock = database_write_lock or threading.RLock()
        with self._connect() as db:
            db.execute("""create table if not exists person_visit_decisions (
                left_id text not null, right_id text not null,
                left_event integer not null references events(id) on delete cascade,
                right_event integer not null references events(id) on delete cascade,
                left_revision text not null, right_revision text not null,
                decision text not null check(decision in ('accept','reject')),
                updated_at text not null, primary key(left_id,right_id))""")

    @contextmanager
    def _connect(self):
        db = connect_main_database(self.database_path, timeout=10, write_lock=self.database_write_lock)
        db.row_factory = sqlite3.Row
        db.execute("pragma foreign_keys = on")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _read(self, db, start: float, end: float) -> tuple[list[dict], bool]:
        events = db.execute("""select id,camera_id,created_at,objects_json from events
            where created_at >= ? and created_at < ?
            and julianday(created_at) between julianday(?) and julianday(?)
            and (exists(select 1 from appearance_embeddings a where a.event_id=events.id and a.model_kind='person')
                 or exists(select 1 from face_observations f where f.event_id=events.id and f.canonical=1)
                 or objects_json like '%"person"%' or objects_json like '%"pedestrian"%')
            order by julianday(created_at) desc,id desc limit ?""", (iso(start - 86400)[:10], iso(end + 172800)[:10], iso(start), iso(end), self.MAX_SIGHTINGS + 1)).fetchall()
        result = []
        truncated = len(events) > self.MAX_SIGHTINGS
        for event in events[:self.MAX_SIGHTINGS]:
            at = epoch(event["created_at"])
            if at is None:
                continue
            rows = db.execute("""select * from appearance_embeddings where event_id=? and model_kind='person'
                and label in ('person','pedestrian') order by id desc limit ?""", (event["id"], self.MAX_SIGHTINGS + 1)).fetchall()
            faces = db.execute("""select f.id,f.candidate_track_id,f.person_id,f.review_status,f.observed_at,
                f.candidate_offset_seconds,f.box_json,p.name from face_observations f
                left join face_people p on p.id=f.person_id where f.event_id=? and f.canonical=1
                order by f.id limit ?""", (event["id"], self.MAX_SIGHTINGS + 1)).fetchall()
            tracks = {}
            for row in rows:
                if row["track_id"] in tracks:
                    continue
                raw = np.frombuffer(row["embedding_blob"], dtype=np.float32) if len(row["embedding_blob"]) % 4 == 0 else np.array([])
                norm = float(np.linalg.norm(raw))
                vector = raw / norm if raw.size == row["embedding_size"] and raw.size and np.all(np.isfinite(raw)) and math.isfinite(norm) and norm > 1e-9 else None
                first, last = epoch(row["first_seen"]), epoch(row["last_seen"])
                first = first if first is not None else at
                last = last if last is not None else first
                if last < first:
                    continue
                tracks[row["track_id"]] = {"id": f"track:{event['id']}:{row['track_id']}", "event_id": event["id"], "track_id": row["track_id"], "camera_id": event["camera_id"], "start": first, "end": last, "vector": vector, "model": row["model_fingerprint"], "quality": row["quality"], "observation_count": row["observation_count"], "threshold": row["match_threshold"], "person_id": None, "person_name": "", "face_id": None}
            try:
                objects = json.loads(event["objects_json"])
            except (ValueError, TypeError):
                objects = []
            objects = objects if isinstance(objects, list) else []
            summaries = []
            for obj in objects:
                if isinstance(obj, dict) and isinstance(obj.get("object_tracking"), dict):
                    summaries.extend(obj["object_tracking"].get("tracks") or [])
            people = [obj for obj in objects if isinstance(obj, dict) and obj.get("label") in ("person", "pedestrian") and obj.get("incident_eligible") is not False]
            for index, track in enumerate(summaries or (people if not tracks else [])):
                if not isinstance(track, dict) or track.get("label") not in ("person", "pedestrian"):
                    continue
                track_id = track.get("track_id")
                if track_id in tracks:
                    continue
                key = track_id if track_id is not None else f"object-{index}"
                first = epoch(track.get("first_seen")) or at
                last = epoch(track.get("last_seen")) or first
                if last < first:
                    continue
                tracks[key] = {"id": f"track:{event['id']}:{key}", "event_id": event["id"], "track_id": key, "camera_id": event["camera_id"], "start": first, "end": last, "vector": None, "model": "", "quality": 0, "observation_count": 0, "threshold": 1, "person_id": None, "person_name": "", "face_id": None, "geometry_revision": hashlib.sha256(json.dumps(track, sort_keys=True).encode()).hexdigest()}
            person_count = sum(1 for obj in objects if isinstance(obj, dict) and obj.get("label") in ("person", "pedestrian") and obj.get("incident_eligible") is not False) if isinstance(objects, list) else 0
            for face in faces:
                # Historical face track IDs are independent of body track IDs.
                # Attach only when the event has exactly one of each, otherwise
                # retain a separate face sighting instead of guessing ownership.
                when = epoch(face["observed_at"])
                when = (when if when is not None else at) + float(face["candidate_offset_seconds"] or 0)
                only_track = next(iter(tracks.values())) if len(tracks) == 1 else None
                attached = only_track is not None and len(faces) == 1 and person_count == 1 and only_track["start"] - .5 <= when <= only_track["end"] + .5
                if attached:
                    node = next(iter(tracks.values()))
                else:
                    node = {"id": f"face:{face['id']}", "event_id": event["id"], "track_id": None, "camera_id": event["camera_id"], "start": when, "end": when, "vector": None, "model": "", "person_id": None, "person_name": ""}
                node["face_id"] = face["id"]
                if face["person_id"] and face["name"] and face["review_status"] in ("confirmed", "auto_identified"):
                    node.update(person_id=face["person_id"], person_name=face["name"], anchor_status="confirmed" if face["review_status"] == "confirmed" else "recognized")
                if not attached:
                    result.append(node)
            result.extend(tracks.values())
            if len(result) > self.MAX_SIGHTINGS:
                truncated = True
                break
        result = sorted(result, key=lambda item: (item["start"], item["id"]), reverse=True)[:self.MAX_SIGHTINGS]
        for node in result:
            vector_hash = hashlib.sha256(node["vector"].tobytes()).hexdigest() if node["vector"] is not None else ""
            node["evidence_revision"] = hashlib.sha256(json.dumps({k: v for k, v in node.items() if k not in {"vector", "person_id", "person_name", "anchor_status"}}, sort_keys=True).encode() + vector_hash.encode()).hexdigest()
            node["revision"] = hashlib.sha256(json.dumps({k: v for k, v in node.items() if k != "vector"}, sort_keys=True).encode() + vector_hash.encode()).hexdigest()
        return result, truncated

    @staticmethod
    def _decisions(db, nodes):
        found = {}
        if not nodes:
            return found
        placeholders = ','.join('?' for _ in nodes)
        for row in db.execute(f"select * from person_visit_decisions where left_id in ({placeholders})", tuple(nodes)):
            left, right = nodes.get(row["left_id"]), nodes.get(row["right_id"])
            if left and right and left["evidence_revision"] == row["left_revision"] and right["evidence_revision"] == row["right_revision"]:
                found[(row["left_id"], row["right_id"])] = row["decision"]
        return found

    def list(self, start: float, end: float, config) -> dict:
        if not all(math.isfinite(x) for x in (start, end)) or not 0 < end - start <= 86400:
            raise ValueError("Choose a time range of at most 24 hours")
        with self._connect() as db:
            db.execute("begin")
            sightings, truncated = self._read(db, start, end)
            decisions = self._decisions(db, {item["id"]: item for item in sightings})
        effective = config.model_copy(update={"visit_auto_link_enabled": False}) if truncated else config
        result = project_visits(sightings, decisions, effective)
        return {**result, "start": start, "end": end, "truncated": truncated, "auto_link_enabled": config.visit_auto_link_enabled, "routes_configured": sum(route.enabled for route in config.camera_transition_routes)}

    def decide(self, left_id: str, right_id: str, left_revision: str, right_revision: str, decision: str, start: float, end: float, config) -> dict:
        if decision not in {"accept", "reject", "reset"} or left_id == right_id:
            raise ValueError("Invalid visit link decision")
        if not all(math.isfinite(x) for x in (start, end)) or not 0 < end - start <= 86400:
            raise ValueError("Choose a time range of at most 24 hours")
        with self._connect() as db:
            db.execute("begin immediate")
            sightings, _ = self._read(db, start, end)
            nodes = {item["id"]: item for item in sightings}
            left, right = nodes.get(left_id), nodes.get(right_id)
            if not left or not right or left["revision"] != left_revision or right["revision"] != right_revision:
                raise ValueError("Evidence changed or expired; refresh visits before reviewing")
            pair = _pair(left_id, right_id)
            decisions = self._decisions(db, nodes)
            if decision == "accept":
                decisions[pair] = "accept"
                projected = project_visits(sightings, decisions, config)
                if not any({left_id, right_id} <= {item["id"] for item in visit["sightings"]} for visit in projected["visits"]):
                    raise ValueError("This link conflicts with identity, time, or a rejected link")
            if decision == "reset":
                db.execute("delete from person_visit_decisions where left_id=? and right_id=?", pair)
            else:
                a, b = nodes[pair[0]], nodes[pair[1]]
                db.execute("""insert into person_visit_decisions values (?,?,?,?,?,?,?,?)
                    on conflict(left_id,right_id) do update set left_revision=excluded.left_revision,
                    right_revision=excluded.right_revision,decision=excluded.decision,updated_at=excluded.updated_at""", (*pair, a["event_id"], b["event_id"], a["evidence_revision"], b["evidence_revision"], decision, datetime.now(timezone.utc).isoformat()))
        return {"updated": True}
