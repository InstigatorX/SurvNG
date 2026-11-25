"""Group captured faces by person, so each person is named once.

Each event's faces are grouped per person track: the track's mean embedding
joins the most similar group when the cosine similarity reaches the threshold,
otherwise it starts a new group. A group keeps the running mean of its faces.
Naming a group links it to a person; naming another group with the same name
puts both under that person, which is how a person split across groups is
joined again. An ignored group (not a person, or junk) keeps attracting
similar detections out of the way. Faces a person later gains are named by the
group they join.

The tables live in the face capture store's database.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

# Cosine similarity of a track's mean face to a group's mean face. Measured
# here without labels: at 0.35 no event-level cluster mixed two people; this
# errs further on the side of splitting, which naming merges again.
DEFAULT_GROUP_THRESHOLD = 0.40
SAMPLE_FACES = 6
# An unnamed group suggests the named person it most resembles from this
# similarity on (well below the grouping threshold: a person asks, not decides).
SUGGEST_THRESHOLD = 0.25

GROUP_SCHEMA = """
create table if not exists capture_people (
    id integer primary key autoincrement,
    name text not null unique collate nocase,
    created_at text not null,
    updated_at text not null
);
create table if not exists capture_groups (
    id integer primary key autoincrement,
    person_id integer,
    ignored integer not null default 0,
    centroid blob not null,
    faces integer not null default 0,
    first_epoch real not null,
    last_epoch real not null,
    created_at text not null,
    updated_at text not null
);
create index if not exists capture_groups_person on capture_groups(person_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def ensure_group_schema(connection) -> None:
    connection.executescript(GROUP_SCHEMA)
    columns = {row[1] for row in connection.execute("pragma table_info(face_captures)")}
    if "group_id" not in columns:
        connection.execute("alter table face_captures add column group_id integer")
    if "assignment" not in columns:
        # auto: placed by similarity; moved: placed by a person (kept on regroup)
        connection.execute("alter table face_captures add column assignment text not null default ''")
    connection.execute("create index if not exists face_captures_group on face_captures(group_id)")
    # The frame the face box is in (faces stored before this have none).
    # Each column on its own: an interrupted upgrade must not leave one
    # without the other, which would fail every later insert.
    for column in ("frame_width", "frame_height"):
        if column not in columns:
            try:
                connection.execute(f"alter table face_captures add column {column} integer")
            except sqlite3.OperationalError as exc:
                # Another process opening the store added it first.
                if "duplicate column" not in str(exc).lower():
                    raise


def _centroids(connection) -> tuple[list[int], np.ndarray]:
    ids, vectors = [], []
    for group_id, blob in connection.execute("select id, centroid from capture_groups"):
        ids.append(int(group_id))
        vectors.append(np.frombuffer(blob, dtype=np.float32))
    return ids, (np.vstack(vectors) if vectors else np.zeros((0, 512), np.float32))


def recompute_group(connection, group_id: int, drop_empty: bool = True) -> None:
    """Refresh a group's mean face, count and dates from its faces; drop it when empty.

    With ``drop_empty`` false an emptied group is kept (at zero faces, its mean
    face and name intact) so an event's replacement faces can rejoin it; the
    next grouping pass drops it if nothing did.
    """
    rows = connection.execute(
        "select embedding, captured_epoch from face_captures where group_id = ?", (group_id,)).fetchall()
    if not rows:
        if drop_empty:
            connection.execute("delete from capture_groups where id = ?", (group_id,))
        else:
            connection.execute(
                "update capture_groups set faces = 0, updated_at = ? where id = ?", (_now(), group_id))
        return
    mean = _unit(np.mean([np.frombuffer(row[0], dtype=np.float32) for row in rows], axis=0)).astype(np.float32)
    epochs = [float(row[1]) for row in rows]
    connection.execute(
        "update capture_groups set centroid = ?, faces = ?, first_epoch = ?, last_epoch = ?, updated_at = ?"
        " where id = ?",
        (mean.tobytes(), len(rows), min(epochs), max(epochs), _now(), group_id),
    )


def assign_faces(connection, threshold: float = DEFAULT_GROUP_THRESHOLD, event_id: int | None = None) -> int:
    """Group faces that have no group yet (one event's, or all); returns how many."""
    query = "select id, event_id, track, embedding, captured_epoch from face_captures where group_id is null"
    args: tuple[Any, ...] = ()
    if event_id is not None:
        query += " and event_id = ?"
        args = (int(event_id),)
    tracks: dict[tuple[int, int], list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    for face_id, face_event, track, blob, epoch in connection.execute(query + " order by id", args):
        tracks[(int(face_event), int(track))].append((int(face_id), np.frombuffer(blob, dtype=np.float32), float(epoch)))
    if not tracks:
        drop_empty_groups(connection)
        return 0
    group_ids, centroids = _centroids(connection)
    # Running sums per group touched, so each track updates its group's mean
    # face in memory; the stored row is refreshed once per group at the end.
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    placed = 0
    now = _now()
    for faces in tracks.values():
        mean = _unit(np.mean([vector for _, vector, _ in faces], axis=0)).astype(np.float32)
        best = None
        if len(group_ids):
            similarities = centroids @ mean
            index = int(np.argmax(similarities))
            if float(similarities[index]) >= threshold:
                best = group_ids[index]
        epochs = [epoch for _, _, epoch in faces]
        if best is None:
            cursor = connection.execute(
                "insert into capture_groups (centroid, faces, first_epoch, last_epoch, created_at, updated_at)"
                " values (?, 0, ?, ?, ?, ?)", (mean.tobytes(), min(epochs), max(epochs), now, now))
            best = int(cursor.lastrowid)
            group_ids.append(best)
            centroids = np.vstack([centroids, mean[None, :]])
        if best not in sums:
            existing = [np.frombuffer(row[0], dtype=np.float32) for row in connection.execute(
                "select embedding from face_captures where group_id = ?", (best,))]
            sums[best] = np.sum(existing, axis=0, dtype=np.float64) if existing else np.zeros(mean.shape, np.float64)
            counts[best] = len(existing)
        connection.executemany(
            "update face_captures set group_id = ?, assignment = 'auto' where id = ?",
            [(best, face_id) for face_id, _, _ in faces],
        )
        sums[best] = sums[best] + np.sum([vector for _, vector, _ in faces], axis=0, dtype=np.float64)
        counts[best] += len(faces)
        centroids[group_ids.index(best)] = _unit(sums[best] / counts[best]).astype(np.float32)
        placed += len(faces)
    for group_id in sums:
        recompute_group(connection, group_id)
    drop_empty_groups(connection)
    return placed


def drop_empty_groups(connection) -> None:
    """Drop groups left without faces (kept through an event's recapture), and names left without groups."""
    if connection.execute(
        "delete from capture_groups where faces = 0"
        " and id not in (select group_id from face_captures where group_id is not null)").rowcount:
        _drop_unused_people(connection)


class GroupNotFound(KeyError):
    pass


class FaceGroups:
    """Reads and edits on the groups, through the face capture store's connections."""

    def __init__(self, store: Any) -> None:
        self.store = store
        with store._lock, store._connect() as connection:
            ensure_group_schema(connection)

    # Grouping --------------------------------------------------------------

    def assign(self, event_id: int | None = None, threshold: float = DEFAULT_GROUP_THRESHOLD) -> int:
        with self.store._lock, self.store._connect() as connection:
            # Read first: the capture loop calls this every pass, and most
            # passes have nothing to group, so they take no write lock.
            pending = connection.execute(
                "select exists(select 1 from face_captures where group_id is null)"
                " or exists(select 1 from capture_groups where faces = 0)").fetchone()[0]
            if not pending:
                return 0
            connection.execute("begin immediate")
            return assign_faces(connection, threshold, event_id)

    # Reading ---------------------------------------------------------------

    def _group_rows(self, connection, where: str, args: tuple, limit: int, offset: int) -> list[dict[str, Any]]:
        rows = [dict(row) for row in connection.execute(
            "select g.id, g.person_id, g.ignored, g.faces, g.first_epoch, g.last_epoch, p.name as person_name"
            " from capture_groups g left join capture_people p on p.id = g.person_id"
            f" where {where} order by g.faces desc, g.last_epoch desc limit ? offset ?",
            (*args, int(limit), int(offset)))]
        for row in rows:
            row["samples"] = [int(r[0]) for r in connection.execute(
                "select id from face_captures where group_id = ? order by norm desc limit ?",
                (row["id"], SAMPLE_FACES))]
            row["cameras"] = [r[0] for r in connection.execute(
                "select camera_id from face_captures where group_id = ? group by camera_id order by count(*) desc",
                (row["id"],))]
            row["events"] = connection.execute(
                "select count(distinct event_id) from face_captures where group_id = ?", (row["id"],)).fetchone()[0]
            row["ignored"] = bool(row["ignored"])
        unnamed = [row for row in rows if row["person_id"] is None and not row["ignored"]]
        if unnamed:
            people = connection.execute(
                "select g.person_id, p.name, g.centroid from capture_groups g join capture_people p"
                " on p.id = g.person_id where g.ignored = 0").fetchall()
            if people:
                matrix = np.vstack([np.frombuffer(blob, dtype=np.float32) for _, _, blob in people])
                for row in unnamed:
                    centroid = connection.execute(
                        "select centroid from capture_groups where id = ?", (row["id"],)).fetchone()[0]
                    similarities = matrix @ np.frombuffer(centroid, dtype=np.float32)
                    index = int(np.argmax(similarities))
                    if float(similarities[index]) >= SUGGEST_THRESHOLD:
                        row["suggestion"] = {"person_id": int(people[index][0]), "name": people[index][1],
                                             "similarity": round(float(similarities[index]), 3)}
        return rows

    def groups(self, status: str, limit: int = 60, offset: int = 0) -> dict[str, Any]:
        where = {
            "unnamed": "g.person_id is null and g.ignored = 0",
            "named": "g.person_id is not null and g.ignored = 0",
            "ignored": "g.ignored = 1",
        }[status] + " and g.faces > 0"  # not a group emptied by a recapture, until regrouped
        with self.store._connect() as connection:
            total = connection.execute(f"select count(*) from capture_groups g where {where}").fetchone()[0]
            return {"total": int(total), "groups": self._group_rows(connection, where, (), limit, offset)}

    def group(self, group_id: int) -> dict[str, Any]:
        with self.store._connect() as connection:
            rows = self._group_rows(connection, "g.id = ?", (int(group_id),), 1, 0)
            if not rows:
                raise GroupNotFound(group_id)
            group = rows[0]
            group["faces_list"] = [dict(row) for row in connection.execute(
                "select id, event_id, camera_id, captured_epoch, norm, box_json from face_captures"
                " where group_id = ? order by captured_epoch desc limit 300", (int(group_id),))]
            group["visits"] = _visits(connection, "group_id = ?", (int(group_id),))
        return group

    def people(self) -> list[dict[str, Any]]:
        with self.store._connect() as connection:
            rows = [dict(row) for row in connection.execute(
                "select p.id, p.name, count(distinct g.id) as groups, coalesce(sum(g.faces), 0) as faces,"
                " max(g.last_epoch) as last_epoch from capture_people p left join capture_groups g"
                " on g.person_id = p.id and g.ignored = 0 group by p.id order by last_epoch desc, p.name")]
            for row in rows:
                cover = connection.execute(
                    "select f.id from face_captures f join capture_groups g on g.id = f.group_id"
                    " where g.person_id = ? order by f.norm desc limit 1", (row["id"],)).fetchone()
                row["cover"] = int(cover[0]) if cover else None
        return rows

    def person(self, person_id: int) -> dict[str, Any]:
        with self.store._connect() as connection:
            row = connection.execute("select id, name from capture_people where id = ?", (int(person_id),)).fetchone()
            if row is None:
                raise GroupNotFound(person_id)
            person = dict(row)
            person["groups"] = self._group_rows(connection, "g.person_id = ?", (int(person_id),), 500, 0)
            person["visits"] = _visits(
                connection, "group_id in (select id from capture_groups where person_id = ?)", (int(person_id),))
        return person

    def for_events(self, event_ids: list[int]) -> dict[str, Any]:
        """Who was captured in these events: per event, the groups seen; per group,
        its person (or suggestion) and best faces in these events."""
        ids = sorted({int(i) for i in event_ids})[:500]
        if not ids:
            return {"events": {}, "groups": []}
        marks = ",".join("?" * len(ids))
        with self.store._connect() as connection:
            rows = connection.execute(
                f"select id, event_id, group_id, norm from face_captures where event_id in ({marks})"
                " and group_id is not null order by norm desc", ids).fetchall()
            by_event: dict[int, list[int]] = defaultdict(list)
            faces: dict[int, list[int]] = defaultdict(list)
            for face_id, event_id, group_id, _ in rows:
                if int(group_id) not in by_event[int(event_id)]:
                    by_event[int(event_id)].append(int(group_id))
                if len(faces[int(group_id)]) < SAMPLE_FACES:
                    faces[int(group_id)].append(int(face_id))
            groups = self._group_rows(
                connection, f"g.id in ({','.join('?' * len(faces))})", tuple(faces), 500, 0) if faces else []
        for group in groups:
            group["incident_faces"] = faces[group["id"]]
        return {"events": {str(event): group_ids for event, group_ids in by_event.items()}, "groups": groups}

    def for_incident(
        self, incident: dict[str, Any], main_size: Any,
    ) -> dict[str, Any]:
        """``for_events`` for the incident's events, plus which scene object
        (tracked person) each group's faces were bound to."""
        from .binding import bind_faces

        event_ids = incident.get("event_ids") or [event.get("id") for event in incident.get("events") or []]
        ids = sorted({int(i) for i in event_ids if i is not None})
        # A busy incident can hold more events than one lookup takes.
        events: dict[str, list[int]] = {}
        groups: dict[int, dict[str, Any]] = {}
        for start in range(0, len(ids), 500):
            part = self.for_events(ids[start:start + 500])
            events.update(part["events"])
            for group in part["groups"]:
                seen = groups.get(group["id"])
                if seen is None:
                    groups[group["id"]] = group
                else:
                    merged = list(dict.fromkeys(seen["incident_faces"] + group["incident_faces"]))
                    seen["incident_faces"] = merged[:SAMPLE_FACES]
        found = {"events": events, "groups": list(groups.values())}
        objects: dict[str, list[int]] = {}
        if ids:
            rows = []
            with self.store._connect() as connection:
                for start in range(0, len(ids), 500):
                    chunk = ids[start:start + 500]
                    rows.extend(dict(row) for row in connection.execute(
                        "select id, event_id, camera_id, captured_epoch, box_json, frame_width, frame_height, group_id"
                        f" from face_captures where event_id in ({','.join('?' * len(chunk))}) and group_id is not null",
                        chunk))
            bound = bind_faces(rows, incident.get("scene_objects") or [], main_size)
            group_of = {row["id"]: row["group_id"] for row in rows}
            for face_id, object_id in bound.items():
                key = str(object_id)
                if group_of[face_id] not in objects.setdefault(key, []):
                    objects[key].append(group_of[face_id])
        return {**found, "objects": objects}

    def face_crop_path(self, face_id: int) -> str | None:
        with self.store._connect() as connection:
            row = connection.execute("select crop_path from face_captures where id = ?", (int(face_id),)).fetchone()
        return str(row[0]) if row else None

    # Edits -----------------------------------------------------------------

    def _person_id(self, connection, name: str) -> int:
        clean = " ".join(str(name).split())
        if not clean:
            raise ValueError("a name is required")
        if len(clean) > 80:
            raise ValueError("names are at most 80 characters")
        row = connection.execute("select id from capture_people where name = ?", (clean,)).fetchone()
        if row is not None:
            return int(row[0])
        now = _now()
        return int(connection.execute(
            "insert into capture_people (name, created_at, updated_at) values (?, ?, ?)", (clean, now, now)).lastrowid)

    def _require_group(self, connection, group_id: int) -> None:
        if connection.execute("select 1 from capture_groups where id = ?", (int(group_id),)).fetchone() is None:
            raise GroupNotFound(group_id)

    def name_group(self, group_id: int, name: str) -> dict[str, Any]:
        with self.store._lock, self.store._connect() as connection:
            connection.execute("begin immediate")
            self._require_group(connection, group_id)
            person_id = self._person_id(connection, name)
            connection.execute(
                "update capture_groups set person_id = ?, ignored = 0, updated_at = ? where id = ?",
                (person_id, _now(), int(group_id)))
            _drop_unused_people(connection)  # the group's previous name, if nothing else has it
        return self.group(group_id)

    def unname_group(self, group_id: int) -> dict[str, Any]:
        with self.store._lock, self.store._connect() as connection:
            connection.execute("begin immediate")
            self._require_group(connection, group_id)
            connection.execute(
                "update capture_groups set person_id = null, updated_at = ? where id = ?", (_now(), int(group_id)))
            _drop_unused_people(connection)
        return self.group(group_id)

    def ignore_group(self, group_id: int, ignored: bool) -> dict[str, Any]:
        with self.store._lock, self.store._connect() as connection:
            connection.execute("begin immediate")
            self._require_group(connection, group_id)
            connection.execute(
                "update capture_groups set ignored = ?, person_id = case when ? then null else person_id end,"
                " updated_at = ? where id = ?", (1 if ignored else 0, 1 if ignored else 0, _now(), int(group_id)))
            _drop_unused_people(connection)
        return self.group(group_id)

    def merge_groups(self, source_id: int, target_id: int) -> dict[str, Any]:
        if int(source_id) == int(target_id):
            raise ValueError("a group cannot be merged into itself")
        with self.store._lock, self.store._connect() as connection:
            connection.execute("begin immediate")
            self._require_group(connection, source_id)
            self._require_group(connection, target_id)
            connection.execute(
                "update face_captures set group_id = ?, assignment = 'moved' where group_id = ?",
                (int(target_id), int(source_id)))
            source_person = connection.execute(
                "select person_id from capture_groups where id = ?", (int(source_id),)).fetchone()[0]
            connection.execute("delete from capture_groups where id = ?", (int(source_id),))
            if source_person is not None:
                connection.execute(
                    "update capture_groups set person_id = coalesce(person_id, ?) where id = ? and ignored = 0",
                    (source_person, int(target_id)))
            recompute_group(connection, int(target_id))
            _drop_unused_people(connection)
        return self.group(target_id)

    def remove_face(self, face_id: int) -> dict[str, Any]:
        """Take a wrongly grouped face out into a group of its own."""
        with self.store._lock, self.store._connect() as connection:
            connection.execute("begin immediate")
            row = connection.execute(
                "select group_id, embedding, captured_epoch from face_captures where id = ?", (int(face_id),)).fetchone()
            if row is None:
                raise GroupNotFound(face_id)
            old_group, blob, epoch = row
            now = _now()
            new_group = int(connection.execute(
                "insert into capture_groups (centroid, faces, first_epoch, last_epoch, created_at, updated_at)"
                " values (?, 1, ?, ?, ?, ?)", (blob, epoch, epoch, now, now)).lastrowid)
            connection.execute(
                "update face_captures set group_id = ?, assignment = 'moved' where id = ?", (new_group, int(face_id)))
            if old_group is not None:
                recompute_group(connection, int(old_group))
            _drop_unused_people(connection)
        return {"face_id": int(face_id), "group_id": new_group, "previous_group_id": old_group}

    def rename_person(self, person_id: int, name: str) -> dict[str, Any]:
        """Rename; renaming to another person's name joins the two people."""
        clean = " ".join(str(name).split())
        if not clean or len(clean) > 80:
            raise ValueError("a name of 1 to 80 characters is required")
        result_id = int(person_id)
        with self.store._lock, self.store._connect() as connection:
            connection.execute("begin immediate")
            if connection.execute("select 1 from capture_people where id = ?", (int(person_id),)).fetchone() is None:
                raise GroupNotFound(person_id)
            other = connection.execute(
                "select id from capture_people where name = ? and id != ?", (clean, int(person_id))).fetchone()
            if other is not None:
                result_id = int(other[0])
                connection.execute(
                    "update capture_groups set person_id = ? where person_id = ?", (result_id, int(person_id)))
                connection.execute("delete from capture_people where id = ?", (int(person_id),))
            else:
                connection.execute(
                    "update capture_people set name = ?, updated_at = ? where id = ?", (clean, _now(), int(person_id)))
        return self.person(result_id)

    def forget_person(self, person_id: int) -> None:
        """Remove the name; the person's groups go back to "to name"."""
        with self.store._lock, self.store._connect() as connection:
            connection.execute("begin immediate")
            connection.execute("update capture_groups set person_id = null where person_id = ?", (int(person_id),))
            connection.execute("delete from capture_people where id = ?", (int(person_id),))

    def summary(self) -> dict[str, int]:
        with self.store._connect() as connection:
            row = connection.execute(
                "select sum(person_id is null and ignored = 0), sum(person_id is not null and ignored = 0),"
                " sum(ignored = 1) from capture_groups where faces > 0").fetchone()
            people = connection.execute("select count(*) from capture_people").fetchone()[0]
            ungrouped = connection.execute("select count(*) from face_captures where group_id is null").fetchone()[0]
        return {"unnamed_groups": int(row[0] or 0), "named_groups": int(row[1] or 0),
                "ignored_groups": int(row[2] or 0), "people": int(people), "ungrouped_faces": int(ungrouped)}


def _visits(connection, where: str, args: tuple, limit: int = 200) -> list[dict[str, Any]]:
    """The events a group or person was seen in, newest first, each with its best face."""
    events: dict[int, dict[str, Any]] = {}
    for event_id, camera_id, epoch, face_id, norm in connection.execute(
        f"select event_id, camera_id, captured_epoch, id, norm from face_captures where {where}", args,
    ):
        visit = events.setdefault(int(event_id), {
            "event_id": int(event_id), "camera_id": camera_id, "epoch": float(epoch), "faces": 0,
            "best_face": int(face_id), "_norm": float(norm),
        })
        visit["faces"] += 1
        visit["epoch"] = min(visit["epoch"], float(epoch))
        if float(norm) > visit["_norm"]:
            visit["best_face"], visit["_norm"] = int(face_id), float(norm)
    ordered = sorted(events.values(), key=lambda visit: visit["epoch"], reverse=True)[:limit]
    for visit in ordered:
        visit.pop("_norm")
    return ordered


def _drop_unused_people(connection) -> None:
    connection.execute(
        "delete from capture_people where id not in (select person_id from capture_groups where person_id is not null)")

