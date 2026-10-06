"""Persistent, reversible event stories. Original incidents remain authoritative."""
from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

from .main_database import connect_main_database


class StoryConflict(ValueError):
    pass


class StorylineStore:
    def __init__(self, database_path, write_lock=None):
        self.database_path = database_path
        self.write_lock = write_lock or threading.RLock()
        with self._connect() as db:
            db.execute("""create table if not exists storylines (
                id text primary key, revision integer not null,
                payload_json text not null, updated_at real not null)""")
            db.execute("create index if not exists storylines_updated on storylines(updated_at desc)")

    @contextmanager
    def _connect(self):
        db = connect_main_database(self.database_path, timeout=10, write_lock=self.write_lock)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _read(db, story_id):
        row = db.execute("select * from storylines where id=?", (story_id,)).fetchone()
        if row is None:
            raise LookupError("Storyline was not found")
        return json.loads(row["payload_json"]) | {"id": row["id"], "revision": row["revision"], "updated_at": row["updated_at"]}

    @staticmethod
    def _check(story, revision):
        if story["revision"] != revision:
            raise StoryConflict("Storyline changed; refresh before editing")

    @staticmethod
    def _save(db, story):
        story = copy.deepcopy(story)
        story["updated_at"] = time.time()
        db.execute("insert into storylines values(?,?,?,?) on conflict(id) do update set revision=excluded.revision,payload_json=excluded.payload_json,updated_at=excluded.updated_at",
                   (story["id"], story["revision"], json.dumps(story, allow_nan=False, separators=(",", ":")), story["updated_at"]))
        return story

    def list(self, limit=50, offset=0):
        with self._connect() as db:
            rows = db.execute("select id from storylines order by updated_at desc,id limit ? offset ?", (min(100, max(1, limit)), max(0, offset))).fetchall()
            return {"items": [self._read(db, row[0]) for row in rows], "total": db.execute("select count(*) from storylines").fetchone()[0]}

    def get(self, story_id):
        with self._connect() as db:
            return self._read(db, story_id)

    def create(self, payload):
        with self._connect() as db:
            return self._save(db, payload | {"id": "story-" + uuid.uuid4().hex, "revision": 1, "created_at": time.time(), "dismissed": {}, "ai_review": None})

    def update(self, story_id, revision, changes):
        with self._connect() as db:
            db.execute("begin immediate")
            story = self._read(db, story_id)
            self._check(story, revision)
            # AI annotations are evidence-versioned and cannot survive membership edits.
            if "members" in changes and changes["members"] != story["members"]:
                story["ai_review"] = None
            return self._save(db, story | changes | {"revision": revision + 1})

    def delete(self, story_id, revision):
        with self._connect() as db:
            db.execute("begin immediate")
            self._check(self._read(db, story_id), revision)
            db.execute("delete from storylines where id=?", (story_id,))

    def split(self, story_id, revision, incident_ids):
        with self._connect() as db:
            db.execute("begin immediate")
            story = self._read(db, story_id)
            self._check(story, revision)
            selected = set(incident_ids)
            members = story["members"]
            if not selected or not selected < {m["incident_id"] for m in members}:
                raise ValueError("Choose a nonempty proper subset of the Storyline")
            original = self._save(db, story | {"members": [m for m in members if m["incident_id"] not in selected], "revision": revision + 1, "ai_review": None})
            separated = self._save(db, story | {"id": "story-" + uuid.uuid4().hex, "title": story["title"][:112] + " (split)", "members": [m for m in members if m["incident_id"] in selected], "revision": 1, "created_at": time.time(), "ai_review": None, "dismissed": {}})
            return {"story": original, "separated": separated}

    def merge(self, story_id, revision, source_id, source_revision):
        if source_id == story_id:
            raise ValueError("Choose another Storyline")
        with self._connect() as db:
            db.execute("begin immediate")
            story, source = self._read(db, story_id), self._read(db, source_id)
            self._check(story, revision)
            self._check(source, source_revision)
            members = {m["incident_id"]: m for m in story["members"] + source["members"]}
            if len(members) > 64:
                raise ValueError("A Storyline supports at most 64 incidents")
            result = self._save(db, story | {"members": list(members.values()), "revision": revision + 1, "ai_review": None})
            db.execute("delete from storylines where id=?", (source_id,))
            return result
