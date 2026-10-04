"""Durable scene membership. Alert policy never decides what was observed.

All writes share the event transaction. Observations are append-only; object
associations and operator corrections are revisioned projections of that evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from datetime import datetime, timezone
from ..incident_utils import portable_media_path
from ..scene_identity import observation_identity
from .scene_history import legacy_track_observations

# Each camera episode's recorded analysis is recomputed from all of its
# observations, so one episode is bounded. Longer activity continues the same
# incident in a new episode.
MAX_SCENE_EPISODE_SECONDS = 900.0


def _display_confidence(item):
    """Hide below-threshold detections, without changing retained evidence or alert policy."""
    if item.get("confidence_eligible") is False:
        return False
    threshold = item.get("confidence_threshold")
    if threshold is None:
        return True  # Legacy evidence did not retain its confidence policy.
    try:
        score, threshold = float(item.get("confidence", 0)), float(threshold)
    except (TypeError, ValueError):
        return False
    return math.isfinite(score) and math.isfinite(threshold) and score >= threshold


# Equivalent predicate for bounded card/label queries, before loading payloads.
_DISPLAY_CONFIDENCE_SQL = (
    "coalesce(json_extract(o.payload_json,'$.confidence_eligible'),1)!=0 and "
    "(json_extract(o.payload_json,'$.confidence_threshold') is null or "
    "cast(json_extract(o.payload_json,'$.confidence') as real)>="
    "cast(json_extract(o.payload_json,'$.confidence_threshold') as real))"
)


def _incident_display_summary(objects):
    names = []
    for label in sorted({item["label"] for item in objects} - {"face"}):
        possible = all(item["certainty"] == "possible" for item in objects if item["label"] == label)
        names.append(("possible " if possible else "") + label.replace("_", " "))
    if not names:
        return "Activity observed"
    text = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
    return text[:1].upper() + text[1:] + " observed"


class SceneConflict(ValueError):
    """A correction was based on an obsolete incident revision."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _epoch(value, fallback=0.0):
    try:
        result = float(value) if isinstance(value, (float, int)) else datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        return result if math.isfinite(result) else fallback
    except (ValueError, TypeError, OverflowError):
        return fallback


def _iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _objects(raw):
    try:
        value = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _card_cover_objects(raw):
    """Labeled boxes for compact thumbs. Keeps the list card payload small."""
    cover = []
    for item in _objects(raw):
        label = item.get("label")
        box = item.get("box")
        if not label or not isinstance(box, dict) or not _display_confidence(item):
            continue
        try:
            coords = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in coords) or coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        entry = {
            "label": label,
            "box": {"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3]},
        }
        for key in (
            "confidence",
            "zones",
            "detection_frame_width",
            "detection_frame_height",
            "incident_eligible",
            "track_id",
            "snapshot_visible",
        ):
            if key in item:
                entry[key] = item[key]
        cover.append(entry)
    return cover


def _scene_pixels(item):
    return int(item.get("detection_frame_width") or 0) * int(
        item.get("detection_frame_height") or 0
    )


def _scene_observation_cover(representative, observations, *, recorded_pixels=None):
    """Return the canonical retained-observation cover, if it outranks the event."""
    if representative is None:
        return None
    available = [
        (row, json.loads(row["payload_json"]))
        for row in observations
        if row["snapshot_path"] and _display_confidence(json.loads(row["payload_json"]))
    ]
    if not available:
        return None
    current = _objects(representative["objects_json"])
    current_pixels = max((_scene_pixels(item) for item in current), default=0)
    visible = any(
        item.get("label") and item.get("snapshot_visible") is not False
        for item in current
    )
    if recorded_pixels is None:
        decoded = [json.loads(row["payload_json"]) for row in observations]
        recorded_pixels = max(
            (
                _scene_pixels(item)
                for item in decoded
                if item.get("frame_source") == "recorded_main"
            ),
            default=0,
        )
    cover, evidence = max(
        available,
        key=lambda pair: (_scene_pixels(pair[1]), float(pair[1].get("confidence") or 0)),
    )
    promoted_cover = bool(representative["snapshot_path"]) and recorded_pixels > _scene_pixels(evidence)
    if promoted_cover or (visible and _scene_pixels(evidence) <= current_pixels):
        return None
    return {
        "row": cover,
        "objects": [
            {**item, "snapshot_visible": True}
            for row, item in available
            if row["snapshot_path"] == cover["snapshot_path"]
        ],
    }


def _overlap(a, b):
    try:
        aa, bb = a["box"], b["box"]
        def normalized(box, item):
            w, h = float(item.get("detection_frame_width") or 1), float(item.get("detection_frame_height") or 1)
            return [float(box[k]) / (w if k.startswith("x") else h) for k in ("x1", "y1", "x2", "y2")]
        ax, ay, ar, ab = normalized(aa, a)
        bx, by, br, bb = normalized(bb, b)
        area = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
        return area / max(1e-10, (ar-ax)*(ab-ay)+(br-bx)*(bb-by)-area)
    except (KeyError, TypeError, ValueError):
        return 0.0


class EventStoreSceneMixin:
    _scene_outbox_legacy_rowid = 0

    def _init_scene_db(self):
        with self._lock, self._connect() as conn:
            conn.executescript("""
                create table if not exists scene_incidents (
                    id text primary key, revision integer not null default 0,
                    start_epoch real not null, end_epoch real not null,
                    state text not null default 'active', historical integer not null default 0
                );
                create index if not exists scene_incident_time on scene_incidents(start_epoch,end_epoch);
                create index if not exists scene_incident_end on scene_incidents(end_epoch,start_epoch);
                create table if not exists scene_episodes (
                    id text primary key, incident_id text not null references scene_incidents(id),
                    camera_id text not null, start_epoch real not null, end_epoch real not null,
                    last_activity_epoch real not null, coverage_json text not null,
                    boundary_locked integer not null default 0
                );
                create index if not exists scene_episode_camera on scene_episodes(camera_id,start_epoch,end_epoch);
                create index if not exists scene_episode_incident on scene_episodes(incident_id);
                create table if not exists scene_event_membership (
                    event_id integer primary key references events(id) on delete cascade,
                    episode_id text not null references scene_episodes(id)
                );
                create index if not exists scene_members_episode on scene_event_membership(episode_id);
                create table if not exists scene_objects (
                    id text primary key, episode_id text not null references scene_episodes(id),
                    label_override text, association_locked integer not null default 0,
                    facet_label text not null default ''
                );
                create index if not exists scene_object_episode on scene_objects(episode_id);
                create table if not exists scene_facet_zones (value text primary key);
                create table if not exists scene_facet_labels (value text primary key);
                create table if not exists scene_facet_cameras (value text primary key);
                create index if not exists scene_object_label_override on scene_objects(label_override);
                create table if not exists scene_track_aliases (
                    episode_id text not null,track_key text not null,object_id text not null,
                    primary key(episode_id,track_key)
                );
                create table if not exists scene_observations (
                    id text primary key, event_id integer not null references events(id) on delete cascade,
                    episode_id text not null references scene_episodes(id),
                    object_id text not null references scene_objects(id), camera_id text not null,
                    captured_epoch real not null, track_key text not null,
                    object_index integer, payload_json text not null,
                    snapshot_path text not null, recording_path text not null
                );
                create index if not exists scene_observation_object on scene_observations(object_id,captured_epoch);
                create index if not exists scene_observation_episode on scene_observations(episode_id,captured_epoch);
                create index if not exists scene_observation_event_time on scene_observations(event_id,captured_epoch);
                drop index if exists scene_observation_event;
                create index if not exists scene_observation_media on scene_observations(snapshot_path);
                create table if not exists scene_alert_decisions (
                    event_id integer primary key references events(id) on delete cascade,
                    payload_json text not null
                );
                create table if not exists scene_observation_decisions (
                    id text primary key,
                    observation_id text not null references scene_observations(id) on delete cascade,
                    source text not null, payload_json text not null
                );
                create index if not exists scene_decision_observation on scene_observation_decisions(observation_id);
                create table if not exists scene_alert_entries (
                    event_id integer not null references events(id) on delete cascade,
                    object_id text not null, outcome text not null, explicit integer not null,
                    first_epoch real not null, first_index integer not null,
                    first_observation text not null, first_decision text not null,
                    last_epoch real not null, last_index integer not null,
                    last_observation text not null, last_decision text not null,
                    payload_json text not null,
                    primary key(event_id,object_id,outcome)
                );
                create table if not exists scene_alert_state (
                    event_id integer primary key references events(id) on delete cascade
                );
                -- scene_alert_entries summarizes one event's decisions per subject
                -- and policy outcome while its scene_alert_state row exists. New
                -- decisions fold in here; anything that reorders, reassigns or
                -- removes evidence drops the state so the next refresh rebuilds.
                -- Positions order by (epoch, index, observation, decision rowid).
                -- A new rowid exceeds every existing one, so it only breaks ties
                -- toward the newest decision. Entries never store rowids because
                -- VACUUM may renumber them.
                create trigger if not exists scene_alert_fold after insert on scene_observation_decisions
                when exists(select 1 from scene_observations o join scene_alert_state s on s.event_id=o.event_id
                            where o.id=new.observation_id)
                begin
                    delete from scene_alert_entries where new.source='explicit' and explicit=0
                        and (event_id,object_id)=(select event_id,object_id from scene_observations where id=new.observation_id);
                    insert or ignore into scene_alert_entries
                        select o.event_id,o.object_id,json_remove(new.payload_json,'$.label'),new.source='explicit',
                               o.captured_epoch,coalesce(o.object_index,2147483647),o.id,new.id,
                               o.captured_epoch,coalesce(o.object_index,2147483647),o.id,new.id,new.payload_json
                        from scene_observations o where o.id=new.observation_id
                        and (new.source='explicit' or not exists(select 1 from scene_alert_entries e
                             where e.event_id=o.event_id and e.object_id=o.object_id and e.explicit=1));
                    update scene_alert_entries set first_epoch=o.captured_epoch,first_index=coalesce(o.object_index,2147483647),
                        first_observation=o.id,first_decision=new.id
                        from scene_observations o where o.id=new.observation_id
                        and scene_alert_entries.event_id=o.event_id and scene_alert_entries.object_id=o.object_id
                        and scene_alert_entries.outcome=json_remove(new.payload_json,'$.label')
                        and scene_alert_entries.explicit=(new.source='explicit')
                        and (o.captured_epoch,coalesce(o.object_index,2147483647),o.id)
                            < (scene_alert_entries.first_epoch,scene_alert_entries.first_index,scene_alert_entries.first_observation);
                    update scene_alert_entries set last_epoch=o.captured_epoch,last_index=coalesce(o.object_index,2147483647),
                        last_observation=o.id,last_decision=new.id,payload_json=new.payload_json
                        from scene_observations o where o.id=new.observation_id
                        and scene_alert_entries.event_id=o.event_id and scene_alert_entries.object_id=o.object_id
                        and scene_alert_entries.outcome=json_remove(new.payload_json,'$.label')
                        and scene_alert_entries.explicit=(new.source='explicit')
                        and (o.captured_epoch,coalesce(o.object_index,2147483647),o.id)
                            >= (scene_alert_entries.last_epoch,scene_alert_entries.last_index,scene_alert_entries.last_observation);
                end;
                create trigger if not exists scene_alert_decision_changed after update on scene_observation_decisions
                begin
                    delete from scene_alert_state where event_id in
                        (select event_id from scene_observations where id in (old.observation_id,new.observation_id));
                end;
                create trigger if not exists scene_alert_decision_removed after delete on scene_observation_decisions
                begin
                    delete from scene_alert_state where event_id in
                        (select event_id from scene_observations where id=old.observation_id);
                end;
                create trigger if not exists scene_alert_observation_moved
                after update of event_id,object_id,captured_epoch,object_index,id on scene_observations
                when old.event_id is not new.event_id or old.object_id is not new.object_id or old.id is not new.id
                    or old.captured_epoch is not new.captured_epoch or old.object_index is not new.object_index
                begin
                    delete from scene_alert_state where event_id in (old.event_id,new.event_id);
                end;
                create trigger if not exists scene_alert_observation_removed after delete on scene_observations
                begin
                    delete from scene_alert_state where event_id=old.event_id;
                end;
                create table if not exists scene_event_decision_backfill (
                    event_id integer primary key references events(id) on delete cascade
                );
                create table if not exists scene_aliases (alias text primary key, incident_id text not null);
                create table if not exists scene_corrections (
                    id integer primary key, incident_id text not null, revision integer not null,
                    payload_json text not null, created_at real not null
                );
                create table if not exists scene_notification_outbox (
                    incident_id text not null, revision integer not null, payload_json text not null,
                    primary key(incident_id,revision)
                );
                create table if not exists scene_changes (
                    id integer primary key autoincrement, incident_id text not null,
                    revision integer not null, created_at real not null
                );
                create table if not exists scene_discovery_baselines (
                    camera_id text primary key, captured_epoch real not null, objects_json text not null,
                    changed integer not null default 0, event_id integer
                );
                create table if not exists scene_identities (
                    event_id integer primary key references events(id) on delete cascade, payload_json text not null
                );
                create table if not exists scene_migrations (name text primary key,cursor_event_id integer not null);
                create table if not exists scene_expired_snapshots (snapshot_path text primary key);
                create table if not exists scene_snapshot_assets (
                    snapshot_path text primary key,camera_id text not null,created_at text not null,
                    snapshot_size_bytes integer not null default 0
                );
            """)
            self._init_scene_tracking_schema(conn)
            self._init_scene_admission_schema(conn)
            columns = {r[1] for r in conn.execute("pragma table_info(scene_observations)")}
            if "source_observation_id" not in columns:
                conn.execute("alter table scene_observations add column source_observation_id text references acquired_observations(id)")
            object_columns = {r[1] for r in conn.execute("pragma table_info(scene_objects)")}
            if "facet_label" not in object_columns:
                conn.execute("alter table scene_objects add column facet_label text not null default ''")
            conn.execute("create index if not exists scene_object_facet_label on scene_objects(facet_label)")
            conn.execute("create index if not exists scene_observation_source on scene_observations(source_observation_id)")
            self._scene_outbox_legacy_rowid = self._mark_legacy_scene_notifications(conn)
        # Bounded transactions, resumable and silent. Never infer historical
        # video or replay notifications during migration.
        while True:
            with self._lock, self._connect() as conn:
                rows = conn.execute("select e.* from events e where e.id in (select id from events "
                                    "except select event_id from scene_event_membership "
                                    "except select event_id from scene_event_establishment) "
                                    "order by e.created_at,e.id limit 200").fetchall()
                if not rows:
                    break
                for row in rows:
                    self._scene_ingest(conn, row, historical=True, notify=False)

        # Earlier partial imports did not read legacy track histories. Upgrade
        # mapped events as well, with a durable cursor and no alert publication.
        while True:
            with self._lock,self._connect() as conn:
                progress=conn.execute("select cursor_event_id from scene_migrations where name='legacy_tracks_v1'").fetchone()
                cursor=progress[0] if progress else 0
                rows=conn.execute("select e.* from events e join scene_event_membership m on m.event_id=e.id "
                                  "where e.id>? order by e.id limit 200",(cursor,)).fetchall()
                if not rows:
                    break
                for row in rows:
                    if legacy_track_observations(_objects(row["objects_json"])):
                        self._scene_ingest(conn,row,historical=True,notify=False,activity=False)
                conn.execute("insert into scene_migrations values('legacy_tracks_v1',?) on conflict(name) do update set cursor_event_id=excluded.cursor_event_id",(rows[-1]["id"],))
        self._backfill_scene_facets()

    def _backfill_scene_facets(self):
        """Build feed filters once. Later observations maintain them as they are stored."""
        with self._lock, self._connect() as conn:
            have_zones = conn.execute("select 1 from scene_migrations where name='scene_facets_v1'").fetchone()
            have_labels = conn.execute("select 1 from scene_migrations where name='scene_facet_labels_v1'").fetchone()
            have_cameras = conn.execute("select 1 from scene_migrations where name='scene_facet_cameras_v1'").fetchone()
            if have_zones and have_labels and have_cameras:
                return
            conn.execute("begin immediate")
            if not have_zones:
                conn.execute(
                    "update scene_objects set facet_label=coalesce(("
                    "select json_extract(o.payload_json,'$.label') from scene_observations o "
                    "where o.object_id=scene_objects.id order by o.captured_epoch,o.id limit 1),'') "
                    "where facet_label=''"
                )
                conn.execute(
                    "insert or ignore into scene_facet_zones(value) "
                    "select distinct j.value from scene_observations o, json_each("
                    "case when json_type(o.payload_json,'$.zones')='array' "
                    "then json_extract(o.payload_json,'$.zones') else '[]' end) j "
                    "where j.value!=''"
                )
                conn.execute("insert into scene_migrations values('scene_facets_v1',0)")
            if not have_labels:
                conn.execute(
                    "insert or ignore into scene_facet_labels(value) "
                    "select distinct coalesce(nullif(s.label_override,''), nullif(s.facet_label,'')) "
                    "from scene_objects s join scene_episodes p on p.id=s.episode_id "
                    "join scene_incidents i on i.id=p.incident_id "
                    "where i.state!='unconfirmed' and coalesce(nullif(s.label_override,''), nullif(s.facet_label,''))!=''"
                )
                conn.execute("insert into scene_migrations values('scene_facet_labels_v1',0)")
            if not have_cameras:
                conn.execute(
                    "insert or ignore into scene_facet_cameras(value) "
                    "select distinct p.camera_id from scene_episodes p "
                    "join scene_incidents i on i.id=p.incident_id "
                    "where i.state!='unconfirmed' and p.camera_id!=''"
                )
                conn.execute("insert into scene_migrations values('scene_facet_cameras_v1',0)")

    def _note_scene_camera(self, conn, camera_id):
        if isinstance(camera_id, str) and camera_id:
            conn.execute("insert or ignore into scene_facet_cameras(value) values(?)", (camera_id,))

    def _note_scene_label(self, conn, label):
        if isinstance(label, str) and label:
            conn.execute("insert or ignore into scene_facet_labels(value) values(?)", (label,))

    def _forget_unused_scene_label(self, conn, label):
        if not isinstance(label, str) or not label:
            return
        used = conn.execute(
            "select 1 from scene_objects s join scene_episodes p on p.id=s.episode_id "
            "join scene_incidents i on i.id=p.incident_id "
            "where s.label_override=? and i.state!='unconfirmed' limit 1",
            (label,),
        ).fetchone() or conn.execute(
            "select 1 from scene_objects s join scene_episodes p on p.id=s.episode_id "
            "join scene_incidents i on i.id=p.incident_id "
            "where s.facet_label=? and coalesce(s.label_override,'')='' and i.state!='unconfirmed' limit 1",
            (label,),
        ).fetchone()
        if not used:
            conn.execute("delete from scene_facet_labels where value=?", (label,))

    def _forget_unused_scene_camera(self, conn, camera_id):
        if not isinstance(camera_id, str) or not camera_id:
            return
        used = conn.execute(
            "select 1 from scene_episodes p join scene_incidents i on i.id=p.incident_id "
            "where p.camera_id=? and i.state!='unconfirmed' limit 1",
            (camera_id,),
        ).fetchone()
        if not used:
            conn.execute("delete from scene_facet_cameras where value=?", (camera_id,))

    def _scene_facet_values(self, conn, incident_id):
        cameras = [row[0] for row in conn.execute(
            "select distinct camera_id from scene_episodes where incident_id=? and camera_id!=''",
            (incident_id,),
        )]
        labels = [row[0] for row in conn.execute(
            "select distinct coalesce(nullif(s.label_override,''), nullif(s.facet_label,'')) "
            "from scene_objects s join scene_episodes p on p.id=s.episode_id "
            "where p.incident_id=? and coalesce(nullif(s.label_override,''), nullif(s.facet_label,''))!=''",
            (incident_id,),
        )]
        return cameras, labels

    def _release_scene_facets(self, conn, incident_id):
        cameras, labels = self._scene_facet_values(conn, incident_id)
        for camera_id in cameras:
            self._forget_unused_scene_camera(conn, camera_id)
        for label in labels:
            self._forget_unused_scene_label(conn, label)

    def _retain_scene_facets(self, conn, incident_id):
        cameras, labels = self._scene_facet_values(conn, incident_id)
        for camera_id in cameras:
            self._note_scene_camera(conn, camera_id)
        for label in labels:
            self._note_scene_label(conn, label)

    def _note_scene_zones(self, conn, item):
        zones = item.get("zones") if isinstance(item, dict) else None
        if not isinstance(zones, list):
            return
        conn.executemany(
            "insert or ignore into scene_facet_zones(value) values(?)",
            [(str(zone),) for zone in zones if isinstance(zone, str) and zone],
        )

    def _scene_resolve(self, conn, incident_id):
        seen = set()
        while incident_id and incident_id not in seen:
            seen.add(incident_id)
            alias = conn.execute("select incident_id from scene_aliases where alias=?", (incident_id,)).fetchone()
            if not alias:
                return incident_id
            incident_id = alias[0]
        return None

    @staticmethod
    def _scene_join_subjects(conn, retained, duplicate):
        if retained==duplicate:
            return True
        subjects=conn.execute("select id,association_locked,label_override from scene_objects where id in (?,?)",(retained,duplicate)).fetchall()
        if len(subjects)!=2 or any(s["association_locked"] or s["label_override"] for s in subjects):
            return False
        simultaneous=conn.execute(
            "select a.payload_json as left_payload,b.payload_json as right_payload "
            "from scene_observations a join scene_observations b on b.object_id=? "
            "and b.camera_id=a.camera_id and abs(b.captured_epoch-a.captured_epoch)<=0.01 "
            "where a.object_id=?",(duplicate,retained)).fetchall()
        if any(_overlap(json.loads(pair["left_payload"]),json.loads(pair["right_payload"]))<0.5
               for pair in simultaneous):
            # A tracker switching between simultaneous people must not turn
            # one shared frame into an association of their entire histories.
            return False
        conn.execute("update scene_observations set object_id=? where object_id=?",(retained,duplicate))
        conn.execute("update scene_track_aliases set object_id=? where object_id=?",(retained,duplicate))
        columns={r[1] for r in conn.execute("pragma table_info(appearance_embeddings)")}
        if "scene_object_id" in columns:
            conn.execute("update appearance_embeddings set scene_object_id=? where scene_object_id=?",(retained,duplicate))
        return True

    @staticmethod
    def _scene_remap_appearance(conn, object_ids):
        columns={r[1] for r in conn.execute("pragma table_info(appearance_embeddings)")}
        if not {"scene_object_id","observation_id"}<=columns:
            return
        for object_id in set(object_ids):
            conn.execute("update appearance_embeddings set scene_object_id=(select o.object_id from scene_observations o "
                         "where o.id=appearance_embeddings.observation_id) where scene_object_id=? "
                         "and exists(select 1 from scene_observations o where o.id=appearance_embeddings.observation_id)",(object_id,))

    @staticmethod
    def _scene_rebuild_corrected_aliases(conn, object_ids):
        # A separated track can now describe more than one subject. Drop that
        # ambiguous alias rather than allowing replay to undo the correction.
        for object_id in set(object_ids):
            conn.execute("delete from scene_track_aliases where object_id=?",(object_id,))
        for object_id in set(object_ids):
            keys=conn.execute("select distinct episode_id,track_key from scene_observations where object_id=? and track_key!=''",(object_id,)).fetchall()
            for key in keys:
                subjects=conn.execute("select distinct object_id from scene_observations where episode_id=? and track_key=?",
                                      (key["episode_id"],key["track_key"])).fetchall()
                if len(subjects)==1:
                    conn.execute("insert or replace into scene_track_aliases values(?,?,?)",(key["episode_id"],key["track_key"],subjects[0][0]))
                else:
                    conn.execute("delete from scene_track_aliases where episode_id=? and track_key=?",(key["episode_id"],key["track_key"]))

    def _scene_same_source_frame(self, row, item, *, event_id, captured):
        original=json.loads(row["payload_json"])
        if (row["event_id"]==event_id and abs(row["captured_epoch"]-captured)<=1e-6
                and "legacy" in {original.get("frame_source","legacy"),item.get("frame_source","legacy")}
                and _overlap(original,item)>=1-1e-9):
            # Legacy covers have no frame provenance. Only an exact same-event
            # timestamp and box bridge them to acquired observations.
            return True
        recording=portable_media_path(self.storage_dir,str(item.get("recording_path") or ""))
        return bool(recording and recording==row["recording_path"]
                    and original.get("frame_timestamp_exact") and item.get("frame_timestamp_exact"))

    def _scene_merge_shared_objects(self, conn, incident_id):
        # Overlapping episode windows may retain separate associations to one
        # acquired frame. Once the incidents merge, that shared frame supports
        # one subject unless an operator correction forbids the association.
        rows=conn.execute("select o.id,coalesce(json_extract(o.payload_json,'$.source_observation_id'),o.id) as source_id "
                          "from scene_observations o join scene_episodes p on p.id=o.episode_id where p.incident_id=? "
                          "order by o.captured_epoch,o.id",(incident_id,)).fetchall()
        seen={}
        for row in rows:
            object_id=conn.execute("select object_id from scene_observations where id=?",(row["id"],)).fetchone()[0]
            if row["source_id"] in seen:
                self._scene_join_subjects(conn,seen[row["source_id"]],object_id)
            else:
                seen[row["source_id"]]=object_id

    def _scene_record_decision(self, conn, observation_id, item, *, legacy=False):
        """Append policy interpretation without rewriting acquired evidence."""
        explicit = "alert_eligible" in item
        if not explicit and not (legacy and "incident_eligible" in item):
            return False
        decision = {"label": item["label"],
                    "eligible": bool(item["alert_eligible"] if explicit else item["incident_eligible"]),
                    "reasons": item.get("alert_reasons", item.get("incident_ineligible_reasons", [])),
                    "zones": item.get("zones", [])}
        payload = _json(decision)
        source = "explicit" if explicit else "legacy"
        decision_id = hashlib.sha256(_json([observation_id, source, decision]).encode()).hexdigest()
        return bool(conn.execute("insert or ignore into scene_observation_decisions values(?,?,?,?)",
                                 (decision_id, observation_id, source, payload)).rowcount)

    @staticmethod
    def _scene_candidates(row, objects=None):
        row = dict(row)
        at = _epoch(row["created_at"])
        objects = _objects(row.get("objects_json")) if objects is None else objects
        candidates = []
        presentation = []
        for index, item in enumerate(objects):
            if item.get("label"):
                presentation.append((index, dict(item)))
            if item.get("status") == "scene_observations":
                candidates.extend((None, dict(o)) for o in item.get("observations", []) if isinstance(o,dict) and o.get("label"))
            if item.get("status") == "object_tracking" and isinstance(item.get("object_tracking"),dict):
                candidates.extend((None, dict(o)) for o in item["object_tracking"].get("scene_observations",[]) if isinstance(o,dict) and o.get("label"))
        legacy_candidates = [(None,o) for o in legacy_track_observations(objects)]
        if not candidates:
            candidates=presentation
        else:
            # Raw acquisition owns model observations. A consensus label or
            # cover promotion is interpretation, never an extra detected object.
            # Attach a cover only to a matching source frame and geometry.
            for index, cover in presentation:
                if cover.get("snapshot_visible") is False:
                    continue
                cover_absolute=cover.get("captured_at_epoch",cover.get("snapshot_captured_at"))
                cover_at=_epoch(cover_absolute,at) if cover_absolute is not None else at+float(cover.get("temporal_sample_offset_seconds",0) or 0)
                matches=[]
                for position,(_, candidate) in enumerate(candidates):
                    if candidate.get("snapshot_visible") is False:
                        continue
                    absolute=candidate.get("captured_at_epoch",candidate.get("frame_captured_at_epoch",candidate.get("captured_at")))
                    candidate_at=_epoch(absolute,at) if absolute is not None else at+float(candidate.get("offset_seconds",0) or 0)
                    if abs(candidate_at-cover_at)<=0.05 and _overlap(candidate,cover)>=0.95:
                        matches.append(position)
                # Repeated copies of the same acquisition frame are one
                # match, not ambiguous evidence. Attach its image to each copy.
                matched_frames={(candidates[p][1].get("frame_source","legacy"),
                                 _json({k:float(v) for k,v in candidates[p][1].get("box",{}).items()}),
                                 candidates[p][1].get("captured_at_epoch",candidates[p][1].get("frame_captured_at_epoch",candidates[p][1].get("captured_at"))),
                                 candidates[p][1].get("offset_seconds",0),
                                 candidates[p][1].get("label")) for p in matches}
                if len(matched_frames)==1:
                    for position in matches:
                        candidates[position]=(index,{**candidates[position][1],"snapshot_path":row.get("snapshot_path") or ""})
        candidates.extend(legacy_candidates)
        return objects, candidates, presentation

    def _scene_refresh_alerts(self, conn, event_id):
        """Store the event's alert decisions; return whether they changed.

        Explicit decisions supersede legacy ones for the same subject. One
        alert remains per subject and policy outcome, positioned at its first
        observation and carrying its latest. Decision payloads are canonical
        JSON with exactly label/eligible/reasons/zones, so the payload without
        its label identifies the policy outcome.
        """
        if conn.execute("insert or ignore into scene_alert_state values(?)", (event_id,)).rowcount:
            self._scene_rebuild_alert_entries(conn, event_id)
        rows = conn.execute("select e.last_observation,e.object_id,e.payload_json from scene_alert_entries e "
                            "join scene_observation_decisions d on d.id=e.first_decision where e.event_id=? "
                            "order by e.first_epoch,e.first_index,e.first_observation,d.rowid", (event_id,)).fetchall()
        alerts = [{**json.loads(o["payload_json"]), "object_id":o["object_id"], "observation_id":o["last_observation"]}
                  for o in rows]
        alert_payload = _json({"event_id":event_id,"objects":alerts,"eligible":any(o["eligible"] for o in alerts)})
        old_alert = conn.execute("select payload_json from scene_alert_decisions where event_id=?",(event_id,)).fetchone()
        conn.execute("insert into scene_alert_decisions values(?,?) on conflict(event_id) do update set payload_json=excluded.payload_json",
                     (event_id,alert_payload))
        return old_alert is None or old_alert[0] != alert_payload

    @staticmethod
    def _scene_rebuild_alert_entries(conn, event_id):
        conn.execute("delete from scene_alert_entries where event_id=?", (event_id,))
        conn.execute("""
            insert into scene_alert_entries
            with policy as (
                select o.event_id,o.object_id,o.captured_epoch epoch,coalesce(o.object_index,2147483647) position,
                       o.id observation_id,d.id decision_id,d.rowid decision_rowid,d.source,d.payload_json,
                       json_remove(d.payload_json,'$.label') outcome
                from scene_observation_decisions d join scene_observations o on o.id=d.observation_id
                where o.event_id=?),
            kept as (
                select *,
                       row_number() over (partition by object_id,outcome
                           order by epoch,position,observation_id,decision_rowid) earliest,
                       row_number() over (partition by object_id,outcome
                           order by epoch desc,position desc,observation_id desc,decision_rowid desc) latest
                from policy
                where source='explicit' or object_id not in (select object_id from policy where source='explicit'))
            select f.event_id,f.object_id,f.outcome,f.source='explicit',
                   f.epoch,f.position,f.observation_id,f.decision_id,
                   l.epoch,l.position,l.observation_id,l.decision_id,l.payload_json
            from kept f join kept l on l.object_id=f.object_id and l.outcome=f.outcome and l.latest=1
            where f.earliest=1""", (event_id,))

    def _scene_project(self, conn, row, *, historical=False, notify=True, activity=True, force_revision=False, activity_epoch=None, target_episode_id=None, context_only=False, defer_alerts=False, objects=None, evidence_notify=True):
        row = dict(row)
        at = _epoch(row["created_at"])
        event_id, camera_id = int(row["id"]), str(row["camera_id"])
        semantic_media_before = int(conn.execute(
            "select count(*) from scene_observations where event_id=? and snapshot_path!=''",
            (event_id,),
        ).fetchone()[0])
        episode = conn.execute("select p.* from scene_episodes p join scene_event_membership m "
                               "on m.episode_id=p.id where m.event_id=?", (event_id,)).fetchone()
        new_event = episode is None
        if episode is None and target_episode_id is not None:
            episode = conn.execute("select * from scene_episodes where id=?", (target_episode_id,)).fetchone()
        if new_event:
            continued_incident = None
            if episode is None:
                episode = conn.execute("select p.* from scene_episodes p join scene_incidents i on i.id=p.incident_id "
                                   "where i.state!='unconfirmed' and p.camera_id=? and p.boundary_locked=0 "
                                   "and p.start_epoch<=? and p.last_activity_epoch>=? order by p.start_epoch desc limit 1",
                                   (camera_id, at+45, at-45)).fetchone()
                if episode is not None and at - float(episode["start_epoch"]) >= MAX_SCENE_EPISODE_SECONDS:
                    conn.execute("update scene_episodes set boundary_locked=1 where id=?", (episode["id"],))
                    continued_incident, episode = episode["incident_id"], None
            if episode is None:
                incident_id, episode_id = continued_incident or "incident-"+uuid.uuid4().hex, "episode-"+uuid.uuid4().hex
                if continued_incident is None:
                    conn.execute("insert into scene_incidents(id,start_epoch,end_epoch,historical,state) values(?,?,?,?,?)",
                                 (incident_id, at, at, int(historical), "complete" if historical else "active"))
                coverage = {"state": "historical" if historical else "incomplete", "analyzed_through": None,
                            "gaps": [], "reason": "legacy evidence only" if historical else "analysis pending"}
                conn.execute("insert into scene_episodes(id,incident_id,camera_id,start_epoch,end_epoch,last_activity_epoch,coverage_json) "
                             "values(?,?,?,?,?,?,?)", (episode_id,incident_id,camera_id,at,at,at,_json(coverage)))
                self._note_scene_camera(conn, camera_id)
                episode = conn.execute("select * from scene_episodes where id=?", (episode_id,)).fetchone()
            conn.execute("insert into scene_event_membership values(?,?)", (event_id,episode["id"]))
            conn.execute("insert or ignore into scene_aliases values(?,?)", (f"incident-{camera_id}-{event_id}", episode["incident_id"]))
        episode_id, incident_id = episode["id"], episode["incident_id"]
        objects, candidates, presentation = self._scene_candidates(row, objects)
        changed = new_event or force_revision
        cover_observations = {}
        earliest = latest = at
        active_at = float(activity_epoch) if activity_epoch is not None else float(episode["last_activity_epoch"])
        for index, item in candidates:
            try:
                confidence = float(item.get("confidence") or 0)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(confidence) or confidence <= 0:
                continue
            absolute = item.get("captured_at_epoch", item.get("frame_captured_at_epoch",item.get("captured_at", item.get("snapshot_captured_at"))))
            captured = _epoch(absolute, at)
            if absolute is None:
                captured = at + float(item.get("offset_seconds",item.get("temporal_sample_offset_seconds",0)) or 0)
            earliest = min(earliest, captured)
            latest = max(latest, captured)
            track = item.get("scene_track_key", item.get("track_id"))
            track_key = f"{event_id}:{track}" if track is not None else ""
            box = item.get("box")
            if isinstance(box,dict):
                box={k:float(v) for k,v in box.items()}
            observation_id = observation_identity(camera_id, captured, item)
            source_observation_id = observation_id
            owner=conn.execute("select episode_id from scene_observations where id=?",(observation_id,)).fetchone()
            if owner and owner[0] != episode_id:
                # The same acquired frame may be in overlapping episode
                # windows. Each episode needs its own association to it.
                observation_id="observation-"+hashlib.sha256(_json([source_observation_id,episode_id]).encode()).hexdigest()[:32]
            item["source_observation_id"] = source_observation_id
            if index is not None:
                cover_observations[index] = observation_id
            existing=conn.execute("select o.snapshot_path,o.object_id,s.association_locked,s.label_override "
                                  "from scene_observations o join scene_objects s on s.id=o.object_id where o.id=?", (observation_id,)).fetchone()
            if existing:
                changed = self._scene_record_decision(conn, observation_id, item, legacy=index is not None) or changed
                if track_key:
                    prior_track=conn.execute("select object_id from scene_track_aliases where episode_id=? and track_key=?",(episode_id,track_key)).fetchone()
                    if prior_track and prior_track[0]!=existing["object_id"]:
                        changed=self._scene_join_subjects(conn,existing["object_id"],prior_track[0]) or changed
                    if prior_track or not (existing["association_locked"] or existing["label_override"]):
                        conn.execute("insert or ignore into scene_track_aliases values(?,?,?)",(episode_id,track_key,existing["object_id"]))
                supporting=portable_media_path(self.storage_dir,str(item.get("snapshot_path") or (row.get("snapshot_path") if index is not None and item.get("snapshot_visible") is not False else "") or ""))
                if supporting and not existing["snapshot_path"] and not conn.execute("select 1 from scene_expired_snapshots where snapshot_path=?",(supporting,)).fetchone():
                    conn.execute("update scene_observations set snapshot_path=? where id=?",(supporting,observation_id))
                    conn.execute("insert or ignore into scene_snapshot_assets values(?,?,?,?)",(supporting,camera_id,_iso(captured),int(row.get("snapshot_size_bytes") or 0)))
                    changed=True
                continue
            previous = None
            if track_key:
                previous = conn.execute("select o.* from scene_observations o join scene_track_aliases a on a.object_id=o.object_id "
                                        "where a.episode_id=? and o.episode_id=a.episode_id and a.track_key=? order by o.captured_epoch desc limit 1",
                                        (episode_id,track_key)).fetchone()
            # The widened range lets the episode/time index seek; abs() stays exact.
            shared=conn.execute("select o.* from scene_observations o join scene_objects s on s.id=o.object_id "
                                "where o.episode_id=? and o.captured_epoch between ? and ? "
                                "and abs(o.captured_epoch-?)<=0.01 and s.association_locked=0 and s.label_override is null",
                                (episode_id,captured-0.02,captured+0.02,captured)).fetchall()
            shared=[o for o in shared if self._scene_same_source_frame(o,item,event_id=event_id,captured=captured)
                    and json.loads(o["payload_json"]).get("label")==item["label"] and _overlap(item,json.loads(o["payload_json"]))>=0.95]
            shared_ids={o["object_id"] for o in shared}
            if previous is not None:
                shared_ids.discard(previous["object_id"])
            if len(shared_ids)==1:
                retained=next(iter(shared_ids))
                if previous is None or self._scene_join_subjects(conn,retained,previous["object_id"]):
                    previous=next(o for o in shared if o["object_id"]==retained)
            if previous is None:
                window = 2 if activity else 45
                nearby = conn.execute("select o.* from scene_observations o join scene_objects s on s.id=o.object_id "
                                      "where o.episode_id=? and o.captured_epoch between ? and ? "
                                      "and abs(o.captured_epoch-?)<=? and s.association_locked=0 and s.label_override is null "
                                      "and o.captured_epoch=(select max(n.captured_epoch) from scene_observations n where n.object_id=o.object_id) "
                                      "order by o.captured_epoch desc limit 100",
                                      (episode_id,captured-window-0.01,captured+window+0.01,captured,window)).fetchall()
                matches = [o for o in nearby if json.loads(o["payload_json"]).get("label")==item["label"]
                           and abs(o["captured_epoch"]-captured)>0.01
                           and o["captured_epoch"]>float(item.get("association_after_epoch",float("-inf")))
                           and _overlap(item,json.loads(o["payload_json"])) >= 0.65]
                # Ambiguous association remains separate evidence.
                if len({o["object_id"] for o in matches}) == 1:
                    previous = matches[0]
            object_id = previous["object_id"] if previous else "object-"+uuid.uuid4().hex
            if previous is None:
                label = str(item.get("label") or "")
                conn.execute("insert into scene_objects(id,episode_id,facet_label) values(?,?,?)",
                             (object_id,episode_id,label))
                self._note_scene_label(conn, label)
            if track_key:
                conn.execute("insert or ignore into scene_track_aliases values(?,?,?)",(episode_id,track_key,object_id))
            explicit_snapshot = str(item.get("snapshot_path") or "")
            snapshot = explicit_snapshot or (str(row.get("snapshot_path") or "") if index is not None and item.get("snapshot_visible") is not False else "")
            snapshot = portable_media_path(self.storage_dir,snapshot)
            if snapshot and conn.execute("select 1 from scene_expired_snapshots where snapshot_path=?",(snapshot,)).fetchone():
                snapshot = ""
            if snapshot:
                conn.execute("insert into scene_snapshot_assets values(?,?,?,?) on conflict(snapshot_path) do update "
                             "set created_at=max(created_at,excluded.created_at),snapshot_size_bytes=max(snapshot_size_bytes,excluded.snapshot_size_bytes)",
                             (snapshot,camera_id,_iso(captured),int(row.get("snapshot_size_bytes") or 0)))
            recording = portable_media_path(self.storage_dir,str(item.get("recording_path") or row.get("recording_path") or ""))
            item.pop("snapshot_path",None)
            item.pop("recording_path",None)
            conn.execute("insert into scene_observations(id,event_id,episode_id,object_id,camera_id,captured_epoch,track_key,object_index,payload_json,snapshot_path,recording_path,source_observation_id) values(?,?,?,?,?,?,?,?,?,?,?,?)",
                         (observation_id,event_id,episode_id,object_id,camera_id,captured,track_key,index,_json(item),snapshot,recording,source_observation_id))
            self._note_scene_zones(conn, item)
            self._scene_record_decision(conn, observation_id, item, legacy=index is not None)
            changed = True
        for index, cover in presentation:
            supporting_id = cover_observations.get(index)
            if supporting_id is None:
                absolute = cover.get("captured_at_epoch", cover.get("snapshot_captured_at"))
                captured = _epoch(absolute,at) if absolute is not None else at+float(cover.get("temporal_sample_offset_seconds",0) or 0)
                matches = conn.execute("select id,payload_json from scene_observations where event_id=? "
                                       "and captured_epoch between ? and ? and abs(captured_epoch-?)<=0.05",
                                       (event_id,captured-0.06,captured+0.06,captured)).fetchall()
                matches = [o for o in matches if _overlap(cover,json.loads(o["payload_json"]))>=0.95]
                if len(matches)==1:
                    supporting_id=matches[0]["id"]
            if supporting_id is not None:
                changed = self._scene_record_decision(conn, supporting_id, cover, legacy=True) or changed
        # Historical observations may predate the decision table. Only explicit
        # policy annotations qualify; raw confidence/zone metadata is not policy.
        # Later observations record their own decision when inserted above.
        if conn.execute("insert or ignore into scene_event_decision_backfill values(?)", (event_id,)).rowcount:
            for observed in conn.execute("select id,payload_json from scene_observations where event_id=?", (event_id,)):
                self._scene_record_decision(conn, observed["id"], json.loads(observed["payload_json"]))
        # A deferring caller already forced this revision and refreshes alerts
        # once after its context projection, which recomputes them anyway.
        if not (defer_alerts and changed):
            changed = self._scene_refresh_alerts(conn, event_id) or changed
        tracking = next((o.get("object_tracking") for o in objects if o.get("status")=="object_tracking"),None)
        coverage = json.loads(episode["coverage_json"])
        if isinstance(tracking,dict):
            coverage = {"state":"sampled" if tracking.get("state") in {"complete","completed"} else "incomplete",
                        "analyzed_through":tracking.get("analyzed_through"), "gaps":list(tracking.get("coverage_gaps",[])),
                        "reason":str(tracking.get("state") or "analysis pending")}
            if isinstance(tracking.get("scene_analysis_job"),dict):
                job=conn.execute("select state,last_error,coverage_gaps_json from scene_analysis_jobs where episode_id=?",(episode_id,)).fetchone()
                if job:
                    for gap in json.loads(job["coverage_gaps_json"]):
                        if gap not in coverage["gaps"]:
                            coverage["gaps"].append(gap)
                if coverage["gaps"]:
                    coverage["gaps"].sort(key=lambda gap:gap.get("start_epoch",0) if isinstance(gap,dict) else 0)
                    coverage["state"]="incomplete"
                if job and job["state"]!="complete":
                    coverage["state"]="incomplete"
                    coverage["reason"]=job["last_error"] or "recorded analysis queued"
        elif candidates and not historical and not context_only:
            coverage.update(analyzed_through=_iso(latest), reason="sampled detector frames; continuous coverage not established")
        coverage_changed = _json(coverage) != episode["coverage_json"]
        conn.execute("update scene_episodes set start_epoch=min(start_epoch,?),end_epoch=max(end_epoch,?), "
                     "last_activity_epoch=max(last_activity_epoch,?),coverage_json=? where id=?",
                     (earliest,latest,active_at if activity else episode["last_activity_epoch"],_json(coverage),episode_id))
        conn.execute("update scene_incidents set start_epoch=min(start_epoch,?),end_epoch=max(end_epoch,?) where id=?",(earliest,latest,incident_id))
        if activity and not historical and active_at>=time.time()-45:
            conn.execute("update scene_incidents set state='active',historical=0 where id=?",(incident_id,))
        if changed or coverage_changed:
            self._scene_changed(conn,incident_id,notify=notify)
        semantic_media_after = int(conn.execute(
            "select count(*) from scene_observations where event_id=? and snapshot_path!=''",
            (event_id,),
        ).fetchone()[0])
        if not historical and semantic_media_after > semantic_media_before:
            conn.execute(
                "update events set scene_media_revision=scene_media_revision+1 where id=?",
                (event_id,),
            )
            current = conn.execute("select * from events where id=?", (event_id,)).fetchone()
            if current is not None:
                self._evidence_outbox(
                    conn, current, "evidence_updated", reason="scene_observation_added", notify=evidence_notify
                )
        return incident_id

    def _mark_legacy_scene_notifications(self, conn):
        """Retire per-revision payload rows from before outbox coalescing.

        Legacy rows embedded a full snapshot per revision. They are purged in
        small batches after startup; unfinished incidents are re-queued as one
        pending marker each so their latest state is still published.
        """
        row = conn.execute(
            "select cursor_event_id from scene_migrations where name='notification_outbox_coalesce_v1'"
        ).fetchone()
        if row is not None:
            return int(row[0])
        cutoff = int(conn.execute("select coalesce(max(rowid),0) from scene_notification_outbox").fetchone()[0])
        if cutoff:
            pending = conn.execute(
                "select id, revision from scene_incidents where state not in ('complete','merged','unconfirmed') "
                "and id in (select distinct incident_id from scene_notification_outbox)"
            ).fetchall()
            for incident in pending:
                conn.execute(
                    "delete from scene_notification_outbox where incident_id=? and revision=?",
                    (incident["id"], incident["revision"]),
                )
                conn.execute(
                    "insert into scene_notification_outbox values(?,?,'')",
                    (incident["id"], incident["revision"]),
                )
        conn.execute(
            "insert into scene_migrations values('notification_outbox_coalesce_v1',?)", (cutoff,),
        )
        return cutoff

    def purge_legacy_scene_notifications(self, limit=10):
        """Delete a bounded batch of retired outbox rows; returns rows removed."""
        if not self._scene_outbox_legacy_rowid:
            return 0
        with self._lock, self._connect() as conn:
            removed = conn.execute(
                "delete from scene_notification_outbox where rowid in (select rowid from "
                "scene_notification_outbox where rowid <= ? and payload_json != '' limit ?)",
                (self._scene_outbox_legacy_rowid, max(1, int(limit))),
            ).rowcount
        if not removed:
            self._scene_outbox_legacy_rowid = 0
        return max(0, int(removed))

    def _scene_changed(self,conn,incident_id,*,notify=True):
        incident_id=self._scene_resolve(conn,incident_id)
        conn.execute("update scene_incidents set revision=revision+1 where id=?",(incident_id,))
        incident = conn.execute("select * from scene_incidents where id=?",(incident_id,)).fetchone()
        conn.execute("insert into scene_changes(incident_id,revision,created_at) values(?,?,?)",(incident_id,incident["revision"],time.time()))
        self._scene_facet_cache = {}
        if notify and incident["state"] != "unconfirmed":
            # One pending marker per incident. Publication builds the newest
            # snapshot outside this commit, so superseded revisions coalesce.
            conn.execute(
                "delete from scene_notification_outbox where incident_id=? and payload_json=''",
                (incident_id,),
            )
            conn.execute("insert or ignore into scene_notification_outbox values(?,?,'')",(incident_id,incident["revision"]))

    def _scene_notification_payload(self, conn, incident_id):
        payload = self._scene_payload(conn, incident_id)
        if payload is None:
            return None
        # Notifications carry the entire roster, with representative
        # evidence per object. The canonical detail owns per-frame history.
        for subject in payload.get("scene_objects",[]):
            evidence=subject["observations"]
            subject["observations"]=[max(evidence,key=lambda o:(o.get("snapshot_available",False),o.get("confidence",0)))] if evidence else []
        return payload

    def _scene_payload(self, conn, incident_id):
        from ..incident_presenter import _event_row, _incident_row
        incident_id = self._scene_resolve(conn,incident_id)
        incident = conn.execute("select * from scene_incidents where id=?",(incident_id,)).fetchone()
        if not incident:
            return None
        episodes = conn.execute("select * from scene_episodes where incident_id=? order by start_epoch,id",(incident_id,)).fetchall()
        rows = conn.execute("select e.* from events e join scene_event_membership m on m.event_id=e.id "
                            "join scene_episodes p on p.id=m.episode_id where p.incident_id=? order by e.created_at,e.id",(incident_id,)).fetchall()
        if not rows:
            return None
        payload = _incident_row(rows[0]["camera_id"],[_event_row(dict(r)) for r in rows])
        observations = conn.execute("select o.*,s.label_override,s.association_locked from scene_observations o join scene_objects s on s.id=o.object_id "
                                    "join scene_episodes p on p.id=o.episode_id where p.incident_id=? order by o.captured_epoch,o.id",(incident_id,)).fetchall()
        observations = [row for row in observations if _display_confidence(json.loads(row["payload_json"]))]
        grouped = {}
        for row in observations:
            item = json.loads(row["payload_json"])
            item.update(id=row["id"],event_id=row["event_id"],camera_id=row["camera_id"],captured_at=_iso(row["captured_epoch"]),
                        object_index=row["object_index"],snapshot_available=bool(row["snapshot_path"]),recording_available=bool(row["recording_path"]))
            if not item.get("image"):
                item["image"] = {"width":item.get("detection_frame_width"), "height":item.get("detection_frame_height"),
                                 "source":item.get("frame_source", "legacy"), "captured_at":item["captured_at"],
                                 "analyzed_frame":item.get("frame_source") in {"recorded_main","live_discovery","live_fast_path","tracking"}}
            subject = grouped.setdefault(row["object_id"],{"id":row["object_id"],"observations":[],"label":row["label_override"] or item["label"],
                                                         "first_seen_at":item["captured_at"],"confidence":0.0,"certainty":"possible",
                                                         "association_corrected":bool(row["association_locked"]),"continuity_uncertain":False})
            subject["observations"].append(item)
            subject["last_seen_at"] = item["captured_at"]
            if float(item.get("confidence") or 0) > subject["confidence"]:
                subject.update(confidence=float(item["confidence"]),source_event_id=row["event_id"],object_index=row["object_index"])
            if row["label_override"] or item.get("historical_track_confirmed") or (item.get("confidence_provenance") != "track_summary" and float(item.get("confidence") or 0)>=float(item.get("scene_confirmation_threshold", .45)) and int(item.get("temporal_observations") or item.get("track_observations") or 1)>=2):
                subject["certainty"] = "observed"
        for subject in grouped.values():
            subject["observation_count"]=len(subject["observations"])
            strong_times = {o["captured_at"] for o in subject["observations"] if o.get("confidence_provenance") != "track_summary" and float(o.get("confidence") or 0)>=float(o.get("scene_confirmation_threshold", .45))}
            if len(strong_times)>=2:
                subject["certainty"]="observed"
        scene_objects = list(grouped.values())
        # Distinct supported tracks need not mean distinct physical people.
        # Nonconcurrent sightings with unresolved identity remain accessible,
        # without turning them into a confident count of unique subjects.
        observed_frames = {s["id"]:{(o["camera_id"],round(_epoch(o["captured_at"]),2)) for o in s["observations"]} for s in scene_objects}
        for index, subject in enumerate(scene_objects):
            for other in scene_objects[index+1:]:
                if (subject["label"]==other["label"]
                        and not (subject["association_corrected"] and other["association_corrected"])
                        and not observed_frames[subject["id"]] & observed_frames[other["id"]]):
                    subject["continuity_uncertain"] = other["continuity_uncertain"] = True
        labels = sorted({s["label"] for s in scene_objects})
        zones = sorted({str(z) for s in scene_objects for o in s["observations"] for z in o.get("zones",[])})
        episode_payloads = [{"id":p["id"],"camera_id":p["camera_id"],"start_at":_iso(p["start_epoch"]),"end_at":_iso(p["end_epoch"]),
                             "event_ids":[r[0] for r in conn.execute("select event_id from scene_event_membership where episode_id=? order by event_id",(p["id"],))],
                             "last_activity_at":_iso(p["last_activity_epoch"]),"coverage":json.loads(p["coverage_json"])} for p in episodes]
        activity = []
        supported_positions = {key for row in conn.execute(
            "select d.evidence_json from scene_activity_decisions d join scene_activity_admissions a on a.decision_id=d.id "
            "join scene_episodes p on p.id=a.episode_id where p.incident_id=?", (incident_id,))
            for key in json.loads(row[0]).get("supporting_observation_ids", [])}
        for subject in scene_objects:
            per_camera = {}
            for observation in subject["observations"]:
                per_camera.setdefault(observation["camera_id"], []).append(observation)
            for camera_id, evidence in per_camera.items():
                def note(kind, observation, **extra):
                    activity.append({"kind":kind,"object_id":subject["id"],"label":subject["label"],
                                     "camera_id":camera_id,"captured_at":observation["captured_at"],
                                     "observation_id":observation["id"],**extra})
                note("appeared", evidence[0])
                for previous, current in zip(evidence, evidence[1:]):
                    physical = current.get("source_observation_id", current["id"]) in supported_positions
                    if physical and previous.get("box") and current.get("box") and _overlap(previous,current)<.7:
                        note("changed_position", current)
                    old_zones, new_zones = sorted(previous.get("zones",[])), sorted(current.get("zones",[]))
                    if physical and old_zones != new_zones:
                        note("zone_changed", current, from_zones=old_zones, to_zones=new_zones)
                if len(evidence)>1:
                    note("last_seen", evidence[-1])
        activity.sort(key=lambda item:(item["captured_at"],item["object_id"],item["kind"]))
        cameras = list(dict.fromkeys(p["camera_id"] for p in episodes))
        uncertain_labels = {s["label"] for s in scene_objects if s["continuity_uncertain"]}
        coverage = {"state":"historical" if incident["historical"] else ("incomplete" if any(p["coverage"]["state"]!="sampled" for p in episode_payloads) else "sampled"),
                    "analyzed_through":max((p["coverage"].get("analyzed_through") or "" for p in episode_payloads),default="") or None,
                    "gaps":[g for p in episode_payloads for g in p["coverage"].get("gaps",[])]}
        alerts = [json.loads(r[0]) for r in conn.execute("select a.payload_json from scene_alert_decisions a join scene_event_membership m on m.event_id=a.event_id "
                                                      "join scene_episodes p on p.id=m.episode_id where p.incident_id=?",(incident_id,))]
        current_subjects = {o["id"]:o["object_id"] for o in observations}
        for event_decision in alerts:
            for decision in event_decision.get("objects",[]):
                if decision.get("observation_id") in current_subjects:
                    decision["object_id"] = current_subjects[decision["observation_id"]]
        identities=[identity for r in conn.execute("select s.payload_json from scene_identities s join scene_event_membership m on m.event_id=s.event_id "
                                                  "join scene_episodes p on p.id=m.episode_id where p.incident_id=?",(incident_id,)) for identity in json.loads(r[0])]
        payload["establishment"] = self._scene_establishment(conn, incident_id)
        payload.update(id=incident_id,incident_id=incident_id,revision=incident["revision"],schema_version=3,state=incident["state"],historical=bool(incident["historical"]),
                       start_at=_iso(incident["start_epoch"]),end_at=_iso(incident["end_epoch"]),start_epoch=incident["start_epoch"],last_epoch=incident["end_epoch"],
                       duration_seconds=incident["end_epoch"]-incident["start_epoch"],event_ids=[r["id"] for r in rows],
                       camera_ids=cameras,scene_objects=scene_objects,episodes=episode_payloads,labels=labels,zones=zones,has_objects=bool(scene_objects),
                       summary=_incident_display_summary(scene_objects),
                       continuity_uncertain=bool(uncertain_labels),activity=activity,coverage=coverage,alert_decisions=alerts,identities=identities)
        representative_event = next(
            (event for event in rows if int(event["id"]) == int(payload.get("representative_event_id") or 0)),
            None,
        )
        if representative_event is not None:
            payload["camera_id"] = representative_event["camera_id"]
        if incident["state"] == "unconfirmed":
            payload["summary"] = "Activity was not established from these observations."
        # Media quality is independent of alert admission. Retain the exact
        # detector images and expose a larger analyzed image as separate evidence.
        available = [(row,json.loads(row["payload_json"])) for row in observations if row["snapshot_path"]]
        if available:
            representative = next(
                (row for row in rows if int(row["id"]) == int(payload.get("representative_event_id") or 0)),
                rows[-1],
            )
            selected = _scene_observation_cover(representative, observations)
            if selected is not None:
                cover = selected["row"]
                cover_event = next(row for row in rows if row["id"] == cover["event_id"])
                payload.update(
                    representative_event_id=cover["event_id"], camera_id=cover["camera_id"],
                    created_at=cover_event["created_at"], snapshot_path="available",
                    snapshot_observation_id=cover["id"], snapshot_captured_at=_iso(cover["captured_epoch"]),
                    snapshot_url=f"/api/incidents/observations/{cover['id']}/snapshot",
                    objects=selected["objects"],
                    object_tracking=None,
                )
            main_images = [(row,item) for row,item in available if item.get("frame_source") == "recorded_main"]
            if main_images:
                review, image_item = max(main_images,key=lambda pair:_scene_pixels(pair[1]))
                for subject in scene_objects:
                    for observation in subject["observations"]:
                        if observation["camera_id"] == review["camera_id"] and _scene_pixels(observation) < _scene_pixels(image_item):
                            observation["review_image"] = {"url":f"/api/incidents/observations/{review['id']}/snapshot",
                                "width":image_item.get("detection_frame_width"), "height":image_item.get("detection_frame_height"),
                                "source":"recorded_main", "captured_at":_iso(review["captured_epoch"]), "analyzed_frame":True}
        # The public incident includes event-frame boxes as well as its scene
        # roster. Apply the same confidence rule to both, without rewriting DB rows.
        for event in [payload, *payload.get("events", [])]:
            event["objects"] = [item for item in event.get("objects", []) if _display_confidence(item)]
            if "labels" in event and event is not payload:
                event["labels"] = sorted({item["label"] for item in event["objects"] if item.get("label")})
        visible_ids = {row["id"] for row in observations}
        for decision in payload.get("alert_decisions", []):
            decision["objects"] = [item for item in decision.get("objects", [])
                                   if item.get("observation_id") in visible_ids]
        return payload

    def scene_episode_for_event(self, event_id, episode_id):
        with self._connect() as conn:
            row = conn.execute("select p.* from scene_episodes p join scene_event_membership m on m.episode_id=p.id "
                               "where p.id=? and m.event_id=?", (episode_id, event_id)).fetchone()
            return dict(row) if row else None

    def scene_incident(self,incident_id=None,*,event_id=None):
        with self._connect() as conn:
            if event_id is not None:
                row=conn.execute("select p.incident_id from scene_episodes p join scene_event_membership m on m.episode_id=p.id where m.event_id=?",(event_id,)).fetchone()
                if not row:
                    return None
                incident_id=row[0]
            return self._scene_payload(conn,incident_id)

    def scene_incident_metadata(self, event_ids):
        """Return canonical incident metadata for events without expanding scenes."""
        normalized = sorted({int(event_id) for event_id in event_ids if int(event_id) > 0})
        if not normalized:
            return {}
        result = {}
        with self._connect() as conn:
            for offset in range(0, len(normalized), 500):
                chunk = normalized[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    "select m.event_id,p.incident_id from scene_event_membership m "
                    "join scene_episodes p on p.id=m.episode_id "
                    f"where m.event_id in ({placeholders})",
                    chunk,
                ).fetchall()
                incident_ids = sorted({str(row["incident_id"]) for row in rows})
                decisions = {}
                if incident_ids:
                    incident_placeholders = ",".join("?" for _ in incident_ids)
                    decision_rows = conn.execute(
                        "select p.incident_id,d.* from scene_activity_decisions d "
                        "join scene_event_establishment x on x.decision_id=d.id "
                        "join scene_event_membership m on m.event_id=x.event_id "
                        "join scene_episodes p on p.id=m.episode_id "
                        f"where p.incident_id in ({incident_placeholders}) "
                        "order by p.incident_id,(d.verdict='supported') desc,d.created_at",
                        incident_ids,
                    ).fetchall()
                    for decision in decision_rows:
                        decisions.setdefault(str(decision["incident_id"]), decision)
                for row in rows:
                    incident_id = str(row["incident_id"])
                    result[int(row["event_id"])] = {
                        "incident_id": incident_id,
                        "establishment": self._scene_establishment_payload(
                            decisions.get(incident_id)
                        ),
                    }
        return result

    @staticmethod
    def _scene_query_where(*, start_epoch=None, end_epoch=None, camera_id="", event_type="all", object_label="", zone=""):
        clauses=["i.state != 'unconfirmed'", "exists(select 1 from scene_episodes p join scene_event_membership m on m.episode_id=p.id where p.incident_id=i.id)"]
        args=[]
        if start_epoch is not None:
            clauses.append("i.end_epoch>=?"); args.append(start_epoch)
        if end_epoch is not None:
            clauses.append("i.start_epoch<?"); args.append(end_epoch)
        if camera_id:
            clauses.append("exists(select 1 from scene_episodes p where p.incident_id=i.id and p.camera_id=?)");args.append(camera_id)
        observed="select 1 from scene_observations o join scene_episodes p on p.id=o.episode_id where p.incident_id=i.id and " + _DISPLAY_CONFIDENCE_SQL
        if event_type in {"object","motion"}:
            clauses.append(("not " if event_type=="motion" else "")+"exists("+observed+")")
        if object_label:
            clauses.append("exists(select 1 from scene_observations o join scene_objects s on s.id=o.object_id "
                           "join scene_episodes p on p.id=o.episode_id where p.incident_id=i.id and "+_DISPLAY_CONFIDENCE_SQL+" and "+EventStoreSceneMixin._scene_label_sql()+"=?)")
            args.append(object_label)
        if zone:
            clauses.append("exists(select 1 from scene_observations o join scene_episodes p on p.id=o.episode_id "
                           "join json_each(coalesce(json_extract(o.payload_json,'$.zones'),'[]')) z "
                           "where p.incident_id=i.id and z.value=?)")
            args.append(zone)
        return " and ".join(clauses),args

    @staticmethod
    def _scene_label_sql():
        return ("coalesce(s.label_override,(select json_extract(first.payload_json,'$.label') from scene_observations first "
                "where first.object_id=o.object_id order by first.captured_epoch,first.id limit 1))")

    def scene_incident_facets(self, *, start_epoch=None, end_epoch=None):
        key = (
            None if start_epoch is None else float(start_epoch),
            None if end_epoch is None else float(end_epoch),
        )
        cache = getattr(self, "_scene_facet_cache", None)
        if isinstance(cache, dict) and key in cache:
            return {name: list(values) for name, values in cache[key].items()}
        result = self._query_scene_incident_facets(start_epoch=start_epoch, end_epoch=end_epoch)
        if not isinstance(cache, dict):
            cache = {}
            self._scene_facet_cache = cache
        if len(cache) > 16:
            cache.clear()
        cache[key] = result
        return {name: list(values) for name, values in result.items()}

    def _query_scene_incident_facets(self, *, start_epoch=None, end_epoch=None):
        where,args=self._scene_query_where(start_epoch=start_epoch,end_epoch=end_epoch)
        with self._connect() as conn:
            if start_epoch is not None or end_epoch is not None:
                # Drive from the day's incidents. A scan of scene_objects, or of
                # observation JSON, reads the whole history before the date applies.
                scope=("with scoped as materialized (select i.id from scene_incidents i where "+where+") , "
                       "episodes as materialized (select p.id,p.camera_id from scoped cross join scene_episodes p on p.incident_id=scoped.id) ")
                cameras=[r[0] for r in conn.execute(scope+"select distinct camera_id from episodes order by camera_id",args)]
                labels=[r[0] for r in conn.execute(
                    scope+"select distinct coalesce(nullif(s.label_override,''), nullif(s.facet_label,'')) "
                    "from episodes p cross join scene_objects s on s.episode_id=p.id "
                    "where coalesce(nullif(s.label_override,''), nullif(s.facet_label,''))!='' order by 1", args) if r[0]]
                zones=[r[0] for r in conn.execute(scope+"select distinct z.value from episodes p cross join scene_observations o on o.episode_id=p.id "
                        "join json_each(coalesce(json_extract(o.payload_json,'$.zones'),'[]')) z order by z.value",args) if r[0]]
                return {"camera_ids":cameras,"labels":labels,"zones":zones}
            cameras=[r[0] for r in conn.execute("select value from scene_facet_cameras order by value") if r[0]]
            # The live feed asks for every incident. Display labels live on the
            # object, and zone names are inserted as observations arrive, so this
            # does not walk historical observation payloads.
            labels=[r[0] for r in conn.execute("select value from scene_facet_labels order by value") if r[0]]
            zones=[r[0] for r in conn.execute("select value from scene_facet_zones order by value") if r[0]]
        return {"camera_ids":cameras,"labels":labels,"zones":zones}

    def count_scene_incidents(self, **filters):
        where,args=self._scene_query_where(**filters)
        with self._connect() as conn:
            return conn.execute("select count(*) from scene_incidents i where "+where,args).fetchone()[0]

    def list_scene_incidents(self,*,start_epoch=None,end_epoch=None,start_at=None,end_at=None,camera_id="",limit=200,offset=0,
                             event_type="all",object_label="",zone="",summary_only=False):
        if start_at is not None:
            start_epoch=_epoch(start_at)
        if end_at is not None:
            end_epoch=_epoch(end_at)
        where,args=self._scene_query_where(start_epoch=start_epoch,end_epoch=end_epoch,camera_id=camera_id,
                                          event_type=event_type,object_label=object_label,zone=zone)
        with self._connect() as conn:
            ids=conn.execute("select i.id from scene_incidents i where "+where+" order by i.start_epoch desc,i.id limit ? offset ?",
                             (*args,max(1,min(int(limit),1000)),max(0,int(offset)))).fetchall()
            if summary_only:
                return [{"id":row[0],"events":[dict(r) for r in conn.execute(
                    "select e.id,e.camera_id,e.created_at from events e join scene_event_membership m on m.event_id=e.id "
                    "join scene_episodes p on p.id=m.episode_id where p.incident_id=?",(row[0],))]} for row in ids]
            return [self._scene_payload(conn,row[0]) for row in ids]

    def list_scene_incident_notifications(self,limit=200):
        """Recent incidents in the notification form that lifecycle events carry."""
        where,args=self._scene_query_where()
        with self._connect() as conn:
            ids=conn.execute("select i.id from scene_incidents i where "+where+" order by i.start_epoch desc,i.id limit ?",
                             (*args,max(1,min(int(limit),1000)))).fetchall()
            payloads=(self._scene_notification_payload(conn,row[0]) for row in ids)
            return [payload for payload in payloads if payload is not None]

    def list_scene_incident_cards(self,*,start_epoch=None,end_epoch=None,start_at=None,end_at=None,camera_id="",limit=200,offset=0,
                                  event_type="all",object_label="",zone=""):
        """Page rows for the incident rail: cover, labels, and time, without observation history."""
        from ..incident_presenter import _best_incident_event, _event_row
        if start_at is not None:
            start_epoch=_epoch(start_at)
        if end_at is not None:
            end_epoch=_epoch(end_at)
        where,args=self._scene_query_where(start_epoch=start_epoch,end_epoch=end_epoch,camera_id=camera_id,
                                          event_type=event_type,object_label=object_label,zone=zone)
        with self._connect() as conn:
            ids=[row[0] for row in conn.execute(
                "select i.id from scene_incidents i where "+where+" order by i.start_epoch desc,i.id limit ? offset ?",
                (*args,max(1,min(int(limit),1000)),max(0,int(offset))))]
            if not ids:
                return []
            placeholders=",".join("?"*len(ids))
            incidents={row["id"]:row for row in conn.execute(
                f"select id,revision,start_epoch,end_epoch from scene_incidents where id in ({placeholders})", ids)}
            episode_rows=conn.execute(
                f"select incident_id,camera_id from scene_episodes where incident_id in ({placeholders}) order by start_epoch,id",
                ids).fetchall()
            label_rows=conn.execute(
                "select p.incident_id, coalesce(nullif(s.label_override,''), nullif(s.facet_label,'')) label "
                "from scene_objects s join scene_episodes p on p.id=s.episode_id "
                f"where p.incident_id in ({placeholders}) and exists(select 1 from scene_observations o "
                "where o.object_id=s.id and " + _DISPLAY_CONFIDENCE_SQL + ")", ids).fetchall()
            event_rows=conn.execute(
                "select p.incident_id,e.id,e.camera_id,e.kind,e.topic,e.created_at,e.snapshot_path,e.evidence_revision,e.objects_json "
                "from events e join scene_event_membership m on m.event_id=e.id "
                "join scene_episodes p on p.id=m.episode_id "
                f"where p.incident_id in ({placeholders}) order by e.created_at,e.id", ids).fetchall()
            cover_rows=conn.execute(
                "select incident_id,id,event_id,camera_id,captured_epoch,payload_json,snapshot_path from ("
                "select p.incident_id,o.*,row_number() over(partition by p.incident_id order by "
                "coalesce(cast(json_extract(o.payload_json,'$.detection_frame_width') as integer),0)*"
                "coalesce(cast(json_extract(o.payload_json,'$.detection_frame_height') as integer),0) desc,"
                "coalesce(cast(json_extract(o.payload_json,'$.confidence') as real),0) desc,"
                "o.captured_epoch asc,o.id asc) rank "
                "from scene_observations o join scene_episodes p on p.id=o.episode_id "
                f"where p.incident_id in ({placeholders}) and o.snapshot_path!='') where rank=1", ids).fetchall()
            recorded_rows=conn.execute(
                "select p.incident_id,max(coalesce(cast(json_extract(o.payload_json,'$.detection_frame_width') as integer),0)*"
                "coalesce(cast(json_extract(o.payload_json,'$.detection_frame_height') as integer),0)) pixels "
                "from scene_observations o join scene_episodes p on p.id=o.episode_id "
                f"where p.incident_id in ({placeholders}) and json_extract(o.payload_json,'$.frame_source')='recorded_main' "
                "group by p.incident_id", ids).fetchall()
            observation_rows=[]
            if cover_rows:
                clauses=" or ".join("(p.incident_id=? and o.snapshot_path=?)" for _ in cover_rows)
                cover_args=[value for row in cover_rows for value in (row["incident_id"],row["snapshot_path"])]
                observation_rows=conn.execute(
                    "select p.incident_id,o.id,o.event_id,o.camera_id,o.captured_epoch,o.payload_json,o.snapshot_path "
                    "from scene_observations o join scene_episodes p on p.id=o.episode_id where "+clauses,
                    cover_args).fetchall()
        cameras_by={}
        for row in episode_rows:
            cameras_by.setdefault(row["incident_id"], [])
            if row["camera_id"] and row["camera_id"] not in cameras_by[row["incident_id"]]:
                cameras_by[row["incident_id"]].append(row["camera_id"])
        labels_by={}
        for row in label_rows:
            if row["label"]:
                labels_by.setdefault(row["incident_id"], set()).add(row["label"])
        events_by={}
        for row in event_rows:
            events_by.setdefault(row["incident_id"], []).append(row)
        observations_by={}
        for row in observation_rows:
            observations_by.setdefault(row["incident_id"], []).append(row)
        recorded_pixels_by={row["incident_id"]:int(row["pixels"] or 0) for row in recorded_rows}
        cards=[]
        for incident_id in ids:
            incident=incidents.get(incident_id)
            if incident is None:
                continue
            members=events_by.get(incident_id, [])
            labels=sorted(labels_by.get(incident_id, ()))
            presented_members=[_event_row(dict(event)) for event in members]
            best=(
                _best_incident_event(presented_members)
                if presented_members else None
            )
            representative=next(
                (event for event in members if event["id"] == best["id"]),
                members[-1] if members else None,
            )
            opener=members[0] if members else None
            raw_trigger=str((opener["topic"] if opener else None) or "camera").lower()
            trigger="ema" if raw_trigger in {"ema","adaptive","visual_backup","adaptive/visual_backup"} else "camera"
            snapshot_observation_id=None
            snapshot_url=None
            selected = _scene_observation_cover(
                representative, observations_by.get(incident_id, []),
                recorded_pixels=recorded_pixels_by.get(incident_id, 0),
            )
            if selected is not None:
                cover = selected["row"]
                representative = next(
                    (event for event in members if event["id"] == cover["event_id"]),
                    representative,
                )
                snapshot_observation_id=cover["id"]
                snapshot_url=f"/api/incidents/observations/{snapshot_observation_id}/snapshot"
                snapshot_path="available"
                cover_objects=_card_cover_objects(_json(selected["objects"]))
            elif representative is not None and representative["snapshot_path"]:
                snapshot_path="available"
                cover_objects=_card_cover_objects(representative["objects_json"])
            else:
                snapshot_path=""
                cover_objects=[]
            cover_event_id=int(representative["id"]) if representative is not None and snapshot_path else None
            cards.append({
                "id":incident_id,
                "incident_id":incident_id,
                "revision":int(incident["revision"] or 0),
                "schema_version":3,
                "camera_id":str(
                    (representative["camera_id"] if representative is not None else "")
                    or (cameras_by.get(incident_id) or [""])[0]
                ),
                "camera_ids":cameras_by.get(incident_id, []),
                "start_at":_iso(incident["start_epoch"]),
                "end_at":_iso(incident["end_epoch"]),
                "start_epoch":incident["start_epoch"],
                "last_epoch":incident["end_epoch"],
                "created_at":representative["created_at"] if representative is not None else _iso(incident["start_epoch"]),
                "representative_event_id":representative["id"] if representative is not None else None,
                "evidence_revision":int(representative["evidence_revision"] or 0) if representative is not None else 0,
                "snapshot_path":snapshot_path,
                "snapshot_url":snapshot_url,
                "snapshot_observation_id":snapshot_observation_id,
                "labels":labels,
                "zones":[],
                "has_objects":bool(labels),
                "kind":"object" if labels else "motion",
                "trigger_source":trigger,
                "event_count":len(members),
                "event_ids":[event["id"] for event in members],
                "objects":cover_objects,
                "events":[{
                    "id":event["id"],
                    "camera_id":event["camera_id"],
                    "kind":event["kind"],
                    "created_at":event["created_at"],
                    "evidence_revision":int(event["evidence_revision"] or 0),
                    "has_objects":bool(labels),
                    "labels":labels,
                    "trigger_source":trigger if event is opener else "camera",
                    **({"objects":cover_objects} if cover_event_id is not None and int(event["id"]) == cover_event_id else {}),
                } for event in members],
            })
        return cards

    def record_scene_observations(self,event_id,observations,*,activity=True):
        with self._lock,self._connect() as conn:
            conn.execute("begin immediate")
            row=conn.execute("select * from events where id=?",(event_id,)).fetchone()
            if row:
                retained=[]
                for observation in observations:
                    item=dict(observation)
                    if item.get("snapshot_path"):
                        retained_path=self._acquired_media_path(
                            conn,item["snapshot_path"]
                        )
                        item["snapshot_path"]=retained_path
                        if not retained_path:
                            item["snapshot_visible"]=False
                    retained.append(item)
                projected=dict(row)
                projected["objects_json"]=_json([*_objects(row["objects_json"]),{"status":"scene_observations","observations":retained}])
                self._scene_ingest(conn,projected,activity=activity)

    def scene_observation(self,observation_id):
        with self._connect() as conn:
            row=conn.execute("select o.*,s.label_override,s.association_locked from scene_observations o join scene_objects s on s.id=o.object_id where o.id=?",(observation_id,)).fetchone()
            return dict(row) if row else None

    def scene_has_search_observation(self, event_id):
        with self._connect() as conn:
            return conn.execute(
                "select 1 from scene_observations where event_id=? and snapshot_path!='' limit 1",
                (int(event_id),),
            ).fetchone() is not None

    def scene_observation_media(self, event_id):
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(
                "select id, snapshot_path from scene_observations where event_id=? and snapshot_path!=''",
                (int(event_id),),
            )]

    def scene_observation_payloads(self, observation_ids):
        identifiers = [str(item) for item in observation_ids if item]
        if not identifiers:
            return []
        rows = []
        with self._connect() as conn:
            for start in range(0, len(identifiers), 200):
                chunk = identifiers[start:start + 200]
                placeholders = ",".join("?" * len(chunk))
                rows.extend(dict(row) for row in conn.execute(
                    f"select * from scene_observations where id in ({placeholders})", chunk,
                ))
        return rows

    def scene_search_observations(self,event_id=None,after_id="",limit=100):
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(
                "select * from scene_observations where snapshot_path!='' and id>? "
                + ("and event_id=? " if event_id is not None else "") + "order by id limit ?",
                (after_id, *((event_id,) if event_id is not None else ()), max(1,min(int(limit),1000))))]

    def update_scene_identities(self,event_id,identities,*,routes=()):
        """Attach accepted identity evidence and conservatively connect episodes.

        Appearance candidates never enter this path. Repeated stationary presence
        cannot bridge incidents: an observation must contain measured movement.
        """
        accepted=[dict(i) for i in identities if i.get("status") in {"confirmed","automatic"} and int(i.get("identity_id") or 0)>0]
        with self._lock,self._connect() as conn:
            conn.execute("begin immediate")
            member=conn.execute("select p.*,e.created_at from scene_episodes p join scene_event_membership m on m.episode_id=p.id "
                                "join events e on e.id=m.event_id where e.id=?",(event_id,)).fetchone()
            if member is None:
                return
            raw=_json(accepted)
            old=conn.execute("select payload_json from scene_identities where event_id=?",(event_id,)).fetchone()
            identity_changed = old is None or old[0] != raw
            connected = False
            conn.execute("insert into scene_identities values(?,?) on conflict(event_id) do update set payload_json=excluded.payload_json",(event_id,raw))
            incident_id=member["incident_id"]
            at=_epoch(member["created_at"])
            def person_interval(anchor):
                people = conn.execute(
                    "select count(distinct object_id),min(captured_epoch),max(captured_epoch) from scene_observations "
                    "where event_id=? and json_extract(payload_json,'$.label') in ('person','pedestrian')",(anchor,)).fetchone()
                if people[0] != 1:
                    return None
                return people[1], people[2]
            current_interval = person_interval(event_id)
            route_values = [r.model_dump() if hasattr(r,"model_dump") else r for r in routes]
            search_window = max([45.0, *[float(r.get("max_seconds",45)) for r in route_values if r.get("enabled",True)]])
            def active(anchor):
                # Identity links use the same physical evidence as episode
                # establishment. Detector-box drift and newly-seen flags do
                # not turn a stationary person into a cross-camera bridge.
                decisions = conn.execute("select d.evidence_json from scene_activity_decisions d "
                    "join scene_activity_admissions a on a.decision_id=d.id where a.event_id=?", (anchor,))
                supporting = set()
                for row in decisions:
                    evidence = json.loads(row[0])
                    zone = evidence.get("zone_interpretation")
                    # A recorded zone rejection cannot admit this camera. Decisions
                    # made before zone establishment remain as they were recorded.
                    if isinstance(zone, dict) and zone.get("establishment_eligible") is False:
                        continue
                    supporting.update(evidence.get("supporting_observation_ids") or [])
                for key in supporting:
                    observed = conn.execute("select payload_json from acquired_observations where id=?", (key,)).fetchone()
                    if observed and json.loads(observed[0]).get("label") in {"person","pedestrian"}:
                        return True
                return False
            if len({i["identity_id"] for i in accepted})==1 and current_interval and not member["boundary_locked"] and active(event_id):
                candidates=conn.execute("select s.event_id,s.payload_json,p.*,e.created_at from scene_identities s "
                                        "join events e on e.id=s.event_id join scene_event_membership m on m.event_id=e.id "
                                        "join scene_episodes p on p.id=m.episode_id where p.camera_id!=? and e.created_at>=? and e.created_at<=? "
                                        "and p.boundary_locked=0 order by e.created_at desc",(member["camera_id"],_iso(at-search_window),_iso(at+search_window))).fetchall()
                known={int(i["identity_id"]) for i in accepted}
                matches={}
                for c in candidates:
                    other_known={int(i["identity_id"]) for i in json.loads(c["payload_json"])}
                    other_interval=person_interval(c["event_id"])
                    if known != other_known or not other_interval or c["incident_id"]==incident_id:
                        continue
                    other_at=_epoch(c["created_at"])
                    # Conflicting simultaneous sightings do not establish a
                    # transition. Overlapping camera views need human review.
                    if min(current_interval[1],other_interval[1]) >= max(current_interval[0],other_interval[0]):
                        continue
                    transition_gap=max(current_interval[0],other_interval[0])-min(current_interval[1],other_interval[1])
                    previous_camera,next_camera=(c["camera_id"],member["camera_id"]) if other_at<=at else (member["camera_id"],c["camera_id"])
                    windows=[]
                    for route in route_values:
                        if not route.get("enabled",True):
                            continue
                        if (route["from_camera"],route["to_camera"])==(previous_camera,next_camera) or (route.get("bidirectional") and (route["to_camera"],route["from_camera"])==(previous_camera,next_camera)):
                            windows.append((route.get("min_seconds",0),route.get("max_seconds",45)))
                    if not any(low<=transition_gap<=high for low,high in (windows or [(0,45)])):
                        continue
                    if active(c["event_id"]):
                        matches[c["incident_id"]]=c
                # Competing accepted identities/episodes require human review.
                if len(matches)==1:
                    connected = True
                    source=next(iter(matches))
                    conn.execute("update scene_episodes set incident_id=? where incident_id=?",(incident_id,source))
                    conn.execute("insert or replace into scene_aliases values(?,?)",(source,incident_id))
                    conn.execute("update scene_aliases set incident_id=? where incident_id=?",(incident_id,source))
                    conn.execute("update scene_incidents set state='merged' where id=?",(source,))
                    conn.execute("delete from scene_notification_outbox where incident_id=?",(source,))
                    self._scene_merge_shared_objects(conn,incident_id)
                    other_event=matches[source]["event_id"]
                    people=[]
                    for anchor in (other_event,event_id):
                        ids={o["object_id"] for o in conn.execute("select object_id,payload_json from scene_observations where event_id=?",(anchor,)) if json.loads(o["payload_json"]).get("label") in {"person","pedestrian"}}
                        people.append(ids)
                    if all(len(ids)==1 for ids in people) and len(known)==1:
                        self._scene_join_subjects(conn,next(iter(people[0])),next(iter(people[1])))
                    conn.execute("insert into scene_corrections(incident_id,revision,payload_json,created_at) values(?,0,?,?)",
                                 (incident_id,_json({"operation":"identity_connection","source_incident_id":source,"event_id":event_id}),time.time()))
                    self._scene_recalculate(conn,incident_id)
            if identity_changed or connected:
                self._scene_changed(conn,incident_id)

    def scene_pending_notifications(self,limit=100):
        """Newest snapshot for each incident with an unpublished change.

        Each entry carries the revision it was built at; acknowledging it
        clears every pending revision up to that one.
        """
        entries, orphaned = [], []
        with self._connect() as conn:
            conn.execute("begin")
            try:
                rows = conn.execute(
                    "select incident_id, revision from scene_notification_outbox where payload_json='' order by rowid limit ?",
                    (limit,),
                ).fetchall()
                for row in rows:
                    incident_id = str(row["incident_id"])
                    payload = (
                        self._scene_notification_payload(conn, incident_id)
                        if self._scene_resolve(conn, incident_id) == incident_id else None
                    )
                    if payload is None:
                        orphaned.append((incident_id, int(row["revision"])))
                        continue
                    entries.append({"incident_id": incident_id, "revision": int(payload["revision"]), "payload": payload})
            finally:
                conn.rollback()
        for incident_id, revision in orphaned:
            self.acknowledge_scene_notification(incident_id, revision)
        return entries

    def acknowledge_scene_notification(self,incident_id,revision):
        with self._lock,self._connect() as conn:
            conn.execute(
                "delete from scene_notification_outbox where incident_id=? and revision<=? and payload_json=''",
                (incident_id, revision),
            )

    def settle_scene_incidents(self,now=None):
        now=time.time() if now is None else now
        with self._lock,self._connect() as conn:
            ids=conn.execute("select i.id from scene_incidents i where state='active' and "
                             "exists(select 1 from scene_episodes p where p.incident_id=i.id) and "
                             "not exists(select 1 from scene_episodes p where p.incident_id=i.id and p.last_activity_epoch>?)",(now-45,)).fetchall()
            for row in ids:
                conn.execute("update scene_incidents set state='complete' where id=?",(row[0],))
                historical=conn.execute("select historical from scene_incidents where id=?",(row[0],)).fetchone()[0]
                self._scene_changed(conn,row[0],notify=not historical)

    def correct_scene_incident(self,incident_id,expected_revision,correction):
        """Optimistic, atomic corrections. Raw model evidence is never rewritten."""
        with self._lock,self._connect() as conn:
            conn.execute("begin immediate")
            incident_id=self._scene_resolve(conn,incident_id)
            current=conn.execute("select * from scene_incidents where id=?",(incident_id,)).fetchone()
            if not current:
                raise LookupError("incident not found")
            if current["revision"]!=expected_revision:
                raise SceneConflict("incident changed; refresh before correcting")
            operation=correction.get("operation")
            if operation=="merge":
                for source in dict.fromkeys(correction.get("incident_ids",[])):
                    source=self._scene_resolve(conn,source)
                    if source==incident_id:
                        continue
                    revision=conn.execute("select revision from scene_incidents where id=?",(source,)).fetchone()
                    if not revision:
                        raise LookupError("merge incident not found")
                    if correction.get("expected_revisions",{}).get(source)!=revision[0]:
                        raise SceneConflict("merge incident changed; refresh before correcting")
                    conn.execute("update scene_episodes set incident_id=?,boundary_locked=1 where incident_id=?",(incident_id,source))
                    conn.execute("insert or replace into scene_aliases values(?,?)",(source,incident_id))
                    conn.execute("update scene_aliases set incident_id=? where incident_id=?",(incident_id,source))
                    conn.execute("update scene_incidents set state='merged' where id=?",(source,))
                    conn.execute("delete from scene_notification_outbox where incident_id=?",(source,))
                self._scene_merge_shared_objects(conn,incident_id)
            elif operation=="split":
                selected=set(correction.get("episode_ids",[]))
                all_ids={r[0] for r in conn.execute("select id from scene_episodes where incident_id=?",(incident_id,))}
                if not selected or not selected<all_ids:
                    raise ValueError("select some, but not all, camera episodes to split")
                new_id="incident-"+uuid.uuid4().hex
                conn.execute("insert into scene_incidents(id,start_epoch,end_epoch,historical,state) values(?,?,?,?,?)",(new_id,current["start_epoch"],current["end_epoch"],current["historical"],current["state"]))
                for episode_id in selected:
                    conn.execute("update scene_episodes set incident_id=?,boundary_locked=1 where id=?",(new_id,episode_id))
                conn.execute("update scene_episodes set boundary_locked=1 where incident_id=?",(incident_id,))
                # An association may span camera episodes. After splitting,
                # subjects must be independently owned by each incident.
                subjects=conn.execute("select distinct o.object_id from scene_observations o join scene_episodes p on p.id=o.episode_id where p.incident_id=?",(new_id,)).fetchall()
                for subject in subjects:
                    owner=conn.execute("select * from scene_objects where id=?",(subject[0],)).fetchone()
                    moved=conn.execute("select o.id,o.episode_id from scene_observations o join scene_episodes p on p.id=o.episode_id where p.incident_id=? and o.object_id=?",(new_id,subject[0])).fetchall()
                    remaining=conn.execute("select 1 from scene_observations o join scene_episodes p on p.id=o.episode_id where p.incident_id=? and o.object_id=? limit 1",(incident_id,subject[0])).fetchone()
                    if remaining:
                        cloned="object-"+uuid.uuid4().hex
                        conn.execute("insert into scene_objects(id,episode_id,label_override,association_locked,facet_label) values(?,?,?,1,?)",
                                     (cloned,moved[0]["episode_id"],owner["label_override"],owner["facet_label"]))
                        self._note_scene_label(conn, owner["label_override"] or owner["facet_label"])
                        conn.executemany("update scene_observations set object_id=? where id=?",[(cloned,o["id"]) for o in moved])
                        for moved_episode in {o["episode_id"] for o in moved}:
                            conn.execute("update scene_track_aliases set object_id=? where episode_id=? and object_id=?",(cloned,moved_episode,subject[0]))
                        self._scene_remap_appearance(conn,[subject[0]])
                        original_episode=conn.execute("select episode_id from scene_observations where object_id=? limit 1",(subject[0],)).fetchone()[0]
                        conn.execute("update scene_objects set episode_id=?,association_locked=1 where id=?",(original_episode,subject[0]))
                    else:
                        conn.execute("update scene_objects set episode_id=?,association_locked=1 where id=?",(moved[0]["episode_id"],subject[0]))
                self._scene_recalculate(conn,new_id)
                self._refresh_scene_establishment(conn,new_id)
                self._scene_changed(conn,new_id)
            elif operation in {"label","associate","separate"}:
                owned={r[0] for r in conn.execute("select s.id from scene_objects s join scene_episodes p on p.id=s.episode_id where p.incident_id=?",(incident_id,))}
                if operation=="label":
                    label=str(correction.get("label") or "").strip()
                    if correction.get("object_id") not in owned or not label or len(label)>100:
                        raise ValueError("a scene object and a label of 1 to 100 characters are required")
                    previous=conn.execute("select label_override,facet_label from scene_objects where id=?",(correction["object_id"],)).fetchone()
                    conn.execute("update scene_objects set label_override=? where id=?",(label,correction["object_id"]))
                    self._note_scene_label(conn, label)
                    if previous:
                        old=previous["label_override"] or previous["facet_label"]
                        if old and old!=label:
                            self._forget_unused_scene_label(conn, old)
                elif operation=="associate":
                    ids=list(dict.fromkeys(correction.get("object_ids",[])))
                    if len(ids)<2 or not set(ids)<=owned:
                        raise ValueError("select at least two objects from this incident")
                    for source in ids[1:]:
                        conn.execute("update scene_observations set object_id=? where object_id=?",(ids[0],source))
                        conn.execute("update scene_track_aliases set object_id=? where object_id=?",(ids[0],source))
                    self._scene_remap_appearance(conn,ids)
                    conn.execute("update scene_objects set association_locked=1 where id=?",(ids[0],))
                else:
                    ids=list(dict.fromkeys(correction.get("observation_ids",[])))
                    if not ids:
                        raise ValueError("select observations to separate")
                    new_object="object-"+uuid.uuid4().hex
                    affected_objects={new_object}
                    for index,observation_id in enumerate(ids):
                        row=conn.execute("select * from scene_observations where id=?",(observation_id,)).fetchone()
                        if row is None or row["object_id"] not in owned:
                            raise ValueError("observation does not belong to this incident")
                        if index==0:
                            separated_label=str(json.loads(row["payload_json"]).get("label") or "")
                            conn.execute("insert into scene_objects(id,episode_id,association_locked,facet_label) values(?,?,1,?)",
                                         (new_object,row["episode_id"],separated_label))
                            self._note_scene_label(conn, separated_label)
                        affected_objects.add(row["object_id"])
                        conn.execute("update scene_objects set association_locked=1 where id=?",(row["object_id"],))
                        conn.execute("update scene_observations set object_id=? where id=?",(new_object,observation_id))
                    self._scene_rebuild_corrected_aliases(conn,affected_objects)
                    self._scene_remap_appearance(conn,affected_objects)
            else:
                raise ValueError("unknown correction operation")
            self._scene_recalculate(conn,incident_id)
            self._refresh_scene_establishment(conn,incident_id)
            conn.execute("insert into scene_corrections(incident_id,revision,payload_json,created_at) values(?,?,?,?)",
                         (incident_id,expected_revision+1,_json(correction),time.time()))
            self._scene_changed(conn,incident_id)
            return self._scene_payload(conn,incident_id)

    @staticmethod
    def _scene_recalculate(conn,incident_id):
        conn.execute("update scene_incidents set start_epoch=(select min(start_epoch) from scene_episodes where incident_id=?),"
                     "end_epoch=(select max(end_epoch) from scene_episodes where incident_id=?) where id=?",(incident_id,incident_id,incident_id))
