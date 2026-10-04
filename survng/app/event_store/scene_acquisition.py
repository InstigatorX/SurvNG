"""Acquisition and verification ownership independent of incident admission."""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid

from ..database_polling import polling_connection
from ..incident_utils import portable_media_path, snapshot_deletion_claimed
from ..scene_identity import observation_identity

# Five evenly spaced confirmation frames must remain close enough for the
# ten-second physical comparison horizon. Longer activity uses later jobs.
SCENE_CONFIRMATION_MAX_WINDOW_SECONDS = 40.0


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _epoch(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("finite timestamp required")
    return value


class EventStoreSceneAcquisitionMixin:
    def _init_scene_acquisition_db(self):
        with self._lock, self._connect() as conn:
            had_seeds = conn.execute("select 1 from sqlite_master where name='scene_candidate_seeds'").fetchone() is not None
            conn.executescript("""
                create table if not exists acquired_samples (
                    id text primary key, camera_id text not null, captured_epoch real not null,
                    source text not null, status text not null check(status in ('complete','failed')),
                    metadata_json text not null, snapshot_path text not null, recording_path text not null,
                    created_at real not null
                );
                create index if not exists acquired_sample_camera_id on acquired_samples(camera_id,captured_epoch,id);
                drop index if exists acquired_sample_camera;
                create index if not exists acquired_sample_snapshot on acquired_samples(snapshot_path);
                create table if not exists scene_activity_measurements (
                    id text primary key, sample_id text not null references acquired_samples(id),
                    payload_json text not null, created_at real not null
                );
                create index if not exists scene_measurement_sample on scene_activity_measurements(sample_id);
                create table if not exists acquired_observations (
                    id text primary key, camera_id text not null, captured_epoch real not null,
                    payload_json text not null, snapshot_path text not null, recording_path text not null
                );
                create index if not exists acquired_observation_camera on acquired_observations(camera_id,captured_epoch,id);
                create index if not exists acquired_observation_snapshot on acquired_observations(snapshot_path);
                create table if not exists acquired_sample_observations (
                    sample_id text not null references acquired_samples(id),
                    observation_id text not null references acquired_observations(id),
                    primary key(sample_id,observation_id)
                );
                create table if not exists acquired_sample_events (
                    sample_id text not null references acquired_samples(id), event_id integer not null,
                    primary key(sample_id,event_id)
                );
                create index if not exists acquired_source_event on acquired_sample_events(event_id);
                create table if not exists scene_activity_decisions (
                    id text primary key, camera_id text not null, sample_ids_json text not null,
                    verdict text not null check(verdict in ('pending','supported','unsupported','incomplete')),
                    activity_epoch real, reason text not null, policy_version text not null,
                    evidence_json text not null, created_at real not null
                );
                create index if not exists scene_activity_pending on scene_activity_decisions(camera_id,verdict,activity_epoch);
                create table if not exists scene_activity_admissions (
                    decision_id text primary key references scene_activity_decisions(id),
                    event_id integer not null, episode_id text not null, created_at real not null
                );
                create table if not exists scene_candidate_jobs (
                    id text primary key, camera_id text not null, start_epoch real not null,
                    end_epoch real not null, seed_sample_ids_json text not null,
                    deadline_epoch real not null, available_at_epoch real not null,
                    state text not null, attempts integer not null default 0,
                    lease_owner text not null default '', lease_token integer not null default 0,
                    lease_expires_at_epoch real, generation integer not null default 1,
                    lease_generation integer not null default 0, reason text not null default '',
                    decision_id text, created_at real not null, updated_at real not null
                );
                create index if not exists scene_candidate_due on scene_candidate_jobs(camera_id,state,available_at_epoch);
                create table if not exists scene_candidate_seeds (
                    sample_id text primary key references acquired_samples(id),
                    candidate_id text not null references scene_candidate_jobs(id)
                );
                create table if not exists scene_candidate_admissions (
                    candidate_id text not null references scene_candidate_jobs(id),
                    generation integer not null, decision_id text not null references scene_activity_decisions(id),
                    event_id integer not null references events(id),
                    primary key(candidate_id,generation)
                );
                create table if not exists scene_acquisition_migrations (name text primary key,cursor integer not null);
                create table if not exists acquired_expired_snapshots (snapshot_path text primary key);
            """)
            if not had_seeds:
                conn.execute("insert or ignore into scene_candidate_seeds select s.value,j.id "
                             "from scene_candidate_jobs j,json_each(j.seed_sample_ids_json) s")

    @staticmethod
    def _sample_payload(conn, sample_id):
        row = conn.execute("select * from acquired_samples where id=?", (sample_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        witnesses = [json.loads(r[0]) for r in conn.execute("select payload_json from scene_activity_measurements where sample_id=? order by id", (sample_id,))]
        if witnesses:
            result["metadata"]["activity_witnesses"] = witnesses
        rows = conn.execute("select o.* from acquired_observations o join acquired_sample_observations a "
                            "on a.observation_id=o.id where a.sample_id=? order by o.captured_epoch,o.id", (sample_id,)).fetchall()
        result["observations"] = [{**json.loads(o["payload_json"]), "id": o["id"],
                                   "snapshot_path": o["snapshot_path"], "recording_path": o["recording_path"]} for o in rows]
        result["observation_ids"] = [o["id"] for o in rows]
        return result

    @staticmethod
    def _sample_payloads(conn, sample_ids):
        """Load a bounded page of samples without per-sample SQL round trips."""
        identifiers = list(dict.fromkeys(str(value) for value in sample_ids if value))
        if not identifiers:
            return []
        placeholders = ",".join("?" for _ in identifiers)
        samples = {
            str(row["id"]): dict(row)
            for row in conn.execute(
                f"select * from acquired_samples where id in ({placeholders})",
                identifiers,
            )
        }
        for sample in samples.values():
            sample["metadata"] = json.loads(sample.pop("metadata_json"))
            sample["observations"] = []
            sample["observation_ids"] = []
        witnesses = {key: [] for key in samples}
        for row in conn.execute(
            f"select sample_id,payload_json from scene_activity_measurements "
            f"where sample_id in ({placeholders}) order by sample_id,id",
            identifiers,
        ):
            key = str(row["sample_id"])
            if key in witnesses:
                witnesses[key].append(json.loads(row["payload_json"]))
        for key, values in witnesses.items():
            if values:
                samples[key]["metadata"]["activity_witnesses"] = values
        for row in conn.execute(
            "select a.sample_id,o.* from acquired_sample_observations a "
            "join acquired_observations o on o.id=a.observation_id "
            f"where a.sample_id in ({placeholders}) order by a.sample_id,o.captured_epoch,o.id",
            identifiers,
        ):
            sample = samples.get(str(row["sample_id"]))
            if sample is None:
                continue
            sample["observations"].append({
                **json.loads(row["payload_json"]),
                "id": row["id"],
                "snapshot_path": row["snapshot_path"],
                "recording_path": row["recording_path"],
            })
            sample["observation_ids"].append(row["id"])
        return [samples[key] for key in identifiers if key in samples]

    def scene_sample(self, sample_id):
        with self._connect() as conn:
            return self._sample_payload(conn, sample_id)

    _scene_sample = _sample_payload

    def _scene_samples(self, conn, camera_id, start_epoch, end_epoch, *, limit=200, after_id=""):
        rows=conn.execute("select id from acquired_samples where camera_id=? and captured_epoch>=? "
                          "and captured_epoch<=? and id>? order by id limit ?",
                          (camera_id,_epoch(start_epoch),_epoch(end_epoch),after_id,max(1,min(int(limit),1000)))).fetchall()
        return self._sample_payloads(conn, [row[0] for row in rows])

    def scene_acquired_observation(self, observation_id):
        with self._connect() as conn:
            row=conn.execute("select * from acquired_observations where id=?",(observation_id,)).fetchone()
            return dict(row) if row else None

    def _acquired_media_path(self, conn, path):
        path = portable_media_path(self.storage_dir, str(path or ""))
        has_claims = path and conn.execute(
            "select 1 from sqlite_master where type='table' and name='media_deletion_claims'"
        ).fetchone()
        if has_claims and snapshot_deletion_claimed(conn, self.storage_dir, path):
            return ""
        if path and conn.execute("select 1 from acquired_expired_snapshots where snapshot_path=?", (path,)).fetchone():
            return ""
        if path and conn.execute("select 1 from sqlite_master where name='scene_expired_snapshots'").fetchone():
            if conn.execute("select 1 from scene_expired_snapshots where snapshot_path=?", (path,)).fetchone():
                return ""
        return path

    def _register_acquired_snapshot(self, conn, path, camera_id, captured, size_bytes=0):
        if path and conn.execute("select 1 from sqlite_master where name='scene_snapshot_assets'").fetchone():
            from datetime import datetime, timezone
            conn.execute("insert into scene_snapshot_assets values(?,?,?,?) on conflict(snapshot_path) do update "
                         "set created_at=max(created_at,excluded.created_at),snapshot_size_bytes=max(snapshot_size_bytes,excluded.snapshot_size_bytes)",
                         (path, camera_id, datetime.fromtimestamp(captured, timezone.utc).isoformat(),max(0,int(size_bytes or 0))))

    @staticmethod
    def _record_activity_measurements(conn, sample_id, metadata):
        inserted = 0
        for witness in (metadata or {}).get("activity_witnesses", []):
            raw = _json(witness)
            measurement_id = hashlib.sha256(_json([sample_id,witness]).encode()).hexdigest()
            inserted += conn.execute("insert or ignore into scene_activity_measurements values(?,?,?,?)", (measurement_id,sample_id,raw,time.time())).rowcount
        return inserted

    def _acquire_scene_sample(self, conn, *, sample_id, camera_id, captured_epoch, source, status,
                              observations, metadata=None, snapshot_path="", recording_path="",
                              request_confirmation=None):
        if not sample_id or not camera_id or not source or status not in {"complete", "failed"}:
            raise ValueError("sample identity, camera, source and valid status required")
        captured_epoch = _epoch(captured_epoch)
        existing = self._sample_payload(conn, sample_id)
        if existing:
            if (existing["camera_id"], existing["captured_epoch"], existing["source"]) != (camera_id, captured_epoch, source):
                raise ValueError("acquisition identity collision")
            refresh = self._record_activity_measurements(conn, sample_id, metadata) > 0
            snapshot = self._acquired_media_path(conn, snapshot_path)
            if snapshot and not existing["snapshot_path"]:
                refresh = True
                conn.execute("update acquired_samples set snapshot_path=? where id=?", (snapshot,sample_id))
                conn.execute("update acquired_observations set snapshot_path=? where snapshot_path='' and id in "
                             "(select observation_id from acquired_sample_observations where sample_id=?)", (snapshot,sample_id))
                self._register_acquired_snapshot(conn,snapshot,camera_id,captured_epoch,(metadata or {}).get("snapshot_size_bytes",0))
            if request_confirmation:
                self._enqueue_scene_candidate(conn, camera_id, seed_sample_id=sample_id, **request_confirmation)
            return {**(self._sample_payload(conn,sample_id) if refresh else existing), "created": False}
        if status == "failed" and observations:
            raise ValueError("failed acquisition cannot contain detections")
        snapshot = self._acquired_media_path(conn, snapshot_path)
        recording = portable_media_path(self.storage_dir, str(recording_path or ""))
        conn.execute("insert into acquired_samples values(?,?,?,?,?,?,?,?,?)",
                     (sample_id,camera_id,captured_epoch,source,status,_json(metadata or {}),snapshot,recording,time.time()))
        self._record_activity_measurements(conn, sample_id, metadata)
        self._register_acquired_snapshot(conn, snapshot, camera_id, captured_epoch,(metadata or {}).get("snapshot_size_bytes",0))
        for raw in observations:
            item = dict(raw)
            at = _epoch(item.get("captured_at_epoch", item.get("frame_captured_at_epoch", captured_epoch)))
            box = item.get("box")
            if isinstance(box, dict):
                box = {key: _epoch(value) for key, value in box.items()}
                item["box"] = box
            if not item.get("label"):
                raise ValueError("acquired observation must have a label")
            observation_id = observation_identity(camera_id, at, item)
            image = self._acquired_media_path(conn, item.pop("snapshot_path", ""))
            movie = portable_media_path(self.storage_dir, str(item.pop("recording_path", "") or recording))
            conn.execute("insert or ignore into acquired_observations values(?,?,?,?,?,?)",
                         (observation_id,camera_id,at,_json(item),image,movie))
            # Media acquisition may become available later; raw detector output
            # remains the first acquired result for this source observation.
            if image:
                conn.execute("update acquired_observations set snapshot_path=? where id=? and snapshot_path=''", (image,observation_id))
                self._register_acquired_snapshot(conn,image,camera_id,at,item.get("snapshot_size_bytes",0))
            conn.execute("insert into acquired_sample_observations values(?,?) on conflict do nothing",(sample_id,observation_id))
        if request_confirmation:
            self._enqueue_scene_candidate(conn, camera_id, seed_sample_id=sample_id, **request_confirmation)
        return {**self._sample_payload(conn,sample_id), "created": True}

    def acquire_scene_sample(self, **kwargs):
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            return self._acquire_scene_sample(conn, **kwargs)

    def scene_acquired_observations(self, camera_id, start_epoch, end_epoch, *, limit=200, after_id=""):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("select * from acquired_observations where camera_id=? "
                        "and captured_epoch>=? and captured_epoch<=? and id>? order by id limit ?",
                        (camera_id,_epoch(start_epoch),_epoch(end_epoch),after_id,max(1,min(int(limit),1000))))]

    @staticmethod
    def _activity_payload(row):
        if row is None:
            return None
        result = dict(row)
        result["sample_ids"] = json.loads(result.pop("sample_ids_json"))
        result["evidence"] = json.loads(result.pop("evidence_json"))
        return result

    def _record_scene_activity_decision(self, conn, *, decision_id, sample_ids, verdict, activity_epoch,
                                        reason, policy_version, evidence=None):
        sample_ids = sorted(set(sample_ids))
        if not decision_id or not sample_ids or verdict not in {"pending","supported","unsupported","incomplete"}:
            raise ValueError("decision identity, source samples and valid verdict required")
        samples = conn.execute("select camera_id from acquired_samples where id in (select value from json_each(?))", (_json(sample_ids),)).fetchall()
        if len(samples) != len(sample_ids) or len({s[0] for s in samples}) != 1:
            raise ValueError("decision samples must exist and belong to one camera")
        activity_epoch = _epoch(activity_epoch) if verdict == "supported" else None
        values = (decision_id,samples[0][0],_json(sample_ids),verdict,activity_epoch,str(reason),str(policy_version),_json(evidence or {}))
        previous = conn.execute("select * from scene_activity_decisions where id=?",(decision_id,)).fetchone()
        if previous and tuple(previous)[:-1] != values:
            raise ValueError("activity decision identity collision")
        conn.execute("insert or ignore into scene_activity_decisions values(?,?,?,?,?,?,?,?,?)",(*values,time.time()))
        return self._activity_payload(conn.execute("select * from scene_activity_decisions where id=?",(decision_id,)).fetchone())

    def record_scene_activity_decision(self, **kwargs):
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            return self._record_scene_activity_decision(conn, **kwargs)

    def scene_activity_decision(self, decision_id):
        with self._connect() as conn:
            return self._activity_payload(conn.execute("select * from scene_activity_decisions where id=?",(decision_id,)).fetchone())

    def scene_pending_activity_decisions(self, camera_id="", *, limit=100):
        with self._connect() as conn:
            return [self._activity_payload(row) for row in conn.execute(
                "select d.* from scene_activity_decisions d left join scene_activity_admissions a on a.decision_id=d.id "
                "where a.decision_id is null and d.verdict='supported' " + ("and d.camera_id=? " if camera_id else "") +
                "order by d.activity_epoch,d.id limit ?", (*((camera_id,) if camera_id else ()),max(1,min(int(limit),1000))))]

    @staticmethod
    def _link_scene_acquisition(conn, *, sample_id, event_id):
        if not conn.execute("select 1 from acquired_samples where id=?",(sample_id,)).fetchone():
            raise LookupError("sample not found")
        conn.execute("insert or ignore into acquired_sample_events values(?,?)",(sample_id,event_id))

    def _admit_scene_activity(self, conn, *, decision_id, event_id, episode_id):
        decision = conn.execute("select * from scene_activity_decisions where id=?",(decision_id,)).fetchone()
        if decision is None or decision["verdict"] != "supported":
            raise ValueError("only supported activity can be admitted")
        prior = conn.execute("select * from scene_activity_admissions where decision_id=?",(decision_id,)).fetchone()
        if prior and (prior["event_id"],prior["episode_id"]) != (event_id,episode_id):
            raise ValueError("activity already admitted elsewhere")
        conn.execute("insert or ignore into scene_activity_admissions values(?,?,?,?)",(decision_id,event_id,episode_id,time.time()))
        for sample_id in json.loads(decision["sample_ids_json"]):
            self._link_scene_acquisition(conn,sample_id=sample_id,event_id=event_id)

    @staticmethod
    def _candidate_payload(row):
        if row is None:
            return None
        result = dict(row)
        result["seed_sample_ids"] = json.loads(result.pop("seed_sample_ids_json"))
        return result

    def _enqueue_scene_candidate(self, conn, camera_id, *, seed_sample_id, start_epoch, end_epoch,
                                  deadline_epoch, available_at_epoch=None):
        start_epoch,end_epoch,deadline_epoch = map(_epoch,(start_epoch,end_epoch,deadline_epoch))
        if end_epoch < start_epoch or end_epoch-start_epoch > SCENE_CONFIRMATION_MAX_WINDOW_SECONDS:
            raise ValueError("candidate verification window must span at most 40 seconds")
        sample = conn.execute("select camera_id,captured_epoch,source from acquired_samples where id=?",(seed_sample_id,)).fetchone()
        if sample is None or sample[0] != camera_id:
            raise ValueError("candidate seed must belong to camera")
        if sample["source"] == "live_discovery":
            previous = conn.execute("select captured_epoch from acquired_samples where camera_id=? and captured_epoch<? "
                                    "and source='live_discovery' and status='complete' order by captured_epoch desc limit 1",
                                    (camera_id,sample["captured_epoch"])).fetchone()
            if previous:
                # Include the preceding discovery instant: a person can arrive
                # between samples and then stand still throughout a short
                # near-trigger window. The detector's label change alone is
                # not proof; the recorded baseline allows physical comparison.
                start_epoch = max(end_epoch-SCENE_CONFIRMATION_MAX_WINDOW_SECONDS,min(start_epoch,previous[0]))
        now = time.time()
        duplicate = conn.execute("select j.* from scene_candidate_seeds s join scene_candidate_jobs j on j.id=s.candidate_id "
                                 "where s.sample_id=?",(seed_sample_id,)).fetchone()
        if duplicate:
            return self._candidate_payload(duplicate)
        prior = conn.execute("select * from scene_candidate_jobs where camera_id=? and state in ('pending','running') "
                             "and deadline_epoch>? and start_epoch<=? and end_epoch>=? and max(end_epoch,?)-min(start_epoch,?)<=? "
                             "order by created_at limit 1",(camera_id,now,end_epoch,start_epoch,end_epoch,start_epoch,SCENE_CONFIRMATION_MAX_WINDOW_SECONDS)).fetchone()
        if prior:
            seeds = sorted(set(json.loads(prior["seed_sample_ids_json"])) | {seed_sample_id})
            conn.execute("update scene_candidate_jobs set start_epoch=min(start_epoch,?),end_epoch=max(end_epoch,?),"
                         "seed_sample_ids_json=?,generation=generation+1,updated_at=? where id=?",
                         (start_epoch,end_epoch,_json(seeds),now,prior["id"]))
            job_id = prior["id"]
        else:
            job_id = "candidate-"+uuid.uuid4().hex
            conn.execute("insert into scene_candidate_jobs(id,camera_id,start_epoch,end_epoch,seed_sample_ids_json,"
                         "deadline_epoch,available_at_epoch,state,created_at,updated_at) values(?,?,?,?,?,?,?,'pending',?,?)",
                         (job_id,camera_id,start_epoch,end_epoch,_json([seed_sample_id]),deadline_epoch,
                          now if available_at_epoch is None else _epoch(available_at_epoch),now,now))
        conn.execute("insert into scene_candidate_seeds values(?,?)", (seed_sample_id,job_id))
        return self._candidate_payload(conn.execute("select * from scene_candidate_jobs where id=?",(job_id,)).fetchone())

    def enqueue_scene_candidate(self, camera_id, **kwargs):
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            return self._enqueue_scene_candidate(conn,camera_id,**kwargs)

    def scene_candidate_admission(self, candidate_id, generation):
        """Recover committed admission before replaying any inference or alerts."""
        with self._connect() as conn:
            row = conn.execute("select a.decision_id,e.* from scene_candidate_admissions a "
                               "join events e on e.id=a.event_id where a.candidate_id=? and a.generation=?",
                               (candidate_id,generation)).fetchone()
            return dict(row) if row else None

    def claim_scene_candidate(self, camera_id, *, lease_owner, lease_seconds=60):
        if not lease_owner:
            raise ValueError("lease owner required")
        now=time.time()
        # Idle camera workers must not reserve the shared SQLite writer.
        with polling_connection(self.db_path) as conn:
            due=conn.execute("select 1 from scene_candidate_jobs where camera_id=? and state in ('pending','running') "
                             "and (lease_expires_at_epoch is null or lease_expires_at_epoch<=?) "
                             "and (available_at_epoch<=? or deadline_epoch<=?) limit 1",(camera_id,now,now,now)).fetchone()
        if due is None:
            return None
        with self._lock,self._connect() as conn:
            conn.execute("begin immediate")
            expired=conn.execute("select * from scene_candidate_jobs where camera_id=? "
                                 "and state in ('pending','running') and deadline_epoch<=? "
                                 "and (lease_expires_at_epoch is null or lease_expires_at_epoch<=?)",(camera_id,now,now)).fetchall()
            for job in expired:
                decision_id=job["id"]+":deadline"
                self._record_scene_activity_decision(conn,decision_id=decision_id,
                    sample_ids=json.loads(job["seed_sample_ids_json"]),verdict="incomplete",
                    activity_epoch=None,reason="verification_deadline",policy_version="verification_lifecycle_v1")
                conn.execute("update scene_candidate_jobs set state='incomplete',reason='verification_deadline',"
                             "decision_id=case when exists(select 1 from scene_activity_admissions a where a.decision_id=scene_candidate_jobs.decision_id) "
                             "then decision_id else ? end,lease_owner='',lease_expires_at_epoch=null,updated_at=? where id=?",(decision_id,now,job["id"]))
            row=conn.execute("select * from scene_candidate_jobs where camera_id=? and state in ('pending','running') "
                             "and available_at_epoch<=? and deadline_epoch>? "
                             "and (lease_expires_at_epoch is null or lease_expires_at_epoch<=?) "
                             "order by available_at_epoch,created_at limit 1",(camera_id,now,now,now)).fetchone()
            if row is None:
                return None
            conn.execute("update scene_candidate_jobs set state='running',attempts=attempts+1,lease_owner=?,"
                         "lease_token=lease_token+1,lease_generation=generation,lease_expires_at_epoch=?,updated_at=? where id=?",
                         (lease_owner,min(row["deadline_epoch"],now+max(1,float(lease_seconds))),now,row["id"]))
            return self._candidate_payload(conn.execute("select * from scene_candidate_jobs where id=?",(row["id"],)).fetchone())

    @staticmethod
    def assert_scene_candidate_lease(conn, candidate_id, lease_owner, lease_token):
        row=conn.execute("select * from scene_candidate_jobs where id=? and state='running' and lease_owner=? "
                         "and lease_token=? and lease_expires_at_epoch>?",
                         (candidate_id,lease_owner,lease_token,time.time())).fetchone()
        if row is None:
            raise ValueError("candidate verification lease expired or superseded")
        return dict(row)

    def finish_scene_candidate(self, candidate_id, *, lease_owner, lease_token, decision_id):
        now=time.time()
        with self._lock,self._connect() as conn:
            conn.execute("begin immediate")
            job=conn.execute("select * from scene_candidate_jobs where id=? and state='running' and lease_owner=? "
                             "and lease_token=? and lease_expires_at_epoch>?",(candidate_id,lease_owner,lease_token,now)).fetchone()
            if job is None:
                return False
            decision=conn.execute("select * from scene_activity_decisions where id=?",(decision_id,)).fetchone()
            if decision is None or decision["camera_id"] != job["camera_id"]:
                raise ValueError("candidate requires a recorded decision from its camera")
            if (job["generation"] == job["lease_generation"] and
                    not set(json.loads(job["seed_sample_ids_json"])) <= set(json.loads(decision["sample_ids_json"]))):
                raise ValueError("decision must reference all claimed candidate evidence")
            state = "complete" if decision["verdict"] in {"supported","unsupported"} else "incomplete"
            if job["generation"] != job["lease_generation"]:
                state="pending" if now < job["deadline_epoch"] else "incomplete"
            conn.execute("update scene_candidate_jobs set state=?,decision_id=?,reason=?,lease_owner='',"
                         "lease_expires_at_epoch=null,available_at_epoch=?,updated_at=? where id=?",
                         (state,decision_id,decision["reason"],now,now,candidate_id))
            return True

    def defer_scene_candidate(self, candidate_id, *, lease_owner, lease_token, reason, retry_delay_seconds=15):
        now=time.time()
        with self._lock,self._connect() as conn:
            return bool(conn.execute("update scene_candidate_jobs set state=case when deadline_epoch<=? then 'incomplete' else 'pending' end,"
                         "reason=?,available_at_epoch=?,lease_owner='',lease_expires_at_epoch=null,updated_at=? "
                         "where id=? and state='running' and lease_owner=? and lease_token=? and lease_expires_at_epoch>?",
                         (now,str(reason),now+max(0,float(retry_delay_seconds)),now,candidate_id,lease_owner,lease_token,now)).rowcount)

    def _expire_acquired_snapshots(self, conn, paths):
        for path in paths:
            conn.execute("insert or ignore into acquired_expired_snapshots values(?)",(path,))
            conn.execute("update acquired_samples set snapshot_path='' where snapshot_path=?",(path,))
            conn.execute("update acquired_observations set snapshot_path='' where snapshot_path=?",(path,))

    def backfill_existing_scene_acquisitions(self, *, batch_size=200):
        """Copy one historical batch silently; caller repeats while remaining.

        No acquisition is purged on a confirmation deadline. Media and metadata
        continue to follow the application's existing retention policy.
        """
        with self._lock,self._connect() as conn:
            conn.execute("begin immediate")
            columns={row[1] for row in conn.execute("pragma table_info(scene_observations)")}
            if not columns:
                return {"processed":0,"complete":True}
            if "source_observation_id" not in columns:
                conn.execute("alter table scene_observations add column source_observation_id text references acquired_observations(id)")
                conn.execute("create index if not exists scene_observation_source on scene_observations(source_observation_id)")
            progress=conn.execute("select cursor from scene_acquisition_migrations where name='scene_observations_v1'").fetchone()
            cursor=progress[0] if progress else 0
            rows=conn.execute("select rowid as migration_rowid,* from scene_observations where rowid>? order by rowid limit ?",
                              (cursor,max(1,min(int(batch_size),1000)))).fetchall()
            for row in rows:
                if row["source_observation_id"] and conn.execute("select 1 from acquired_observations where id=?", (row["source_observation_id"],)).fetchone():
                    continue
                raw=json.loads(row["payload_json"])
                source_id=row["source_observation_id"] or raw.get("source_observation_id") or row["id"]
                sample_id="legacy-scene:"+source_id
                conn.execute("insert or ignore into acquired_samples values(?,?,?,?,?,?,?,?,?)",(sample_id,row["camera_id"],row["captured_epoch"],
                             "historical_projection","complete",_json({"historical":True}),"","",time.time()))
                image=self._acquired_media_path(conn,row["snapshot_path"])
                conn.execute("insert or ignore into acquired_observations values(?,?,?,?,?,?)",
                             (source_id,row["camera_id"],row["captured_epoch"],row["payload_json"],image,row["recording_path"]))
                if image:
                    conn.execute("update acquired_observations set snapshot_path=? where id=? and snapshot_path=''",(image,source_id))
                conn.execute("insert or ignore into acquired_sample_observations values(?,?)",(sample_id,source_id))
                self._link_scene_acquisition(conn,sample_id=sample_id,event_id=row["event_id"])
                conn.execute("update scene_observations set source_observation_id=? where id=?",(source_id,row["id"]))
                self._register_acquired_snapshot(conn,image,row["camera_id"],row["captured_epoch"])
            if rows:
                conn.execute("insert into scene_acquisition_migrations values('scene_observations_v1',?) on conflict(name) do update set cursor=excluded.cursor",(rows[-1]["migration_rowid"],))
            return {"processed":len(rows),"complete":len(rows)<max(1,min(int(batch_size),1000))}
