"""The boundary between immutable acquisition and a human activity episode.

Only a recorded activity decision can create membership or advance activity.
Associating an object, changing a cover, and changing alert policy cannot.
"""
from __future__ import annotations

import hashlib
import json
import time

from .scenes import _epoch, _iso, _json, _objects
from ..incident_utils import DEFAULT_INCIDENT_GAP_SECONDS
from ..scene_identity import analysis_observation_ids, observation_identity
from ..scene_zone_admission import evaluate_scene_establishment


def _identity(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()[:32]


class EventStoreSceneAdmissionMixin:
    def _init_scene_admission_schema(self, conn):
        conn.executescript("""
            create table if not exists acquired_sample_episodes (
                sample_id text not null references acquired_samples(id), episode_id text not null references scene_episodes(id),
                primary key(sample_id,episode_id)
            );
            create table if not exists scene_event_establishment (
                event_id integer primary key references events(id) on delete cascade,
                decision_id text not null references scene_activity_decisions(id)
            );
            create table if not exists scene_context_projection_jobs (
                episode_id text primary key references scene_episodes(id) on delete cascade,
                anchor_event_id integer not null references events(id) on delete cascade,
                context_start real not null, context_end real not null,
                created_at real not null
            );
        """)
        columns = {row[1] for row in conn.execute("pragma table_info(scene_context_projection_jobs)")}
        if "notify" not in columns:
            conn.execute("alter table scene_context_projection_jobs add column notify integer not null default 1")

    def _acquire_event_samples(self, conn, row, objects=None):
        """Compatibility sources also acquire before incident projection."""
        row = dict(row)
        at = _epoch(row["created_at"])
        objects, candidates, _ = self._scene_candidates(row, objects)
        groups = {}
        for index, original in candidates:
            item = dict(original)
            absolute = item.get("captured_at_epoch", item.get("frame_captured_at_epoch", item.get("captured_at", item.get("snapshot_captured_at"))))
            captured = _epoch(absolute, at) if absolute is not None else at + float(item.get("offset_seconds", item.get("temporal_sample_offset_seconds", 0)) or 0)
            item["captured_at_epoch"] = captured
            if index is not None and item.get("snapshot_visible") is not False:
                item.setdefault("snapshot_path", row.get("snapshot_path") or "")
            groups.setdefault((captured, str(item.get("frame_source") or "legacy")), []).append(item)
        # A successful notice with no retained detections is still a source
        # record. It is not proof of an analyzed empty frame.
        if not groups:
            groups[(at, "event_notice")] = []
        samples = []
        frame_metadata = {}
        for item in objects:
            batches = item.get("samples", []) if item.get("status") == "scene_observations" else item.get("object_tracking", {}).get("scene_samples", []) if item.get("status") == "object_tracking" else []
            for sample in batches:
                metadata = dict(sample.get("metadata") or {})
                source = str(metadata.get("source") or sample.get("source") or "recorded_main")
                frame_metadata[(float(sample["captured_epoch"]), source)] = (sample, metadata)
                groups.setdefault((float(sample["captured_epoch"]), source), [])
        for (captured, source), observations in groups.items():
            batch, metadata = frame_metadata.get((captured, source), ({}, {}))
            sample_id = batch.get("id") or "sample-event-" + _identity([row["id"], captured, source, observations])
            # A failed retry must not inherit detections from earlier successful
            # analyses. Retain those separately below without erasing the failure.
            sample = self._acquire_scene_sample(
                conn, sample_id=sample_id, camera_id=row["camera_id"], captured_epoch=captured,
                source=source, status=batch.get("status", "complete"),
                observations=[] if batch.get("status") == "failed" else observations,
                metadata={"legacy_event_id": row["id"], "analysis_unknown": source in {"legacy", "event_notice"}, **metadata},
                recording_path=row.get("recording_path") or "",
            )
            self._link_scene_acquisition(conn, sample_id=sample_id, event_id=row["id"])
            samples.append(sample)
            # Retained events can reference an earlier analysis of this capture
            # while later tracking contributes more detections. Preserve the
            # original acquisition and append the extra analysis explicitly.
            # This also supports replay of sample IDs persisted by older code.
            retained = set(sample["observation_ids"])
            additional = [item for item in observations
                          if observation_identity(row["camera_id"], captured, item) not in retained]
            if additional:
                variant_id = "sample-analysis-" + _identity([
                    sample_id, analysis_observation_ids(row["camera_id"], captured, additional)])
                variant = self._acquire_scene_sample(
                    conn, sample_id=variant_id, camera_id=row["camera_id"], captured_epoch=captured,
                    source=source, status="complete", observations=additional,
                    metadata={"source":source,"analysis_of_sample_id":sample_id},
                    recording_path=row.get("recording_path") or "",
                )
                self._link_scene_acquisition(conn, sample_id=variant_id, event_id=row["id"])
                samples.append(variant)
        return samples

    def _scene_ingest(self, conn, row, *, historical=False, notify=True, activity=True, force_revision=False, objects=None):
        """``objects`` is the decoded ``row["objects_json"]`` when the caller has it; it is only read."""
        from ..scene_activity import evaluate_scene_activity

        row = dict(row)
        objects = _objects(row.get("objects_json")) if objects is None else objects
        qualification = next((o.get("motion_qualification", {}) for o in objects if o.get("status") == "motion_qualification"), {})
        sample_ids = qualification.get("scene_sample_ids") or []
        samples = [self._scene_sample(conn, key) for key in sample_ids]
        samples = [s for s in samples if s is not None]
        samples.extend(self._acquire_event_samples(conn, row, objects))
        sample_ids = list(dict.fromkeys(s["id"] for s in samples))
        at = _epoch(row["created_at"])
        decision_id = qualification.get("scene_activity_decision_id")
        decision = self._activity_payload(conn.execute("select * from scene_activity_decisions where id=?", (decision_id,)).fetchone()) if decision_id else None
        membership = conn.execute("select p.* from scene_episodes p join scene_event_membership m on m.episode_id=p.id where m.event_id=?", (row["id"],)).fetchone()
        previous_row = conn.execute("select d.* from scene_activity_decisions d join scene_event_establishment x on x.decision_id=d.id where x.event_id=?", (row["id"],)).fetchone()
        previous = self._activity_payload(previous_row) if previous_row else None
        if previous and (decision is None or (previous.get("activity_epoch") or 0) >= (decision.get("activity_epoch") or 0)):
            decision = previous
            decision_id = decision["id"]
        policy = qualification.get("establishment_zone_policy")
        if historical:
            # Historical import keeps the decision recorded with the evidence.
            # Today's zone configuration must not reopen or reject those episodes.
            assessment = evaluate_scene_activity(samples)
            if assessment["status"] != "supported" and not qualification.get("scene_discovery") and not qualification.get("scene_confirmation"):
                camera_notice = qualification.get("trigger_source", "camera") == "camera"
                assessment = {"status": "supported", "reason": "admitted_camera_notice",
                              "summary": "Historical camera activity; original verification is unavailable.", "activity_epoch": at,
                              "supporting_observation_ids": [], "policy_version": 1,
                              "evidence_kind": "camera_reported", "historical_unverified": True}
            if qualification.get("scene_discovery") and assessment["status"] == "pending":
                assessment.update(status="incomplete", reason="historical_activity_unverified",
                                  summary="Activity could not be established from retained historical observations.")
        else:
            notice = None
            if not qualification.get("scene_discovery") and not qualification.get("scene_confirmation"):
                notice = {"source": "camera" if qualification.get("trigger_source", "camera") == "camera" else "motion", "epoch": at}
            memory = None
            if callable(getattr(type(self), "scene_context_snapshot", None)):
                memory = self.scene_context_snapshot(conn, str(row.get("camera_id") or ""))
            created = row.get("created_at")
            event_key = created.isoformat() if hasattr(created, "isoformat") else str(created or "")
            assessment = evaluate_scene_establishment(
                samples,
                policy=policy if isinstance(policy, dict) else None,
                notice=notice,
                scene_context_memory=memory,
                event_key=event_key,
                observed_at_epoch=at,
            )
        newer_activity = assessment["status"] == "supported" and (
            decision is None or decision["verdict"] != "supported" or (assessment.get("activity_epoch") or 0) > (decision.get("activity_epoch") or 0))
        if decision is None or (newer_activity and activity and not historical):
            decision_id = "activity-event-" + _identity([row["id"], assessment, sample_ids])
            decision = self._record_scene_activity_decision(
                conn, decision_id=decision_id, sample_ids=sample_ids, verdict=assessment["status"],
                activity_epoch=assessment.get("activity_epoch"), reason=assessment["reason"],
                policy_version=assessment.get("policy_version", 1), evidence=assessment,
            )
        if decision["camera_id"] != row["camera_id"]:
            raise ValueError("activity decision belongs to another camera")
        candidate = None
        if qualification.get("scene_candidate_id") and membership is None and not historical:
            candidate = self.assert_scene_candidate_lease(conn, qualification["scene_candidate_id"],
                lease_owner=qualification["scene_candidate_lease_owner"], lease_token=qualification["scene_candidate_lease_token"])
        supported = decision["verdict"] == "supported"
        if membership is None and not supported:
            # Existing raw event links may exist from older deployments; they
            # remain queryable acquisitions without manufacturing an episode.
            # Observations that fall inside an already-open episode stay in that
            # scene, but ineligible activity does not move its activity clock.
            conn.execute("insert into scene_event_establishment values(?,?) on conflict(event_id) do update set decision_id=excluded.decision_id", (row["id"], decision_id))
            self._attach_open_episode_context(conn, row, samples)
            return None
        if supported:
            prior = conn.execute("select * from scene_activity_admissions where decision_id=?", (decision_id,)).fetchone()
            if prior and prior["event_id"] != row["id"]:
                raise ValueError("activity decision already has an event")
        previous_revision = conn.execute("select revision from scene_incidents where id=?", (membership["incident_id"],)).fetchone()[0] if membership else 0
        incident_id = self._scene_project(
            conn, row, historical=historical, notify=False, activity=supported and activity,
            activity_epoch=decision.get("activity_epoch") if supported and activity else None,
            force_revision=force_revision, defer_alerts=force_revision, objects=objects,
            evidence_notify=notify,
        )
        alerts_pending = force_revision
        episode = conn.execute("select p.* from scene_episodes p join scene_event_membership m on m.episode_id=p.id where m.event_id=?", (row["id"],)).fetchone()
        conn.execute("insert into scene_event_establishment values(?,?) on conflict(event_id) do update set decision_id=excluded.decision_id", (row["id"], decision_id))
        if supported:
            self._admit_scene_activity(conn, decision_id=decision_id, event_id=row["id"], episode_id=episode["id"])
            if candidate:
                conn.execute("insert into scene_candidate_admissions values(?,?,?,?)",
                             (candidate["id"],candidate["lease_generation"],decision_id,row["id"]))
                conn.execute("update scene_candidate_jobs set decision_id=? where id=?", (decision_id,candidate["id"]))
        for key in sample_ids:
            self._link_scene_acquisition(conn, sample_id=key, event_id=row["id"])
        # The complete bounded scene joins the episode, regardless of whether
        # these observations helped establish it or qualify for an alert.
        context_start = min(episode["start_epoch"], min(s["captured_epoch"] for s in samples))
        context_end = max(episode["end_epoch"], max(s["captured_epoch"] for s in samples))
        ids = conn.execute("select s.id from acquired_samples s where s.camera_id=? and s.captured_epoch>=? and s.captured_epoch<=? "
                           "and not exists(select 1 from acquired_sample_episodes a where a.sample_id=s.id and a.episode_id=?) "
                           "order by s.captured_epoch,s.id limit 200", (row["camera_id"],context_start,context_end,episode["id"])).fetchall()
        if ids:
            if self._project_acquired_context(
                conn, episode, row, self._sample_payloads(conn, [key[0] for key in ids]), notify=notify
            ):
                alerts_pending = False
        if len(ids) == 200:
            conn.execute(
                "insert into scene_context_projection_jobs "
                "(episode_id,anchor_event_id,context_start,context_end,created_at,notify) values(?,?,?,?,?,?) "
                "on conflict(episode_id) do update set "
                "context_start=min(context_start,excluded.context_start),"
                "context_end=max(context_end,excluded.context_end),"
                "anchor_event_id=excluded.anchor_event_id,"
                "notify=max(notify,excluded.notify)",
                (episode["id"], row["id"], context_start, context_end, time.time(), int(notify)),
            )
        if alerts_pending:
            self._scene_refresh_alerts(conn, row["id"])
        self._refresh_scene_establishment(conn, incident_id)
        revision = conn.execute("select revision from scene_incidents where id=?", (incident_id,)).fetchone()[0]
        if revision != previous_revision:
            self._scene_changed(conn, incident_id, notify=notify and supported)
        return incident_id

    def project_pending_scene_context(self) -> int:
        """Project one durable context page without monopolizing the DB lock."""
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            job = conn.execute(
                "select * from scene_context_projection_jobs order by created_at,episode_id limit 1"
            ).fetchone()
            if job is None:
                return 0
            episode = conn.execute(
                "select * from scene_episodes where id=?", (job["episode_id"],),
            ).fetchone()
            anchor = conn.execute(
                "select * from events where id=?", (job["anchor_event_id"],),
            ).fetchone()
            if episode is None or anchor is None:
                conn.execute(
                    "delete from scene_context_projection_jobs where episode_id=?",
                    (job["episode_id"],),
                )
                return 0
            ids = conn.execute(
                "select s.id from acquired_samples s where s.camera_id=? "
                "and s.captured_epoch>=? and s.captured_epoch<=? and not exists("
                "select 1 from acquired_sample_episodes a where a.sample_id=s.id and a.episode_id=?) "
                "order by s.captured_epoch,s.id limit 200",
                (episode["camera_id"], job["context_start"], job["context_end"], episode["id"]),
            ).fetchall()
            before = int(conn.execute(
                "select revision from scene_incidents where id=?", (episode["incident_id"],),
            ).fetchone()[0])
            if ids:
                self._project_acquired_context(
                    conn, episode, anchor,
                    self._sample_payloads(conn, [row[0] for row in ids]),
                    notify=bool(job["notify"]),
                )
            if len(ids) < 200:
                conn.execute(
                    "delete from scene_context_projection_jobs where episode_id=?",
                    (episode["id"],),
                )
            self._refresh_scene_establishment(conn, episode["incident_id"])
            after = int(conn.execute(
                "select revision from scene_incidents where id=?", (episode["incident_id"],),
            ).fetchone()[0])
            if after != before:
                self._scene_changed(conn, episode["incident_id"], notify=bool(job["notify"]))
            return len(ids)

    def _project_acquired_context(self, conn, episode, anchor, samples, *, notify=True):
        evidence = []
        for sample in samples:
            if conn.execute("select 1 from acquired_sample_episodes where sample_id=? and episode_id=?", (sample["id"],episode["id"])).fetchone():
                continue
            conn.execute("insert into acquired_sample_episodes values(?,?)", (sample["id"],episode["id"]))
            self._link_scene_acquisition(conn, sample_id=sample["id"], event_id=anchor["id"])
            for observation in sample["observations"]:
                # Capture clocks may retain more precision than a detector's
                # timestamp. Source identity belongs to the retained observation;
                # projection must not replace that timestamp with the sample's.
                evidence.append({**observation, "captured_at_epoch": observation.get("captured_at_epoch",
                                 observation.get("frame_captured_at_epoch", sample["captured_epoch"])),
                                 "image": sample.get("metadata", {}).get("image")})
        if evidence:
            projected = dict(anchor)
            projected["objects_json"] = _json([{"status": "scene_observations", "observations": evidence}])
            self._scene_project(conn, projected, activity=False, notify=False, target_episode_id=episode["id"], context_only=True,
                                evidence_notify=notify)
        return bool(evidence)

    def _attach_open_episode_context(self, conn, row, samples):
        """Keep in-window observations without prolonging the episode."""
        for sample in samples:
            captured = sample.get("captured_epoch")
            if captured is None:
                continue
            episodes = conn.execute(
                "select p.* from scene_episodes p join scene_incidents i on i.id=p.incident_id "
                "where i.state!='unconfirmed' and p.camera_id=? and p.boundary_locked=0 "
                "and p.start_epoch<=? and p.last_activity_epoch+?>=?",
                (row["camera_id"], captured, DEFAULT_INCIDENT_GAP_SECONDS, captured),
            ).fetchall()
            for episode in episodes:
                anchor = conn.execute(
                    "select e.* from events e join scene_event_membership m on m.event_id=e.id "
                    "where m.episode_id=? order by e.id limit 1",
                    (episode["id"],),
                ).fetchone()
                if anchor:
                    self._project_acquired_context(conn, episode, anchor, [sample])

    def associate_scene_samples(self, sample_ids):
        """Attach context to existing bounds; never create or extend activity."""
        updates = {}
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            for sample_id in sample_ids:
                sample = self._scene_sample(conn, sample_id)
                if not sample:
                    continue
                episodes = conn.execute("select p.* from scene_episodes p join scene_incidents i on i.id=p.incident_id "
                                        "where i.state!='unconfirmed' and p.camera_id=? and p.start_epoch<=? "
                                        "and p.last_activity_epoch+45>=?", (sample["camera_id"], sample["captured_epoch"], sample["captured_epoch"])).fetchall()
                for episode in episodes:
                    anchor = conn.execute("select e.* from events e join scene_event_membership m on m.event_id=e.id where m.episode_id=? order by e.id limit 1", (episode["id"],)).fetchone()
                    if anchor:
                        before = conn.execute("select revision from scene_incidents where id=?", (episode["incident_id"],)).fetchone()[0]
                        self._project_acquired_context(conn, episode, anchor, [sample])
                        revision = conn.execute("select revision from scene_incidents where id=?", (episode["incident_id"],)).fetchone()[0]
                        if revision != before:
                            updates[episode["incident_id"]] = {"incident_id":episode["incident_id"], "revision":revision, "event_id":anchor["id"], "camera_id":sample["camera_id"]}
        return list(updates.values())

    def _refresh_scene_establishment(self, conn, incident_id):
        supported = conn.execute("select 1 from scene_event_establishment x join scene_event_membership m on m.event_id=x.event_id "
                                 "join scene_episodes p on p.id=m.episode_id join scene_activity_decisions d on d.id=x.decision_id "
                                 "where p.incident_id=? and d.verdict='supported' limit 1", (incident_id,)).fetchone()
        if not supported:
            legacy_notice = conn.execute("select 1 from events e join scene_event_membership m on m.event_id=e.id join scene_episodes p on p.id=m.episode_id "
                "where p.incident_id=? and e.topic!='scene/discovery' and not exists(select 1 from scene_event_establishment x where x.event_id=e.id) limit 1", (incident_id,)).fetchone()
            if legacy_notice:
                return
            before = conn.execute("select state,revision from scene_incidents where id=?", (incident_id,)).fetchone()
            conn.execute("update scene_incidents set state='unconfirmed' where id=?", (incident_id,))
            conn.execute("delete from scene_notification_outbox where incident_id=?", (incident_id,))
            if before and before["state"] != "unconfirmed":
                conn.execute("insert into scene_corrections(incident_id,revision,payload_json,created_at) values(?,?,?,?)",
                    (incident_id,before["revision"]+1,_json({"operation":"establishment_reassessment", "previous_state":before["state"], "state":"unconfirmed", "policy_version":1}),time.time()))
                self._release_scene_facets(conn, incident_id)
        else:
            restored = conn.execute(
                "update scene_incidents set state='complete' where id=? and state='unconfirmed'",
                (incident_id,),
            ).rowcount
            if restored:
                self._retain_scene_facets(conn, incident_id)

    def _scene_establishment_payload(self, row):
        if row is None:
            return {"status": "unverified", "reason": "legacy_evidence", "summary": "Historical evidence; activity verification is unavailable.", "supporting_observation_ids": []}
        decision = self._activity_payload(row); evidence = decision["evidence"]
        return {"status": "unverified" if evidence.get("historical_unverified") else {"supported":"established", "unsupported":"not_established", "pending":"unverified", "incomplete":"incomplete"}[decision["verdict"]],
                "reason": decision["reason"], "summary": evidence.get("summary") or "Activity was not established from these observations.",
                "supporting_observation_ids": evidence.get("supporting_observation_ids", []), "policy_version": decision["policy_version"],
                "zone_interpretation": evidence.get("zone_interpretation"),
                "physical_evidence": evidence.get("physical_evidence")}

    def _scene_establishment(self, conn, incident_id):
        row = conn.execute("select d.* from scene_activity_decisions d join scene_event_establishment x on x.decision_id=d.id "
                           "join scene_event_membership m on m.event_id=x.event_id join scene_episodes p on p.id=m.episode_id "
                           "where p.incident_id=? order by (d.verdict='supported') desc,d.created_at limit 1", (incident_id,)).fetchone()
        return self._scene_establishment_payload(row)

    def migrate_scene_establishment(self, *, batch_size=100):
        """Reclassify legacy discovery without inference, alerts, or deleting links."""
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            progress = conn.execute("select cursor from scene_acquisition_migrations where name='discovery_establishment_v1'").fetchone()
            cursor = progress[0] if progress else 0
            rows = conn.execute("select e.* from events e where e.id>? and e.topic='scene/discovery' "
                                "and not exists(select 1 from scene_event_establishment x where x.event_id=e.id) order by e.id limit ?", (cursor,max(1,min(batch_size,500)))).fetchall()
            for row in rows:
                # Membership and acquisition were already imported. Replaying
                # object projection here would rewrite evidence merely to
                # classify its establishment, with quadratic history work.
                sample_ids = [r[0] for r in conn.execute("select sample_id from acquired_sample_events where event_id=? order by sample_id", (row["id"],))]
                if not sample_ids:
                    sample_ids = [s["id"] for s in self._acquire_event_samples(conn,row)]
                decision_id = "historical-activity-" + str(row["id"])
                self._record_scene_activity_decision(conn,decision_id=decision_id,sample_ids=sample_ids,
                    verdict="incomplete",activity_epoch=None,reason="historical_activity_unverified",policy_version=1,
                    evidence={"summary":"Activity could not be established from retained historical observations.",
                              "supporting_observation_ids":[],"historical_unverified":True})
                conn.execute("insert into scene_event_establishment values(?,?)", (row["id"],decision_id))
            incident_ids = {r[0] for row in rows for r in conn.execute("select p.incident_id from scene_event_membership m join scene_episodes p on p.id=m.episode_id where m.event_id=?", (row["id"],))}
            for incident_id in incident_ids:
                self._refresh_scene_establishment(conn,incident_id)
                self._scene_changed(conn,incident_id,notify=False)
                incident = conn.execute("select * from scene_incidents where id=? and state='unconfirmed'", (incident_id,)).fetchone()
                if not incident:
                    continue
                # Archived records remain discoverable without queuing any
                # historical inference. Explicit reanalysis is a separate act.
                sources = conn.execute("select distinct a.sample_id from acquired_sample_events a join scene_event_membership m on m.event_id=a.event_id join scene_episodes p on p.id=m.episode_id where p.incident_id=? order by a.sample_id", (incident_id,)).fetchall()
                decision = conn.execute("select x.decision_id,e.camera_id from scene_event_establishment x join events e on e.id=x.event_id join scene_event_membership m on m.event_id=x.event_id join scene_episodes p on p.id=m.episode_id where p.incident_id=? order by e.id desc limit 1", (incident_id,)).fetchone()
                if not sources or not decision:
                    continue
                now = time.time()
                conn.execute("insert into scene_candidate_jobs(id,camera_id,start_epoch,end_epoch,seed_sample_ids_json,deadline_epoch,available_at_epoch,state,reason,decision_id,created_at,updated_at) "
                    "values(?,?,?,?,?,?,?,'incomplete','historical_activity_unverified',?,?,?) on conflict(id) do update set seed_sample_ids_json=excluded.seed_sample_ids_json,decision_id=excluded.decision_id,updated_at=excluded.updated_at",
                    ("review-legacy-"+incident_id,decision["camera_id"],incident["start_epoch"],incident["end_epoch"],_json([r[0] for r in sources]),now,now,decision["decision_id"],now,now))
            if rows:
                conn.execute("insert into scene_acquisition_migrations values('discovery_establishment_v1',?) on conflict(name) do update set cursor=excluded.cursor", (rows[-1]['id'],))
            return {"processed":len(rows), "complete":len(rows)<max(1,min(batch_size,500))}
