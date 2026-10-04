"""Read models for acquisition review; these records do not imply incidents."""
from __future__ import annotations

from datetime import datetime, timezone


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


class EventStoreSceneReviewMixin:
    @staticmethod
    def _review_status_sql():
        return "case when exists(select 1 from scene_activity_admissions a where a.decision_id=j.decision_id) then 'established' when j.state='incomplete' then 'incomplete' when j.state='complete' then 'not_established' else 'pending' end"

    def _scene_review_payload(self, conn, row, *, detail):
        job = self._candidate_payload(row)
        decision = self._activity_payload(conn.execute("select * from scene_activity_decisions where id=?", (job["decision_id"],)).fetchone()) if job.get("decision_id") else None
        admission = conn.execute("select p.incident_id from scene_activity_admissions a join scene_episodes p on p.id=a.episode_id where a.decision_id=?", (job.get("decision_id"),)).fetchone()
        preserved = None
        if job["id"].startswith("review-legacy-"):
            preserved = conn.execute("select id from scene_incidents where id=?", (job["id"][len("review-legacy-"):],)).fetchone()
        status = "established" if admission else "incomplete" if job["state"] == "incomplete" else "not_established" if job["state"] == "complete" else "pending"
        summaries = {"established":"Physical activity was established from these observations.",
                     "incomplete":"Analysis is incomplete; activity remains unresolved.",
                     "not_established":"Activity was not established from these observations.",
                     "pending":"Awaiting confirmation of physical activity."}
        sample_ids = list(dict.fromkeys([*job["seed_sample_ids"], *(decision["sample_ids"] if decision else [])]))
        observations, seen, review_images = [], set(), []
        for sample_id in sample_ids:
            sample = self._scene_sample(conn, sample_id)
            if not sample:
                continue
            image = sample.get("metadata", {}).get("image") or {"source":sample["source"], "captured_at":iso(sample["captured_epoch"])}
            for raw in sample["observations"]:
                if raw["id"] in seen:
                    continue
                seen.add(raw["id"])
                observation = {key:value for key,value in raw.items() if key not in {"snapshot_path", "recording_path"}}
                observation.update(camera_id=sample["camera_id"], captured_at=iso(sample["captured_epoch"]),
                    certainty="possible", snapshot_available=bool(raw.get("snapshot_path")),
                    image=image, snapshot_url=f"/api/incidents/observations/{raw['id']}/snapshot" if raw.get("snapshot_path") else None)
                observations.append(observation)
            if sample.get("snapshot_path") and sample["source"] == "recorded_main":
                review_images.append({**image, "url":f"/api/observations/{sample_id}/snapshot", "analyzed_frame":True})
        result = {"id":job["id"], "camera_id":job["camera_id"], "captured_at":iso(job["start_epoch"]),
                  "status":status, "summary":summaries[status], "reason":job["reason"],
                  "incident_id":admission[0] if admission else preserved[0] if preserved else None,
                  "observations":observations if detail else observations[:8], "observation_count":len(observations),
                  "establishment":{"status":status, "reason":job["reason"], "summary":summaries[status]},
                  "coverage":{"state":"incomplete" if job["state"] != "complete" or (decision or {}).get("evidence",{}).get("diagnostics",{}).get("failed_sample_count") else "sampled",
                              "reason":job["reason"]}}
        if detail and review_images:
            result["review_images"] = review_images
            # A later full-resolution frame is additional evidence, not an
            # alternate rendering of a low-resolution detector observation.
            for observation in result["observations"]:
                if observation.get("image",{}).get("source") != "recorded_main":
                    observation["review_image"] = review_images[0]
        return result

    def scene_observation_reviews(self, *, start_epoch, end_epoch, camera_id="", status="", limit=25, offset=0):
        where = ["j.end_epoch>=?", "j.start_epoch<?"]
        args = [start_epoch,end_epoch]
        if camera_id:
            where.append("j.camera_id=?"); args.append(camera_id)
        if status:
            where.append(self._review_status_sql()+"=?"); args.append(status)
        clause = " and ".join(where)
        with self._connect() as conn:
            total = conn.execute("select count(*) from scene_candidate_jobs j where "+clause,args).fetchone()[0]
            rows = conn.execute("select j.* from scene_candidate_jobs j where "+clause+" order by j.start_epoch desc,j.id limit ? offset ?",(*args,max(1,min(int(limit),100)),max(0,int(offset)))).fetchall()
            return {"items":[self._scene_review_payload(conn,row,detail=False) for row in rows],"total":total}

    def scene_observation_review(self, record_id):
        with self._connect() as conn:
            row = conn.execute("select * from scene_candidate_jobs where id=?",(record_id,)).fetchone()
            return self._scene_review_payload(conn,row,detail=True) if row else None
