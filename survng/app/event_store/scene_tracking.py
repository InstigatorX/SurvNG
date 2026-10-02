"""Lease-backed recorded scene analysis, independently of trigger lifetime."""
from __future__ import annotations

from datetime import datetime
import json
import math
import time


class EventStoreSceneTrackingMixin:
    def _scene_tracking_extent(self, conn, episode_id, start_epoch, end_epoch, *, pending=False):
        episode = conn.execute("select * from scene_episodes where id=?", (episode_id,)).fetchone()
        if episode is None:
            return
        coverage = json.loads(episode["coverage_json"])
        if pending:
            coverage.update(state="incomplete", reason="recorded analysis queued")
        changed = (start_epoch < episode["start_epoch"] or end_epoch > episode["end_epoch"]
                   or coverage != json.loads(episode["coverage_json"]))
        if not changed:
            return
        conn.execute("update scene_episodes set start_epoch=min(start_epoch,?),end_epoch=max(end_epoch,?),coverage_json=? where id=?",
                     (start_epoch,end_epoch,json.dumps(coverage,sort_keys=True,separators=(",",":")),episode_id))
        conn.execute("update scene_incidents set start_epoch=min(start_epoch,?),end_epoch=max(end_epoch,?) where id=?",
                     (start_epoch,end_epoch,episode["incident_id"]))
        self._scene_changed(conn,episode["incident_id"],notify=False)

    @staticmethod
    def _init_scene_tracking_schema(conn):
        conn.execute("""create table if not exists scene_analysis_jobs (
            episode_id text primary key references scene_episodes(id) on delete cascade,
            event_id integer not null references events(id) on delete cascade,
            camera_id text not null, event_epoch real not null,
            start_epoch real not null, end_epoch real not null, cursor_epoch real,
            state text not null default 'queued', lease_owner text not null default '',
            lease_expires real not null default 0, retry_at real not null default 0,
            attempts integer not null default 0, last_error text not null default ''
        )""")
        columns={r[1] for r in conn.execute("pragma table_info(scene_analysis_jobs)")}
        if "analyzed_epoch" not in columns:
            conn.execute("alter table scene_analysis_jobs add column analyzed_epoch real")
        if "coverage_gaps_json" not in columns:
            conn.execute("alter table scene_analysis_jobs add column coverage_gaps_json text not null default '[]'")
        conn.execute("create index if not exists scene_analysis_jobs_camera on scene_analysis_jobs(camera_id,state,retry_at)")

    def enqueue_scene_tracking(self, event_id, start_epoch, end_epoch):
        if not (math.isfinite(start_epoch) and math.isfinite(end_epoch) and start_epoch < end_epoch):
            raise ValueError("invalid scene analysis window")
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            event = conn.execute("select e.*,m.episode_id from events e join scene_event_membership m "
                                 "on m.event_id=e.id where e.id=?", (event_id,)).fetchone()
            if event is None:
                return None
            event_epoch = datetime.fromisoformat(event["created_at"].replace("Z", "+00:00")).timestamp()
            conn.execute("""insert into scene_analysis_jobs
                (episode_id,event_id,camera_id,event_epoch,start_epoch,end_epoch)
                values(?,?,?,?,?,?) on conflict(episode_id) do update set
                start_epoch=min(start_epoch,excluded.start_epoch),
                end_epoch=max(end_epoch,excluded.end_epoch),
                cursor_epoch=case when excluded.start_epoch<start_epoch then null else cursor_epoch end,
                lease_owner=case when excluded.start_epoch<start_epoch then '' else lease_owner end,
                lease_expires=case when excluded.start_epoch<start_epoch then 0 else lease_expires end,
                state=case when excluded.start_epoch<start_epoch then 'queued'
                    when state='running' then state
                    when cursor_epoch>=excluded.end_epoch then 'complete' else 'queued' end,
                retry_at=case when excluded.end_epoch>end_epoch then 0 else retry_at end
                """, (event["episode_id"],event_id,event["camera_id"],event_epoch,start_epoch,end_epoch))
            job = dict(conn.execute("select * from scene_analysis_jobs where episode_id=?", (event["episode_id"],)).fetchone())
            self._scene_tracking_extent(conn,job["episode_id"],job["start_epoch"],job["end_epoch"],pending=job["state"]!="complete")
            return job

    def claim_scene_tracking(self, camera_id, lease_owner, lease_seconds=60.0):
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            if conn.execute("select 1 from scene_analysis_jobs where camera_id=? and state='running' "
                            "and lease_expires>? limit 1",(camera_id,now)).fetchone():
                return None
            job = conn.execute("""select * from scene_analysis_jobs where camera_id=? and
                ((state='queued' and retry_at<=?) or (state='running' and lease_expires<=?))
                order by start_epoch,episode_id limit 1""", (camera_id,now,now)).fetchone()
            if job is None:
                return None
            conn.execute("update scene_analysis_jobs set state='running',lease_owner=?,lease_expires=?,"
                         "attempts=attempts+1 where episode_id=?",
                         (lease_owner,now+max(30.0,lease_seconds),job["episode_id"]))
            return dict(conn.execute("select * from scene_analysis_jobs where episode_id=?",(job["episode_id"],)).fetchone())

    @staticmethod
    def _scene_tracking_lease_valid(conn, event_id, tracking):
        identity = tracking.get("scene_analysis_job")
        if not isinstance(identity,dict):
            return True
        return conn.execute("select 1 from scene_analysis_jobs where episode_id=? and event_id=? "
                            "and lease_owner=? and state='running'",
                            (identity.get("episode_id"),event_id,identity.get("lease_owner"))).fetchone() is not None

    def _checkpoint_scene_tracking(self, conn, event_id, tracking):
        identity = tracking.get("scene_analysis_job")
        if not isinstance(identity,dict):
            return
        job = conn.execute("select * from scene_analysis_jobs where episode_id=? and event_id=? "
                           "and lease_owner=? and state='running'",
                           (identity.get("episode_id"),event_id,identity.get("lease_owner"))).fetchone()
        if job is None:
            return
        activity = conn.execute("select last_activity_epoch from scene_episodes where id=?",
                                (job["episode_id"],)).fetchone()
        # Measured activity extends analysis even when no further trigger or
        # notification was generated. Compute chunks retain their finite end.
        requested_end = job["end_epoch"]
        if activity and float(activity[0]) > job["event_epoch"]:
            requested_end = max(requested_end,float(activity[0])+45.0)
        cursor = job["cursor_epoch"]
        raw = tracking.get("analyzed_through")
        if raw is not None:
            try:
                analyzed = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
            except (TypeError,ValueError,OverflowError):
                analyzed = None
            if analyzed is not None and math.isfinite(analyzed):
                analyzed=min(analyzed,job["end_epoch"])
                if "scene_cursor_epoch" not in tracking:
                    cursor = max(cursor if cursor is not None else job["start_epoch"],analyzed)
                conn.execute("update scene_analysis_jobs set analyzed_epoch=max(coalesce(analyzed_epoch,?),?) where episode_id=?",(analyzed,analyzed,job["episode_id"]))
        scan_cursor=tracking.get("scene_cursor_epoch")
        if (isinstance(scan_cursor,(int,float)) and math.isfinite(scan_cursor)
                and scan_cursor >= job["start_epoch"]):
            cursor=max(cursor if cursor is not None else job["start_epoch"],min(scan_cursor,job["end_epoch"]))
        gaps=json.loads(job["coverage_gaps_json"])
        for gap in tracking.get("coverage_gaps",[]):
            if gap not in gaps:
                gaps.append(gap)
        conn.execute("update scene_analysis_jobs set coverage_gaps_json=? where episode_id=?",(json.dumps(gaps),job["episode_id"]))
        status = str(tracking.get("state") or "")
        finished = status in {"complete","interrupted","failed","skipped_capacity"}
        # A successfully consumed finite sample window can end just before its
        # requested boundary. Only that explicit completion closes this chunk.
        consumed = status == "complete" and float(tracking.get("window_end_epoch") or 0) >= requested_end
        state = "complete" if consumed else "queued" if finished else "running"
        reason = str(tracking.get("coverage_interruption") or tracking.get("completion_reason") or status)
        continuing = status == "complete" or reason == "processing_budget_exhausted"
        delay = min(60.0,2.0 ** min(int(job["attempts"]),5)) if finished and not consumed and not continuing else 0
        conn.execute("update scene_analysis_jobs set cursor_epoch=?,end_epoch=?,state=?,lease_expires=?,retry_at=?,"
                     "lease_owner=?,last_error=? where episode_id=?",
                     (cursor,requested_end,state,time.time()+60 if not finished else 0,time.time()+delay,
                      job["lease_owner"] if not finished else "",reason if not consumed else "",job["episode_id"]))
        if requested_end > job["end_epoch"]:
            self._scene_tracking_extent(conn,job["episode_id"],job["start_epoch"],requested_end,pending=True)
            return True
        return False

    def scene_track_resume(self, event_id: int) -> dict | None:
        """Last confirmed tracks from a scene job, so the next claim keeps their ids."""
        row = self.get(int(event_id))
        if row is None:
            return None
        try:
            objects = json.loads(str(row.get("objects_json") or "[]"))
        except (TypeError, ValueError):
            return None
        if not isinstance(objects, list):
            return None
        for item in objects:
            if not isinstance(item, dict) or item.get("status") != "object_tracking":
                continue
            tracking = item.get("object_tracking")
            if not isinstance(tracking, dict):
                tracking = item
            resume = tracking.get("scene_track_resume")
            if not isinstance(resume, dict):
                resume = {}
            prior_tracks = [
                track for track in tracking.get("tracks") or []
                if isinstance(track, dict)
            ]
            if not resume.get("tracks") and not prior_tracks:
                return None
            carried = dict(resume)
            # A later claim persists its own summary. Keep the replay already
            # saved so that summary cannot discard the earlier span.
            carried["prior_replay_tracks"] = prior_tracks
            carried["prior_analyzed_from"] = tracking.get("analyzed_from")
            gaps = tracking.get("coverage_gaps")
            carried["prior_coverage_gaps"] = list(gaps) if isinstance(gaps, list) else []
            return carried
        return None

    def release_scene_tracking(self, episode_id, lease_owner, error):
        with self._lock, self._connect() as conn:
            conn.execute("update scene_analysis_jobs set state='queued',lease_owner='',lease_expires=0,"
                         "retry_at=?,last_error=? where episode_id=? and lease_owner=?",
                         (time.time()+2,str(error)[:160],episode_id,lease_owner))
