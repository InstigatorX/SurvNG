#!/usr/bin/env bash
# Read-only: watches refinement jobs change state and prints each one's timeline.
# Usage: MINUTES=10 bash trace-refinement-jobs.sh
set -u
ROOT="${SURVNG_ROOT:-/root/SurvNG}"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
ROOT="$ROOT" MINUTES="${MINUTES:-10}" INTERVAL="${INTERVAL:-0.25}" "$PY" - <<'PY'
import json, os, sqlite3, sys, time
from datetime import datetime

root = os.environ["ROOT"]
minutes, interval = float(os.environ["MINUTES"]), float(os.environ["INTERVAL"])
path = f"{root}/runtime/database/detection-jobs.sqlite3"
conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)

def epoch(text):
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return None

def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]

jobs = {}       # id -> {"camera", "event", "kind", "steps": [(seen, state, attempts, available, stage, error)]}
started = time.time()
print(f"tracing {path} for {minutes:g} min, sampling every {interval:g}s ...", flush=True)
while time.time() - started < minutes * 60:
    now = time.time()
    since = datetime.fromtimestamp(now - 5).astimezone().isoformat()
    try:
        rows = conn.execute(
            "select id, camera_id, state, attempts, available_at, updated_at, last_error, "
            "json_extract(payload_json,'$.event_at'), "
            "json_extract(payload_json,'$.qualification.refinement_resume_stage'), "
            "json_extract(payload_json,'$.qualification.scene_discovery'), "
            "json_extract(payload_json,'$.qualification.scene_confirmation'), "
            "json_extract(payload_json,'$.existing_event_id'), "
            "json_extract(payload_json,'$.refined_outcome.object_detected') "
            "from detection_jobs where state in ('queued','running') "
            "or julianday(updated_at) > julianday(?)", (since,)).fetchall()
    except sqlite3.Error as error:
        print("CHECK read failed:", error)
        time.sleep(1)
        continue
    for (job_id, camera, state, attempts, available, updated, error, event_at,
         stage, discovery, confirmation, existing, detected) in rows:
        if discovery or confirmation:
            continue
        job = jobs.setdefault(job_id, {
            "camera": camera, "event": epoch(event_at),
            "kind": "event" if existing is not None else "probe", "steps": [],
        })
        key = (state, attempts, round(float(available), 2), stage, error or "", detected)
        if job["steps"] and job["steps"][-1][1] == key:
            continue
        job["steps"].append((now, key, epoch(updated)))
    time.sleep(interval)

done = {k: v for k, v in jobs.items()
        if v["event"] and v["steps"][-1][1][0] in ("completed", "failed")
        and any(step[1][0] == "queued" for step in v["steps"])}
print(f"\n{len(jobs)} jobs seen, {len(done)} followed from queued to a terminal state\n")
claim_lag, run_time, passes, totals = [], [], [], []
shown = 0
for job_id, job in sorted(done.items(), key=lambda item: item[1]["steps"][0][0]):
    event = job["event"]
    runs = 0
    lines = []
    previous = None
    for seen, (state, attempts, available, stage, error, detected), updated in job["steps"]:
        # The row's own updated_at is exact; "seen" is only when this sampler noticed.
        at = updated if updated is not None else seen
        note = ""
        if state == "running":
            runs += 1
            if previous and previous[1][0] == "queued":
                lag = at - max(previous[1][2], previous[2] or 0)
                claim_lag.append(lag)
                note = f"claimed {lag:+.2f}s after it became due"
        elif previous and previous[1][0] == "running":
            run_time.append(at - (previous[2] or previous[0]))
            note = f"ran {run_time[-1]:.2f}s"
        lines.append(
            f"    T{at - event:+7.2f}s {state:<9} attempt={attempts} due=T{available - event:+7.2f}s "
            f"stage={stage} detected={detected} {error[:30]} {note}")
        previous = (seen, (state, attempts, available, stage, error, detected), updated)
    passes.append(runs)
    totals.append((job["steps"][-1][2] or job["steps"][-1][0]) - event)
    if shown < 25:
        shown += 1
        print(f"  {job['camera']} {job['kind']} job {job_id[:8]}")
        print("\n".join(lines))
if done:
    def line(label, values):
        if values:
            print(f"  {label:<40} n={len(values):<4} p50={pct(values,.5):7.2f} p90={pct(values,.9):7.2f} max={max(values):7.2f}")
    print("\nsummary (seconds):")
    line("event -> terminal state", totals)
    line("due -> claimed by the worker", claim_lag)
    line("one pass in the worker", run_time)
    line("passes per job (count)", [float(value) for value in passes])
PY
