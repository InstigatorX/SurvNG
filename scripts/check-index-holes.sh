#!/usr/bin/env bash
# Read-only: looks for instants no recording-index row covers, and checks
# whether refinement jobs that waited a long time were looking at one.
# Usage: HOURS=24 bash check-index-holes.sh
set -u
ROOT="${SURVNG_ROOT:-/root/SurvNG}"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
ROOT="$ROOT" HOURS="${HOURS:-24}" STAGES="${STAGES:--0.5 0 0.5 4 4.5 8 8.5}" "$PY" - <<'PY'
import json, os, sqlite3, time
from datetime import datetime

root, hours = os.environ["ROOT"], float(os.environ["HOURS"])
offsets = [float(item) for item in os.environ["STAGES"].split()]
now = time.time()
since = now - hours * 3600
index_path = f"{root}/runtime/recording-index/recordings.sqlite3"
jobs_path = f"{root}/runtime/database/detection-jobs.sqlite3"

def ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)

def epoch(text):
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return None

def find(node, key):
    if isinstance(node, dict):
        if key in node and isinstance(node[key], dict):
            return node[key]
        for value in node.values():
            found = find(value, key)
            if found is not None:
                return found
    return None

index = ro(index_path)
# One index seek per camera; "select distinct" would scan the whole index.
cameras = [row[0] for row in index.execute(
    "with recursive c(id) as (select min(camera_id) from recordings union all "
    "select (select min(camera_id) from recordings where camera_id > c.id) from c where c.id is not null) "
    "select id from c where id is not null")]
print(f"== instants not covered by any main index row, last {hours:g} h")
print("   hole = uncovered stretch of up to 2 s between consecutive segments; gap = longer")
for camera in sorted(cameras):
    rows = index.execute(
        "select start_epoch, end_epoch from recordings where camera_id=? and source='main' "
        "and start_epoch > ? order by start_epoch", (camera, since)).fetchall()
    if len(rows) < 3:
        continue
    steps = {}
    holes, hole_seconds, gaps, gap_seconds, overlaps = 0, 0.0, 0, 0.0, 0
    worst = []
    for (start, end), (next_start, _next_end) in zip(rows, rows[1:]):
        step = round(next_start - start)
        steps[step] = steps.get(step, 0) + 1
        uncovered = next_start - end
        if uncovered > 2.0:
            gaps += 1
            gap_seconds += uncovered
        elif uncovered > 0.01:
            holes += 1
            hole_seconds += uncovered
            if len(worst) < 3:
                worst.append((start, end, next_start))
        elif uncovered < -0.01:
            overlaps += 1
    span = rows[-1][1] - rows[0][0]
    common = ", ".join(f"{step}s x{count}" for step, count in sorted(steps.items(), key=lambda item: -item[1])[:4])
    flag = "CHECK " if holes else "      "
    print(f"  {flag}{camera:<16} rows={len(rows):<6} holes={holes:<5} ({hole_seconds:7.1f}s, "
          f"{100.0 * hole_seconds / span if span > 0 else 0:5.2f}% of time)  gaps={gaps:<3} ({gap_seconds:7.1f}s)  "
          f"overlaps={overlaps:<4} steps: {common}")
    for start, end, next_start in worst:
        print(f"         e.g. segment {datetime.fromtimestamp(start):%H:%M:%S} indexed to +{end - start:.2f}s, "
              f"next starts +{next_start - start:.2f}s")

print(f"\n== refinement passes that waited 9 s or more for a recording, last {hours:g} h")
print(f"   sample offsets assumed: {offsets} (set STAGES=... to change)")
try:
    from datetime import timezone
    # updated_at is a UTC ISO string, so it compares as text; the lower bound
    # keeps the read to the window instead of the whole retention period.
    jobs = ro(jobs_path).execute(
        "select camera_id, payload_json, updated_at from detection_jobs "
        "where state='completed' and updated_at >= ?",
        (datetime.fromtimestamp(since, timezone.utc).isoformat(),))
except sqlite3.Error as error:
    jobs = []
    print("CHECK cannot read jobs database:", error)
causes = {}
shown = 0
long_waits = 0
considered = 0
for camera, payload_json, updated_at in jobs:
    updated = epoch(updated_at)
    if updated is None or updated < since:
        continue
    try:
        payload = json.loads(payload_json)
    except ValueError:
        continue
    qualification = payload.get("qualification") or {}
    if qualification.get("scene_discovery") or qualification.get("scene_confirmation"):
        continue
    phases = find(payload.get("refined_outcome"), "phases_ms") or {}
    wait = phases.get("recording_wait_ms")
    event = epoch(payload.get("event_at"))
    if event is None or not isinstance(wait, (int, float)):
        continue
    considered += 1
    if wait < 9000:
        continue
    long_waits += 1
    rows = index.execute(
        "select start_epoch, end_epoch, duration_seconds, playback_duration_seconds, playable, validated, "
        "health_error, name from recordings where camera_id=? and source='main' "
        "and start_epoch > ? and start_epoch < ? order by start_epoch", (camera, event - 45, event + 30)).fetchall()
    verdicts = []
    for offset in offsets:
        target = event + offset
        covering = [row for row in rows if row[0] <= target < row[1]]
        if covering and all(row[4] for row in covering):
            continue
        if covering:
            verdicts.append((offset, "row not playable: " + str(covering[-1][6] or "")[:40]))
            continue
        before = [row for row in rows if row[0] <= target]
        after = [row for row in rows if row[0] > target]
        if not rows:
            verdicts.append((offset, "no rows near the event (not recording / sub stream only)"))
        elif before and after and after[0][0] - before[-1][1] <= 2.0:
            verdicts.append((offset, f"index hole: segment {before[-1][7]} indexed to +{before[-1][1] - before[-1][0]:.2f}s, "
                                     f"next starts +{after[0][0] - before[-1][0]:.2f}s"))
        elif before and after:
            verdicts.append((offset, f"recording gap of {after[0][0] - before[-1][1]:.1f}s"))
        elif before:
            verdicts.append((offset, "after the newest row near the event"))
        else:
            verdicts.append((offset, "before the oldest row near the event"))
    kind = verdicts[0][1].split(":")[0].split(" of ")[0] if verdicts else "every sample is covered now (indexed late, or the wait was for a later stage)"
    causes[kind] = causes.get(kind, 0) + 1
    if shown < 15:
        shown += 1
        print(f"  {camera:<14} event {datetime.fromtimestamp(event):%m-%d %H:%M:%S} waited {wait / 1000.0:5.1f}s "
              f"workflow {float(phases.get('workflow_ms') or 0) / 1000.0:5.1f}s")
        for offset, verdict in verdicts[:3]:
            print(f"       offset {offset:+.1f}s: {verdict}")
        if not verdicts:
            print("       every sample is covered by a playable row now")
print(f"\n  {long_waits} of {considered} completed jobs had a final pass that waited 9 s or more")
for kind, count in sorted(causes.items(), key=lambda item: -item[1]):
    print(f"    {count:4d}  {kind}")
index.close()
PY
