#!/usr/bin/env bash
# Read-only: measures recorded-refinement delay, segment/GOP shape and decoder evidence.
set -u
ROOT="${SURVNG_ROOT:-/root/SurvNG}"
HOURS="${HOURS:-24}"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
REC_PID="$(pgrep -f survng-recorder | head -1 || true)"
FFDIR=""
[ -n "$REC_PID" ] && FFDIR="$(dirname "$(readlink -f "/proc/$REC_PID/exe" 2>/dev/null)" 2>/dev/null || true)"
FFPROBE="ffprobe"; [ -n "$FFDIR" ] && [ -x "$FFDIR/ffprobe" ] && FFPROBE="$FFDIR/ffprobe"
echo "== versions"
( cd "$ROOT" && git log --oneline -1 2>/dev/null )
"$FFPROBE" -version 2>/dev/null | head -1 || echo "CHECK ffprobe not found ($FFPROBE)"
echo
ROOT="$ROOT" HOURS="$HOURS" FFPROBE="$FFPROBE" "$PY" - <<'PY'
import json, os, sqlite3, subprocess, time
from datetime import datetime

root, hours, ffprobe = os.environ["ROOT"], float(os.environ["HOURS"]), os.environ["FFPROBE"]
now = time.time()

def ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)

def epoch(text):
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return None

def pct(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]

def line(label, values, unit="s"):
    if not values:
        print(f"  {label:<34} n=0")
        return
    print(f"  {label:<34} n={len(values):<5} p50={pct(values,.5):7.2f} p90={pct(values,.9):7.2f} "
          f"max={max(values):7.2f} {unit}")

def find(node, key):
    if isinstance(node, dict):
        if key in node and isinstance(node[key], dict):
            return node[key]
        for value in node.values():
            found = find(value, key)
            if found is not None:
                return found
    return None

jobs_path = f"{root}/runtime/database/detection-jobs.sqlite3"
index_path = f"{root}/runtime/recording-index/recordings.sqlite3"
print(f"== refinement delay, completed jobs in the last {hours:g} h ({jobs_path})")
cameras = set()
try:
    conn = ro(jobs_path)
    cutoff = datetime.fromtimestamp(now - hours * 3600).astimezone().isoformat()
    rows = conn.execute(
        "select camera_id, payload_json, available_at, created_at, updated_at, state, last_error "
        "from detection_jobs").fetchall()
    conn.close()
except sqlite3.Error as error:
    rows = []
    print(f"CHECK cannot read jobs database: {error}")
states = {}
errors = {}
sched, queued, total, after_ready, split = [], [], [], [], 0
phase = {}
per_camera = {}
groups = {}
final_stage = {}
attempt_counts = {}
for camera, payload_json, available_at, created_at, updated_at, state, last_error in rows:
    updated = epoch(updated_at)
    if updated is None or updated < now - hours * 3600:
        continue
    states[state] = states.get(state, 0) + 1
    if state == "failed":
        reason = (last_error or "unknown")[:70]
        errors[reason] = errors.get(reason, 0) + 1
    try:
        payload = json.loads(payload_json)
    except ValueError:
        continue
    qualification = payload.get("qualification") or {}
    if state != "completed" or not payload.get("refined_outcome"):
        continue
    if qualification.get("scene_discovery") or qualification.get("scene_confirmation"):
        continue
    event = epoch(payload.get("event_at"))
    created = epoch(created_at)
    if event is None or created is None:
        continue
    cameras.add(camera)
    resumed = bool(qualification.get("refinement_resume_stage"))
    split += resumed
    total.append(updated - event)
    per_camera.setdefault(camera, []).append(updated - event)
    queued.append(created - event)
    if not resumed:
        # available_at is the predicted "segment closed and indexed" time.
        sched.append(float(available_at) - event)
        after_ready.append(updated - float(available_at))
    phases = find(payload.get("refined_outcome"), "phases_ms") or {}
    refined = payload.get("refined_outcome") or {}
    detected = refined.get("object_detected")
    group = ("resumed" if resumed else "single pass") + (
        ", object" if detected else ", no object" if detected is False else ", undecided")
    bucket = groups.setdefault(group, {"total": [], "workflow": [], "wait": [], "tail": []})
    bucket["total"].append(updated - event)
    # Time from the last scheduled start to completion: worker queueing plus work.
    bucket["tail"].append(updated - float(available_at))
    for name, key in (("workflow", "workflow_ms"), ("wait", "recording_wait_ms")):
        if isinstance(phases.get(key), (int, float)):
            bucket[name].append(float(phases[key]) / 1000.0)
    stage = qualification.get("refinement_resume_stage")
    final_stage[stage] = final_stage.get(stage, 0) + 1
    for key in ("recording_wait_ms", "frame_decode_ms", "decode_budget_wait_ms",
                "detector_request_ms", "temporal_confirmation_wait_ms", "workflow_ms",
                "recording_batch_processes", "recording_fallback_samples",
                "recording_samples_requested", "recording_samples_decoded",
                "recording_open_segment_lookups", "recording_index_hole_lookups"):
        value = phases.get(key)
        if isinstance(value, (int, float)):
            phase.setdefault(key, []).append(float(value))
print("  job states touched in window:", dict(sorted(states.items())) or "none")
line("event -> job queued", queued)
line("event -> segment predicted ready", sched)
line("predicted ready -> job done", after_ready)
line("event -> refinement done (total)", total)
print(f"  jobs resumed for a later stage: {split} (excluded from the two 'ready' rows)")
print("  by path and result (seconds):")
for group in sorted(groups):
    bucket = groups[group]
    print(f"   {group}")
    line("    event -> done", bucket["total"])
    line("    last scheduled start -> done", bucket["tail"])
    line("    workflow of final pass", bucket["workflow"])
    line("    recording wait in final pass", bucket["wait"])
print("  stage the final pass resumed at (None = never resumed):",
      dict(sorted(final_stage.items(), key=lambda item: str(item[0]))))
if errors:
    print("  failed jobs by error:")
    for reason, count in sorted(errors.items(), key=lambda item: -item[1])[:8]:
        print(f"    {count:5d}  {reason}")
print("  phases of the final pass:")
for key in sorted(phase):
    unit = "ms" if key.endswith("_ms") else ""
    line("  " + key, phase[key], unit)
print("  event -> refinement done, by camera:")
for camera in sorted(per_camera):
    line("  " + camera, per_camera[camera])

print(f"\n== main recording segments, last hour ({index_path})")
try:
    size_gb = sum(os.path.getsize(index_path + suffix) for suffix in ("", "-wal")
                  if os.path.exists(index_path + suffix)) / 1e9
    conn = ro(index_path)
    if not cameras:
        cameras = {row[0] for row in conn.execute(
            "select distinct camera_id from recordings where start_epoch > ?", (now - 3600,))}
    print(f"  index size {size_gb:.2f} GB; newest rowid {conn.execute('select max(rowid) from recordings').fetchone()[0]}")
    for camera in sorted(cameras):
        segs = conn.execute(
            "select path, start_epoch, duration_seconds, size_bytes, modified_at from recordings "
            "where camera_id=? and source='main' and start_epoch > ? order by start_epoch",
            (camera, now - 3600)).fetchall()
        if len(segs) < 3:
            print(f"  {camera:<22} no main recording in the last hour (not recorded, or sub stream only)")
            continue
        steps = [b[1] - a[1] for a, b in zip(segs, segs[1:])]
        # mtime - filename start = how long FFmpeg really held the file open.
        held = [s[4] - s[1] for s in segs[:-1]]
        mb = sum(s[3] for s in segs) / len(segs) / 1e6
        gop = "gop n/a"
        path = segs[-2][0]
        try:
            out = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "packet=pts_time,flags:stream=codec_name,avg_frame_rate", "-of", "json", path],
                capture_output=True, text=True, timeout=20)
            data = json.loads(out.stdout or "{}")
            keys = [float(p["pts_time"]) for p in data.get("packets", [])
                    if "K" in p.get("flags", "") and p.get("pts_time") not in (None, "N/A")]
            gaps = [b - a for a, b in zip(keys, keys[1:])]
            stream = (data.get("streams") or [{}])[0]
            gop = (f"{stream.get('codec_name','?')} {stream.get('avg_frame_rate','?')}fps "
                   f"keyframes={len(keys)} gop max={max(gaps):.2f}s" if gaps
                   else f"{stream.get('codec_name','?')} keyframes={len(keys)} (one GOP per segment)" if keys
                   else "gop n/a (no packets read)")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            gop = f"gop n/a ({type(error).__name__})"
        print(f"  {camera:<22} n={len(segs):<4} step p50={pct(steps,.5):5.1f}s max={max(steps):5.1f}s "
              f"open p50={pct(held,.5):5.1f}s max={max(held):5.1f}s avg={mb:5.1f}MB  {gop}")
    conn.close()
except (sqlite3.Error, OSError) as error:
    print(f"CHECK cannot read recording index: {error}")
PY
echo
echo "== decoder evidence in the journal (current service run)"
INV="$(systemctl show survng -p InvocationID --value 2>/dev/null)"
if [ -n "$INV" ]; then
  J="$(journalctl _SYSTEMD_INVOCATION_ID="$INV" -o cat 2>/dev/null)"
  for pattern in "recorded decoder errors" "recorded decode failed" "sampled-frame decode failed" \
                 "decode skipped" "class=unfinalized_input" "class=stalled" "class=timeout" "class=hardware_init"; do
    printf '  %-32s %s\n' "$pattern" "$(printf '%s\n' "$J" | grep -c -- "$pattern")"
  done
  echo "  last 5 decoder lines:"
  printf '%s\n' "$J" | grep -E "recorded decoder errors|recorded decode failed|sampled-frame decode failed" | tail -5 | cut -c1-260 | sed 's/^/    /'
else
  echo "CHECK survng service invocation not found"
fi
echo
echo "== decode budget and refinement counters (survngctl status)"
( cd "$ROOT" && ./survngctl status 2>/dev/null ) | "$PY" -c '
import json, sys
try:
    data = json.load(sys.stdin)
except ValueError:
    print("CHECK survngctl status unavailable"); sys.exit(0)
def walk(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in ("recorded_decode", "decode_budget", "recorded_decode_budget"):
                print(" ", here, json.dumps(value)[:600])
            else:
                walk(value, here)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            walk(value, f"{path}[{index}]")
walk(data)
for camera in data.get("cameras", []) if isinstance(data.get("cameras"), list) else []:
    ref = ((camera.get("motion") or {}).get("refinement") or {})
    stages = ref.get("worker_stage_ms") or {}
    print("  %-22s queue=%s completed=%s resumed=%s failures=%s oldest_ms=%s" % (
        camera.get("id"), ref.get("refinement_queue_depth"), ref.get("refinements_completed"),
        ref.get("refinements_resumed"), ref.get("refinement_failures"), ref.get("oldest_refinement_age_ms")))
    for name, stage in stages.items():
        if isinstance(stage, dict) and stage.get("samples"):
            print("      %-13s n=%-5s p50=%s p95=%s p99=%s ms" % (
                name, stage.get("samples"), stage.get("p50_ms"), stage.get("p95_ms"), stage.get("p99_ms")))
'
