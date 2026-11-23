#!/usr/bin/env bash
# Read-only: checks one camera's newest main segments after enabling
# recording_fragmented, and whether the segment being written is readable.
# Usage: bash check-fragmented-recording.sh CAMERA_ID [main|live]
set -u
CAMERA="${1:?camera id required}"
SOURCE="${2:-main}"
ROOT="${SURVNG_ROOT:-/root/SurvNG}"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
REC_PID="$(pgrep -f survng-recorder | head -1 || true)"
FFDIR=""
[ -n "$REC_PID" ] && FFDIR="$(dirname "$(readlink -f "/proc/$REC_PID/exe" 2>/dev/null)" 2>/dev/null || true)"
FFPROBE="ffprobe"; [ -n "$FFDIR" ] && [ -x "$FFDIR/ffprobe" ] && FFPROBE="$FFDIR/ffprobe"
ROOT="$ROOT" CAMERA="$CAMERA" SOURCE="$SOURCE" FFPROBE="$FFPROBE" "$PY" - <<'PY'
import json, os, sqlite3, struct, subprocess, time
from pathlib import Path

root, camera, source, ffprobe = (os.environ[k] for k in ("ROOT", "CAMERA", "SOURCE", "FFPROBE"))
index_path = f"{root}/runtime/recording-index/recordings.sqlite3"

def boxes(path):
    found, size_total = [], os.path.getsize(path)
    with open(path, "rb") as handle:
        cursor = 0
        while cursor + 8 <= size_total and len(found) < 400:
            handle.seek(cursor)
            header = handle.read(16)
            if len(header) < 8:
                break
            size = struct.unpack(">I", header[:4])[0]
            if size == 1 and len(header) >= 16:
                size = struct.unpack(">Q", header[8:16])[0]
            if size < 8:
                break
            found.append(header[4:8].decode("latin-1"))
            if cursor + size > size_total:
                found[-1] += "(incomplete)"
                break
            cursor += size
    return found

def layout(found):
    if "moof" in found:
        if "moov" not in found:
            return f"fragmented ({found.count('moof')} fragments, NO index)"
        return f"fragmented ({found.count('moof')} fragments, moov {'first' if found.index('moov') < found.index('moof') else 'late'})"
    if "moov" in found:
        return "plain mp4"
    return "no index yet: " + ",".join(found[:4])

def probe(path):
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=codec_name,nb_read_packets:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=20)
        data = json.loads(out.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        return (f"{stream.get('codec_name','?')} packets={stream.get('nb_read_packets','?')} "
                f"duration={(data.get('format') or {}).get('duration','?')}"
                + (f" stderr={out.stderr.strip()[:120]}" if out.returncode else ""))
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return f"probe failed: {type(error).__name__}"

conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True, timeout=5)
rows = conn.execute(
    "select path, start_epoch, duration_seconds, playback_duration_seconds, playable, validated, "
    "health_error, stream_fingerprint from recordings where camera_id=? and source=? "
    "order by start_epoch desc limit 8", (camera, source)).fetchall()
conn.close()
if not rows:
    raise SystemExit(f"CHECK no indexed {source} segments for {camera}")
print(f"== newest indexed {source} segments for {camera} (newest first)")
fingerprints = set()
problems = 0
for path, start, duration, playback, playable, validated, error, fingerprint in rows:
    if not os.path.exists(path):
        print(f"  {Path(path).name}  CHECK file missing")
        problems += 1
        continue
    kind = layout(boxes(path))
    fingerprints.add(fingerprint or "(none yet)")
    flag = ""
    if not playable or error:
        flag = "  CHECK not playable: " + (error or "")
        problems += 1
    print(f"  {Path(path).name}  {kind:<44} index={duration:.2f}s exact={playback} "
          f"validated={validated} | {probe(path)}{flag}")
print("  ('exact' is filled in by index validation or first playback; None before that is normal)")
print(f"  stream fingerprints across these rows: {len(fingerprints)} "
      f"({'same stream identity' if len(fingerprints) == 1 else 'CHECK identity differs between segments'})")

print("\n== segment being written right now (not in the index)")
newest_dir = Path(rows[0][0]).parent
# Only the newest hour directory (and a just-started next hour) is listed.
day_dirs = sorted(item for item in newest_dir.parent.parent.iterdir() if item.is_dir())
hour_dirs = sorted(item for item in day_dirs[-1].iterdir() if item.is_dir()) if day_dirs else []
candidates = sorted(
    {*newest_dir.glob("*.mp4"), *(hour_dirs[-1].glob("*.mp4") if hour_dirs else [])},
    key=lambda item: item.name,
)
open_file = candidates[-1] if candidates else None
if open_file is None or str(open_file) == rows[0][0]:
    print("  no newer file than the newest indexed one; the recorder may be stopped")
else:
    for attempt in range(3):
        found = boxes(open_file)
        size = os.path.getsize(open_file)
        print(f"  {open_file.name} size={size} {layout(found)} | {probe(open_file)}")
        time.sleep(2.0)
    print("  A fragmented camera should show the fragment count and packet count rising between lines.")
    print("  A plain camera shows 'no index yet' and a failed probe until the segment closes.")
print(f"\n{'PASS' if not problems else 'CHECK'}: {problems} problem(s) in indexed segments")
raise SystemExit(1 if problems else 0)
PY
