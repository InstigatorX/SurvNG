"""Export an incident through the production playlist/remux and MP4 builder.

Synthetic independent recordings only; no application server or camera access.
Usage: .venv/bin/python frontend/tests/fixtures/generate-incident-replay.py /tmp/replay
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from survng.app.recording_media import playback_segment_duration, mp4_stream_fingerprint
from survng.app.recording_media_runtime import RecordingMediaRuntime
from survng.app.recording_routes import create_recording_router


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    epoch = 1789500000
    rows = []
    for index, color in enumerate(("red", "lime", "blue")):
        path = root / f"recording-{index}.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color={color}:size=640x360:rate=10:duration=10",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=10",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "100", "-bf", "2", "-c:a", "aac", "-y", str(path),
        ], check=True, capture_output=True, timeout=30)
        rows.append(dict(path=str(path), name=path.name, start_epoch=epoch+index*10,
                         end_epoch=epoch+(index+1)*10, duration_seconds=10, stream_fingerprint=mp4_stream_fingerprint(path)))
    event = dict(id=1, camera_id="test", created_at=datetime.fromtimestamp(epoch, timezone.utc).isoformat(), _recording_rows=rows)
    manager = SimpleNamespace(storage_dir=root, events=SimpleNamespace(get=lambda _:event))
    runtime = RecordingMediaRuntime(SimpleNamespace(get_config=lambda:SimpleNamespace(ffmpeg_path="ffmpeg"),
        get_manager=lambda:manager, ffprobe_path=lambda:"ffprobe"))
    runtime._maintain_recording_cache = lambda _:None
    runtime._recording_storage_path = lambda value, **kwargs:Path(value)
    runtime._event_clip_vaapi_enabled = lambda _:False
    runtime._event_clip_qsv_enabled = lambda _:False
    runtime._build_event_clip(event, 0, 24, root/"test.mp4", source="live")
    playlist = create_recording_router(SimpleNamespace(manager_access=None, manager_lock=None,
        get_manager=lambda:manager, event_clip_window=lambda m,b,a:(b,a),
        recording_day_rows=lambda *args, **kwargs:rows, public_url=lambda x:x,
    )).handlers["event_stream"](1, before=0, after=24, source="live").body.decode()

    def local(uri):
        parsed = urlparse(uri)
        query = parse_qs(parsed.query)
        row = next(row for row in rows if row["name"] == unquote(parsed.path.split("/")[-2]))
        duration = playback_segment_duration(row["start_epoch"], row["duration_seconds"], epoch+24, True)
        init, media = runtime._recording_fmp4_files(Path(row["path"]), duration, float(query["media_offset"][0]))
        is_init = parsed.path.endswith("init.mp4")
        name = f"init{rows.index(row)}.mp4" if is_init else f"test{rows.index(row)}.m4s"
        shutil.copyfile(init if is_init else media, root/name)
        return name
    lines=[]
    for line in playlist.splitlines():
        if line.startswith("#EXT-X-MAP:"):
            line=re.sub(r'URI="([^"]+)"', lambda m:f'URI="{local(m[1])}"', line)
        elif line and not line.startswith("#"):
            line=local(line)
        lines.append(line)
    (root/"test.m3u8").write_text("\n".join(lines)+"\n")
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(root/"test.mp4")]))
    assert abs(float(probe["format"]["duration"])-24)<.2, probe


if __name__ == "__main__":
    generate(Path(sys.argv[1]).resolve())
