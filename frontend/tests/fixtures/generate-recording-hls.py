"""Export real recording-route HLS fixtures for the browser playback checks.

Run from the repository's Python environment (application dependencies required):
    python frontend/tests/fixtures/generate-recording-hls.py --output /tmp/survng-hls
Then run the browser suite with HLS_RECORDING_FIXTURES=/tmp/survng-hls.
FFMPEG_PATH and FFPROBE_PATH can override binaries discovered on PATH. HEVC
fixtures are included when FFmpeg provides libx265; H.264 requires libx264.
Only source fixture generation encodes video. The exported HLS runs through the
same stream-copy remux and playlist generation as the recording API.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

# The script also works when invoked from outside the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from survng.app.recording_media import mp4_stream_fingerprint
from survng.app.recording_media_runtime import RecordingMediaRuntime
from survng.app.recording_routes import create_recording_router


def generate(output: Path, ffmpeg: str, ffprobe: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    sources = output / "sources"
    sources.mkdir(exist_ok=True)
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], check=True,
        capture_output=True, text=True, timeout=10,
    ).stdout
    if "libx264" not in encoders:
        raise RuntimeError("FFmpeg with libx264 is required for the HLS fixtures")
    codecs = [("h264", ["red", "lime", "blue"])]
    if "libx265" in encoders:
        codecs.append(("hevc", ["lime", "blue"]))
    for codec, colors in codecs:
        for color in colors:
            encoder = (
                ["-c:v", "libx264", "-preset", "ultrafast", "-bf", "2", "-g", "50"]
                if codec == "h264" else
                ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params",
                 "pools=1:frame-threads=1:bframes=2:keyint=50:log-level=error", "-tag:v", "hvc1"]
            )
            subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                f"color=c={color}:s=320x180:r=25:d=10", "-f", "lavfi", "-i",
                "sine=frequency=440:sample_rate=48000:duration=10", *encoder,
                "-c:a", "aac", "-movflags", "+faststart", str(sources / f"{codec}-{color}.mp4"),
            ], check=True, capture_output=True, timeout=60)
    runtime = RecordingMediaRuntime(SimpleNamespace(
        get_config=lambda: SimpleNamespace(ffmpeg_path=ffmpeg),
        get_manager=lambda: SimpleNamespace(storage_dir=output / "cache"),
        ffprobe_path=lambda: ffprobe,
    ))
    # Retention needs the full application config and is unrelated to remux QA.
    runtime._maintain_recording_cache = lambda _path: None
    catalog = {
        "h264": [("h264-red", 100), ("h264-lime", 110), ("h264-blue", 120)],
        "gap": [("h264-red", 100), ("h264-lime", 110), ("h264-blue", 150)],
        "unknown": [("h264-red", 100), ("h264-lime", 110), ("h264-blue", 120)],
    }
    if any(codec == "hevc" for codec, _colors in codecs):
        catalog.update({
            "hevc": [("hevc-lime", 100), ("hevc-blue", 110)],
            "mixed": [("h264-red", 100), ("hevc-lime", 110), ("h264-blue", 120)],
        })
    manifest = {}
    for name, selected in catalog.items():
        folder = output / name
        folder.mkdir(exist_ok=True)
        rows = [{
            "name": key + ".mp4", "path": str(sources / (key + ".mp4")),
            "start_epoch": start, "end_epoch": start + 10, "duration_seconds": 10,
            "stream_fingerprint": "" if name == "unknown" else mp4_stream_fingerprint(sources / (key + ".mp4")),
        } for key, start in selected]
        manager = SimpleNamespace(camera=lambda _camera: object())
        dependencies = SimpleNamespace(
            manager_access=None, manager_lock=None, get_manager=lambda: manager,
            recording_day_rows=lambda *_args, **_kwargs: rows,
        )
        playlist = create_recording_router(dependencies).handlers["recording_day_hls_playlist"](
            "gate", 100, 200,
        ).body.decode()

        def local_uri(uri: str) -> str:
            parsed = urlparse(uri)
            source_name = unquote(parsed.path.split("/")[-2])
            suffix = parsed.path.split("/")[-1]
            index = next(index for index, row in enumerate(rows) if row["name"] == source_name)
            offset = float(parse_qs(parsed.query)["media_offset"][0])
            init, media = runtime._recording_fmp4_files(Path(rows[index]["path"]), 10, offset)
            local = f"{index}-{suffix}"
            shutil.copyfile(init if suffix == "init.mp4" else media, folder / local)
            return local

        lines = []
        for line in playlist.splitlines():
            if line.startswith("#EXT-X-MAP:"):
                line = re.sub(r'URI="([^"]+)"', lambda match: f'URI="{local_uri(match[1])}"', line)
            elif line and not line.startswith("#"):
                line = local_uri(line)
            lines.append(line)
        (folder / "index.m3u8").write_text("\n".join(lines) + "\n")
        manifest[name] = {
            "playlist": name + "/index.m3u8", "duration": 10 * len(rows),
            "boundaries": [index * 10 for index in range(1, len(rows))],
            "colors": [key.split("-")[1] for key, _start in selected],
            "codecs": [key.split("-")[0] for key, _start in selected],
            "pdt_epochs": [start for _key, start in selected],
            "discontinuities": lines.count("#EXT-X-DISCONTINUITY"),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_PATH", "ffmpeg"))
    parser.add_argument("--ffprobe", default=os.environ.get("FFPROBE_PATH", "ffprobe"))
    arguments = parser.parse_args()
    exported = generate(arguments.output.resolve(), arguments.ffmpeg, arguments.ffprobe)
    print(f"Exported production recording HLS fixtures to {arguments.output.resolve()}")
    print(", ".join(f"{name}: {entry['duration']}s" for name, entry in exported.items()))
