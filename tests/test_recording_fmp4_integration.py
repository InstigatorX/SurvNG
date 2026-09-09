"""Real recording-muxer MP4s exercise the production HLS timestamp conversion."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
from types import SimpleNamespace

import pytest

from survng.app.recording_media import hls_map_transition, mp4_stream_fingerprint
from survng.app import recording_media_runtime
from survng.app.recording_media_runtime import RecordingMediaRuntime
from survng.app.recording_routes import create_recording_router


@pytest.fixture(scope="module")
def recording_fragments(tmp_path_factory):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg and ffprobe are required")
    root = tmp_path_factory.mktemp("recording-fmp4")
    source = root / "source.mp4"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=160x90:rate=25:duration=6", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=6", "-c:v", "libx264",
        "-g", "50", "-bf", "2", "-c:a", "aac", str(source),
    ], check=True, capture_output=True, timeout=30)
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-map", "0", "-c", "copy", "-f", "segment", "-segment_time", "2",
        "-reset_timestamps", "1", str(root / "segment-%02d.mp4"),
    ], check=True, capture_output=True, timeout=30)
    runtime = RecordingMediaRuntime(SimpleNamespace(
        get_config=lambda: SimpleNamespace(ffmpeg_path=ffmpeg),
        get_manager=lambda: SimpleNamespace(storage_dir=root),
        ffprobe_path=lambda: ffprobe,
    ))
    # Cache retention is independent of timestamp conversion and needs the full
    # application configuration. Keep actual remux, cache writes, and tfdt repair.
    runtime._maintain_recording_cache = lambda _path: None
    return root, ffmpeg, ffprobe, runtime


def packets(ffprobe, path):
    result = subprocess.run([
        ffprobe, "-v", "error", "-show_packets", "-show_streams",
        "-show_data_hash", "sha256", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True, timeout=20)
    return json.loads(result.stdout)


def fragment_packets(root, ffprobe, init, media, name):
    combined = root / name
    combined.write_bytes(init.read_bytes() + media.read_bytes())
    return packets(ffprobe, combined)


def decode_times(runtime, media):
    data = media.read_bytes()
    values = {}
    for box_type, _, payload, end in runtime._mp4_boxes(data):
        if box_type != b"moof":
            continue
        for child_type, _, child_payload, child_end in runtime._mp4_boxes(data, payload, end):
            if child_type != b"traf":
                continue
            track_id = None
            timestamp = None
            for kind, _, start, stop in runtime._mp4_boxes(data, child_payload, child_end):
                if kind == b"tfhd":
                    track_id = struct.unpack_from(">I", data, start + 4)[0]
                elif kind == b"tfdt":
                    timestamp = struct.unpack_from(">Q" if data[start] else ">I", data, start + 4)[0]
            assert track_id is not None and timestamp is not None
            values.setdefault(track_id, []).append(timestamp)
    return values


@pytest.mark.parametrize("segment_index,offset", [(0, 2.0), (1, 7.25)])
def test_fragment_offset_is_applied_once_to_every_track(recording_fragments, segment_index, offset):
    root, _ffmpeg, ffprobe, runtime = recording_fragments
    source = root / f"segment-{segment_index:02d}.mp4"
    initial_init, initial_media = runtime._recording_fmp4_files(source, 2, 0)
    shifted_init, shifted_media = runtime._recording_fmp4_files(source, 2, offset)
    timescales = runtime._mp4_track_timescales(initial_init.read_bytes())
    assert len(timescales) == 2 and len(set(timescales.values())) == 2
    initial_times, shifted_times = decode_times(runtime, initial_media), decode_times(runtime, shifted_media)
    assert initial_times.keys() == shifted_times.keys() == timescales.keys()
    for track, scale in timescales.items():
        assert shifted_times[track] == [value + round(offset * scale) for value in initial_times[track]]
    initial = fragment_packets(root, ffprobe, initial_init, initial_media, f"initial-{segment_index}.mp4")
    shifted = fragment_packets(root, ffprobe, shifted_init, shifted_media, f"shifted-{segment_index}.mp4")
    assert len(initial["packets"]) == len(shifted["packets"])
    assert any(packet["pts"] != packet["dts"] for packet in initial["packets"] if packet["stream_index"] == 0)
    for before, after in zip(initial["packets"], shifted["packets"]):
        assert after["data_hash"] == before["data_hash"]  # compressed media is copied
        assert after["stream_index"] == before["stream_index"]
        for field in ("pts_time", "dts_time"):
            assert float(after[field]) - float(before[field]) == pytest.approx(offset, abs=0.000002)


def test_remux_version_invalidates_disk_cache(recording_fragments, monkeypatch):
    root, _ffmpeg, _ffprobe, runtime = recording_fragments
    source = root / "segment-00.mp4"
    before = runtime._recording_fmp4_files(source, 2, 0)
    monkeypatch.setattr(recording_media_runtime, "RECORDING_FMP4_VERSION", recording_media_runtime.RECORDING_FMP4_VERSION + 1)
    after = runtime._recording_fmp4_files(source, 2, 0)
    assert before[0].parent != after[0].parent
    assert all(path.is_file() for path in (*before, *after))


@pytest.mark.parametrize("codec,b_frames", [("h264", 0), ("h264", 2), ("hevc", 2)])
def test_fmp4_preserves_source_av_timestamps_and_compressed_packets(recording_fragments, codec, b_frames):
    root, ffmpeg, ffprobe, runtime = recording_fragments
    encoder = "libx265" if codec == "hevc" else "libx264"
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], check=True,
        capture_output=True, text=True, timeout=10,
    ).stdout
    if encoder not in encoders:
        pytest.skip(f"FFmpeg encoder {encoder} is unavailable")
    source = root / f"av-{codec}-b{b_frames}.mp4"
    codec_options = (
        ["-x265-params", "pools=1:frame-threads=1:bframes=2:keyint=50:log-level=error", "-tag:v", "hvc1"]
        if codec == "hevc" else ["-bf", str(b_frames), "-g", "50"]
    )
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=160x90:rate=25:duration=2", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=2", "-c:v", encoder,
        "-preset", "ultrafast", *codec_options, "-c:a", "aac", str(source),
    ], check=True, capture_output=True, timeout=30)
    original = packets(ffprobe, source)
    # Check a later segment with its own init, including AAC's negative priming
    # timestamp and video decode timestamps preceding presentation for B-frames.
    init, media = runtime._recording_fmp4_files(source, 2, 10)
    remuxed = fragment_packets(root, ffprobe, init, media, f"av-{codec}-b{b_frames}-result.mp4")
    assert [stream["codec_name"] for stream in remuxed["streams"]] == [codec, "aac"]
    assert len(original["packets"]) == len(remuxed["packets"])
    for stream_index in (0, 1):
        before = [packet for packet in original["packets"] if packet["stream_index"] == stream_index]
        after = [packet for packet in remuxed["packets"] if packet["stream_index"] == stream_index]
        assert len(before) == len(after)
        if stream_index == 1:
            assert float(before[0]["pts_time"]) < 0
        if stream_index == 0 and b_frames:
            assert any(packet["pts"] != packet["dts"] for packet in before)
        for source_packet, remuxed_packet in zip(before, after):
            assert remuxed_packet["data_hash"] == source_packet["data_hash"]
            for field in ("pts_time", "dts_time"):
                assert float(remuxed_packet[field]) - float(source_packet[field]) == pytest.approx(10, abs=.000002)


def test_same_codec_video_has_continuous_pts_across_recording_gaps(recording_fragments):
    root, ffmpeg, ffprobe, runtime = recording_fragments
    rows = [{
        "name": f"segment-{index:02d}.mp4", "path": str(root / f"segment-{index:02d}.mp4"),
        "start_epoch": start, "end_epoch": start + 2, "duration_seconds": 2,
        "stream_fingerprint": mp4_stream_fingerprint(root / f"segment-{index:02d}.mp4"),
    } for index, start in enumerate((100, 102, 120))]
    assert len({row["stream_fingerprint"] for row in rows}) == 1
    manager = SimpleNamespace(camera=lambda _camera_id: object())
    dependencies = SimpleNamespace(
        manager_access=None, manager_lock=None, get_manager=lambda: manager,
        recording_day_rows=lambda *_args, **_kwargs: rows,
    )
    playlist = create_recording_router(dependencies).handlers["recording_day_hls_playlist"](
        "gate", 100, 130,
    ).body.decode()
    assert playlist.count("#EXT-X-MAP:") == 1
    assert "#EXT-X-DISCONTINUITY" not in playlist
    assert playlist.count("#EXTINF:2.000,") == 3
    assert "media_offset=2.000" in playlist and "media_offset=4.000" in playlist
    assert "1970-01-01T00:02:00+00:00" in playlist  # the wall-clock gap stays in PDT
    outputs = [runtime._recording_fmp4_files(Path(row["path"]), 2, index * 2) for index, row in enumerate(rows)]
    first_init = outputs[0][0]
    video_packets = []
    for index, (_init, media) in enumerate(outputs):
        data = fragment_packets(root, ffprobe, first_init, media, f"reused-init-{index}.mp4")
        video_packets.extend(packet for packet in data["packets"] if packet["stream_index"] == 0)
    presentation_times = sorted(float(packet["pts_time"]) for packet in video_packets)
    assert len(presentation_times) == 150
    assert [right - left for left, right in zip(presentation_times, presentation_times[1:])] == pytest.approx([.04] * 149, abs=.000002)
    combined = root / "continuous.mp4"
    combined.write_bytes(first_init.read_bytes() + b"".join(media.read_bytes() for _init, media in outputs))
    decoded = subprocess.run([
        ffmpeg, "-v", "error", "-i", str(combined), "-f", "null", "-",
    ], check=True, capture_output=True, text=True, timeout=20)
    assert not decoded.stderr


def test_changed_real_stream_description_requires_discontinuity(recording_fragments):
    root, ffmpeg, _ffprobe, _runtime = recording_fragments
    resized = root / "resized.mp4"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=192x108:rate=25:duration=2", "-c:v", "libx264", str(resized),
    ], check=True, capture_output=True, timeout=20)
    original_fingerprint = mp4_stream_fingerprint(root / "segment-00.mp4")
    changed_fingerprint = mp4_stream_fingerprint(resized)
    assert original_fingerprint and changed_fingerprint != original_fingerprint
    lines, _ = hls_map_transition(original_fingerprint, changed_fingerprint, "changed/init.mp4")
    assert lines == ["#EXT-X-DISCONTINUITY", '#EXT-X-MAP:URI="changed/init.mp4"']
