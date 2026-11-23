"""Fragmented recording segments: written by the recorder, read like plain ones."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from survng.app.config import CameraConfig
from survng.app.recording_media import (
    mp4_playback_metadata,
    mp4_stream_fingerprint,
    mp4_video_dimensions,
)
from survng.app.recording_process.recorder import Recorder


def _camera(**overrides) -> CameraConfig:
    return CameraConfig(
        id="gate", name="Gate", stream_url="rtsp://example.invalid/main", **overrides,
    )


def test_plain_mp4_is_the_default_and_adds_no_muxer_options():
    camera = _camera()
    assert camera.recording_fragmented is False
    assert Recorder._segment_layout_args(camera) == []


def test_a_fragmented_camera_gets_keyframe_fragments_with_a_delayed_moov():
    args = Recorder._segment_layout_args(_camera(recording_fragmented=True))
    assert args[0] == "-segment_format_options"
    options = dict(item.split("=", 1) for item in args[1].split(":"))
    assert set(options["movflags"].split("+")) - {""} == {
        "frag_keyframe", "delay_moov", "default_base_moof",
    }
    # Each finished fragment reaches the file at once instead of waiting in
    # FFmpeg's write buffer.
    assert options["flush_packets"] == "1"


def test_the_recorder_command_carries_the_layout_per_camera(tmp_path):
    recorder = Recorder("ffmpeg", tmp_path, segment_seconds=10)
    cameras = [
        _camera(recording_fragmented=True),
        CameraConfig(id="door", name="Door", stream_url="rtsp://example.invalid/door"),
    ]
    # An empty stderr ends the recorder's log reader at once.
    processes = [Mock(pid=4321, stderr=iter(())), Mock(pid=4322, stderr=iter(()))]
    for process in processes:
        process.poll.return_value = None
        process.wait.return_value = 0
    with (
        patch("survng.app.recording_process.recorder.subprocess.Popen", side_effect=processes) as popen,
        patch.object(recorder, "_audio_output_args", return_value=[]),
        patch.object(recorder, "_owned_ffmpeg_recorders", return_value={}),
        patch.object(recorder, "_kill_pid"),
        # stop() signals the process group; these pids are not ours.
        patch("survng.app.recording_process.recorder.os.killpg"),
    ):
        try:
            for camera in cameras:
                recorder.start(camera, "main")
        finally:
            for camera in cameras:
                recorder.stop(camera.id, "main")
    fragmented, plain = (call.args[0] for call in popen.call_args_list[:2])
    assert "survng_camera=gate" in fragmented and "survng_camera=door" in plain
    position = fragmented.index("-segment_format_options")
    # Muxer options belong to the output: after -segment_format, before the path.
    assert fragmented.index("-segment_format") < position == len(fragmented) - 3
    assert fragmented[position + 1].startswith("movflags=")
    assert fragmented[-1].endswith(".mp4")
    assert "-segment_format_options" not in plain
    assert len(fragmented) == len(plain) + 2


# --- Real FFmpeg -----------------------------------------------------------


@pytest.fixture(scope="module")
def segments(tmp_path_factory):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg unavailable")
    root = tmp_path_factory.mktemp("fragmented")
    source = root / "source.mkv"
    # Video with B-frames and audio, as cameras send it; a keyframe every second.
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=9",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=9",
        "-c:v", "libx264", "-g", "10", "-bf", "2", "-c:a", "aac", "-y", str(source),
    ], check=True, capture_output=True, timeout=60)
    written: dict[str, list[Path]] = {}
    for layout, camera in (
        ("plain", _camera()),
        ("fragmented", _camera(recording_fragmented=True)),
    ):
        directory = root / layout
        directory.mkdir()
        # The recorder's own segment muxer arguments, fed from a file.
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
            "-f", "segment", "-segment_time", "3", "-reset_timestamps", "1",
            "-segment_format", "mp4", *Recorder._segment_layout_args(camera),
            str(directory / "seg%02d.mp4"),
        ], check=True, capture_output=True, timeout=60)
        written[layout] = sorted(directory.glob("seg*.mp4"))
    assert len(written["plain"]) == len(written["fragmented"]) == 3
    return SimpleNamespace(ffmpeg=ffmpeg, root=root, **written)


def _probe_duration(path: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of", "csv=p=0", str(path),
    ], check=True, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())


def test_fragmented_segments_report_the_same_video_duration_as_plain(segments):
    for plain, fragmented in zip(segments.plain, segments.fragmented, strict=True):
        _fingerprint, plain_duration = mp4_playback_metadata(plain)
        _fingerprint, fragmented_duration = mp4_playback_metadata(fragmented)
        assert plain_duration == pytest.approx(3.0)
        # Before the fragment reader this was None, and playback fell back to
        # durations estimated from file names.
        assert fragmented_duration == pytest.approx(plain_duration)
        assert fragmented_duration == pytest.approx(_probe_duration(fragmented), abs=0.001)


def test_plain_and_fragmented_segments_of_one_stream_share_identity(segments):
    # A different fingerprint would insert a playback discontinuity and split
    # exports wherever a camera's layout changed.
    for plain, fragmented in zip(segments.plain, segments.fragmented, strict=True):
        assert mp4_stream_fingerprint(plain)
        assert mp4_stream_fingerprint(fragmented) == mp4_stream_fingerprint(plain)
        assert mp4_video_dimensions(fragmented) == mp4_video_dimensions(plain) == (160, 120)


def _stream_starts(path: Path) -> list[str]:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time",
        "-of", "csv=p=0", str(path),
    ], check=True, capture_output=True, text=True, timeout=30)
    return sorted(result.stdout.split())


def _joined_frames(ffmpeg: str, paths: list[Path], listing: Path) -> list[str]:
    listing.write_text("".join(f"file '{path}'\n" for path in paths))
    result = subprocess.run([
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(listing), "-ss", "1", "-t", "7", "-an", "-f", "framemd5", "-",
    ], check=True, capture_output=True, text=True, timeout=60)
    return [line for line in result.stdout.splitlines() if not line.startswith("#")]


def test_fragmented_segments_keep_each_tracks_start_offset(segments):
    # With empty_moov FFmpeg writes the index before it has seen a packet, so
    # B-frame video and audio lose their start offsets in every segment.
    offsets = set()
    for plain, fragmented in zip(segments.plain, segments.fragmented, strict=True):
        assert _stream_starts(fragmented) == _stream_starts(plain)
        offsets.update(_stream_starts(plain))
    # The source really has non-zero offsets, or this would prove nothing.
    assert any(float(item.split(",")[1]) > 0 for item in offsets)


def test_joined_playback_is_frame_identical_across_layouts(segments, tmp_path):
    plain = _joined_frames(segments.ffmpeg, segments.plain, tmp_path / "plain.ffconcat")
    fragmented = _joined_frames(segments.ffmpeg, segments.fragmented, tmp_path / "fragmented.ffconcat")
    # A camera switched part-way through: plain, then fragmented, then plain.
    mixed = _joined_frames(
        segments.ffmpeg,
        [segments.plain[0], segments.fragmented[1], segments.plain[2]],
        tmp_path / "mixed.ffconcat",
    )
    assert len(plain) == 70
    # Same frames at the same timestamps: no held frame at a segment join.
    assert fragmented == plain
    assert mixed == plain


def test_fragmented_layout_is_what_was_written(segments):
    def top_level(path: Path) -> list[bytes]:
        data = path.read_bytes()
        boxes, cursor = [], 0
        while cursor + 8 <= len(data):
            size = int.from_bytes(data[cursor:cursor + 4], "big")
            boxes.append(data[cursor + 4:cursor + 8])
            if size < 8:
                break
            cursor += size
        return boxes

    plain = top_level(segments.plain[0])
    fragmented = top_level(segments.fragmented[0])
    assert b"moof" not in plain and plain.index(b"moov") > plain.index(b"mdat")
    # The index comes first and every second of video starts a fragment (the
    # muxer may add a short one to flush trailing audio).
    assert fragmented.index(b"moov") < fragmented.index(b"moof")
    assert fragmented.count(b"moof") >= 3


def test_an_open_fragmented_segment_reports_only_its_completed_fragments(segments, tmp_path):
    complete = segments.fragmented[0].read_bytes()
    fragments = []
    cursor = 0
    while cursor + 8 <= len(complete):
        size = int.from_bytes(complete[cursor:cursor + 4], "big")
        if complete[cursor + 4:cursor + 8] == b"moof":
            fragments.append(cursor)
        cursor += size
    # FFmpeg is part-way through writing the third second's media data.
    third, fourth = fragments[2], fragments[3]
    writing = tmp_path / "open.mp4"
    writing.write_bytes(complete[: third + (fourth - third) // 2])
    _fingerprint, duration = mp4_playback_metadata(writing)
    assert duration == pytest.approx(2.0)
    assert mp4_video_dimensions(writing) == (160, 120)

    # Nothing but the initial moov yet: readable identity, no duration.
    started = tmp_path / "started.mp4"
    started.write_bytes(complete[: complete.index(b"moof") - 4])
    assert mp4_playback_metadata(started) == (mp4_stream_fingerprint(segments.fragmented[0]), None)


def test_index_validation_accepts_a_fragmented_segment(segments, tmp_path):
    recorder = Recorder(segments.ffmpeg, tmp_path, segment_seconds=3)
    duration, error = recorder._probe_recording(segments.fragmented[1])
    assert error == ""
    # Container duration includes the audio tail and B-frame lead-in.
    assert duration == pytest.approx(3.0, abs=0.5)


def test_index_resolves_exact_playback_duration_for_fragmented_rows(segments, tmp_path):
    recorder = Recorder(segments.ffmpeg, tmp_path, segment_seconds=3)
    path = segments.fragmented[2]
    stat = path.stat()
    row = {
        "path": str(path), "start_epoch": 1000.0, "duration_seconds": 3.0, "end_epoch": 1003.0,
        "size_bytes": stat.st_size, "modified_at": stat.st_mtime,
        "stream_fingerprint": "", "fingerprint_checked": 0, "playback_duration_seconds": None,
    }
    recorder.resolve_recording_playback_metadata([row])
    assert row["stream_fingerprint"] == mp4_stream_fingerprint(segments.plain[2])
    assert row["duration_seconds"] == pytest.approx(_probe_duration(path), abs=0.001)
    assert row["end_epoch"] == pytest.approx(1000.0 + row["duration_seconds"])
