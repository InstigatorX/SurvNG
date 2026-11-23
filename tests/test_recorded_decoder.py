"""The shared recorded-video decoder: commands, lifecycle and failure classes."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from survng.app import recorded_decoder, recorded_video_frames
from survng.app.config import CameraConfig
from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector
from survng.app.recorded_decoder import (
    FAILURE_BITSTREAM,
    FAILURE_HARDWARE_INIT,
    FAILURE_MISSING_INPUT,
    FAILURE_OTHER,
    FAILURE_SPAWN,
    FAILURE_STALLED,
    FAILURE_TIMEOUT,
    FAILURE_TIMESTAMP,
    FAILURE_UNFINALIZED_INPUT,
    DecodePlan,
    StreamingDecode,
    build_decode_command,
    classify_decoder_stderr,
    decode_recording_still,
    decoder_error_codecs,
    recorded_decode_plans,
    redacted_command,
    run_decoder,
    warn_decoder_errors,
)
from survng.app.recorded_video_frames import sampled_video_frames, video_frame_at_reference


# Captured from FFmpeg 6.1.1 and 8.1.3; the tag before "moov atom" differs.
@pytest.mark.parametrize(("stderr", "expected"), [
    (
        b"[in#0 @ 0x5564b85d6d00] Error opening input: No such file or directory\n"
        b"Error opening input file missing.mp4.\n",
        FAILURE_MISSING_INPUT,
    ),
    (
        b"[mov,mp4,m4a,3gp,3g2,mj2 @ 0x55ace6af2080] moov atom not found\n"
        b"[in#0 @ 0x55ace6af1f80] Error opening input: Invalid data found when processing input\n",
        FAILURE_UNFINALIZED_INPUT,
    ),
    (
        b"[in#0 @ 0x5558a983d000] moov atom not found\n"
        b"[in#0 @ 0x5558a983cd00] Error opening input: Invalid data found when processing input\n",
        FAILURE_UNFINALIZED_INPUT,
    ),
    (
        b"[VAAPI @ 0x55634c207bc0] No VA display found for device /dev/dri/renderD128.\n"
        b"Device creation failed: -22.\n"
        b"Failed to set value '/dev/dri/renderD128' for option 'vaapi_device': Invalid argument\n",
        FAILURE_HARDWARE_INIT,
    ),
    (
        b"[AVHWDeviceContext @ 0x55d01a7ed100] No VA display found for device /dev/dri/renderD128.\n"
        b"Failed to set value '/dev/dri/renderD128' for option 'qsv_device': Invalid argument\n",
        FAILURE_HARDWARE_INIT,
    ),
    (
        b"[h264 @ 0x5564a34c12c0] top block unavailable for requested intra mode -1\n"
        b"[h264 @ 0x5564a34c12c0] error while decoding MB 6 0, bytestream 5061\n",
        FAILURE_BITSTREAM,
    ),
    (
        b"[hevc_qsv @ 0x1] Error submitting packet to decoder: Invalid data found when processing input\n",
        FAILURE_BITSTREAM,
    ),
    (
        b"[mp4 @ 0x1] Application provided invalid, non monotonically increasing dts to muxer"
        b" in stream 0: 9000 >= 9000\n",
        FAILURE_TIMESTAMP,
    ),
    (b"Conversion failed!\n", FAILURE_OTHER),
    (b"", FAILURE_OTHER),
])
def test_stderr_is_classified_by_cause(stderr, expected):
    assert classify_decoder_stderr(stderr) == expected


def test_an_unopened_input_outranks_everything_reported_after_it():
    stderr = (
        b"[in#0 @ 0x1] moov atom not found\n"
        b"[in#0 @ 0x1] Error opening input: Invalid data found when processing input\n"
        b"[h264 @ 0x2] error while decoding MB 1 1\n"
    )
    assert classify_decoder_stderr(stderr) == FAILURE_UNFINALIZED_INPUT
    assert decoder_error_codecs(stderr) == {"h264": 1}


@pytest.mark.parametrize("mode", [None, "auto", "off"])
def test_recorded_decode_is_software_unless_hardware_is_explicit(mode):
    assert [plan.backend for plan in recorded_decode_plans(mode)] == ["cpu"]
    assert [plan.backend for plan in recorded_decode_plans(mode, scale=(64, 36))] == ["cpu"]


@pytest.mark.parametrize("mode", ["qsv", "vaapi"])
def test_hardware_attempt_is_always_followed_by_software(mode):
    hardware, cpu = recorded_decode_plans(mode)
    assert hardware.backend == "hardware" and cpu == DecodePlan("cpu")
    assert mode in hardware.input_args
    assert hardware.download_filter == "hwdownload,format=nv12"

    scaled, cpu = recorded_decode_plans(mode, scale=(64, 36))
    # scale_qsv re-times frames, so GPU scaling is VAAPI for both modes.
    assert scaled.backend == "vaapi" and cpu == DecodePlan("cpu")
    assert "qsv" not in scaled.input_args
    assert scaled.download_filter.startswith("scale_vaapi=w=64:h=36:")


def test_every_command_carries_the_thread_limits_in_their_positions():
    plan = DecodePlan("hardware", ("-hwaccel", "vaapi"), "hwdownload,format=nv12")
    command = build_decode_command(
        "ffmpeg", Path("segment.mp4"), plan=plan, loglevel="info",
        tolerate_corruption=True, seek="6", copyts=True, start_at_zero=True,
        first_video_stream=True, video_filter="select=1", passthrough_timestamps=True,
        frames=2,
    )
    split = command.index("-i")
    before, after = command[:split], command[split + 2:]
    assert before[before.index("-threads:v") + 1] == "2"
    assert before[before.index("-filter_threads") + 1] == "1"
    assert after[after.index("-threads:v") + 1] == "1"
    for option in ("-fflags", "-err_detect", "-hwaccel", "-ss", "-copyts", "-start_at_zero"):
        assert option in before, option
    assert before.index("-hwaccel") < before.index("-ss")
    assert "-nostdin" in before
    assert after[after.index("-vcodec") + 1] == "bmp"
    assert command[-1] == "pipe:1"

    raw = build_decode_command("ffmpeg", "list.ffconcat", plan=DecodePlan("cpu"),
                               concat_input=True, duration="1.5", transport="raw")
    assert raw[raw.index("-f") + 1] == "concat"
    assert "rawvideo" in raw and "bmp" not in raw
    assert "-copyts" not in raw and "-ss" not in raw
    with pytest.raises(ValueError):
        build_decode_command("ffmpeg", "x.mp4", plan=DecodePlan("cpu"), transport="mjpeg")


def test_logged_commands_mask_credentials():
    logged = redacted_command(["ffmpeg", "-i", "rtsp://admin:hunter2@camera.invalid/main"])
    assert "hunter2" not in logged and "camera.invalid" in logged


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_runner_drains_both_pipes_without_deadlock():
    code = (
        "import sys\n"
        "sys.stderr.write('e' * 400000); sys.stderr.flush()\n"
        "sys.stdout.write('o' * 400000)\n"
    )
    result = run_decoder(_python(code), backend="cpu", timeout=10)
    assert result.ok
    assert len(result.output) == 400000 and len(result.stderr) == 400000


def test_runner_kills_a_decoder_that_outlives_its_timeout():
    started = time.monotonic()
    result = run_decoder(_python("import time; time.sleep(30)"), backend="hardware", timeout=0.2)
    assert time.monotonic() - started < 5
    assert result.failure == FAILURE_TIMEOUT
    assert result.returncode not in (None, 0)
    assert result.describe() == "hardware timed out"


def test_runner_reports_a_missing_binary_instead_of_raising(tmp_path):
    result = run_decoder([str(tmp_path / "no-ffmpeg"), "-i", "x"], backend="cpu", timeout=1)
    assert result.failure == FAILURE_SPAWN and result.returncode is None
    assert "cpu [spawn]" in result.describe()


def test_runner_classifies_a_failed_exit_from_stderr():
    code = "import sys; sys.stderr.write('[in#0 @ 0x1] moov atom not found\\n'); sys.exit(183)"
    result = run_decoder(_python(code), backend="cpu", timeout=10)
    assert result.failure == FAILURE_UNFINALIZED_INPUT and result.returncode == 183
    assert "moov atom not found" in result.describe()


def test_a_hung_streaming_decoder_is_killed_and_reported_as_stalled():
    code = (
        "import sys, time\n"
        "sys.stdout.buffer.write(b'x' * 8); sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    started = time.monotonic()
    with StreamingDecode(_python(code), backend="cpu", stall_timeout=0.3) as decoder:
        assert decoder.read_exact(8) == b"x" * 8
        assert decoder.read_exact(8) == b""
        assert decoder.stalled
        result = decoder.finish()
    assert time.monotonic() - started < 5
    assert result.failure == FAILURE_STALLED
    assert "no decoder output" in result.describe()


def test_a_slow_consumer_is_not_mistaken_for_a_stalled_decoder():
    # The decoder is ready long before each read; only the consumer is slow.
    code = (
        "import sys\n"
        "for _ in range(3):\n"
        "    sys.stdout.buffer.write(b'y' * 8); sys.stdout.flush()\n"
    )
    with StreamingDecode(_python(code), backend="cpu", stall_timeout=0.2) as decoder:
        chunks = []
        for _ in range(3):
            time.sleep(0.4)
            chunks.append(decoder.read_exact(8))
        assert decoder.read_exact(8) == b""
        result = decoder.finish()
    assert chunks == [b"y" * 8] * 3
    assert not decoder.stalled and result.ok


def test_closing_a_streaming_decoder_stops_its_process():
    seen = []
    code = "import sys, time; sys.stderr.write('note\\n'); sys.stderr.flush(); time.sleep(30)"
    decoder = StreamingDecode(_python(code), backend="cpu", on_stderr_end=lambda: seen.append("end"))
    process = decoder._process
    deadline = time.monotonic() + 10
    while not decoder.stderr_tail() and time.monotonic() < deadline:
        time.sleep(0.01)
    decoder.close()
    assert process.poll() is not None
    assert seen == ["end"]
    assert decoder.stderr_tail() == b"note\n"


def test_codec_error_warnings_are_rate_limited_per_recording_directory(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(recorded_decoder, "_warning_state", {})
    clock = [1000.0]
    monkeypatch.setattr(recorded_decoder.time, "monotonic", lambda: clock[0])
    stderr = b"[h264 @ 0x1] error while decoding MB 1 1\n[h264 @ 0x1] concealing 5 errors\n"
    caplog.set_level(logging.WARNING, logger="survng.app.recorded_decoder")

    for name in ("a.mp4", "b.mp4", "c.mp4"):
        assert warn_decoder_errors("tracking", tmp_path / name, "cpu", stderr) == {"h264": 2}
    assert warn_decoder_errors("tracking", tmp_path / "a.mp4", "cpu", b"") == {}
    assert len(caplog.records) == 1
    assert "consumer=tracking" in caplog.text and "h264=2" in caplog.text

    clock[0] += recorded_decoder.DECODER_WARNING_INTERVAL_SECONDS + 1
    warn_decoder_errors("tracking", tmp_path / "d.mp4", "cpu", stderr)
    assert len(caplog.records) == 2
    assert "(suppressed=2)" in caplog.records[1].getMessage()


# --- Real FFmpeg -----------------------------------------------------------


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg unavailable")
    root = tmp_path_factory.mktemp("recorded-decoder")
    good = root / "good.mp4"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=160x120:rate=10:duration=6", "-c:v", "libx264",
        "-g", "20", "-bf", "2", "-y", str(good),
    ], check=True, capture_output=True, timeout=30)
    data = good.read_bytes()
    # A segment the recorder has not finalized has media data but no index.
    unfinalized = root / "unfinalized.mp4"
    unfinalized.write_bytes(data[: len(data) // 2])
    # Damage the coded pictures only; the container index stays intact.
    start = data.index(b"mdat") + 4096
    end = data.rindex(b"moov") - 16
    damaged = bytearray(data)
    for position in range(start, end, 211):
        damaged[position] ^= 0xA5
    corrupt = root / "corrupt.mp4"
    corrupt.write_bytes(bytes(damaged))
    return SimpleNamespace(ffmpeg=ffmpeg, good=good, unfinalized=unfinalized, corrupt=corrupt)


def _detector(ffmpeg: str) -> RecordedMotionObjectDetector:
    return RecordedMotionObjectDetector(
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        SimpleNamespace(config=SimpleNamespace()),
        SimpleNamespace(ffmpeg_path=ffmpeg, hardware_acceleration="off"),
        lambda: None,
    )


def test_still_is_the_same_lossless_frame_refinement_decodes(clips):
    still = decode_recording_still(clips.good, ffmpeg_path=clips.ffmpeg, seek_seconds=2.0)
    refined = _detector(clips.ffmpeg)._read_recorded_frame(clips.good, 2.0)
    assert still is not None and still.shape == (120, 160, 3)
    # An MJPEG round trip would differ from the lossless refinement frame.
    assert np.array_equal(still, refined)


def test_still_past_the_end_is_absent_without_a_failure_warning(clips, caplog, monkeypatch):
    monkeypatch.setattr(recorded_decoder, "_warning_state", {})
    caplog.set_level(logging.WARNING, logger="survng.app.recorded_decoder")
    assert decode_recording_still(clips.good, ffmpeg_path=clips.ffmpeg, seek_seconds=60.0) is None
    assert "recorded decode failed" not in caplog.text


def test_unfinalized_segment_is_reported_as_a_lifecycle_failure(clips, caplog, monkeypatch):
    monkeypatch.setattr(recorded_decoder, "_warning_state", {})
    caplog.set_level(logging.WARNING, logger="survng.app.recorded_decoder")
    assert decode_recording_still(
        clips.unfinalized, ffmpeg_path=clips.ffmpeg, consumer="stream-alignment",
    ) is None
    assert "consumer=stream-alignment" in caplog.text
    assert "class=unfinalized_input" in caplog.text

    command = build_decode_command(clips.ffmpeg, clips.unfinalized, plan=DecodePlan("cpu"), frames=1)
    result = run_decoder(command, backend="cpu", timeout=20)
    assert result.failure == FAILURE_UNFINALIZED_INPUT


def test_batch_timestamps_match_the_frames_they_are_paired_with(clips):
    backend = _detector(clips.ffmpeg)
    frames, processes = backend._read_recorded_frames(clips.good, [0.5, 2.25, 4.0])
    assert processes == 1 and set(frames) == {0.5, 2.25, 4.0}
    for offset, item in frames.items():
        # 10 fps source: the first frame at or after each request.
        assert item.exact_timestamp
        assert offset <= item.actual_offset < offset + 0.1 + 1e-6
        single = backend._read_recorded_frame(clips.good, item.actual_offset)
        assert np.array_equal(single, item.frame)


def test_tracking_sampler_surfaces_stream_corruption(clips, caplog, monkeypatch):
    monkeypatch.setattr(recorded_decoder, "_warning_state", {})
    caplog.set_level(logging.WARNING, logger="survng.app.recorded_decoder")
    samples = list(sampled_video_frames(
        clips.corrupt, start_epoch=100.0, sample_fps=5.0, duration_seconds=5.0,
        ffmpeg_path=clips.ffmpeg,
    ))
    # The decoder conceals the damage and keeps producing frames; before the
    # shared decoder this left no trace anywhere.
    assert samples
    assert "recorded decoder errors consumer=tracking" in caplog.text
    assert "h264=" in caplog.text


def test_sampler_and_exact_frame_agree_through_the_shared_decoder(clips):
    samples = list(sampled_video_frames(
        clips.good, start_epoch=50.0, start_offset_seconds=1.3, sample_fps=2.0,
        duration_seconds=2.0, ffmpeg_path=clips.ffmpeg, maximum_width=80,
    ))
    assert len(samples) >= 3
    for sample in samples:
        assert sample.frame.shape == (60, 80, 3)
        assert sample.captured_at == pytest.approx(50.0 + sample.reference.pts_seconds - 1.3)
        exact = video_frame_at_reference(sample.reference, ffmpeg_path=clips.ffmpeg, maximum_width=80)
        assert exact is not None and np.array_equal(exact.frame, sample.frame)


def test_tracking_sampler_gives_up_on_a_decoder_that_stops_producing(tmp_path, monkeypatch):
    # Stands in for FFmpeg blocked on storage: it starts and then never writes.
    hung = tmp_path / "hung-ffmpeg"
    hung.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    hung.chmod(0o755)
    monkeypatch.setattr(recorded_decoder, "STREAM_STALL_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(
        recorded_video_frames, "_ffprobe_video_metadata",
        lambda path, ffmpeg_path: (64, 36, 1, 90000, 0.0),
    )
    segment = tmp_path / "segment.mp4"
    segment.touch()
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="stalled"):
        list(sampled_video_frames(
            segment, start_epoch=0.0, sample_fps=1.0, duration_seconds=1.0,
            ffmpeg_path=str(hung),
        ))
    assert time.monotonic() - started < 10
    survivors = subprocess.run(
        ["pgrep", "-f", str(hung)], capture_output=True, text=True, check=False,
    ).stdout.split()
    assert [pid for pid in survivors if int(pid) != os.getpid()] == []


# --- Timestamp pairing when decoder errors tear showinfo lines ---------------


def _pairs(lines: list[bytes], tolerance: float = 0.5) -> list[tuple[int, float, bool] | None]:
    timestamps = recorded_video_frames._SampledTimestamps(tolerance)
    for line in lines:
        timestamps.line(line)
    timestamps.end()
    items = []
    while not timestamps.queue.empty():
        items.append(timestamps.queue.get_nowait())
    return items


def _frame(tap: str, n: int, pts: int, seconds: float, checksum: str | None) -> bytes:
    head = f"[showinfo@{tap} @ 0x1] n:{n:4d} pts:{pts:7d} pts_time:{seconds:<7g} fmt:yuv420p type:P "
    if checksum is None:
        return (head + "Reference 5 >= 4\n").encode()
    return (head + f"checksum:{checksum} plane_checksum:[{checksum}]\n").encode()


def test_a_source_checksum_arriving_on_a_later_line_still_pairs_exactly():
    assert _pairs([
        _frame("source", 0, 0, 0.0, "AAAA0000"),
        _frame("source", 1, 9000, 0.1, None),
        b"[h264 @ 0x2] error while decoding MB 3 4, bytestream 91\n",
        b"[showinfo@source @ 0x1] checksum:BBBB0000 plane_checksum:[BBBB0000]\n",
        _frame("sampled", 0, 0, 0.0, "AAAA0000"),
        _frame("sampled", 1, 1, 0.1, "BBBB0000"),
    ]) == [(0, 0.0, True), (9000, 0.1, True), None]


def test_a_torn_sampled_line_never_shifts_later_frames_onto_its_timestamp():
    pairs = _pairs([
        _frame("source", 0, 0, 0.0, "AAAA0000"),
        _frame("source", 1, 45000, 0.5, "BBBB0000"),
        _frame("source", 2, 90000, 1.0, "CCCC0000"),
        _frame("sampled", 0, 0, 0.0, "AAAA0000"),
        _frame("sampled", 1, 1, 0.5, None),
        _frame("sampled", 2, 2, 1.0, "CCCC0000"),
    ])
    # One timestamp per image, in order; only the torn one is inexact. The
    # previous parser skipped the torn line, giving image 1 the timestamp of
    # image 2 and leaving the last image without one.
    assert pairs == [(0, 0.0, True), (45000, 0.5, False), (90000, 1.0, True), None]


def test_a_source_frame_that_lost_its_checksum_is_paired_by_time_as_inexact():
    assert _pairs([
        _frame("source", 0, 0, 0.0, None),
        _frame("source", 1, 45000, 0.5, "BBBB0000"),
        _frame("sampled", 0, 0, 0.0, "AAAA0000"),
        _frame("sampled", 1, 1, 0.5, "BBBB0000"),
    ]) == [(0, 0.0, False), (45000, 0.5, True), None]


def test_a_sampled_frame_with_no_possible_source_ends_the_pairing():
    assert _pairs([
        _frame("source", 0, 0, 0.0, "AAAA0000"),
        _frame("sampled", 0, 0, 0.0, "AAAA0000"),
        _frame("sampled", 1, 1, 0.5, "DDDD0000"),
        _frame("sampled", 2, 2, 1.0, "EEEE0000"),
    ]) == [(0, 0.0, True), None]


def test_corrupt_stream_frames_keep_their_own_timestamps(clips):
    # Repeated because the tearing depends on FFmpeg thread timing.
    for _ in range(6):
        samples = list(sampled_video_frames(
            clips.corrupt, start_epoch=100.0, sample_fps=5.0, duration_seconds=5.0,
            ffmpeg_path=clips.ffmpeg,
        ))
        times = [sample.reference.pts_seconds for sample in samples]
        assert len(samples) == 25
        assert times == sorted(times)
        for index, seconds in enumerate(times):
            assert abs(seconds - index * 0.2) <= 0.2 + 1e-6
