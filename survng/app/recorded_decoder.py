"""The one FFmpeg decoder for recorded video.

Every consumer that reads frames back out of a recording segment builds its
command, picks its hardware attempts, runs its process and classifies its
failure here. The callers keep only what really differs between them: which
frames they select and how they map FFmpeg's timestamps onto their own.

Failure classes separate what the caller can act on:

``unfinalized_input``     the segment has no index yet (recorder lifecycle)
``missing_input``         the file is gone or unreadable (retention, storage)
``hardware_init``         the GPU decoder could not start (host configuration)
``bitstream_corruption``  the codec rejected the camera's stream
``timestamp``             the demuxer or muxer rejected the timeline
``timeout`` / ``stalled`` the process outlived its budget or stopped producing
``spawn``                 FFmpeg itself could not be started
"""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .evidence_work import (
    check_evidence_cancellation,
    evidence_wait_timeout,
    report_evidence_stage,
)
from .ffmpeg_hw import (
    RECORDED_FRAME_INPUT_THREAD_ARGS,
    RECORDED_FRAME_OUTPUT_THREAD_ARGS,
    dri_render_device,
    qsv_enabled,
    recorded_frame_hw_args,
    vaapi_enabled,
)
from .security import redact_secret_text

LOGGER = logging.getLogger(__name__)

FAILURE_SPAWN = "spawn"
FAILURE_TIMEOUT = "timeout"
FAILURE_STALLED = "stalled"
FAILURE_MISSING_INPUT = "missing_input"
FAILURE_UNFINALIZED_INPUT = "unfinalized_input"
FAILURE_HARDWARE_INIT = "hardware_init"
FAILURE_BITSTREAM = "bitstream_corruption"
FAILURE_TIMESTAMP = "timestamp"
FAILURE_OUTPUT = "output"
FAILURE_OTHER = "other"

# A streaming decoder that produces nothing for this long while its consumer
# is waiting is treated as hung and killed.
STREAM_STALL_TIMEOUT_SECONDS = 10.0
STREAM_EXIT_TIMEOUT_SECONDS = 5.0
STREAM_STDERR_TAIL_BYTES = 64 * 1024
RECORDING_STILL_TIMEOUT_SECONDS = 12.0
DECODER_WARNING_INTERVAL_SECONDS = 60.0

_CODEC_ERROR = re.compile(
    rb"\[(hevc|h264)(?:_[A-Za-z0-9]+)? @",
    re.IGNORECASE,
)
# Ordered: an input that never opened explains everything after it, and a
# hardware device that never started is not a statement about the stream.
_FAILURE_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (FAILURE_MISSING_INPUT, re.compile(
        rb"Error opening input[^\r\n]*(?:No such file or directory|Permission denied)",
        re.IGNORECASE,
    )),
    (FAILURE_UNFINALIZED_INPUT, re.compile(
        rb"moov atom not found|Error opening input[^\r\n]*Invalid data found",
        re.IGNORECASE,
    )),
    (FAILURE_HARDWARE_INIT, re.compile(
        rb"Device creation failed|No VA display found|Failed to initiali[sz]e VAAPI"
        rb"|Error initializing an? (?:internal )?MFX session"
        rb"|hwaccel initiali[sz]ation returned error|No device available for decoder"
        rb"|Failed to set value '[^']*' for option '(?:vaapi|qsv)_device'"
        rb"|Impossible to convert between the formats supported by the filter",
        re.IGNORECASE,
    )),
    (FAILURE_BITSTREAM, re.compile(
        rb"\[(?:hevc|h264)(?:_[A-Za-z0-9]+)? @[^\r\n]*"
        rb"|error while decoding|Invalid NAL unit|Error submitting packet to decoder"
        rb"|Invalid data found when processing input",
        re.IGNORECASE,
    )),
    (FAILURE_TIMESTAMP, re.compile(
        rb"non[- ]monoton\w*(?: increasing)? dts|Invalid (?:DTS|PTS)|DTS \d+ [<>] \d+"
        rb"|Timestamps are unset|pts has no value|out of order",
        re.IGNORECASE,
    )),
)


def classify_decoder_stderr(stderr: bytes) -> str:
    """Name the most specific failure class FFmpeg's stderr supports."""
    for name, pattern in _FAILURE_PATTERNS:
        if pattern.search(stderr):
            return name
    return FAILURE_OTHER


def decoder_error_codecs(stderr: bytes) -> Counter[str]:
    """Count codec-tagged stderr lines: bitstream damage the decoder reported."""
    return Counter(
        match.group(1).decode("ascii").lower()
        for match in _CODEC_ERROR.finditer(stderr)
    )


def redacted_command(command: Sequence[str]) -> str:
    """A loggable command line; credentials in any argument are masked."""
    return " ".join(redact_secret_text(part) for part in command)


@dataclass(frozen=True, slots=True)
class DecodePlan:
    """One decoder attempt. ``download_filter`` brings frames to system memory."""

    backend: str
    input_args: tuple[str, ...] = ()
    download_filter: str = ""


def recorded_decode_plans(
    hardware_acceleration: str | None,
    *,
    scale: tuple[int, int] | None = None,
) -> tuple[DecodePlan, ...]:
    """Ordered attempts for one recorded decode; software is always last.

    Recorded decode stays software unless ``qsv`` or ``vaapi`` is explicit.
    With ``scale`` the GPU also resizes, and explicit QSV runs through VAAPI:
    scale_qsv re-times frames onto a 1/fps time base, which would break exact
    source-PTS frame references.
    """
    plans: list[DecodePlan] = []
    if scale is None:
        input_args, filter_args = recorded_frame_hw_args(hardware_acceleration)
        if input_args:
            plans.append(DecodePlan(
                "hardware",
                tuple(input_args),
                filter_args[1] if len(filter_args) > 1 else "",
            ))
    elif qsv_enabled(hardware_acceleration) or vaapi_enabled(hardware_acceleration):
        width, height = scale
        plans.append(DecodePlan(
            "vaapi",
            (
                "-vaapi_device", dri_render_device(),
                "-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi",
            ),
            f"scale_vaapi=w={width}:h={height}:format=nv12,hwdownload,format=nv12",
        ))
    plans.append(DecodePlan("cpu"))
    return tuple(plans)


def build_decode_command(
    ffmpeg_path: str,
    path: Path | str,
    *,
    plan: DecodePlan,
    loglevel: str = "error",
    tolerate_corruption: bool = False,
    concat_input: bool = False,
    seek: str | None = None,
    copyts: bool = False,
    start_at_zero: bool = False,
    first_video_stream: bool = False,
    video_filter: str = "",
    duration: str | None = None,
    passthrough_timestamps: bool = False,
    frames: int | None = None,
    transport: str = "bmp",
) -> list[str]:
    """Build one recorded-frame command. Thread limits are not optional.

    ``transport`` is ``bmp`` (self-delimiting lossless images) or ``raw``
    (bgr24 frames of a size the caller already knows).
    """
    if transport not in {"bmp", "raw"}:
        raise ValueError(f"unknown recorded frame transport: {transport}")
    command = [ffmpeg_path, "-nostdin", "-hide_banner", "-loglevel", loglevel]
    if tolerate_corruption:
        command += ["-fflags", "+discardcorrupt", "-err_detect", "ignore_err"]
    command += [*RECORDED_FRAME_INPUT_THREAD_ARGS, *plan.input_args]
    if concat_input:
        command += ["-f", "concat", "-safe", "0"]
    if seek is not None:
        command += ["-ss", seek]
    if copyts:
        command.append("-copyts")
    if start_at_zero:
        command.append("-start_at_zero")
    command += ["-i", str(path)]
    if first_video_stream:
        command += ["-map", "0:v:0"]
    command += ["-an", "-sn", "-dn"]
    if video_filter:
        command += ["-vf", video_filter]
    if duration is not None:
        command += ["-t", duration]
    if passthrough_timestamps:
        command += ["-fps_mode", "passthrough"]
    if frames is not None:
        command += ["-frames:v", str(int(frames))]
    command += ["-pix_fmt", "bgr24"]
    if transport == "bmp":
        command += ["-f", "image2pipe", "-vcodec", "bmp"]
    else:
        command += ["-f", "rawvideo"]
    command += [*RECORDED_FRAME_OUTPUT_THREAD_ARGS, "pipe:1"]
    return command


@dataclass(slots=True)
class DecodeResult:
    """What one decoder process did. ``failure`` is empty on a clean exit."""

    command: list[str]
    backend: str
    returncode: int | None
    output: Any
    stderr: bytes
    failure: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.failure

    def describe(self) -> str:
        """Short reason for logs: backend, failure class and FFmpeg's own words."""
        if self.failure == FAILURE_TIMEOUT:
            return f"{self.backend} timed out"
        detail = self.error or " ".join(
            self.stderr.decode("utf-8", errors="replace").strip().splitlines()[0:2]
        )
        label = f"{self.backend} [{self.failure}]" if self.failure else self.backend
        return f"{label}: {detail[:180]}"


def _log_failed_command(result: DecodeResult) -> None:
    LOGGER.debug(
        "recorded decoder failed (%s): %s",
        result.describe(),
        redacted_command(result.command),
    )


def read_all(stream: Any) -> bytes:
    return stream.read()


def run_decoder(
    command: list[str],
    *,
    backend: str,
    timeout: float,
    stdout_reader: Callable[[Any], Any] = read_all,
) -> DecodeResult:
    """Run one bounded decoder process to completion.

    Both pipes are drained on their own threads, so a decoder that fills one
    cannot deadlock against the other. Scoped evidence work is polled for
    cancellation; preemption kills the process and propagates. The process
    never outlives this call.
    """
    check_evidence_cancellation()
    report_evidence_stage("frame_decode")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        result = DecodeResult(command, backend, None, None, b"", FAILURE_SPAWN, str(exc))
        _log_failed_command(result)
        return result
    outputs: list[Any] = []
    stderr_chunks: list[bytes] = []
    stream_errors: list[Exception] = []

    def read_stdout() -> None:
        try:
            outputs.append(stdout_reader(process.stdout))
        except Exception as exc:
            stream_errors.append(exc)

    def read_stderr() -> None:
        try:
            stderr_chunks.append(process.stderr.read())
        except Exception as exc:
            stream_errors.append(exc)

    threads = [
        threading.Thread(target=read_stdout, name="survng-decode-out", daemon=True),
        threading.Thread(target=read_stderr, name="survng-decode-err", daemon=True),
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout
    exited = False
    timed_out = False
    try:
        while True:
            check_evidence_cancellation()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                process.wait(timeout=evidence_wait_timeout(remaining))
                exited = True
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        if not exited:
            process.kill()
            process.wait()
        for thread in threads:
            thread.join()
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
    stderr = b"".join(stderr_chunks)
    result = DecodeResult(
        command,
        backend,
        process.returncode,
        outputs[0] if outputs else None,
        stderr,
    )
    if timed_out:
        result.failure = FAILURE_TIMEOUT
    elif stream_errors:
        result.failure = FAILURE_OUTPUT
        result.error = f"decoder pipe read failed: {stream_errors[0]}"
    elif process.returncode != 0:
        result.failure = classify_decoder_stderr(stderr)
    if result.failure:
        _log_failed_command(result)
    return result


class StreamingDecode:
    """A decoder whose frames are consumed while it runs.

    The consumer may pause for as long as it likes between reads (FFmpeg just
    blocks on the full pipe). Only a read that itself waits longer than
    ``stall_timeout`` counts as a stall, and the process is then killed so the
    read returns. ``on_stderr_line`` sees every stderr line and returns true
    for lines it consumed; the rest are kept (bounded) for classification.
    """

    def __init__(
        self,
        command: list[str],
        *,
        backend: str,
        buffer_bytes: int = -1,
        stall_timeout: float | None = None,
        on_stderr_line: Callable[[bytes], bool] | None = None,
        on_stderr_end: Callable[[], None] | None = None,
    ) -> None:
        self.command = command
        self.backend = backend
        self.stalled = False
        self._stall_timeout = float(
            STREAM_STALL_TIMEOUT_SECONDS if stall_timeout is None else stall_timeout
        )
        self._on_stderr_line = on_stderr_line
        self._on_stderr_end = on_stderr_end
        self._tail: deque[bytes] = deque()
        self._tail_bytes = 0
        self._read_started: float | None = None
        self._closed = threading.Event()
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=buffer_bytes,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name="survng-decode-err",
            daemon=True,
        )
        self._stderr_thread.start()
        self._watchdog: threading.Thread | None = None
        if self._stall_timeout > 0:
            self._watchdog = threading.Thread(
                target=self._watch_for_stall,
                name="survng-decode-stall",
                daemon=True,
            )
            self._watchdog.start()

    def _read_stderr(self) -> None:
        stderr = self._process.stderr
        try:
            if stderr is None:
                return
            for line in iter(stderr.readline, b""):
                if self._on_stderr_line is not None and self._on_stderr_line(line):
                    continue
                self._tail.append(line)
                self._tail_bytes += len(line)
                while self._tail_bytes > STREAM_STDERR_TAIL_BYTES and len(self._tail) > 1:
                    self._tail_bytes -= len(self._tail.popleft())
        except (OSError, ValueError):
            # The pipe was closed under the reader during teardown.
            pass
        finally:
            if self._on_stderr_end is not None:
                self._on_stderr_end()

    def _watch_for_stall(self) -> None:
        interval = max(0.01, min(0.25, self._stall_timeout / 4.0))
        while not self._closed.wait(interval):
            started = self._read_started
            if started is None or time.monotonic() - started <= self._stall_timeout:
                continue
            self.stalled = True
            try:
                self._process.kill()
            except OSError:
                pass
            return

    def read_exact(self, size: int) -> bytes:
        """Read ``size`` bytes, or fewer only at end of stream."""
        stdout = self._process.stdout
        if stdout is None:
            raise RuntimeError("recorded decoder output is unavailable")
        payload = bytearray()
        try:
            while len(payload) < size:
                self._read_started = time.monotonic()
                chunk = stdout.read(size - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
        finally:
            self._read_started = None
        return bytes(payload)

    def stderr_tail(self) -> bytes:
        return b"".join(self._tail)

    def finish(self) -> DecodeResult:
        """Wait for the exit after end of stream and classify it."""
        result = DecodeResult(self.command, self.backend, None, None, b"")
        try:
            result.returncode = self._process.wait(timeout=STREAM_EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            result.failure = FAILURE_TIMEOUT
        self._stderr_thread.join(timeout=1.0)
        result.stderr = self.stderr_tail()
        if self.stalled:
            result.failure = FAILURE_STALLED
            result.error = (
                f"no decoder output for {self._stall_timeout:g}s; process killed"
            )
        elif not result.failure and result.returncode != 0:
            result.failure = classify_decoder_stderr(result.stderr)
        if result.failure:
            _log_failed_command(result)
        return result

    def close(self) -> None:
        """Stop the process if it is still running and release its pipes."""
        self._closed.set()
        process = self._process
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._stderr_thread.join(timeout=1.0)
        if process.stderr is not None:
            process.stderr.close()
        if self._watchdog is not None:
            self._watchdog.join(timeout=1.0)

    def __enter__(self) -> StreamingDecode:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_bmp_frames(stream: Any) -> list[np.ndarray]:
    """Read and decode concatenated BMP records without buffering stdout."""
    frames: list[np.ndarray] = []

    def read_exact(size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    while True:
        header = read_exact(6)
        if not header:
            break
        if len(header) != 6 or header[:2] != b"BM":
            break
        frame_size = int.from_bytes(header[2:6], "little")
        if frame_size < 54:
            break
        encoded = header + read_exact(frame_size - len(header))
        if len(encoded) != frame_size:
            break
        frame = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is not None:
            frames.append(frame)
    return frames


def decode_recording_still(
    path: Path,
    *,
    ffmpeg_path: str,
    seek_seconds: float = 1.0,
    consumer: str = "still",
) -> np.ndarray | None:
    """Decode one lossless still from a closed recording segment."""
    if not ffmpeg_path or not path.is_file():
        return None
    plan = DecodePlan("cpu")
    command = build_decode_command(
        ffmpeg_path,
        path,
        plan=plan,
        seek=f"{max(0.0, float(seek_seconds)):.3f}",
        frames=1,
    )
    result = run_decoder(
        command,
        backend=plan.backend,
        timeout=RECORDING_STILL_TIMEOUT_SECONDS,
    )
    warn_decoder_errors(consumer, path, plan.backend, result.stderr)
    if not result.ok:
        warn_decoder_failure(consumer, path, result)
        return None
    if not result.output:
        # A seek past the last frame exits cleanly with nothing to show.
        return None
    image = cv2.imdecode(np.frombuffer(result.output, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or getattr(image, "size", 0) <= 0:
        return None
    return image


_warning_lock = threading.Lock()
_warning_state: dict[tuple[str, str], tuple[float, int]] = {}


def _rate_limited(key: tuple[str, str]) -> tuple[bool, int]:
    """(emit now, occurrences suppressed since the last emission)."""
    now = time.monotonic()
    with _warning_lock:
        last, suppressed = _warning_state.get(key, (float("-inf"), 0))
        if now - last < DECODER_WARNING_INTERVAL_SECONDS:
            _warning_state[key] = (last, suppressed + 1)
            return False, 0
        _warning_state[key] = (now, 0)
        return True, suppressed


def warn_decoder_errors(
    consumer: str,
    path: Path,
    backend: str,
    stderr: bytes,
) -> Counter[str]:
    """Report codec errors from consumers that have no decode budget.

    Continuous consumers decode a segment every few seconds, so one camera
    with a damaged stream is reported at most once a minute per recording
    directory, with the number of reports skipped in between.
    """
    codec_lines = decoder_error_codecs(stderr)
    if not codec_lines:
        return codec_lines
    emit, suppressed = _rate_limited((consumer, str(path.parent)))
    if emit:
        LOGGER.warning(
            "recorded decoder errors consumer=%s recording=%s backend=%s %s%s",
            consumer,
            path,
            backend,
            " ".join(f"{codec}={lines}" for codec, lines in sorted(codec_lines.items())),
            f" (suppressed={suppressed})" if suppressed else "",
        )
    return codec_lines


def warn_decoder_failure(consumer: str, path: Path, result: DecodeResult) -> None:
    """Report a failed decode by class, rate-limited like codec errors."""
    emit, suppressed = _rate_limited((f"{consumer}:{result.failure}", str(path.parent)))
    if emit:
        LOGGER.warning(
            "recorded decode failed consumer=%s recording=%s class=%s %s%s",
            consumer,
            path,
            result.failure,
            result.describe(),
            f" (suppressed={suppressed})" if suppressed else "",
        )
