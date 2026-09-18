"""Live capture through a shared DL Streamer / GStreamer supervisor process."""

from __future__ import annotations

import atexit
import json
import logging
import os
import select
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable
from urllib.parse import urlsplit

import numpy as np

from .camera_capture import (
    CAPTURE_OPEN_LOCK_POLL_SECONDS,
    CAPTURE_OPEN_TIMEOUT_MS,
    CAPTURE_PIPE_READ_CHUNK_BYTES,
    CAPTURE_READ_TIMEOUT_MS,
    CaptureHandle,
    CaptureOpenLimiter,
)
from .dlstreamer_protocol import (
    PROTOCOL_FD_ENV,
    TYPE_DETECTIONS,
    TYPE_FATAL,
    TYPE_FRAME,
    TYPE_JPEG,
    TYPE_STATUS,
    MessageReader,
    decode_frame_payload,
    decode_jpeg_payload,
    decode_json_payload,
    decode_stream_payload,
)
from survng.dlstreamer_live import STREAM_STOP_TIMEOUT_SECONDS, model_instance_id
from survng.native_deepsort import DEFAULT_DEEP_SORT_CONFIG
from .live_detections import DetectionSnapshot
from .dlstreamer_supervisor import (
    DLSTREAMER_INFERENCE_STALL_SECONDS,
    _SharedLiveProcess,
    _StreamInbox,
    _diagnostic_tail,
    _start_child,
)
from .redact import redact_secret_text

LOGGER = logging.getLogger(__name__)
# RTSP setup/keyframe wait also needs the native startup window when capture
# does not compile a model. Keep parent and child budgets in agreement.
DLSTREAMER_STARTUP_TIMEOUT_MS = 30000


def adjacent_model_proc(model_path: str) -> str:
    path = Path(model_path)
    if not path.name:
        return ""
    for candidate in (
        path.with_name(f"{path.stem}.json"),
        path.with_name(f"{path.stem}_proc.json"),
        path.parent / "model-proc.json",
    ):
        if candidate.is_file():
            return str(candidate)
    return ""


@dataclass(frozen=True, slots=True)
class DlStreamerCaptureOptions:
    python_executable: str = ""
    decoder: str = "va"
    open_timeout_ms: int = CAPTURE_OPEN_TIMEOUT_MS
    read_timeout_ms: int = CAPTURE_READ_TIMEOUT_MS
    admission_poll_seconds: float = CAPTURE_OPEN_LOCK_POLL_SECONDS
    rtsp_transport: str = "tcp"
    frame_rate: Callable[[], float] | None = None
    detection_frame_rate: Callable[[], float] | None = None
    main_frame_rate: Callable[[], float] | None = None
    model_path: str = ""
    labels_path: str = ""
    labels: tuple[str, ...] = ()
    model_proc_path: str = ""
    inference_device: str = "GPU"
    detect_enabled: bool = False
    batch_size: int = 1
    inference_interval: int = 1
    inference_requests: int = 4
    inference_streams: int = 2
    native_tracking: str = "off"
    tracking_classes: tuple[str, ...] | None = None
    reid_model_path: str = ""
    reid_device: str = "CPU"
    deep_sort_config: str = DEFAULT_DEEP_SORT_CONFIG
    frame_width: int = 320
    jpeg_fps: float = 1.0
    confidence_threshold: float = 0.1
    nms_threshold: float = 0.45


def live_python_executable(preferred: str = "") -> str:
    """Prefer system Python so GStreamer GI plugins resolve.

    Isolated live capture runs under ``/usr/bin/python3``, where ``python3-gi``
    is installed. The SurvNG venv has pydantic and the rest of the app stack;
    the child must not import that.
    """
    candidates = [
        preferred,
        os.environ.get("SURVNG_DLSTREAMER_PYTHON", ""),
        "/usr/bin/python3",
        sys.executable,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return sys.executable


class DlStreamerCaptureHandle:
    """One isolated live pipeline with a dedicated frame and metadata pipe."""

    def __init__(self, *, read_timeout_ms: int) -> None:
        self._read_timeout_seconds = max(0.001, read_timeout_ms / 1000.0)
        self._process: subprocess.Popen[bytes] | None = None
        self._output: BinaryIO | None = None
        self._stdout_thread: threading.Thread | None = None
        self._reader = MessageReader()
        self._prefetched: np.ndarray | None = None
        self._prefetched_identity: tuple[int, float, str] | None = None
        self._last_frame_identity: tuple[int, float, str] | None = None
        self._parsed_frame_identity: tuple[int, float, str] | None = None
        self._local_inbox = _StreamInbox()
        self.source_role = "live"
        self.h264_decoder_compliance = "auto"
        self.frame_width: int | None = None
        self._stderr = bytearray()
        self._native_stdout = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._status: dict[str, object] = {}
        self._detections: list[dict[str, object]] = []
        self._detections_lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._shared: _SharedLiveProcess | None = None
        self._stream_id = ""
        self._inbox: _StreamInbox | None = None

    def is_opened(self) -> bool:
        if self._inbox is not None:
            shared = self._shared
            return (
                self._inbox.alive
                and shared is not None
                and shared.is_running()
            )
        return self._process is not None and self._process.poll() is None

    def set_buffer_size(self, size: int) -> None:
        del size

    def set_source_role(self, source: str) -> None:
        if source not in {"live", "main"}:
            raise ValueError("invalid capture source role")
        self.source_role = source

    def set_frame_width(self, width: int) -> None:
        if width != 0 and not 240 <= width <= 960:
            raise ValueError("capture frame width must be zero (native) or between 240 and 960")
        self.frame_width = int(width)

    def start(self, command: list[str], source_url: str) -> None:
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (repo_root, existing) if part
        )
        self._process, self._output = _start_child(command, env, repo_root)
        assert self._process.stdin is not None
        self._process.stdin.write(f"{source_url}\n".encode("utf-8"))
        self._process.stdin.close()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="dlstreamer-live-stderr",
            daemon=True,
        )
        self._stdout_thread = threading.Thread(
            target=self._drain_stderr, args=(self._process.stdout, self._native_stdout),
            name="dlstreamer-native-stdout", daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def attach(
        self,
        shared: _SharedLiveProcess,
        stream_id: str,
        inbox: _StreamInbox,
    ) -> None:
        self._shared = shared
        self._stream_id = stream_id
        self._inbox = inbox

    def prefetch(self, timeout_ms: int, cancelled: Callable[[], bool]) -> bool:
        frame = self._next_frame(
            max(0.001, timeout_ms / 1000.0),
            cancelled=cancelled,
        )
        if frame is None:
            return False
        self._prefetched = frame[0]
        self._prefetched_identity = (frame[1], frame[2], frame[3])
        self._harvest_available_messages()
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._prefetched is not None:
            frame, self._prefetched = self._prefetched, None
            self._last_frame_identity, self._prefetched_identity = self._prefetched_identity, None
            self._harvest_available_messages()
            return True, frame
        received = self._next_frame(self._read_timeout_seconds)
        frame = None if received is None else received[0]
        if received is not None:
            self._harvest_available_messages()
            self._last_frame_identity = (received[1], received[2], received[3])
        return (frame is not None), frame

    def pipeline_status(self) -> dict[str, object]:
        inbox = self._inbox
        if inbox is not None:
            return dict(inbox.status)
        return dict(self._status)

    def pop_detections(self) -> list[dict[str, object]]:
        inbox = self._inbox
        if inbox is not None:
            return inbox.pop_detections()
        with self._detections_lock:
            detections = list(self._detections)
            self._detections = []
        return detections

    def pop_detection_snapshots(self) -> list[DetectionSnapshot]:
        return (self._inbox or self._local_inbox).pop_detection_snapshots()

    def pop_frame_identity(self) -> tuple[int, float, str] | None:
        identity, self._last_frame_identity = self._last_frame_identity, None
        return identity

    def pop_jpeg(self) -> bytes | None:
        inbox = self._inbox
        if inbox is not None:
            return inbox.pop_jpeg()
        with self._detections_lock:
            jpeg, self._jpeg = self._jpeg, None
            return jpeg

    def close(self) -> None:
        shared, stream_id, inbox = self._shared, self._stream_id, self._inbox
        self._shared = None
        self._stream_id = ""
        self._inbox = None
        if shared is not None:
            if inbox is not None:
                self._status = dict(inbox.status)
                if inbox.error:
                    self._status["error"] = str(self._status.get("error") or inbox.error)
                self._stderr = bytearray(shared.stderr_text().encode("utf-8"))
            shared.remove_stream(stream_id)
            return
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        stderr_thread, self._stderr_thread = self._stderr_thread, None
        if stderr_thread is not None:
            stderr_thread.join(timeout=1.0)
        stdout_thread, self._stdout_thread = self._stdout_thread, None
        if stdout_thread is not None:
            stdout_thread.join(timeout=1.0)
        for stream in (self._output, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self._output = None

    def error_detail(self) -> str:
        inbox = self._inbox
        if inbox is not None:
            shared = self._shared
            status_error = str(inbox.status.get("error") or inbox.error or "").strip()
            detail = shared.stderr_text() if shared is not None else ""
            process = shared._process if shared is not None else None
            return_code = process.poll() if process is not None else None
        else:
            process = self._process
            return_code = process.poll() if process is not None else None
            status_error = str(self._status.get("error") or "").strip()
            detail = _diagnostic_tail(self._stderr, self._native_stdout)
        parts = [part for part in (status_error, detail) if part]
        combined = ": ".join(parts)
        if return_code is None:
            return combined
        outcome = (
            f"DL Streamer exited from signal {-return_code}"
            if return_code < 0
            else f"DL Streamer exited with status {return_code}"
        )
        return f"{outcome}: {combined}" if combined else outcome

    def _drain_stderr(self, stream=None, buffer=None) -> None:
        process = self._process
        if stream is None:
            stream = process.stderr if process is not None else None
        if stream is None:
            return
        if buffer is None:
            buffer = self._stderr
        while True:
            chunk = stream.read1(CAPTURE_PIPE_READ_CHUNK_BYTES)
            if not chunk:
                return
            buffer.extend(chunk)
            if len(buffer) > 8192:
                del buffer[:-8192]

    def _next_frame(
        self,
        timeout_seconds: float,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[np.ndarray, int, float, str] | None:
        inbox = self._inbox
        if inbox is not None:
            return inbox.get_frame(timeout_seconds, cancelled=cancelled)
        process = self._process
        if process is None or self._output is None:
            return None
        deadline = time.monotonic() + timeout_seconds
        while True:
            if cancelled is not None and cancelled():
                return None
            popped = self._reader.pop()
            if popped is not None:
                frame = self._apply_message(*popped)
                if frame is not None:
                    sequence, pts, session = self._parsed_frame_identity or (0, float("nan"), "")
                    return frame, sequence, pts, session
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            readable, _, _ = select.select(
                [self._output],
                [],
                [],
                min(remaining, CAPTURE_OPEN_LOCK_POLL_SECONDS),
            )
            if not readable:
                continue
            chunk = self._output.read1(CAPTURE_PIPE_READ_CHUNK_BYTES)
            if not chunk:
                return None
            self._reader.feed(chunk)

    def _harvest_available_messages(self) -> None:
        if self._inbox is not None:
            return
        process = self._process
        if process is None or self._output is None:
            return
        while True:
            popped = self._reader.pop()
            if popped is not None:
                self._apply_message(*popped)
                continue
            readable, _, _ = select.select([self._output], [], [], 0)
            if not readable:
                return
            chunk = self._output.read1(CAPTURE_PIPE_READ_CHUNK_BYTES)
            if not chunk:
                return
            self._reader.feed(chunk)

    def _apply_message(self, message_type: int, payload: bytes) -> np.ndarray | None:
        if message_type == TYPE_FRAME:
            width, height, sequence, pts, pixels = decode_frame_payload(payload)
            session = self._local_inbox.qualify_pts("frame", pts)
            self._parsed_frame_identity = (sequence, pts, session)
            pixel_count = width * height
            if len(pixels) == pixel_count:
                return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width).copy()
            return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()
        if message_type == TYPE_JPEG:
            _width, _height, _sequence, _pts, jpeg = decode_jpeg_payload(payload)
            with self._detections_lock:
                self._jpeg = jpeg
            return None
        if message_type == TYPE_DETECTIONS:
            decoded = decode_json_payload(payload)
            objects = decoded.get("objects")
            if isinstance(objects, list):
                with self._detections_lock:
                    normalized = [item for item in objects if isinstance(item, dict)]
                    self._detections = normalized
                    if all(key in decoded for key in ("source_pts", "inference_sequence", "width", "height")):
                        try:
                            self._local_inbox.add_detection_snapshot(decoded)
                        except ValueError:
                            self._status["invalid_detection_snapshots"] = int(self._status.get("invalid_detection_snapshots", 0)) + 1
            return None
        if message_type == TYPE_STATUS:
            decoded = decode_json_payload(payload)
            if "invalid_detection_snapshots" in self._status:
                decoded["invalid_detection_snapshots"] = self._status["invalid_detection_snapshots"]
            self._status = decoded
            error = decoded.get("error")
            if decoded.get("ok") is False and error:
                raise RuntimeError(str(error))
            return None
        raise RuntimeError(f"unsupported live-capture message type {message_type}")


class DlStreamerCaptureBackend:
    """Shared GStreamer live capture with optional in-pipeline gvadetect."""

    def __init__(
        self,
        limiter: CaptureOpenLimiter,
        options: DlStreamerCaptureOptions | None = None,
    ) -> None:
        self.limiter = limiter
        self.options = options or DlStreamerCaptureOptions()
        if self.options.rtsp_transport not in {"tcp", "udp"}:
            raise ValueError("rtsp_transport must be tcp or udp")
        if self.options.decoder not in {"auto", "va"}:
            raise ValueError("decoder must be auto or va")
        if not 1 <= int(self.options.batch_size) <= 4:
            raise ValueError("batch_size must be between 1 and 4")
        if not 1 <= int(self.options.inference_interval) <= 5:
            raise ValueError("inference_interval must be between 1 and 5")
        if self.options.native_tracking not in {"off", "short-term-imageless", "deep-sort"}:
            raise ValueError("native_tracking must be off, short-term-imageless, or deep-sort")
        if self.options.native_tracking == "deep-sort":
            if not self.options.reid_model_path.strip():
                raise ValueError("deep-sort requires a ReID model path")
            if self.options.tracking_classes != ("person",):
                raise ValueError("deep-sort experiment requires person-only tracking")
        self._credential_warning_lock = threading.Lock()
        self._credential_warning_hosts: set[str] = set()
        self._shared: _SharedLiveProcess | None = None
        self._shared_lock = threading.Lock()
        atexit.register(self.close)

    def create_handle(self) -> CaptureHandle:
        return DlStreamerCaptureHandle(read_timeout_ms=self.options.read_timeout_ms)

    @property
    def startup_timeout_ms(self) -> int:
        return max(self.options.open_timeout_ms, DLSTREAMER_STARTUP_TIMEOUT_MS)

    def close(self) -> None:
        with self._shared_lock:
            shared, self._shared = self._shared, None
        if shared is not None:
            shared.close()

    def open(
        self,
        handle: CaptureHandle,
        source_url: str,
        cancelled: Callable[[], bool],
        *,
        open_timeout_ms: int | None = None,
    ) -> bool:
        if not isinstance(handle, DlStreamerCaptureHandle):
            raise TypeError("DlStreamerCaptureBackend requires DlStreamerCaptureHandle")
        while not cancelled():
            if not self.limiter.acquire(self.options.admission_poll_seconds):
                continue
            try:
                if cancelled():
                    return False
                timeout_ms = max(
                    1,
                    int(
                        self.startup_timeout_ms
                        if open_timeout_ms is None
                        else open_timeout_ms
                    ),
                )
                self.warn_credentialed_url(source_url)
                command = self.command()
                if "--supervisor" in command:
                    opened = self._open_shared(
                        handle,
                        source_url,
                        cancelled,
                        timeout_ms=timeout_ms,
                    )
                else:
                    handle.start(command, source_url)
                    if cancelled():
                        handle.close()
                        return False
                    opened = handle.prefetch(timeout_ms, cancelled)
                    if not opened:
                        handle.close()
                return opened
            finally:
                self.limiter.release()
        return False

    def _open_shared(
        self,
        handle: DlStreamerCaptureHandle,
        source_url: str,
        cancelled: Callable[[], bool],
        *,
        timeout_ms: int,
    ) -> bool:
        with self._shared_lock:
            if self._shared is None or not self._shared.is_running():
                if self._shared is not None:
                    self._shared.close()
                self._shared = _SharedLiveProcess(
                    self.command(),
                    read_timeout_ms=self.options.read_timeout_ms,
                    inference_stall_seconds=DLSTREAMER_INFERENCE_STALL_SECONDS,
                )
                self._shared.start()
            shared = self._shared
        stream_id = uuid.uuid4().hex
        handle.attach(shared, stream_id, shared.add_stream(
            stream_id, source_url, source_role=handle.source_role, frame_width=handle.frame_width,
            detection_enabled=getattr(handle, "detection_enabled", True),
            h264_decoder_compliance=handle.h264_decoder_compliance,
            spatial_plan=getattr(handle, "spatial_plan", None),
        ))
        if cancelled():
            handle.close()
            return False
        if not handle.prefetch(timeout_ms, cancelled):
            handle.close()
            return False
        return True

    def command(self) -> list[str]:
        requested_rate = (
            self.options.frame_rate()
            if self.options.frame_rate is not None
            else 5.0
        )
        frame_rate = min(10.0, max(0.5, float(requested_rate)))
        requested_detection_rate = (
            self.options.detection_frame_rate()
            if self.options.detection_frame_rate is not None
            else frame_rate
        )
        detection_rate = min(10.0, max(0.5, float(requested_detection_rate)))
        main_rate = self.options.main_frame_rate() if self.options.main_frame_rate else frame_rate
        open_timeout = max(0.001, self.startup_timeout_ms / 1000.0)
        command = [
            live_python_executable(self.options.python_executable),
            "-m",
            "survng.dlstreamer_live",
            "--fps",
            f"{frame_rate:.6f}",
            "--detect-fps",
            f"{detection_rate:.6f}",
            "--main-fps",
            f"{min(10.0, max(0.5, float(main_rate))):.6f}",
            "--threshold",
            str(self.options.confidence_threshold),
            "--nms-threshold",
            str(self.options.nms_threshold),
            "--open-timeout",
            f"{open_timeout:.3f}",
            "--rtsp-transport",
            self.options.rtsp_transport,
            "--decoder",
            self.options.decoder,
            "--device",
            self.options.inference_device or "GPU",
            "--frame-width",
            str(0 if self.options.frame_width == 0 else max(240, min(960, int(self.options.frame_width or 320)))),
            "--jpeg-fps",
            f"{max(0.0, min(5.0, float(self.options.jpeg_fps))):.6f}",
            "--supervisor",
        ]
        model_path = self.options.model_path.strip()
        if self.options.detect_enabled and model_path:
            command.extend(["--model", model_path])
            command.extend(["--batch-size", str(int(self.options.batch_size))])
            command.extend(["--inference-interval", str(int(self.options.inference_interval))])
            command.extend(["--inference-requests", str(self.options.inference_requests)])
            command.extend(["--inference-streams", str(self.options.inference_streams)])
            command.extend(["--native-tracking", self.options.native_tracking])
            if self.options.native_tracking == "deep-sort":
                command.extend(["--reid-model", self.options.reid_model_path.strip()])
                command.extend(["--reid-device", self.options.reid_device or "CPU"])
                command.extend(["--deep-sort-config", self.options.deep_sort_config])
            if self.options.tracking_classes is not None:
                command.extend(["--tracking-classes", json.dumps(self.options.tracking_classes)])
            command.extend(
                [
                    "--model-instance-id",
                    model_instance_id(
                        model_path,
                        self.options.inference_device or "GPU",
                    ),
                ]
            )
            labels_path = self.options.labels_path.strip()
            if labels_path:
                command.extend(["--labels", labels_path])
            elif self.options.labels:
                command.extend(["--labels-list", ",".join(self.options.labels)])
            model_proc = (
                self.options.model_proc_path.strip()
                or adjacent_model_proc(model_path)
            )
            if model_proc:
                command.extend(["--model-proc", model_proc])
        else:
            command.append("--no-detect")
        return command

    def warn_credentialed_url(self, source_url: str) -> None:
        try:
            parsed = urlsplit(source_url)
        except ValueError:
            return
        if parsed.username is None and parsed.password is None:
            return
        host = parsed.hostname or "unknown"
        with self._credential_warning_lock:
            if host in self._credential_warning_hosts:
                return
            self._credential_warning_hosts.add(host)
        LOGGER.warning(
            "camera capture URL for host %s contains credentials; route the "
            "camera through a credential-free go2rtc restream so the isolated "
            "live pipeline does not ingest a passworded URL",
            host,
        )
