"""Optional encoded-main collector lifecycle and budgeted evidence requests."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np

from .main_evidence_ring import receive_control, send_control
from .motion_pipeline.recorded_decode_budget import RecordedDecodeBudget


@dataclass(frozen=True)
class MainEvidenceFrame:
    frame: np.ndarray
    captured_at_epoch: float
    reference: dict[str, Any]
    timestamp_uncertainty_seconds: float
    timestamp_exact: bool = False  # Source PTS can be exact; UTC is estimated.


@dataclass(frozen=True)
class MainEvidenceBatch:
    status: str
    frames: dict[float, MainEvidenceFrame] = field(default_factory=dict)
    reason: str = ""
    memory_lease: Any | None = field(default=None, repr=False, compare=False)

    def release_memory(self) -> None:
        if self.memory_lease is not None:
            self.memory_lease.release()


class MainEvidenceProvider:
    """One opt-in camera; never starts until its lifecycle owner calls start().

    ``max_bytes`` covers current ring, pinned old bytes during export, and one
    exported encoded window (one third each).
    Successful batches carry their actual-geometry memory lease until the
    caller finishes consuming them; failures release it here. Only one
    exported request exists per camera. Main-stream receive UTC is explicitly
    approximate; the configured uncertainty is a policy allowance, not measured
    camera-clock calibration.
    """

    def __init__(self, camera_id: str, source_url: str, decode_budget: RecordedDecodeBudget,
                 *, history_seconds: float = 20.0, max_bytes: int = 64 * 1024 * 1024,
                 max_timestamp_uncertainty_seconds: float = 1.0,
                 decoder: str = "auto", python_executable: str = "/usr/bin/python3"):
        if (not math.isfinite(history_seconds) or history_seconds <= 0
                or max_bytes < 2 * 1024 * 1024
                or not math.isfinite(max_timestamp_uncertainty_seconds)
                or max_timestamp_uncertainty_seconds <= 0
                or decoder not in {"auto", "va", "cpu"}):
            raise ValueError("invalid main evidence configuration")
        self.camera_id = str(camera_id)
        self.source_url = str(source_url)
        self.decode_budget = decode_budget
        self.max_bytes = int(max_bytes)
        self.history_seconds = float(history_seconds)
        self.uncertainty = float(max_timestamp_uncertainty_seconds)
        self.decoder = decoder
        self.python_executable = python_executable
        self._state = threading.Lock()
        self._request = threading.Lock()
        self._stop = threading.Event()
        self._process: subprocess.Popen | None = None
        self._decoder_process: subprocess.Popen | None = None
        self._socket: socket.socket | None = None
        self._status: dict[str, Any] = {"state": "stopped", "camera_id": self.camera_id}
        self._counts: dict[str, int] = {}
        self._started_at = 0.0

    @property
    def running(self) -> bool:
        with self._state:
            return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        with self._state:
            if self._process is not None and self._process.poll() is None:
                return
            if self._socket is not None:
                self._socket.close()
            self._stop.clear()
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                process = subprocess.Popen(
                    [self.python_executable, "-m", "survng.main_evidence_worker", "collect",
                     "--control-fd", str(child.fileno())],
                    cwd=str(Path(__file__).resolve().parents[2]),
                    pass_fds=(child.fileno(),), stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True)
                config = {"source_url": self.source_url, "max_bytes": self.max_bytes,
                          "history_seconds": self.history_seconds,
                          "timestamp_uncertainty_seconds": self.uncertainty}
                assert process.stdin is not None
                process.stdin.write(json.dumps(config).encode() + b"\n")
                process.stdin.close()
                self._process, self._socket = process, parent
                self._started_at = time.monotonic()
                self._status = {"state": "warming", "camera_id": self.camera_id}
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()

    def stop(self) -> None:
        self._stop.set()
        with self._state:
            processes = [self._process, self._decoder_process]
            sock, self._socket = self._socket, None
            self._process = None
            self._status = {"state": "stopped", "camera_id": self.camera_id}
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # Already closed by a failed request.
            sock.close()
        for process in processes:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass  # The helper exited between poll and signal.
        deadline = time.monotonic() + 2.0
        for process in processes:
            if process is not None:
                try:
                    process.wait(timeout=max(0.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=1)

    def _exchange(self, request: dict, timeout: float):
        with self._state:
            sock = self._socket
        if sock is None or self._stop.is_set():
            raise EOFError("collector unavailable")
        sock.settimeout(max(0.01, timeout))
        send_control(sock, request)
        return receive_control(sock)

    def refresh_status(self) -> None:
        """Lifecycle polling only; observers read the cached snapshot below."""
        if time.monotonic() - self._started_at < 5.0:
            return  # GI/plugin initialization is not a failed control channel.
        if self._request.acquire(blocking=False):
            try:
                if self.running:
                    value, fd = self._exchange({"op": "status"}, 0.5)
                    if fd is not None:
                        os.close(fd)
                    self._status = {**value, "camera_id": self.camera_id}
                    if value.get("failure"):
                        self.stop()
                        self._status = {**value, "state": "unavailable", "camera_id": self.camera_id}
            except (OSError, EOFError, ValueError):
                # A timeout leaves framing ambiguous; never reuse the channel.
                self.stop()
                self._status = {"state": "unavailable", "camera_id": self.camera_id,
                                "failure": "collector_control_unavailable"}
            finally:
                self._request.release()

    def status(self) -> dict[str, Any]:
        return {**self._status, "running": self.running,
                "quota_bytes": self.max_bytes, "outcomes": dict(self._counts)}

    def frames_at(self, target_epochs: list[float], *, deadline: float,
                  cancelled: Callable[[], bool] = lambda: False) -> MainEvidenceBatch:
        targets = sorted(set(float(value) for value in target_epochs))
        if not targets or len(targets) > 32 or not all(math.isfinite(v) for v in targets):
            return MainEvidenceBatch("unsupported", reason="invalid_targets")
        if cancelled() or self._stop.is_set() or not self.running:
            return MainEvidenceBatch("miss", reason="collector_unavailable")
        if not self._request.acquire(blocking=False):
            return MainEvidenceBatch("capacity_denied", reason="request_active")
        fd = output_fd = None
        lease = None
        memory_lease = None
        exported = False
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return MainEvidenceBatch("capacity_denied", reason="deadline")
            metadata, fd = self._exchange({"op": "window", "targets": targets}, min(1.0, remaining))
            state = str(metadata.get("status") or "miss")
            self._counts[state] = self._counts.get(state, 0) + 1
            if state != "ready" or fd is None:
                return MainEvidenceBatch(state, reason=str(metadata.get("reason") or ""))
            exported = True
            self._status = {key: value for key, value in metadata.items()
                            if key not in {"units", "caps"}}
            self._status["state"] = "ready"
            self._status["timestamp_mapping"] = "uncalibrated_host_receive_estimate"
            width, height = int(metadata.get("width") or 0), int(metadata.get("height") or 0)
            if width <= 0 or height <= 0 or width * height * 3 > 64 * 1024 * 1024:
                return MainEvidenceBatch("unsupported", reason="geometry")
            frame_bytes = width * height * 3
            self.decode_budget.observe_frame_bytes(frame_bytes)
            # Selected frames plus their immutable output transport can coexist.
            # Twenty full BGR frames conservatively cover codec DPB, conversion
            # and bounded appsink/worker scratch (accepted 8-bit H264/H265).
            memory_lease = self.decode_budget.reserve_workflow(
                maximum_frames=2 * len(targets) + 20, frame_bytes=frame_bytes,
                camera_id=self.camera_id, incident_epoch=min(targets), deadline=deadline,
                cancelled=lambda: cancelled() or self._stop.is_set())
            if memory_lease is None:
                return MainEvidenceBatch("capacity_denied", reason="memory_budget")
            lease = self.decode_budget.acquire_process(incident_epoch=min(targets), deadline=deadline,
                                                       cancelled=lambda: cancelled() or self._stop.is_set())
            if lease is None:
                return MainEvidenceBatch("capacity_denied", reason="decode_budget")
            timeout = min(10.0, deadline - time.monotonic())
            if timeout <= 0:
                return MainEvidenceBatch("capacity_denied", reason="deadline")
            output_fd = os.memfd_create("survng-evidence-frames", os.MFD_CLOEXEC)
            config = {**metadata, "targets": targets, "decoder": self.decoder,
                      "timeout_seconds": max(0.1, timeout - 0.1),
                      "maximum_frame_offset_seconds": 0.25,
                      "maximum_output_bytes": width * height * 3 * len(targets)}
            process = subprocess.Popen(
                [self.python_executable, "-m", "survng.main_evidence_worker", "decode",
                 "--input-fd", str(fd), "--output-fd", str(output_fd)],
                cwd=str(Path(__file__).resolve().parents[2]), pass_fds=(fd, output_fd),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True)
            with self._state:
                self._decoder_process = process
            request_data = json.dumps(config).encode() + b"\n"
            end = time.monotonic() + timeout
            try:
                while True:
                    try:
                        stdout, _ = process.communicate(input=request_data, timeout=min(0.1, max(0.01, end - time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        request_data = None
                        if cancelled() or self._stop.is_set() or time.monotonic() >= end:
                            process.kill()
                            process.communicate()
                            return MainEvidenceBatch("pending", reason="decode_cancelled_or_timeout")
                if process.returncode != 0 or len(stdout) > 1024 * 1024:
                    return MainEvidenceBatch("miss", reason="decoder_failed")
                result = json.loads(stdout)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=1)
                with self._state:
                    self._decoder_process = None
            if result.get("status") != "ready":
                return MainEvidenceBatch(str(result.get("status") or "miss"), reason=str(result.get("reason") or ""))
            frames = {}
            for item in result["frames"]:
                frame_width, frame_height = int(item["width"]), int(item["height"])
                size = frame_width * frame_height * 3
                if size != item["size"] or size > width * height * 3:
                    return MainEvidenceBatch("miss", reason="invalid_frame_geometry")
                data = os.pread(output_fd, size, int(item["offset"]))
                if len(data) != size:
                    return MainEvidenceBatch("miss", reason="truncated_frame")
                unit = item["unit"]
                reference = {"source": "buffered_main", "camera_id": self.camera_id,
                    "session": metadata["session"], "generation": metadata["generation"],
                    "ordinal": unit["ordinal"], "pts_ns": unit["pts_ns"],
                    "dts_ns": unit.get("dts_ns"), "time_base_num": 1, "time_base_den": 1_000_000_000,
                    "source_timestamp_exact": True, "utc_timestamp_exact": False,
                    "timestamp_method": metadata["timestamp_method"]}
                frame = MainEvidenceFrame(np.frombuffer(data, dtype=np.uint8).reshape(frame_height, frame_width, 3),
                    float(unit["epoch"]), reference, float(metadata["timestamp_uncertainty_seconds"]))
                for target in item["targets"]:
                    frames[float(target)] = frame
            if set(frames) != set(targets):
                return MainEvidenceBatch("miss", reason="incomplete_targets")
            result = MainEvidenceBatch("ready", frames, memory_lease=memory_lease)
            memory_lease = None  # Ownership passes to the result's consumer.
            return result
        except (OSError, EOFError, ValueError, KeyError, TypeError):
            self._counts["provider_error"] = self._counts.get("provider_error", 0) + 1
            self.stop()
            return MainEvidenceBatch("miss", reason="provider_error")
        finally:
            if lease is not None:
                lease.release()
            if memory_lease is not None:
                memory_lease.release()
            for descriptor in (fd, output_fd):
                if descriptor is not None:
                    os.close(descriptor)
            if exported:
                try:
                    _, extra_fd = self._exchange({"op": "release"}, 0.2)
                    if extra_fd is not None:
                        os.close(extra_fd)
                except (OSError, EOFError, ValueError):
                    self.stop()
            self._request.release()
