"""Whisper transcription in a supervised child process.

Whisper takes seconds per segment, so it runs in its own spawned process,
not in the detector pool or the API process: a slow or crashed transcription
can never delay incident inference. The parent keeps a bounded queue per
camera and dispatches one segment at a time, round-robin across cameras.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import SpeechConfig
from ..inference_runtime.worker_topology import openvino_gpu_compile_config

LOGGER = logging.getLogger("uvicorn.error")

START_TIMEOUT_SECONDS = 180.0
# Generous ceiling for a 30 s segment on a slow CPU; beyond it the worker is
# presumed wedged and restarted.
REQUEST_TIMEOUT_SECONDS = 120.0
RESTART_DELAYS_SECONDS = (5.0, 15.0, 60.0, 300.0)
# Transcription is enrichment: it runs below surveillance work in the OS
# scheduler and, on CPU, never takes every core.
WORKER_NICE_INCREMENT = 10
AUTO_CPU_THREADS_MAX = 4


def transcription_cpu_threads(configured: int, cpu_count: int | None = None) -> int:
    """Whisper CPU threads: the configured value, or half the cores (1-4)."""
    if configured > 0:
        return int(configured)
    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, min(AUTO_CPU_THREADS_MAX, cores // 2))
# Whisper fills silence and noise with stock phrases learned from subtitles.
HALLUCINATION_PHRASES = frozenset({
    "thank you.",
    "thank you",
    "thanks for watching!",
    "thanks for watching.",
    "thank you for watching.",
    "thank you for watching!",
    "please subscribe.",
    "you",
    "bye.",
    ".",
})


@dataclass(slots=True)
class TranscriptionJob:
    segment_id: int
    camera_id: str
    audio: np.ndarray


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    segment_id: int
    camera_id: str
    status: str
    text: str = ""
    language: str = ""
    score: float | None = None
    model: str = ""
    device: str = ""
    reason: str = ""


def restart_delay(failures: int) -> float:
    index = max(0, min(int(failures) - 1, len(RESTART_DELAYS_SECONDS) - 1))
    return RESTART_DELAYS_SECONDS[index]


def classify_transcript(text: str) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return "no_text"
    if normalized.lower() in HALLUCINATION_PHRASES:
        return "low_confidence"
    return "done"


def _model_label(path: str) -> str:
    name = Path(path).name
    return name[len("whisper-"):] if name.startswith("whisper-") else name


def _send(connection, message: tuple) -> bool:
    try:
        connection.send(message)
        return True
    except (BrokenPipeError, EOFError, OSError):
        # The parent stopped or restarted us; exit quietly.
        return False


def transcription_worker_main(
    connection,
    candidates: list[tuple[str, str]],
    cpu_threads: int = 0,
) -> None:
    """Child process entry point. ``candidates`` is [(device, model_path), ...]."""
    try:
        from ..inference_runtime.process import _disable_worker_core_dumps, _set_worker_process_name

        _disable_worker_core_dumps()
        _set_worker_process_name("speech")
    except Exception:
        pass
    try:
        os.nice(WORKER_NICE_INCREMENT)
    except OSError:
        pass
    try:
        import openvino_genai
    except ImportError:
        _send(connection, ("failed", "transcription runtime (openvino-genai) is not installed"))
        return
    pipeline = None
    device = ""
    model_path = ""
    errors: list[str] = []
    for candidate_device, candidate_path in candidates:
        if not candidate_path or not Path(candidate_path).is_dir():
            errors.append(f"{candidate_device}: model not installed")
            continue
        properties = openvino_gpu_compile_config(candidate_device)
        if candidate_device == "CPU" and cpu_threads > 0:
            properties["INFERENCE_NUM_THREADS"] = int(cpu_threads)
        try:
            try:
                pipeline = openvino_genai.WhisperPipeline(candidate_path, candidate_device, **properties)
            except Exception:
                if candidate_device != "CPU" or "INFERENCE_NUM_THREADS" not in properties:
                    raise
                # A runtime that rejects the thread cap still transcribes.
                pipeline = openvino_genai.WhisperPipeline(candidate_path, candidate_device)
            device, model_path = candidate_device, candidate_path
            break
        except Exception as error:  # device or model load failure
            errors.append(f"{candidate_device}: {type(error).__name__}")
    if pipeline is None:
        _send(connection, ("failed", "; ".join(errors) or "no transcription model"))
        return
    model = _model_label(model_path)
    if not _send(connection, ("ready", device, model)):
        return
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if not message or message[0] != "transcribe":
            return
        _kind, segment_id, payload = message
        try:
            audio = np.frombuffer(payload, dtype=np.float32)
            # English-only (.en) models reject language/task overrides.
            result = pipeline.generate(audio.tolist())
            texts = list(getattr(result, "texts", []) or [])
            scores = list(getattr(result, "scores", []) or [])
            text = " ".join(str(texts[0]).split()) if texts else ""
            score = float(scores[0]) if scores else None
        except Exception as error:
            if not _send(connection, ("error", segment_id, type(error).__name__)):
                return
            continue
        if not _send(connection, ("ok", segment_id, text, "en", score)):
            return


class TranscriptionService:
    def __init__(
        self,
        config: SpeechConfig,
        *,
        on_result: Callable[[TranscriptionResult], None],
        on_dropped: Callable[[TranscriptionJob, str], None],
        context: Any | None = None,
    ) -> None:
        self.config = config
        self.on_result = on_result
        self.on_dropped = on_dropped
        self._context = context or multiprocessing.get_context("spawn")
        self._queues: OrderedDict[str, deque[TranscriptionJob]] = OrderedDict()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None
        self._connection = None
        self._state = "stopped"
        self._reason = ""
        self._device = ""
        self._model = ""
        self._completed = 0
        self._failed = 0
        self._dropped = 0
        self._restarts = 0

    def candidates(self) -> list[tuple[str, str]]:
        requested = self.config.transcription_device
        result = [(requested, self.config.whisper_model_path(requested))]
        if requested != "CPU":
            # GPU unavailable or failing: fall back to the CPU-sized model.
            result.append(("CPU", self.config.whisper_model_path("CPU")))
        return result

    # Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._state = "starting"
        self._thread = threading.Thread(target=self._run, name="speech-transcriber", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None:
            # The supervisor thread owns the pipe and process; it notices the
            # stop within one poll interval and shuts the worker down itself.
            thread.join(timeout)
            if thread.is_alive():
                process = self._process
                if process is not None and process.is_alive():
                    process.kill()
                thread.join(2.0)
            self._thread = None
        with self._condition:
            pending = [job for queue in self._queues.values() for job in queue]
            self._queues.clear()
        for job in pending:
            self.on_dropped(job, "shutdown")
        self._state = "stopped"

    def submit(self, job: TranscriptionJob) -> None:
        dropped: TranscriptionJob | None = None
        with self._condition:
            if self._state == "unavailable":
                dropped_reason = self._reason or "transcriber_unavailable"
                dropped = job
            else:
                queue = self._queues.setdefault(job.camera_id, deque())
                if len(queue) >= self.config.max_queue_per_camera:
                    dropped = queue.popleft()
                    self._dropped += 1
                queue.append(job)
                dropped_reason = "queue_full"
                self._condition.notify()
        if dropped is not None:
            self.on_dropped(dropped, dropped_reason)

    def status(self) -> dict[str, Any]:
        with self._condition:
            depth = {camera_id: len(queue) for camera_id, queue in self._queues.items() if queue}
        return {
            "state": self._state,
            "reason": self._reason,
            "requested_device": self.config.transcription_device,
            "cpu_threads": transcription_cpu_threads(self.config.transcription_cpu_threads),
            "device": self._device,
            "model": self._model,
            "queue_depth": sum(depth.values()),
            "queue_by_camera": depth,
            "completed": self._completed,
            "failed": self._failed,
            "dropped": self._dropped,
            "restarts": self._restarts,
        }

    # Worker supervision -----------------------------------------------

    def _next_job(self) -> TranscriptionJob | None:
        with self._condition:
            while not self._stop.is_set():
                for camera_id in list(self._queues):
                    queue = self._queues[camera_id]
                    if queue:
                        job = queue.popleft()
                        # Rotate so one chatty camera cannot starve the others.
                        self._queues.move_to_end(camera_id)
                        return job
                self._condition.wait(timeout=1.0)
        return None

    def _run(self) -> None:
        # Consecutive failures without a successful transcription. A worker
        # that crashes on every request must not reload Whisper every few
        # seconds, so the backoff keeps growing until one job succeeds.
        failures = 0
        while not self._stop.is_set():
            if not self._start_process():
                if self._state == "unavailable":
                    self._drain_unavailable()
                    return
                failures += 1
                self._stop.wait(restart_delay(failures))
                continue
            completed_before = self._completed
            healthy = self._serve()
            self._stop_process()
            if self._completed > completed_before:
                failures = 0
            if not healthy and not self._stop.is_set():
                failures += 1
                self._restarts += 1
                self._state = "restarting"
                self._stop.wait(restart_delay(failures))

    def _start_process(self) -> bool:
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=transcription_worker_main,
            args=(
                child,
                self.candidates(),
                transcription_cpu_threads(self.config.transcription_cpu_threads),
            ),
            name="survng-speech",
            daemon=True,
        )
        try:
            process.start()
        except Exception as error:
            self._state = "restarting"
            self._reason = f"worker start failed: {type(error).__name__}"
            LOGGER.warning("Speech transcriber %s", self._reason)
            return False
        child.close()
        self._process, self._connection = process, parent
        try:
            if not self._poll(parent, START_TIMEOUT_SECONDS):
                if not self._stop.is_set():
                    self._reason = "worker did not become ready"
                    self._state = "restarting"
                self._stop_process()
                return False
            message = parent.recv()
        except (EOFError, OSError):
            self._reason = "worker exited during startup"
            self._state = "restarting"
            self._stop_process()
            return False
        if message[0] == "ready":
            _kind, self._device, self._model = message
            self._state = "ready"
            self._reason = ""
            if self._device != self.config.transcription_device:
                self._reason = f"{self.config.transcription_device} unavailable; using {self._device}"
                LOGGER.warning("Speech transcriber %s", self._reason)
            LOGGER.info("Speech transcriber ready: whisper %s on %s", self._model, self._device)
            return True
        # Missing runtime or model: a configuration problem, not transient.
        self._state = "unavailable"
        self._reason = str(message[1])[:300]
        LOGGER.warning("Speech transcription unavailable: %s", self._reason)
        self._stop_process()
        return False

    def _serve(self) -> bool:
        connection = self._connection
        while not self._stop.is_set():
            job = self._next_job()
            if job is None:
                return True
            self.on_result(TranscriptionResult(job.segment_id, job.camera_id, "transcribing"))
            try:
                connection.send(("transcribe", job.segment_id, job.audio.astype(np.float32).tobytes()))
                if not self._poll(connection, REQUEST_TIMEOUT_SECONDS):
                    self._fail(job, "shutdown" if self._stop.is_set() else "timeout")
                    return self._stop.is_set()
                message = connection.recv()
            except (EOFError, OSError, BrokenPipeError):
                self._fail(job, "worker_exited")
                return False
            if message[0] == "ok":
                _kind, _segment_id, text, language, score = message
                self._completed += 1
                self.on_result(TranscriptionResult(
                    job.segment_id,
                    job.camera_id,
                    classify_transcript(text),
                    text=text,
                    language=language,
                    score=score,
                    model=self._model,
                    device=self._device,
                ))
            else:
                self._fail(job, str(message[2] if len(message) > 2 else "error"))
        return True

    def _poll(self, connection, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if connection.poll(min(1.0, remaining)):
                return True
        return False

    def _fail(self, job: TranscriptionJob, reason: str) -> None:
        self._failed += 1
        self.on_result(TranscriptionResult(
            job.segment_id, job.camera_id, "failed",
            model=self._model, device=self._device, reason=reason[:200],
        ))

    def _drain_unavailable(self) -> None:
        with self._condition:
            pending = [job for queue in self._queues.values() for job in queue]
            self._queues.clear()
        for job in pending:
            self.on_dropped(job, self._reason or "transcriber_unavailable")

    def _stop_process(self) -> None:
        process, connection = self._process, self._connection
        self._process = self._connection = None
        if connection is not None:
            try:
                connection.send(("stop",))
            except (OSError, ValueError):
                pass
            try:
                connection.close()
            except OSError:
                pass
        if process is None:
            return
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
