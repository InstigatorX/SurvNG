"""Silero voice activity detection on OpenVINO (CPU).

The model is ``silero_vad_openvino_16k.onnx`` from the MIT-licensed Silero VAD
package. Each call consumes 512 new 16 kHz samples plus the previous 64 samples
of context and returns a speech probability. One compiled model is shared;
every camera owns its own infer request and recurrent state.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .segmenter import FRAME_SAMPLES, SAMPLE_RATE

CONTEXT_SAMPLES = 64
STATE_SHAPE = (2, 1, 128)
# Frames quieter than about -60 dBFS cannot hold intelligible speech; skipping
# them keeps idle cameras nearly free.
SILENCE_RMS = 0.001


class VadUnavailable(RuntimeError):
    pass


class SileroVadModel:
    def __init__(self, model_path: str | Path, *, device: str = "CPU") -> None:
        path = Path(str(model_path or "").strip())
        if not str(model_path or "").strip() or not path.is_file():
            raise VadUnavailable("voice activity model is not installed")
        try:
            import openvino as ov
        except ImportError as error:  # pragma: no cover - OpenVINO is a core dependency
            raise VadUnavailable("OpenVINO is not available") from error
        core = ov.Core()
        model = core.read_model(str(path))
        input_names = {item.get_any_name() for item in model.inputs}
        output_names = {item.get_any_name() for item in model.outputs}
        if not {"input", "state"} <= input_names or not {"output", "stateN"} <= output_names:
            raise VadUnavailable("voice activity model has an unexpected interface")
        self._compiled = core.compile_model(model, device)
        self._lock = threading.Lock()
        self.path = str(path)

    def stream(self) -> "SileroVadStream":
        with self._lock:
            request = self._compiled.create_infer_request()
        return SileroVadStream(request, self._compiled)


class SileroVadStream:
    """Stateful per-camera detector; not thread-safe."""

    sample_rate = SAMPLE_RATE
    frame_samples = FRAME_SAMPLES

    def __init__(self, request, compiled) -> None:
        self._request = request
        self._output = compiled.output("output")
        self._state_output = compiled.output("stateN")
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"voice activity frames must hold {FRAME_SAMPLES} samples")
        if float(np.sqrt(np.mean(np.square(frame)))) < SILENCE_RMS:
            self._context = frame[-CONTEXT_SAMPLES:].copy()
            return 0.0
        model_input = np.concatenate([self._context, frame])[np.newaxis, :]
        result = self._request.infer({"input": model_input, "state": self._state})
        self._state = np.asarray(result[self._state_output], dtype=np.float32)
        self._context = frame[-CONTEXT_SAMPLES:].copy()
        return float(np.asarray(result[self._output]).reshape(-1)[0])
