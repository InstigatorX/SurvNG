"""Bounded transcription windows from a single contiguous PCM stream."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from .segmenter import FRAME_SAMPLES, SAMPLE_RATE

FRAME_MS = 1000 * FRAME_SAMPLES / SAMPLE_RATE
CHUNK_FRAMES = 30 * SAMPLE_RATE // FRAME_SAMPLES
OVERLAP_FRAMES = math.ceil(1000 / FRAME_MS)
TAIL_FRAMES = math.ceil(500 / FRAME_MS)


@dataclass(frozen=True, slots=True)
class ConversationChunk:
    conversation_id: str
    chunk_index: int
    start_sample: int
    end_sample: int
    audio: np.ndarray
    final: bool
    interrupted: bool


class ConversationCapture:
    """One per listener stream. VAD opens a session but PCM fills its pauses."""

    def __init__(
        self, *, idle_ms: int, pre_roll_ms: int, min_speech_ms: int, release: float
    ) -> None:
        self._idle_frames = math.ceil(idle_ms / FRAME_MS)
        self._release = release
        self._recent: deque[np.ndarray] = deque(
            maxlen=math.ceil(pre_roll_ms / FRAME_MS) + math.ceil(min_speech_ms / FRAME_MS)
        )
        self._frames: list[np.ndarray] = []
        self._quiet: list[np.ndarray] = []
        self._id: str | None = None
        self._start_sample = 0
        self._index = 0

    @property
    def active(self) -> bool:
        return self._id is not None

    def push(
        self,
        frame: np.ndarray,
        position: int,
        probability: float,
        *,
        speech_start: int | None,
        speaking: bool,
        candidate: bool,
    ) -> list[ConversationChunk]:
        self._recent.append(frame)
        if self._id is None:
            if speech_start is None:
                return []
            oldest_sample = position - (len(self._recent) - 1) * FRAME_SAMPLES
            self._start_sample = max(speech_start, oldest_sample)
            self._frames = list(self._recent)[(self._start_sample - oldest_sample) // FRAME_SAMPLES:]
            self._id = uuid4().hex
            self._index = 0
            return []

        if not speaking or probability < self._release:
            self._quiet.append(frame)
            # A possible onset may begin just before the timeout. Keep its
            # audio until the segmenter confirms or rejects it.
            if len(self._quiet) >= self._idle_frames and not speaking and not candidate:
                # The optional terminal pause should not create another job.
                tail = min(TAIL_FRAMES, CHUNK_FRAMES - len(self._frames))
                self._frames.extend(self._quiet[:tail])
                return self._finish(interrupted=False)
            return []

        self._frames.extend(self._quiet)
        self._quiet.clear()
        self._frames.append(frame)
        return self._take_full_chunks()

    def flush(self, tail: np.ndarray | None = None) -> list[ConversationChunk]:
        """Close a session at EOF, preserving any audio before the interruption."""
        if self._id is None:
            return []
        self._frames.extend(self._quiet)
        if tail is not None:
            self._frames.append(tail)
        return self._finish(interrupted=True)

    def _take_full_chunks(self) -> list[ConversationChunk]:
        chunks: list[ConversationChunk] = []
        # Keep at least one frame beyond the overlap for a final chunk, and
        # let an exactly full chunk carry the final flag itself.
        while len(self._frames) > CHUNK_FRAMES:
            chunks.append(self._chunk(self._frames[:CHUNK_FRAMES], final=False))
            stride = CHUNK_FRAMES - OVERLAP_FRAMES
            self._frames = self._frames[stride:]
            self._start_sample += stride * FRAME_SAMPLES
            self._index += 1
        return chunks

    def _finish(self, *, interrupted: bool) -> list[ConversationChunk]:
        chunks = self._take_full_chunks()
        chunks.append(self._chunk(self._frames, final=True, interrupted=interrupted))
        self._id = None
        self._frames.clear()
        self._quiet.clear()
        return chunks

    def _chunk(
        self, frames: list[np.ndarray], *, final: bool, interrupted: bool = False
    ) -> ConversationChunk:
        assert self._id is not None
        audio = np.concatenate(frames)
        return ConversationChunk(
            self._id,
            self._index,
            self._start_sample,
            self._start_sample + audio.size,
            audio,
            final,
            interrupted,
        )
