"""Turn per-frame voice probabilities into bounded speech segments.

This module is pure: it owns no clocks, processes, or models, so its state
machine can be tested with synthetic probability sequences.

States::

    idle --p>=threshold--> candidate --voiced >= min_speech--> speaking
    candidate --p<release--> idle                (too short: discarded)
    speaking --silence >= hangover--> idle       (segment ends)
    speaking --length >= max_segment--> speaking (segment split)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
# Hysteresis keeps brief dips inside a word from ending speech.
RELEASE_MARGIN = 0.15


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    start_sample: int


@dataclass(frozen=True, slots=True)
class SpeechConfirmed:
    """Enough voiced audio in one utterance to request an incident check.

    Emitted at most once per utterance (a long utterance split at
    max_segment_seconds is still one utterance). ``start_sample`` is where
    the utterance began, so the incident time is when speech started.
    """

    start_sample: int


@dataclass(frozen=True, slots=True)
class SpeechEnded:
    start_sample: int
    end_sample: int
    peak_probability: float
    # True when the segment was cut at max_segment_seconds and speech continues.
    split: bool
    # 16 kHz mono float32 audio, present only when audio retention is enabled.
    audio: np.ndarray | None

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / SAMPLE_RATE


SegmenterEvent = SpeechStarted | SpeechConfirmed | SpeechEnded


def _frames_for_ms(milliseconds: float, frame_samples: int) -> int:
    return max(0, math.ceil(milliseconds * SAMPLE_RATE / 1000.0 / frame_samples))


class SpeechSegmenter:
    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 300,
        hangover_ms: int = 800,
        pre_roll_ms: int = 500,
        max_segment_seconds: float = 30.0,
        keep_audio: bool = False,
        frame_samples: int = FRAME_SAMPLES,
        confirm_ms: int | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.release = max(0.01, self.threshold - RELEASE_MARGIN)
        self.frame_samples = int(frame_samples)
        self.min_speech_frames = max(1, _frames_for_ms(min_speech_ms, frame_samples))
        self.hangover_frames = max(1, _frames_for_ms(hangover_ms, frame_samples))
        self.max_segment_frames = max(
            self.min_speech_frames + 1,
            _frames_for_ms(max_segment_seconds * 1000.0, frame_samples),
        )
        self.keep_audio = bool(keep_audio)
        self.confirm_frames = (
            max(1, _frames_for_ms(confirm_ms, frame_samples)) if confirm_ms else None
        )
        self._voiced = 0
        self._utterance_start = 0
        self._confirmed = False
        self._pre_roll: deque[np.ndarray | None] = deque(
            maxlen=_frames_for_ms(pre_roll_ms, frame_samples)
        )
        self._frame_index = 0
        self._state = "idle"
        self._segment_start = 0
        self._candidate_start = 0
        self._last_voiced = 0
        self._silence = 0
        self._peak = 0.0
        self._audio: list[np.ndarray] = []

    @property
    def speaking(self) -> bool:
        return self._state == "speaking"

    @property
    def frames_processed(self) -> int:
        return self._frame_index

    def push(self, frame: np.ndarray | None, probability: float) -> list[SegmenterEvent]:
        """Consume one frame and its speech probability (frame may be None)."""
        index = self._frame_index
        self._frame_index += 1
        stored = frame if self.keep_audio else None
        events: list[SegmenterEvent] = []
        probability = float(probability)

        if self._state == "idle":
            if probability >= self.threshold:
                self._state = "candidate"
                self._candidate_start = index
                self._segment_start = index - len(self._pre_roll)
                self._audio = [item for item in self._pre_roll if item is not None]
                self._pre_roll.clear()
                self._peak = probability
                self._last_voiced = index
                self._utterance_start = self._segment_start
                self._voiced = 1
                self._confirmed = False
                self._append(stored)
                if self.min_speech_frames <= 1:
                    self._state = "speaking"
                    events.append(SpeechStarted(self._segment_start * self.frame_samples))
                    self._maybe_confirm(events)
            else:
                self._pre_roll.append(stored)
            return events

        self._append(stored)
        self._peak = max(self._peak, probability)
        if probability >= self.threshold:
            self._last_voiced = index
            self._voiced += 1

        if self._state == "candidate":
            if probability < self.release:
                # Too short to count; keep the tail as pre-roll for the next try.
                tail = self._audio if self.keep_audio else [None] * (index - self._segment_start + 1)
                self._reset_to_idle()
                self._pre_roll.extend(tail[-self._pre_roll.maxlen:] if self._pre_roll.maxlen else [])
                return events
            if index - self._candidate_start + 1 >= self.min_speech_frames:
                self._state = "speaking"
                self._silence = 0
                events.append(SpeechStarted(self._segment_start * self.frame_samples))
                self._maybe_confirm(events)
            return events

        # speaking
        self._maybe_confirm(events)
        if probability < self.release:
            self._silence += 1
        else:
            self._silence = 0
        if self._silence >= self.hangover_frames:
            events.append(self._finish(self._last_voiced + 1, split=False))
            self._reset_to_idle()
            return events
        if index - self._segment_start + 1 >= self.max_segment_frames:
            events.append(self._finish(index + 1, split=True))
            # Continue the utterance as a new segment without a pre-roll gap.
            self._segment_start = index + 1
            self._audio = []
            self._peak = 0.0
            events.append(SpeechStarted(self._segment_start * self.frame_samples))
        return events

    def flush(self) -> SpeechEnded | None:
        """End any confirmed speech, for example when the audio stream stops."""
        if self._state != "speaking":
            self._reset_to_idle()
            return None
        ended = self._finish(self._last_voiced + 1, split=False)
        self._reset_to_idle()
        return ended

    def _append(self, frame: np.ndarray | None) -> None:
        if frame is not None:
            self._audio.append(frame)

    def _finish(self, end_frame: int, *, split: bool) -> SpeechEnded:
        start_frame = self._segment_start
        end_frame = max(end_frame, start_frame + 1)
        audio = None
        if self.keep_audio and self._audio:
            joined = np.concatenate(self._audio).astype(np.float32, copy=False)
            audio = joined[: (end_frame - start_frame) * self.frame_samples]
        return SpeechEnded(
            start_sample=start_frame * self.frame_samples,
            end_sample=end_frame * self.frame_samples,
            peak_probability=round(self._peak, 4),
            split=split,
            audio=audio,
        )

    def _maybe_confirm(self, events: list[SegmenterEvent]) -> None:
        if (
            self.confirm_frames is not None
            and not self._confirmed
            and self._voiced >= self.confirm_frames
        ):
            self._confirmed = True
            events.append(SpeechConfirmed(self._utterance_start * self.frame_samples))

    def _reset_to_idle(self) -> None:
        self._voiced = 0
        self._confirmed = False
        self._state = "idle"
        self._silence = 0
        self._peak = 0.0
        self._audio = []
        self._pre_roll.clear()
