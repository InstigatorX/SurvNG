"""Exercise the listener PCM path with deterministic voice probabilities."""

import io
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest

from survng.app.config import CameraConfig
from survng.app.speech.conversation import CHUNK_FRAMES, OVERLAP_FRAMES, TAIL_FRAMES, ConversationChunk
from survng.app.speech.listener import SpeechListener
from survng.app.speech.segmenter import FRAME_SAMPLES, SAMPLE_RATE, SpeechEnded


class _AmplitudeVad:
    def stream(self):
        return self

    def probability(self, frame: np.ndarray) -> float:
        if frame[0] > 0.25:
            return 0.9
        return 0.4 if frame[0] > 0.125 else 0.0


def _listener(events, *, idle_ms=3200) -> SpeechListener:
    clock = count()
    return SpeechListener(
        CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://example.test/gate",
            speech={
                "enabled": True,
                "transcribe": True,
                "min_speech_ms": 128,
                "hangover_ms": 160,
                "pre_roll_ms": 128,
                "max_segment_seconds": 5,
            },
        ),
        ffmpeg_path="ffmpeg",
        vad_model=_AmplitudeVad(),
        keep_audio=False,
        conversation_idle_ms=idle_ms,
        clock=lambda: 1000.0 + next(clock) * FRAME_SAMPLES / SAMPLE_RATE,
        on_event=events.append,
    )


def _consume(listener, frame_values, tail=()):
    samples = np.repeat(np.asarray(frame_values, dtype="<i2"), FRAME_SAMPLES)
    pcm = samples.tobytes() + np.asarray(tail, dtype="<i2").tobytes()
    listener._consume(SimpleNamespace(stdout=io.BytesIO(pcm)))
    return samples.astype(np.float32) / 32768


def test_listener_keeps_pcm_through_pauses_without_delaying_detection():
    events = []
    listener = _listener(events)
    first_onset = 18
    second_end = 18 + 20 + 63 + 20
    pcm = _consume(listener, [0] * 18 + [11000] * 20 + list(range(63)) + [12000] * 20 + [0] * 110)

    utterances = [item.event for item in events if isinstance(item.event, SpeechEnded)]
    assert len(utterances) == 2
    assert utterances[0].end_sample == (first_onset + 20) * FRAME_SAMPLES
    assert utterances[1].end_sample == second_end * FRAME_SAMPLES

    [item] = [item for item in events if isinstance(item.event, ConversationChunk)]
    chunk = item.event
    assert chunk.final and not chunk.interrupted
    assert chunk.start_sample == (first_onset - 4) * FRAME_SAMPLES
    assert chunk.end_sample == (second_end + TAIL_FRAMES) * FRAME_SAMPLES
    np.testing.assert_array_equal(chunk.audio, pcm[chunk.start_sample:chunk.end_sample])
    assert item.ended_at - item.started_at == pytest.approx(chunk.audio.size / SAMPLE_RATE)


def test_below_onset_audio_after_an_utterance_does_not_hold_the_session_open():
    events = []
    listener = _listener(events)
    _consume(listener, [11000] * 20 + [0] * 10 + [4500] * 110)

    [chunk] = [item.event for item in events if isinstance(item.event, ConversationChunk)]
    assert chunk.final and not chunk.interrupted
    assert chunk.end_sample == (20 + TAIL_FRAMES) * FRAME_SAMPLES


def test_resuming_just_before_idle_expiry_preserves_the_pause():
    events = []
    listener = _listener(events)
    pcm = _consume(listener, [11000] * 20 + [0] * 99 + [12000] * 20 + [0] * 110)

    [chunk] = [item.event for item in events if isinstance(item.event, ConversationChunk)]
    assert chunk.final and not chunk.interrupted
    assert chunk.end_sample == (20 + 99 + 20 + TAIL_FRAMES) * FRAME_SAMPLES
    np.testing.assert_array_equal(chunk.audio, pcm[:chunk.end_sample])


def test_listener_chunks_long_speech_with_no_gaps():
    events = []
    listener = _listener(events)
    pcm = _consume(listener, [11000 + index for index in range(3 * CHUNK_FRAMES)] + [0] * 110)

    chunks = [item.event for item in events if isinstance(item.event, ConversationChunk)]
    assert len(chunks) == 4
    assert len({chunk.conversation_id for chunk in chunks}) == 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(4))
    assert [chunk.final for chunk in chunks] == [False, False, False, True]
    assert all(not chunk.interrupted for chunk in chunks)
    for previous, chunk in zip(chunks, chunks[1:]):
        assert previous.end_sample - chunk.start_sample == OVERLAP_FRAMES * FRAME_SAMPLES
    for chunk in chunks:
        assert chunk.audio.size <= 30 * SAMPLE_RATE
        np.testing.assert_array_equal(chunk.audio, pcm[chunk.start_sample:chunk.end_sample])


def test_listener_emits_one_final_chunk_at_exact_boundary():
    events = []
    listener = _listener(events)
    _consume(listener, [11000] * CHUNK_FRAMES + [0] * 110)
    [chunk] = [item.event for item in events if isinstance(item.event, ConversationChunk)]
    assert chunk.final and not chunk.interrupted
    assert chunk.audio.size == CHUNK_FRAMES * FRAME_SAMPLES


def test_listener_finalizes_interrupted_sessions_and_starts_fresh_after_eof():
    events = []
    listener = _listener(events)
    _consume(listener, [11000] * 12 + [0] * 20, tail=[81, 82, 83])
    _consume(listener, [12000] * 12)

    first, second = [item.event for item in events if isinstance(item.event, ConversationChunk)]
    assert first.final and first.interrupted
    assert second.final and second.interrupted
    assert first.conversation_id != second.conversation_id
    assert first.chunk_index == second.chunk_index == 0
    assert first.end_sample == 32 * FRAME_SAMPLES + 3
    np.testing.assert_array_equal(first.audio[-3:], np.asarray([81, 82, 83]) / 32768)
