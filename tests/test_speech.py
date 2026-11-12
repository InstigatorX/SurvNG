from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from survng.app.config import AppConfig, SpeechConfig
from survng.app.local_observability import _speech_snapshot
from survng.app.security import required_api_scope
from survng.app.speech.lifecycle import SpeechLifecycle
from survng.app.speech.listener import (
    ListenerEvent,
    SpeechListener,
    audio_listener_command,
    classify_ffmpeg_failure,
)
from survng.app.speech.segmenter import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    SpeechEnded,
    SpeechSegmenter,
    SpeechStarted,
)
from survng.app.speech.store import SpeechStore
from survng.app.speech.transcriber import (
    TranscriptionJob,
    TranscriptionResult,
    TranscriptionService,
    classify_transcript,
)

FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE  # 32 ms


def _config(**camera_speech) -> AppConfig:
    return AppConfig.model_validate({
        "speech": {"enabled": True, "vad_model_path": "/models/vad.onnx"},
        "cameras": [
            {
                "id": "gate",
                "name": "Gate",
                "stream_url": "rtsp://user:secret@camera.local/main",
                "speech": {"enabled": True, **camera_speech},
            },
            {"id": "yard", "name": "Yard", "stream_url": "rtsp://camera.local/yard"},
        ],
    })


def _run(segmenter: SpeechSegmenter, probabilities: list[float]):
    events = []
    for index, probability in enumerate(probabilities):
        frame = np.full(FRAME_SAMPLES, index, dtype=np.float32)
        events.extend(segmenter.push(frame, probability))
    return events


# Configuration -------------------------------------------------------------


def test_speech_is_off_by_default_and_old_configs_load() -> None:
    config = AppConfig.model_validate({
        "cameras": [{"id": "gate", "name": "Gate", "stream_url": "rtsp://camera.local/main"}],
    })
    assert config.speech.enabled is False
    assert config.cameras[0].speech.enabled is False
    assert config.cameras[0].speech.transcribe is False
    assert config.speech.mqtt_publish_transcripts is False


def test_whisper_model_follows_selected_device() -> None:
    cpu = SpeechConfig(whisper_models_dir="/models/whisper")
    gpu = SpeechConfig(whisper_models_dir="/models/whisper", transcription_device="GPU")
    pinned = SpeechConfig(
        whisper_models_dir="/models/whisper",
        transcription_device="GPU",
        transcription_model="tiny.en",
    )
    assert cpu.whisper_model_path() == "/models/whisper/whisper-base.en"
    assert gpu.whisper_model_path() == "/models/whisper/whisper-small.en"
    assert gpu.whisper_model_path("CPU") == "/models/whisper/whisper-base.en"
    assert pinned.whisper_model_path() == "/models/whisper/whisper-tiny.en"
    assert SpeechConfig().whisper_model_path() == ""


# Segmenter ---------------------------------------------------------------


def test_brief_noise_does_not_become_speech() -> None:
    segmenter = SpeechSegmenter(min_speech_ms=300, hangover_ms=300)
    events = _run(segmenter, [0.0] * 5 + [0.9] * 3 + [0.0] * 20)
    assert events == []
    assert segmenter.flush() is None


def test_segment_includes_pre_roll_and_ends_after_hangover() -> None:
    segmenter = SpeechSegmenter(
        min_speech_ms=200, hangover_ms=320, pre_roll_ms=128, keep_audio=True
    )
    probabilities = [0.0] * 10 + [0.9] * 20 + [0.0] * 12
    events = _run(segmenter, probabilities)

    assert isinstance(events[0], SpeechStarted)
    ended = events[-1]
    assert isinstance(ended, SpeechEnded)
    # 128 ms pre-roll = 4 frames before speech begins at frame 10.
    assert ended.start_sample == 6 * FRAME_SAMPLES
    # Ends after the last voiced frame (29), not after the hangover silence.
    assert ended.end_sample == 30 * FRAME_SAMPLES
    assert ended.split is False
    assert ended.peak_probability == pytest.approx(0.9)
    assert ended.audio is not None
    assert len(ended.audio) == ended.end_sample - ended.start_sample
    # Audio starts with the first pre-roll frame (each frame holds its index).
    assert ended.audio[0] == 6
    assert ended.audio[-1] == 29


def test_short_dips_inside_speech_do_not_split_it() -> None:
    segmenter = SpeechSegmenter(min_speech_ms=100, hangover_ms=320)
    pattern = [0.9] * 10 + [0.2] * 5 + [0.9] * 10 + [0.0] * 12
    events = _run(segmenter, pattern)
    assert [type(event) for event in events] == [SpeechStarted, SpeechEnded]


def test_long_speech_is_split_at_the_maximum_length() -> None:
    segmenter = SpeechSegmenter(min_speech_ms=100, max_segment_seconds=5.0, pre_roll_ms=0)
    frames_per_segment = int(np.ceil(5.0 * 1000 / FRAME_MS))
    events = _run(segmenter, [0.9] * (frames_per_segment * 2 + 10))
    ended = [event for event in events if isinstance(event, SpeechEnded)]
    assert len(ended) == 2
    assert all(event.split for event in ended)
    assert ended[0].end_sample == ended[1].start_sample
    tail = segmenter.flush()
    assert tail is not None and tail.split is False


def test_flush_closes_confirmed_speech_when_the_stream_stops() -> None:
    segmenter = SpeechSegmenter(min_speech_ms=100)
    _run(segmenter, [0.9] * 10)
    ended = segmenter.flush()
    assert ended is not None
    assert ended.audio is None  # audio is only retained when transcribing
    assert segmenter.flush() is None


# Store -------------------------------------------------------------------


def test_store_round_trip_search_and_retention(tmp_path: Path) -> None:
    store = SpeechStore(tmp_path)
    now = time.time()
    old = store.insert_segment(camera_id="gate", started_at=now - 90000, ended_at=now - 89998, peak_probability=0.8, status="detected")
    first = store.insert_segment(camera_id="gate", started_at=now - 10, ended_at=now - 8, peak_probability=0.9, status="queued")
    second = store.insert_segment(camera_id="yard", started_at=now - 5, ended_at=now - 3, peak_probability=0.7, status="queued")
    store.set_transcript(first, status="done", text="Please leave the package by the door", language="en", score=-0.2, model="base.en", device="CPU")

    assert [row["id"] for row in store.list_segments(query="package")] == [first]
    # FTS operators in user input are treated as literal terms.
    assert store.list_segments(query='package" OR "x') == []
    assert [row["id"] for row in store.list_segments(camera_ids=["yard"])] == [second]
    assert [row["id"] for row in store.list_segments(start=now - 6)] == [second]
    assert [row["id"] for row in store.list_segments(statuses=["done"])] == [first]

    assert store.abandon_pending("restart") == 1
    assert store.get(second)["status"] == "skipped"
    assert store.purge_before(now - 86400) == 1
    assert store.get(old) is None
    assert store.list_segments(query="package")[0]["text"] == "Please leave the package by the door"


# Listener ----------------------------------------------------------------


def test_listener_command_extracts_mono_16k_audio_only() -> None:
    camera = _config().cameras[0]
    command = audio_listener_command(camera, "ffmpeg")
    assert command[command.index("-map") + 1] == "0:a:0"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-ac") + 1] == "1"
    assert "-vn" in command and command[-1] == "pipe:1"


@pytest.mark.parametrize(
    ("line", "category"),
    [
        ("Stream map '0:a:0' matches no streams.", "no_audio_track"),
        ("rtsp://camera: 401 Unauthorized", "stream_unauthorized"),
        ("Connection refused", "stream_unavailable"),
        ("Invalid data found when processing input", "audio_decode_error"),
        ("something unexpected", "ffmpeg_error"),
    ],
)
def test_ffmpeg_failures_are_classified(line: str, category: str) -> None:
    assert classify_ffmpeg_failure(line) == category


class _AmplitudeVad:
    """Fake VAD: loud frames are speech."""

    def stream(self):
        return self

    def probability(self, frame: np.ndarray) -> float:
        return 0.95 if float(np.abs(frame).max()) > 0.1 else 0.0


class _FakeProcess:
    def __init__(self, pcm: bytes, stderr: bytes = b"") -> None:
        self.stdout = io.BytesIO(pcm)
        self.stderr = io.BytesIO(stderr)
        self.pid = 999_999_999
        self.returncode = 0

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


def _pcm(seconds_silence: float, seconds_speech: float) -> bytes:
    silence = np.zeros(int(seconds_silence * SAMPLE_RATE), dtype="<i2")
    speech = (np.sin(np.arange(int(seconds_speech * SAMPLE_RATE)) / 5.0) * 12000).astype("<i2")
    return np.concatenate([silence, speech, silence]).tobytes()


def test_listener_emits_timed_segments_with_audio() -> None:
    camera = _config(transcribe=True, min_speech_ms=100, hangover_ms=200, pre_roll_ms=0).cameras[0]
    events: list[ListenerEvent] = []
    done = threading.Event()
    spawned = []

    def popen(command, **kwargs):
        spawned.append(command)
        if len(spawned) > 1:
            done.set()
        return _FakeProcess(_pcm(1.0, 1.0))

    clock = iter(1000.0 + index * FRAME_SAMPLES / SAMPLE_RATE for index in range(100_000))
    listener = SpeechListener(
        camera,
        ffmpeg_path="ffmpeg",
        vad_model=_AmplitudeVad(),
        keep_audio=True,
        on_event=events.append,
        clock=lambda: next(clock),
        popen=popen,
        sleep=lambda event, seconds: event.is_set(),
    )
    listener.start()
    assert done.wait(5.0)
    listener.stop()

    started = [item for item in events if isinstance(item.event, SpeechStarted)]
    ended = [item for item in events if isinstance(item.event, SpeechEnded)]
    assert started and ended
    first = ended[0]
    assert first.camera_id == "gate"
    assert first.started_at == pytest.approx(1001.0, abs=0.07)
    assert first.ended_at - first.started_at == pytest.approx(1.0, abs=0.07)
    assert first.event.audio is not None
    assert listener.status()["state"] == "stopped"
    assert listener.status()["restarts"] >= 1


def test_listener_waits_without_ffmpeg_when_camera_has_no_audio() -> None:
    camera = _config().cameras[0]
    spawned = []
    waited = threading.Event()

    def wait(event, seconds):
        waited.set()
        return event.wait(5.0)

    listener = SpeechListener(
        camera,
        ffmpeg_path="ffmpeg",
        vad_model=_AmplitudeVad(),
        keep_audio=False,
        on_event=lambda item: None,
        audio_available=lambda camera: False,
        popen=lambda *args, **kwargs: spawned.append(args),
        sleep=wait,
    )
    listener.start()
    assert waited.wait(2.0)
    assert listener.status()["state"] == "no_audio_track"
    listener.stop()
    assert spawned == []


# Transcriber -------------------------------------------------------------


def test_transcript_classification_flags_whisper_filler() -> None:
    assert classify_transcript("  ") == "no_text"
    assert classify_transcript("Thanks for watching!") == "low_confidence"
    assert classify_transcript("Is anybody home?") == "done"


def test_full_queue_drops_the_oldest_segment_and_rotates_cameras() -> None:
    dropped: list[tuple[int, str]] = []
    service = TranscriptionService(
        SpeechConfig(max_queue_per_camera=2),
        on_result=lambda result: None,
        on_dropped=lambda job, reason: dropped.append((job.segment_id, reason)),
    )
    audio = np.zeros(16000, dtype=np.float32)
    for segment_id in (1, 2, 3):
        service.submit(TranscriptionJob(segment_id, "gate", audio))
    service.submit(TranscriptionJob(4, "yard", audio))
    assert dropped == [(1, "queue_full")]
    service._stop.set()
    service._stop.clear()
    order = [service._next_job().segment_id for _ in range(3)]
    assert order == [2, 4, 3]
    assert service.status()["dropped"] == 1


def test_missing_model_makes_transcription_unavailable(tmp_path: Path) -> None:
    pytest.importorskip("openvino_genai")
    dropped: list[str] = []
    service = TranscriptionService(
        SpeechConfig(whisper_models_dir=str(tmp_path / "missing")),
        on_result=lambda result: None,
        on_dropped=lambda job, reason: dropped.append(reason),
    )
    service.submit(TranscriptionJob(7, "gate", np.zeros(8000, dtype=np.float32)))
    service.start()
    deadline = time.monotonic() + 60
    while service.status()["state"] != "unavailable" and time.monotonic() < deadline:
        time.sleep(0.1)
    service.stop()
    assert service.status()["reason"].startswith("CPU: model not installed")
    assert dropped and dropped[0].startswith("CPU: model not installed")


# Lifecycle ---------------------------------------------------------------


class _FakeListener:
    instances: list["_FakeListener"] = []

    def __init__(self, camera, **kwargs) -> None:
        self.camera = camera
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        _FakeListener.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def status(self) -> dict:
        return {"state": "listening", "transcribe": self.camera.speech.transcribe}


class _FakeTranscriber:
    def __init__(self, config, *, on_result, on_dropped) -> None:
        self.on_result = on_result
        self.on_dropped = on_dropped
        self.jobs: list[TranscriptionJob] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def submit(self, job: TranscriptionJob) -> None:
        self.jobs.append(job)

    def status(self) -> dict:
        return {"state": "ready", "queue_depth": len(self.jobs)}


def _lifecycle(tmp_path: Path, config: AppConfig, enabled: dict[str, bool], published: list):
    _FakeListener.instances = []
    return SpeechLifecycle(
        config,
        database_dir=tmp_path,
        camera_enabled=lambda camera_id: enabled.get(camera_id, False),
        publish=lambda suffix, payload: published.append((suffix, payload)),
        vad_factory=lambda path: object(),
        listener_factory=_FakeListener,
        transcriber_factory=_FakeTranscriber,
    )


def _ended(seconds: float, *, audio: bool) -> SpeechEnded:
    samples = int(seconds * SAMPLE_RATE)
    return SpeechEnded(0, samples, 0.9, False, np.zeros(samples, np.float32) if audio else None)


def test_disabled_speech_starts_nothing_and_creates_no_database(tmp_path: Path) -> None:
    config = _config()
    config.speech.enabled = False
    lifecycle = _lifecycle(tmp_path, config, {"gate": True}, [])
    lifecycle.start()
    assert lifecycle.status()["state"] == "disabled"
    assert _FakeListener.instances == []
    assert lifecycle.segments() == []
    assert not (tmp_path / "speech.sqlite3").exists()
    lifecycle.stop()


def test_missing_vad_model_leaves_speech_unavailable(tmp_path: Path) -> None:
    config = _config()
    lifecycle = SpeechLifecycle(
        config, database_dir=tmp_path, camera_enabled=lambda camera_id: True
    )
    lifecycle.start()
    status = lifecycle.status()
    assert status["state"] == "unavailable"
    assert "not installed" in status["reason"]
    lifecycle.stop()


def test_listeners_follow_camera_power_and_only_speech_cameras(tmp_path: Path) -> None:
    enabled = {"gate": False, "yard": True}
    published: list = []
    lifecycle = _lifecycle(tmp_path, _config(), enabled, published)
    lifecycle.start()
    try:
        assert _FakeListener.instances == []  # gate is off; yard has no speech
        enabled["gate"] = True
        lifecycle._reconcile()
        assert [item.camera.id for item in _FakeListener.instances] == ["gate"]
        assert _FakeListener.instances[0].kwargs["keep_audio"] is False
        enabled["gate"] = False
        lifecycle._reconcile()
        assert _FakeListener.instances[0].stopped
        assert published[-1] == ("camera/gate/speech", {"camera_id": "gate", "state": "OFF", "speech": False})
        assert lifecycle.status()["cameras"]["gate"]["state"] == "paused"
    finally:
        lifecycle.stop()


def test_detection_only_cameras_record_segments_and_publish_state(tmp_path: Path) -> None:
    published: list = []
    lifecycle = _lifecycle(tmp_path, _config(), {"gate": True}, published)
    lifecycle.start()
    try:
        lifecycle._listener_event(ListenerEvent("gate", SpeechStarted(0), 100.0, None))
        lifecycle._listener_event(ListenerEvent("gate", _ended(2.0, audio=False), 100.0, 102.0))
        rows = lifecycle.segments()
        assert len(rows) == 1 and rows[0]["status"] == "detected"
        assert lifecycle.transcriber is None
        assert [payload["state"] for _suffix, payload in published] == ["ON", "OFF"]
        assert published[-1][1]["segment_id"] == rows[0]["id"]
    finally:
        lifecycle.stop()


@pytest.mark.parametrize("publish_transcripts", [False, True])
def test_transcription_results_update_rows_and_mqtt_is_opt_in(tmp_path: Path, publish_transcripts: bool) -> None:
    config = _config(transcribe=True)
    config.speech.mqtt_publish_transcripts = publish_transcripts
    published: list = []
    lifecycle = _lifecycle(tmp_path, config, {"gate": True}, published)
    lifecycle.start()
    try:
        assert _FakeListener.instances[0].kwargs["keep_audio"] is True
        lifecycle._listener_event(ListenerEvent("gate", _ended(0.2, audio=True), 100.0, 100.2))
        lifecycle._listener_event(ListenerEvent("gate", _ended(2.0, audio=True), 110.0, 112.0))
        transcriber = lifecycle.transcriber
        rows = {row["status"]: row for row in lifecycle.segments()}
        assert rows["skipped"]["reason"] == "too_short"
        assert [job.segment_id for job in transcriber.jobs] == [rows["queued"]["id"]]

        segment_id = rows["queued"]["id"]
        transcriber.on_result(TranscriptionResult(segment_id, "gate", "transcribing"))
        assert lifecycle.segment(segment_id)["status"] == "transcribing"
        transcriber.on_result(TranscriptionResult(
            segment_id, "gate", "done", text="Hello there", language="en",
            score=-0.3, model="base.en", device="CPU",
        ))
        row = lifecycle.segment(segment_id)
        assert (row["status"], row["text"], row["model"]) == ("done", "Hello there", "base.en")

        transcript_topics = [payload for suffix, payload in published if suffix.endswith("/transcript")]
        if publish_transcripts:
            assert transcript_topics == [{"camera_id": "gate", "segment_id": segment_id, "text": "Hello there", "language": "en"}]
        else:
            assert transcript_topics == []
    finally:
        lifecycle.stop()


# Security and observability ---------------------------------------------


def test_transcripts_require_admin_scope_but_status_is_readable() -> None:
    assert required_api_scope("GET", "/api/speech/segments") == "admin"
    assert required_api_scope("GET", "/api/speech/segments/5") == "admin"
    assert required_api_scope("GET", "/api/speech/status") == "read"


def test_observability_snapshot_excludes_transcripts_and_unknown_values() -> None:
    snapshot = _speech_snapshot({
        "enabled": True,
        "state": "running",
        "cameras": {
            "gate": {
                "state": "listening",
                "failure": "rtsp://user:pw@host failed",
                "transcribe": True,
                "restarts": 2,
                "segments": 5,
                "text": "secret conversation",
            },
        },
        "transcriber": {"state": "ready", "device": "GPU", "model": "small.en", "queue_depth": 1, "text": "x"},
    })
    serialized = repr(snapshot)
    assert "secret conversation" not in serialized and "pw@host" not in serialized
    assert snapshot["cameras"]["gate"] == {
        "state": "listening",
        "failure": "other",
        "transcribe": True,
        "restarts": 2,
        "segments_since_restart": 5,
    }
    assert snapshot["transcriber"]["device"] == "GPU"
    assert "text" not in snapshot["transcriber"]


def test_status_reports_installed_models_without_paths(tmp_path: Path) -> None:
    vad = tmp_path / "vad.onnx"
    vad.write_bytes(b"x")
    whisper = tmp_path / "whisper"
    (whisper / "whisper-base.en").mkdir(parents=True)
    (whisper / "whisper-base.en" / "openvino_encoder_model.xml").write_text("<net/>")
    (whisper / "whisper-small.en").mkdir()  # incomplete download
    config = _config()
    config.speech.vad_model_path = str(vad)
    config.speech.whisper_models_dir = str(whisper)
    config.speech.transcription_device = "GPU"
    lifecycle = _lifecycle(tmp_path, config, {}, [])
    models = lifecycle.status()["models"]
    assert models == {
        "vad_installed": True,
        "whisper_installed": ["base.en"],
        "selected_device": "GPU",
        "selected_model": "small.en",
        "selected_model_installed": False,
    }
    assert str(tmp_path) not in repr(models)


def test_refresh_applies_camera_changes_immediately(tmp_path: Path) -> None:
    enabled = {"gate": True}
    published: list = []
    lifecycle = _lifecycle(tmp_path, _config(), enabled, published)
    lifecycle.refresh()  # not started: no-op
    assert _FakeListener.instances == []
    lifecycle.start()
    try:
        listener = _FakeListener.instances[0]
        enabled["gate"] = False
        lifecycle.refresh()
        assert listener.stopped
        assert lifecycle.status()["cameras"]["gate"]["state"] == "paused"
        enabled["gate"] = True
        lifecycle.refresh()
        assert len(_FakeListener.instances) == 2 and _FakeListener.instances[1].started
    finally:
        lifecycle.stop()


class _Switches:
    def __init__(self, powered: bool, detection: bool) -> None:
        self.powered = powered
        self.detection = detection
        self.calls: list[tuple[str, bool]] = []

    def camera_enabled(self, camera_id: str) -> bool:
        return self.powered

    def detection_enabled(self, camera_id: str) -> bool:
        return self.detection

    def set_detection(self, camera_id: str, enabled: bool) -> bool:
        self.calls.append((camera_id, enabled))
        self.detection = enabled
        return True


class _RefreshCounter:
    def __init__(self) -> None:
        self.count = 0

    def refresh(self) -> None:
        self.count += 1


@pytest.mark.parametrize(
    ("powered", "detection", "active"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_speech_listens_only_with_camera_power_and_detection(powered: bool, detection: bool, active: bool) -> None:
    from survng.app.manager import AppManager

    switches = _Switches(powered, detection)
    manager = AppManager.__new__(AppManager)
    manager.camera_fleet = switches
    manager.camera_controls = switches
    assert manager._speech_camera_active("gate") is active


def test_turning_detection_off_refreshes_speech_immediately() -> None:
    from survng.app.manager import AppManager

    switches = _Switches(True, True)
    refresh = _RefreshCounter()
    manager = AppManager.__new__(AppManager)
    manager.camera_fleet = switches
    manager.camera_controls = switches
    manager.speech = refresh
    assert manager.set_detection("gate", False) is True
    assert switches.calls == [("gate", False)]
    assert refresh.count == 1
    assert manager._speech_camera_active("gate") is False


def test_whisper_cpu_threads_default_to_half_the_cores_capped() -> None:
    from survng.app.speech.transcriber import transcription_cpu_threads

    assert transcription_cpu_threads(0, cpu_count=16) == 4
    assert transcription_cpu_threads(0, cpu_count=4) == 2
    assert transcription_cpu_threads(0, cpu_count=1) == 1
    assert transcription_cpu_threads(6, cpu_count=16) == 6


def test_crashing_worker_backs_off_instead_of_reloading_every_few_seconds() -> None:
    from survng.app.speech.transcriber import RESTART_DELAYS_SECONDS, restart_delay

    delays = [restart_delay(failures) for failures in range(1, 8)]
    assert delays == sorted(delays)
    assert delays[0] == RESTART_DELAYS_SECONDS[0]
    assert delays[-1] == RESTART_DELAYS_SECONDS[-1] >= 300


def test_voice_model_runs_single_threaded() -> None:
    from survng.app.speech.vad import VAD_CPU_PROPERTIES

    assert VAD_CPU_PROPERTIES == {"INFERENCE_NUM_THREADS": 1, "NUM_STREAMS": 1}
