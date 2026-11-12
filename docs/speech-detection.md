# Speech detection and transcription

SurvNG can listen to a camera's audio track, detect when someone is speaking,
and optionally transcribe English speech locally with Whisper. Everything is
**off by default** at two levels: the global `speech.enabled` switch and each
camera's own `speech.enabled`. No audio leaves the host.

> Recording or transcribing conversations is regulated in many places
> (all-party consent laws, GDPR, workplace rules) and is often treated more
> strictly than video. Check what applies to you before enabling it.

## Pipeline

```
go2rtc / camera stream
  └─ survng-audio-listener (FFmpeg, one per camera; 16 kHz mono PCM on a pipe)
       └─ Silero VAD on OpenVINO CPU (32 ms frames, ~free when quiet)
            └─ segmenter (hysteresis, pre-roll, hangover, max length)
                 ├─ speech_segments row  (speech.sqlite3)
                 ├─ MQTT  survng/camera/<id>/speech  ON / OFF
                 └─ transcription queue (bounded per camera)
                      └─ survng-speech worker process (Whisper, OpenVINO GenAI)
```

- **Audio source.** The listener reads `camera.speech.source` (`main` by
  default, because sub-streams often have no audio) through the same input
  helpers as the recorder, so go2rtc restreams are reused. When go2rtc reports
  that the stream has no audio track, the listener stays idle and rechecks every
  10 minutes instead of restart-looping.
- **Voice activity.** `silero_vad_openvino_16k.onnx` (MIT) is shared across
  cameras. Each camera has its own recurrent state. Frames below about −60 dBFS
  skip inference.
- **Segments.** Speech must last `min_speech_ms` to count. `pre_roll_ms` of
  audio before onset is included. A segment ends after `hangover_ms` of
  silence, and long speech is split at `max_segment_seconds`.
- **Transcription.** Whisper runs in its own spawned process, never in the
  detector pool or the API process, so slow transcription cannot delay
  incident inference. Queues are bounded per camera (`max_queue_per_camera`).
  When a queue is full, its oldest segment is marked `skipped` (`queue_full`).
  Cameras are served round-robin. A crashed or wedged worker is restarted with
  backoff.
- **Audio retention.** Audio is held only in memory until transcription.
  Playback uses the camera recording for the segment's time range. Segments
  that were still pending when SurvNG stopped are marked `skipped` at the next
  start.

## Models and devices

| `transcription_device` | `transcription_model: auto` | Fallback |
|---|---|---|
| `CPU` | `base.en` | — |
| `GPU` | `small.en` | If the GPU or `small.en` cannot load, `base.en` runs on CPU. The status `reason` says so. |

Set `transcription_model` to `tiny.en`, `base.en`, `small.en` or `medium.en` to
override the automatic choice. Models are read from
`<whisper_models_dir>/whisper-<model>/`.

Install the models with the model installer. `--speech` is opt-in, and the
installer never turns listening on:

```bash
./install-docker-models.sh --speech --device GPU
```

That writes:

- `speech/silero_vad_openvino_16k.onnx`, extracted from the pinned
  `silero-vad` wheel and SHA-256 verified
- `speech/whisper/whisper-small.en/` and `speech/whisper/whisper-base.en/`, the
  `OpenVINO/whisper-*-int8-ov` packages
- `speech.vad_model_path`, `speech.whisper_models_dir` and
  `speech.transcription_device` in `config.json`

## Configuration

```json
{
  "speech": {
    "enabled": true,
    "vad_model_path": "/models/speech/silero_vad_openvino_16k.onnx",
    "whisper_models_dir": "/models/speech/whisper",
    "transcription_device": "CPU",
    "transcription_model": "auto",
    "max_queue_per_camera": 8,
    "retention_days": 7,
    "mqtt_publish_transcripts": false
  },
  "cameras": [
    {
      "id": "front-door",
      "speech": {
        "enabled": true,
        "transcribe": true,
        "source": "main",
        "vad_threshold": 0.5,
        "min_speech_ms": 300,
        "hangover_ms": 800,
        "pre_roll_ms": 500,
        "max_segment_seconds": 30
      }
    }
  ]
}
```

Speech settings are not hot-applied yet. Changing them goes through the
standard manager reload, which restarts camera workers.

## Outputs

- **API**
  - `GET /api/speech/status` (read scope): service, per-camera and transcriber
    state.
  - `GET /api/speech/segments` (admin scope, because transcripts are
    conversation content): filters are `camera_id` (repeatable), `start` and
    `end` (ISO 8601 with a timezone), `q` (full-text search over transcripts),
    `status`, `before_id` and `limit`.
  - `GET /api/speech/segments/{id}` (admin scope).
- **Segment statuses**
  - `detected`: no transcription requested.
  - `queued`, then `transcribing`, then `done`.
  - `no_text`.
  - `low_confidence`: stock Whisper filler such as "Thanks for watching!".
  - `failed`.
  - `skipped`: `too_short`, `queue_full`, `shutdown`, `restart`, or the model
    is unavailable.
- **MQTT and Home Assistant**
  - Each speech camera gets a `Speech` binary sensor
    (`device_class: sound`) on `survng/camera/<id>/speech`.
  - Transcript text is published to `survng/camera/<id>/speech/transcript` only
    when `speech.mqtt_publish_transcripts` is `true`.
- **`survngctl status`** has a `speech` section with listener state, failure
  category, restart and segment counts, and transcriber device, model, queue
  and drop counters. Transcript text never appears in this snapshot, in logs,
  or in the observability socket.
- **Retention.** Segments older than `speech.retention_days` are deleted
  hourly, independent of video retention.

## Listener states

| State | Meaning |
|---|---|
| `listening` | Audio is flowing; no speech right now. |
| `speaking` | A segment is open. |
| `no_audio_track` | go2rtc or FFmpeg reports no audio on the selected source; rechecked every 10 min. |
| `restarting` | FFmpeg exited. `failure` gives the category: `stream_unavailable`, `stream_unauthorized`, `audio_decode_error`, `ffmpeg_error` or `stream_ended`. |
| `camera_off` | The camera is powered off in SurvNG. |

## Not yet included

- Web UI for transcripts and per-camera speech settings. Today they are
  configured through `config.json` or the configuration API.
- Linking segments to incidents.
- Keyword notifications.
- Hot-applying speech config changes without a manager reload.
- Filtering speech to moments when a person is visible (to suppress TV or
  radio).
