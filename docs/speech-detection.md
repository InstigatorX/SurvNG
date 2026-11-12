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

## Speech-triggered incidents

Each listening camera has an **Incidents** setting:

- **Show speech on incidents** (`"incidents": "attach"`, default): speech
  appears on incidents that video creates; it never creates one.
- **Speech also creates incidents** (`"trigger"`): once one utterance has
  `trigger_min_speech_ms` of voiced audio (1.5 s by default), SurvNG asks
  the camera's motion pipeline for an incident check, dated to when the
  speech began. This does not turn on the Motion sensor, set "last
  motion", or count as an ONVIF event, and it works in every motion mode.
  The check snapshots, runs object detection on the recording and creates
  or extends an incident labelled with a purple **Speech** trigger
  ("The camera heard speech." when nothing else explains it).

**On cameras with ignore or incident zones** speech isn't placed in a zone
(audio has no position), so zones don't decide it. The video can explain
it away instead:

| Video at the time of speech | Result |
|---|---|
| A person in an eligible zone (including a resident sitting still) | Incident: "heard speech while a person was in an eligible zone" |
| Only people in ignore zones (e.g. sidewalk passers-by) | No incident: the speech is attributed to them |
| Nobody visible, or only vehicles/other objects | Incident: "The camera heard speech" |
| Video couldn't be checked (detection failed or unavailable) | Incident, marked "the video could not be checked" |

Known limits: someone talking out of view while a pedestrian is visible in
an ignore zone is missed, and loud out-of-frame speech from the sidewalk
can still create an incident. A check that was also triggered by the camera
itself follows the normal zone rules.

Every check costs a recorded detection job, so speech has its own limits:
at most one check per `trigger_cooldown_seconds` (60 s) and
`trigger_max_per_5m` (5) per camera. A long utterance split into 30 s
segments still counts as one. Checks are requested on a separate thread so
audio reading never waits for the motion pipeline. Admin → Detection →
Speech shows how many checks each camera requested and how many the limits
held back.

**Notifications.** A speech incident with a person in an eligible zone
notifies like any person incident. For speech-only incidents (nobody
eligible visible) each camera's `notify` setting decides: `"motion"`
(default) behaves like motion-only incidents, so the global "exclude
motion" setting applies; `"always"` or `"never"` override it.

**Loudness (recorded only).** Each segment stores its voiced level
(`voiced_dbfs`) and the camera's rolling background level (`noise_dbfs`);
Search → Speech shows them as "-31 dBFS · 18 dB over background".
Nothing is filtered on loudness yet: microphone gain and automatic gain
control make absolute levels unreliable for judging distance, so these
numbers are collected first to calibrate a filter from real data.

## CPU use

Measured on a 4-core host:

| Part | Cost |
|---|---|
| FFmpeg listener | About 0.5% of one core per camera, even with a 1080p video stream underneath (video is not decoded). |
| Voice detection | About 2% of one core per camera with continuous background noise. The model runs single-threaded; OpenVINO's default threading cost about 4× more. Near-silent audio skips inference. |
| Whisper | The only heavy part, and only while a transcribing camera hears speech. It runs in the `survng-speech` process at reduced priority (nice +10) with `transcription_cpu_threads` threads (0 = half the cores, up to 4). A camera near a TV or radio can keep it busy continuously; turn off Transcribe for that camera, or keep listening only. |

A worker that crashes repeatedly backs off 5 s → 15 s → 60 s → 5 min
instead of reloading Whisper every few seconds.

To see where CPU goes on a host:

```bash
ps -o pid,pcpu,etime,comm -C survng-speech,survng-audio-li
```

```bash
./survngctl status
```

`survngctl status` shows the transcriber's queue depth, completed, failed and
restart counts in its `speech` section.

## Configuration

In the web UI, go to **Admin → Detection → Speech**. The top card holds the
global switch, a consent notice, which models are installed, the transcription
device and model, retention, and the MQTT transcript opt-in. The **Cameras**
card lists every camera with Listen and Transcribe switches, the audio source,
the live listener state, and optional tuning. The same settings in
`config.json`:

```json
{
  "speech": {
    "enabled": true,
    "vad_model_path": "/models/speech/silero_vad_openvino_16k.onnx",
    "whisper_models_dir": "/models/speech/whisper",
    "transcription_device": "CPU",
    "transcription_model": "auto",
    "transcription_cpu_threads": 0,
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
        "max_segment_seconds": 30,
        "incidents": "attach",
        "trigger_min_speech_ms": 1500,
        "trigger_cooldown_seconds": 60,
        "trigger_max_per_5m": 5,
        "notify": "motion"
      }
    }
  ]
}
```

A camera listens only while it is powered on **and** its Detection switch is
on (from the Live view, the API, MQTT or Home Assistant). Turning either off
stops its listener immediately, and turning both back on restarts it.

Speech settings are not hot-applied yet. Changing them goes through the
standard manager reload, which restarts camera workers.

## Outputs

- **Search page.** **Search → Speech** lists recent speech and searches
  transcripts. It reuses the page's camera filter, and **Play recording**
  opens the timeline at the moment speech started. The Visual/Speech switch
  appears once speech is enabled. Transcripts are admin-only.
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
- **Retention** (checked hourly, also while speech is turned off):
  - Speech newer than `speech.retention_days` is kept.
  - Older speech that isn't part of an incident is deleted.
  - Older speech that is part of an incident keeps its transcript while a
    recording (main or sub stream) still covers that moment. When the
    recording is removed, the transcript text and its search entry are
    deleted, but the incident still shows that speech happened
    ("Transcript expired with the recording"). On a camera that doesn't
    record, transcripts last `retention_days`.
- **Speech on incidents.** Speech that overlaps an incident's time on the
  same camera (with a 5 s margin) appears on that incident: a microphone
  badge in the list, a purple Speech trigger dot when speech started the
  check, and a Speech transcript panel in the incident view. Incident APIs
  carry only counts; the transcript text comes from the admin-only speech
  API.

## Listener states

| State | Meaning |
|---|---|
| `listening` | Audio is flowing; no speech right now. |
| `speaking` | A segment is open. |
| `no_audio_track` | go2rtc or FFmpeg reports no audio on the selected source; rechecked every 10 min. |
| `restarting` | FFmpeg exited. `failure` gives the category: `stream_unavailable`, `stream_unauthorized`, `audio_decode_error`, `ffmpeg_error` or `stream_ended`. |
| `paused` | The camera is powered off, or its Detection switch is off. Speech follows both. |

## Not yet included

- Linking segments to incidents.
- Keyword notifications.
- Hot-applying speech config changes without a manager reload.
- Filtering speech to moments when a person is visible (to suppress TV or
  radio).
