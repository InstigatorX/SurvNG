# SurvNG Video Processing Pipeline

This is the living technical reference for SurvNG's video-processing path. It
describes the implementation currently in the repository, not an aspirational
design. Update it whenever ingest, recording, motion qualification, inference,
incident generation, media storage, or browser playback behavior changes.

Last reviewed: 2026-07-16

## Pipeline At A Glance

```text
Camera
  |
  +-- main stream --------------------------+--------------------------+
  |                                         |                          |
  |                                  FFmpeg stream copy          go2rtc/WebRTC
  |                                         |                    full-screen live
  |                                         v
  |                                10-second MP4 segments
  |                                         |
  |                         +---------------+----------------+
  |                         |                                |
  |                  recording playback             ONVIF-triggered samples
  |                  fMP4/HLS remux cache             around event time
  |                                                          |
  |                                                          v
  |                                                  OpenVINO detection
  |                                                          |
  +-- live/substream --> OpenCV capture --> grayscale ring    |
  |                            |                    |          |
  |                            |             motion qualifier  |
  |                            |                    |          |
  |                            +--> MJPEG/snapshot  +----------+
  |                                                               
  +-- ONVIF PullPoint --> trigger queue --> burst coalescing ------+
                                                                  |
                                                                  v
                                                  event + clean snapshot
                                                  + object metadata
                                                                  |
                                                                  v
                                                     incidents / MQTT / SSE
```

The main principles are:

- Keep camera ingest shared wherever possible.
- Record encoded camera media without transcoding.
- Treat ONVIF motion as a noisy trigger, not proof of an object.
- Run expensive high-resolution inference only after temporal qualification.
- Store clean snapshots and object coordinates separately; draw annotations in
  the browser.
- Remux recording fragments for browser playback without changing their video
  or audio codecs.

## 1. Camera Sources

Each camera can expose two logical sources:

- `main`: the high-resolution stream configured by `stream_url`.
- `live`: the lower-resolution stream configured by `live_stream_url`. If no
  distinct live URL exists, camera source normalization can resolve it to the
  available source.

RTSP streams are normally provided through the configured go2rtc restream. A
Reolink `reolink://` source can use the native Baichuan integration. URL and
Baichuan behavior is normalized by the camera configuration before workers are
started.

### Source responsibilities

| Work | Preferred source | Reason |
| --- | --- | --- |
| Camera tiles | `live` | Faster startup and lower browser/network cost |
| Full-screen live view | `main` | Highest available detail |
| Motion qualification | `live` | Low-cost temporal analysis |
| Object detection | Recorded `main` | High-resolution evidence near event time |
| Main recording | `main` | Archival quality |
| Optional sub recording | `live` | Fast mobile playback and alternate view |

## 2. Live Capture And Browser Delivery

`CameraWorker` keeps the live source open through OpenCV/FFmpeg. It retains the
latest frame for snapshots and MJPEG while sampling a separate grayscale motion
ring. Main-source OpenCV capture is demand-driven and stops after an idle
period; continuous main recording is handled by its own FFmpeg process.

Browser live-view modes currently include:

- Automatic motion mode: snapshot while idle, WebRTC while camera motion is
  active.
- MJPEG: frames served by SurvNG from the latest OpenCV capture.
- WebRTC: SurvNG relays go2rtc signaling; media remains a shared go2rtc stream.

The intended fallback order is WebRTC, near-live recording where supported,
then snapshot refresh. H265 browser compatibility may require a go2rtc
compatibility stream; SurvNG does not make OpenCV frames into WebRTC media.

## 3. Continuous Recording

The recorder launches one managed FFmpeg process for each enabled
`camera/source` pair. Recording uses stream copy (`-c copy`) so video and audio
remain in the camera's original codecs. The configured segment duration
defaults to 10 seconds and is constrained to 2-300 seconds.

Files are organized as:

```text
STORAGE_DIR/recordings/CAMERA_ID/SOURCE/YYYY-MM-DD/HH/YYYYMMDD-HHMMSS.mp4
```

Examples of `SOURCE` are `main` and `live`.

`recordings.sqlite3` indexes segment paths, source, start/end time, duration,
size, health, and stream fingerprints. Recorder maintenance reconciles disk and
database state, validates fragments, prunes stale rows, and detects duplicate
or orphaned recorder processes.

The camera power control stops the camera worker and its recording processes.
The recording control can stop recording independently while leaving live view
and detection available.

## 4. ONVIF Event Ingestion

Each enabled camera has a long-running ONVIF PullPoint listener. It creates and
renews subscriptions, polls in bounded batches, retries failed subscriptions
with backoff, and reconnects after repeated polling failures.

Topics or messages containing configured motion terms are forwarded to
`CameraWorker.handle_motion_event`. The callback is intentionally lightweight:

1. Normalize the camera timestamp to UTC.
2. Publish the raw motion state through the realtime/MQTT path.
3. Add a trigger to the camera's bounded motion queue.
4. Return immediately so ONVIF polling is not blocked by inference.

The queue holds at most 32 triggers. If it is full, the oldest trigger is
dropped in favor of the newest and the drop is counted in runtime telemetry.

## 5. Motion Qualification

Motion qualification reduces insect, lighting, rain, and transient-noise
triggers before the high-resolution OpenVINO cycle.

### Frame ring

The existing live/substream capture is downscaled to 320 pixels wide, converted
to grayscale, and sampled at `motion_qualification.sample_fps`. This does not
open another camera connection. The ring is sized from the configured sample
rate and analysis window with additional history for timestamp jitter.

### Burst coalescing

ONVIF cameras often emit many messages for one physical event. The per-camera
worker collects triggers until `burst_quiet_seconds` elapses, subject to a hard
deadline, and processes the group as one motion burst.

### Scoring

The qualifier analyzes consecutive grayscale frames with thresholded frame
differences, morphology, and connected contours. Its score combines:

- Persistence across the temporal window.
- Centroid direction and velocity consistency.
- Foreground-area stability.
- Distance from image edges.
- Concentration versus fragmentation of changed pixels.
- A penalty for global image changes such as exposure or lighting shifts.

Sensitivity selects the acceptance threshold:

| Sensitivity | Threshold | Behavior |
| --- | ---: | --- |
| High | 0.36 | More permissive; favors recall |
| Balanced | 0.48 | Default calibration |
| Low | 0.60 | More selective; favors noise reduction |

The names describe motion sensitivity: `high` accepts more candidate motion,
while `low` requires stronger temporal evidence.

### Modes

- `off`: coalesce bursts but skip scoring.
- `audit`: score every burst and report whether it would be suppressed, but
  continue through object detection. This is the default.
- `enforce`: stop rejected bursts before high-resolution sampling and object
  detection.

Semantic ONVIF topics containing person, people, human, vehicle, animal, or
face bypass suppression. Manual test triggers also bypass it. If fewer than
four usable frames surround the event timestamp, qualification fails open so a
capture outage cannot silently suppress a real event.

Each camera can inherit or override the global mode and sensitivity. A sampled
percentage of enforced rejections is saved under:

```text
STORAGE_DIR/motion_samples/CAMERA_ID/
```

The sample filename records timestamp, score, and rejection reason. Retention
is capped at 100 samples per camera.

## 6. High-Resolution Object Detection

An accepted motion burst uses finalized main recording fragments rather than
the low-resolution qualification frames. SurvNG samples five target times
around the ONVIF event:

```text
-1.0s, -0.5s, event time, +0.5s, +1.0s
```

For each available target:

1. Find the indexed main recording containing the target timestamp.
2. Decode one frame with FFmpeg, using configured hardware decoding when
   available and CPU fallback otherwise.
3. Nudge the requested offset when damaged timestamps or keyframe placement
   prevent an exact read.
4. Run the configured OpenVINO/Core ML detector.
5. Apply camera detection zones and per-zone eligibility.

The frame with the strongest eligible object confidence is selected. If all
sampled frames contain no eligible object, the frame nearest the event time is
preferred. SurvNG retries briefly while newly written recording segments are
being finalized. If no recorded frame becomes readable, it fails over to the
latest live frame and marks that source in event metadata.

Model thresholds are applied before zone eligibility. Detection boxes use the
source image coordinate system and are persisted with labels, confidence,
zones, and incident eligibility.

## 7. Event And Incident Persistence

Every processed burst creates a motion event in:

```text
STORAGE_DIR/survng.sqlite3
```

An event stores:

- Camera and UTC event timestamp.
- ONVIF topic and message.
- Clean snapshot path.
- Recording path associated with the selected frame.
- Serialized detections and motion-qualification diagnostics.

Snapshots are stored at:

```text
STORAGE_DIR/snapshots/CAMERA_ID/*.jpg
```

Snapshots are intentionally saved without burned-in boxes. The browser draws
annotations from persisted object coordinates, preserving a clean image for
export, zooming, and future reprocessing.

Incident grouping is a presentation/aggregation layer over events from the
same camera and nearby time range. Object and zone eligibility determine which
detections are promoted through incident, MQTT, and Home Assistant object
paths. Motion and object remain distinct event concepts.

## 8. Incident Video

Incident playback derives its time range from the grouped incident, extending
the configured before/after padding around the first and last associated
events. The server locates every recording segment intersecting that window and
builds a cached MP4 clip. Clip generation is serialized per cache key so
concurrent browser requests do not launch duplicate FFmpeg jobs.

Clip generation prefers stream copy. It may span multiple source files, but it
does not intentionally transcode the recording. Source timestamp damage,
missing references, or browser codec limitations can still affect playback.

## 9. Recording Playback

The recordings page first requests merged availability ranges for a single
camera, source, and local calendar day. Detailed segment rows are requested only
around the playback position, reducing initial payload and DOM size.

For browser playback, each independently recorded MP4 is cold-remuxed to an
fMP4 initialization segment and media fragment with codec copy. Results are
cached under:

```text
STORAGE_DIR/playback-cache/fmp4/
```

The virtual HLS playlist:

- Uses program date/time to map media to wall-clock time.
- Offsets fragment timestamps into one continuous playback timeline.
- Fingerprints stream metadata.
- Emits a new initialization map and discontinuity when stream metadata
  changes after a camera reconnect or codec/configuration change.
- Includes audio when present.

Shaka Player is used for consistent browser integration, with Safari-specific
media behavior handled by the playback layer. The current day's availability
is refreshed incrementally as new recording segments finalize.

Cache limits are controlled by `recording_cache_max_gb` and
`recording_cache_max_days`. Cold fragments incur one no-transcode FFmpeg remux;
subsequent reads use the cache.

## 10. Realtime Outputs

The pipeline publishes state through:

- SSE camera-state and incident updates for the React interface.
- MQTT camera motion and object topics.
- MQTT zone/object topics and Home Assistant Discovery entities.

Motion qualification telemetry includes trigger count, coalesced bursts,
passes, audit rejects, enforced suppressions, priority bypasses, insufficient
frame decisions, dropped triggers, queue depth, ring depth, and the last
decision details.

## 11. Lifecycle And Failure Behavior

SurvNG runs camera capture, motion qualification, ONVIF, recording, indexing,
and maintenance as managed workers. Shutdown order stops command intake,
camera/ONVIF workers, face recognition, and recorder processes before process
exit. Systemd then provides boot startup and crash recovery.

Important failure behavior:

- ONVIF subscription failures retry with exponential backoff.
- Motion queue saturation keeps recent triggers and records drops.
- Qualification lacks frames: fail open.
- Recorded event frame unavailable: retry, then use the latest live frame.
- Hardware frame decode fails: retry with CPU decoding.
- Recorder launch fails, including disk-full errors: isolate the source and
  retry after backoff instead of crashing the application.
- Unreadable mature recording fragments are marked unplayable and excluded
  from normal playback.
- Missing media files are pruned from the recording index during maintenance.

## 12. Configuration Reference

Global motion qualification:

```json
{
  "motion_qualification": {
    "mode": "audit",
    "sensitivity": "balanced",
    "sample_fps": 5.0,
    "window_seconds": 1.6,
    "burst_quiet_seconds": 0.5,
    "rejected_sample_rate": 0.05
  }
}
```

Per-camera override:

```json
{
  "id": "front-door",
  "motion_qualification": {
    "mode": "inherit",
    "sensitivity": "inherit"
  }
}
```

Use `audit` until representative daytime, nighttime, weather, insect, and
headlight events have been observed. Enable `enforce` globally only after
checking false-reject behavior; use camera overrides for difficult scenes.

## 13. Verification

Relevant automated coverage includes:

- Motion scoring for coherent movement, erratic edge movement, global
  brightness changes, and insufficient-frame fail-open behavior.
- Non-blocking event enqueue and ONVIF burst coalescing.
- Recorder ownership, failure isolation, indexing, media validation, recording
  range selection, and playback continuity helpers.
- Incident grouping and event clip behavior.

Run backend tests and build the production UI with:

```bash
.venv/bin/python -m unittest discover -s tests -q
npm --prefix frontend run build
```

Long-running recording playback should also be tested in current Safari and
Chrome across repeated seeks and many segment boundaries.

## 14. Current Boundaries And Future Work

The motion qualifier deliberately remains lightweight. It does not currently
maintain persistent object tracks, calculate dense/sparse optical flow, use a
long-lived MOG2/KNN background model, or consume vendor motion bounding boxes.
Those are possible later stages if audit data shows that the current temporal
score cannot separate scene motion from insects reliably enough.

The next evidence-driven progression is:

1. Run `audit / balanced` and collect representative decisions.
2. Review rejected samples and any audit rejects that still produced objects.
3. Tune per-camera sensitivity before changing global behavior.
4. Enable enforcement for stable cameras.
5. Add component tracking or optical flow only where measured failures justify
   the additional CPU and complexity.

## Documentation Update Checklist

When changing this pipeline, update this file in the same commit if the change
affects any of the following:

- Source selection or camera connection count.
- Worker, queue, retry, or shutdown behavior.
- Recording codecs, segmentation, layout, or indexing.
- Motion qualification features, thresholds, modes, or bypasses.
- Detection sampling, model inputs/outputs, zones, or eligibility.
- Snapshot/annotation persistence.
- Incident grouping or clip generation.
- Browser streaming, remux, cache, or fallback behavior.
- MQTT/SSE payload semantics or observability.
