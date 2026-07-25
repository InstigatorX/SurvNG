# SurvNG Video Processing Pipeline

This is the living technical reference for SurvNG's video-processing path. It
describes the implementation currently in the repository, not an aspirational
design. Update it whenever ingest, recording, motion qualification, inference,
incident generation, media storage, or browser playback behavior changes.

Last reviewed: 2026-07-25

## Motion Pipeline Migration

Motion qualification now enters through the typed, per-camera orchestration
contracts in `survng/app/motion_pipeline`. `AppManager` assembles and injects a
separate `MotionPipeline` for every camera from an instance-scoped stage
registry. The pipeline owns per-camera runtime state, executes stages in order,
and records call, failure, last, average, and maximum timing for each stage.

The production frame-difference path now uses independently registered
preprocessing, frame-difference, threshold, morphology, contour extraction,
minimum-area filtering, dominant-centroid tracking, scoring, event-state, and
trigger-decision stages. Each stage consumes and publishes typed context
artifacts without invoking or depending on concrete neighboring stages. The
original all-in-one `legacy_qualifier` and combined `legacy_motion_scorer`
remain registered as parity/reference implementations.

Continuous background evidence runs through a separate observation pipeline.
Its registered `opencv_mog2_evidence` stage owns the OpenCV model in per-camera
pipeline runtime and publishes samples to an injected, bounded
`MotionEvidenceRepository`. A later `buffered_evidence_fusion` stage selects
samples by event time and adds aggregated evidence to `MotionContext`. The
repository boundary allows additional ONVIF, optical-flow, or AI sources to
run independently and join the same fusion stage without calling each other.

New motion implementations are registered explicitly with a
`MotionStageRegistry`; there is no mutable module-level plugin registry. The
factory validates stage IDs and required context artifacts before a camera
pipeline starts. Object inference, recording lookup, event persistence, MQTT,
and SSE remain outside the motion pipeline.

`MotionDecisionHandler` owns downstream event persistence and notifications.
It receives object evidence from an injected `RecordedMotionObjectDetector`,
which owns recorded-frame selection, decode, zone-aware inference, and live
fallback. `CameraWorker` coordinates these components but no longer implements
their policies.

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
- MSE: if WebRTC fails, SurvNG relays go2rtc fragmented MP4 over the existing
  WebSocket connection before falling back to MJPEG. This keeps go2rtc's API
  private and works through HTTP proxies that support WebSocket upgrades.

The live-view fallback order is WebRTC, MSE, then MJPEG. H265 browser
compatibility may require a go2rtc compatibility stream; SurvNG does not make
OpenCV frames into WebRTC media.

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

The existing live/substream capture is downscaled to the configured
`motion_qualification.frame_width`, converted to grayscale, and sampled at
`motion_qualification.sample_fps`. The global default is 320 pixels; cameras
with small or distant objects can override it independently. This does not open
another camera connection. The ring is sized from the configured sample rate,
analysis window, and post-trigger horizon with additional history for timestamp
jitter.

### MOG2 background audit and blob tracking

When `mog2_audit_enabled` is active and the effective camera mode is `audit`,
the same grayscale frame samples also feed a per-camera OpenCV MOG2 background
model. No additional stream or camera connection is opened. The model warms for
approximately two seconds, separates foreground from its learned background,
and applies morphology before extracting connected blobs.

`CameraWorker` sends each sampled grayscale frame through the lightweight
motion observation pipeline. The MOG2 stage retains its model in that
pipeline's per-camera runtime and writes bounded evidence samples to the
camera's injected repository. The event-time fusion pipeline reads only the
requested time range. `CameraWorker` therefore owns neither the MOG2 model nor
its sample history and has no direct dependency on its aggregation algorithm.

Blobs are associated across samples using normalized centroid distance and
bounding-box overlap. Tracks survive short detection gaps and report
persistence, age, hit count, area stability, direction coherence, edge
occupancy, foreground ratio, and a separate `mog2_score`. The strongest track
within an ONVIF event window is stored with the normal qualification features
using the `mog2_` prefix.

Audit entries also retain up to six active tracks as normalized bounding boxes
and the last 30 centroid positions. The browser draws stable track IDs,
outlines, and trails over the clean audit image; no annotated image is written.

MOG2 evidence is observational only. It does not alter the existing motion
score, acceptance decision, suppression behavior, borderline rescue, or
OpenVINO execution. This allows real daytime, nighttime, weather, and insect
events to establish whether background modeling is reliable before it becomes
part of enforcement. The model history is globally configurable; collection
can be disabled globally or overridden per camera.

### Burst coalescing

ONVIF cameras often emit many messages for one physical event. The per-camera
worker collects triggers until `burst_quiet_seconds` elapses, subject to a hard
deadline, and processes the group as one motion burst.

### Timing And Rolling Windows

Each ONVIF trigger retains both the camera-provided event time and SurvNG's
local receipt time. Because decoded RTSP frames can arrive later than the ONVIF
message, the qualifier evaluates overlapping windows until
`post_trigger_seconds` has elapsed and keeps the strongest result. This avoids
assuming that camera event timestamps and OpenCV frame-receipt timestamps share
the same media latency.

If every usable window has an all-zero foreground score, the result is marked
`no_temporal_signal` and fails open. An apparently static window is treated as
inconclusive rather than proof that the ONVIF trigger was false.

### Scoring

The qualifier analyzes consecutive grayscale frames with thresholded frame
differences, morphology, and connected contours. Its score combines:

- Persistence across the temporal window.
- Centroid direction and velocity consistency.
- Foreground-area stability.
- Distance from image edges, with relief for coherent inward trajectories and
  stable tracks that travel along an edge.
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

### Borderline object rescue

When enabled, a rejected score within `borderline_margin` of the active
threshold is allowed through the high-resolution detector. The motion burst is
accepted only when that detector finds an incident-eligible object. The default
margin is `0.03`; clearly weak motion does not incur inference. Global rescue
settings can be inherited or overridden per camera.

Each camera can inherit or override the global mode and sensitivity. A sampled
percentage of enforced rejections is saved under:

```text
STORAGE_DIR/motion_samples/CAMERA_ID/
```

The sample filename records timestamp, score, and rejection reason. Retention
is capped at 100 samples per camera.

### AI audit advisor

The optional audit advisor supports OpenAI, Google Gemini, and
OpenAI-compatible multimodal APIs. A manual Analyze action sends the selected
audit image plus its motion features, linked object detections, effective
settings, and aggregated recent audit outcomes. Stream credentials, camera
URLs, recordings, and unrelated events are not included.

Provider output is constrained to a typed recommendation schema. SurvNG accepts
changes only for documented motion settings and validates their bounds on the
server. Applying recommendations is disabled by default and always requires an
explicit UI confirmation. Enabling apply writes the selected changes to
`config.json` and reloads camera workers.

API keys are stored in local `config.json`; keep that file out of source
control and restrict its filesystem permissions.

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

Object detection and face embedding execute in two independent spawned
inference processes. The object worker owns the configured detector and uses
the latency-focused GPU device. The face worker owns the embedding and landmark
models and uses the independently configured CPU or AUTO device. Face backlog
therefore cannot queue ahead of a motion-triggered object request.

Each worker has its own reusable 64 MB shared-memory frame buffer and duplex IPC
connection. Uvicorn sends only request metadata and compact results, preserving
source-coordinate boxes without serializing full NumPy frames through a process
queue. Calls within each worker remain intentionally serialized to match the
single-stream latency configuration.

The workers have independent lifecycle supervision, crash counters, restart
delays, and CPU fallback windows. A native face-model failure does not restart
or interrupt object detection, and an object-detector failure does not discard
the face-recognition queue. Worker PID, generation, restart count, configured
device, and fallback state are exposed through detector and face status APIs.
On Linux, the children also set their kernel-visible process names to
`survng-object` and `survng-face`, making GPU ownership identifiable in
`intel_gpu_top`, `top`, `htop`, and `ps`.

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

Rejected decisions are also indexed in the `motion_audits` table inside
`survng.sqlite3` and displayed under Config > Motion Audit. Audit-mode entries
reuse the clean event snapshot and report whether OpenVINO subsequently found
an eligible object. Enforced entries report that detection was skipped and use
the configured rejected-frame sampling rate to attach an image. The viewer is
paginated and filterable by camera and detector outcome. Its card grid uses the
remaining browser height; selecting a thumbnail opens a full-viewport image
overlay with decision details and keyboard previous/next navigation.

## 11. Lifecycle And Failure Behavior

SurvNG runs camera capture, motion qualification, ONVIF, recording, indexing,
and maintenance as managed workers. Shutdown order stops command intake,
camera/ONVIF workers, face recognition, the isolated inference processes, and
recorder processes before process exit. Systemd then provides boot startup and
crash recovery.

The object process owns its OpenVINO `Core`, compiled detector, and GPU context.
The face process separately owns the embedding and landmark models on its
configured CPU or AUTO device. Uvicorn does not perform OpenVINO device or
model probes directly. A native OpenVINO, OpenCL, or IGC fault therefore
terminates only the affected inference child; the other inference workload,
live view, recording, ONVIF, MQTT, and HTTP control plane remain available.

The parent detects a closed IPC connection or dead child and restarts that
worker after a short backoff. Three crashes inside ten minutes activate a
30-minute CPU fallback for only the repeatedly failing worker. Per-worker PID,
generation, restart count, last exit, fallback state, queue depth, and model
timings are included in status responses. Both inference children disable core
dumps so a native failure cannot produce another multi-gigabyte Uvicorn memory
image.

Recording-cache prewarming is stopped before the other workers. An active
prewarm remux runs in its own process group, receives `SIGTERM` immediately,
and is force-killed and reaped if it does not exit within the grace period.
Partial cache output is removed. This prevents a blocking prewarm FFmpeg child
from surviving into native runtime teardown or the next service generation.

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
    "frame_width": 320,
    "sample_fps": 5.0,
    "window_seconds": 1.6,
    "post_trigger_seconds": 2.5,
    "burst_quiet_seconds": 0.5,
    "rejected_sample_rate": 0.05,
    "borderline_rescue_enabled": true,
    "borderline_margin": 0.03,
    "mog2_audit_enabled": true,
    "mog2_history_seconds": 30.0,
    "pipeline": {
      "qualification": [],
      "observation": [],
      "fusion": []
    }
  }
}
```

Per-camera override:

```json
{
  "id": "front-door",
  "motion_qualification": {
    "mode": "inherit",
    "sensitivity": "inherit",
    "frame_width": null,
    "borderline_rescue_enabled": null,
    "borderline_margin": null,
    "mog2_audit_enabled": null,
    "pipeline": {
      "qualification": null,
      "observation": null,
      "fusion": null
    }
  }
}
```

Empty global pipeline lists select SurvNG's built-in graphs. A `null`
per-camera graph inherits the global graph; a non-empty per-camera list
replaces that graph for only that camera. Each configured item contains a
stable stage ID, a registered implementation name, and implementation-specific
options. For example, this swaps one camera back to the parity qualifier
without changing application code:

```json
{
  "motion_qualification": {
    "pipeline": {
      "qualification": [
        {
          "stage_id": "qualification",
          "implementation": "legacy_qualifier",
          "options": {}
        }
      ]
    }
  }
}
```

Configured graphs are checked for non-empty names, duplicate IDs, registered
implementations, and artifact ordering before config is saved or workers are
stopped. Camera status reports the resolved implementation, options, and
whether each graph came from built-in defaults, global config, or a camera
override.

AI audit advisor:

```json
{
  "audit_ai": {
    "enabled": false,
    "provider": "openai",
    "api_key": "",
    "base_url": "",
    "model": "",
    "timeout_seconds": 45.0,
    "allow_apply_recommendations": false
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
- MOG2 warmup, persistent slow-blob tracking, evidence aggregation, and
  audit-only configuration inheritance.
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

The modular observation and fusion paths now maintain audit-only MOG2
foreground tracks, but those tracks do not participate in suppression. The
repository and fusion stage can accept multiple independently produced motion
sources, but SurvNG does not yet calculate dense/sparse optical flow, maintain
semantic object tracks, use a KNN background model, or consume vendor motion
bounding boxes. Those remain possible source stages if audit data shows that
frame differences plus MOG2 cannot separate scene motion from insects reliably
enough.

The next evidence-driven progression is:

1. Run `audit / balanced` and collect representative decisions.
2. Review rejected samples and any audit rejects that still produced objects.
3. Tune per-camera sensitivity before changing global behavior.
4. Enable enforcement for stable cameras.
5. Compare `mog2_*` evidence with object-found and no-object audits before
   defining any consensus suppression rule.
6. Add optical flow only where measured failures justify the additional CPU
   and complexity.

## 15. MQTT Incident Lifecycle

When MQTT and incident publication are enabled, SurvNG publishes non-retained
JSON messages to `<topic_prefix>/events/incidents`. The publisher uses the same
45-second per-camera grouping rule and stable first-event identity as the
Incidents UI.

Each incident emits:

- `new` when its first persisted event arrives.
- `updated` when another event joins the same incident.
- `complete` after 45 seconds without another event, or during a clean shutdown.

Consumers that require exactly one notification per incident should process
only `state == "complete"`. Consumers that favor immediate notification can
use `new` and update or deduplicate by `incident_id` as `updated` messages
arrive. Messages use the configured MQTT QoS and are never retained.

The schema includes `schema_version`, stable `incident_id`, camera identity,
start/end timestamps, event IDs/counts, summarized object classes with maximum
confidence and zones, the representative event, and base-path-aware snapshot
and Incidents URLs. This is intended as the stable Node-RED/Home Assistant
notification boundary; presentation rules, quiet hours, and target devices
remain outside SurvNG.

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
