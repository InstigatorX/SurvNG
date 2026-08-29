# SurvNG Video Processing Pipeline

This is the living technical reference for SurvNG's video-processing path. It
describes the implementation currently in the repository, not an aspirational
design. Update it whenever ingest, recording, motion qualification, inference,
incident generation, media storage, or browser playback behavior changes.

Last reviewed: 2026-08-24

## Motion Pipeline Migration

Motion qualification now enters through the typed, per-camera orchestration
contracts in `survng/app/motion_pipeline`. `AppManager` assembles and injects a
separate `MotionPipeline` for every camera from an instance-scoped stage
registry. The pipeline owns per-camera runtime state, executes stages in order,
and records call, failure, last, average, and maximum timing for each stage.

The adaptive qualification preset uses independently registered grayscale
preprocessing, a selective EMA scene model, background difference, robust
statistical thresholding, morphology, connected-component measurements,
noise-aware blob filtering, persistent multi-region tracking, and adaptive
credibility scoring. It learns per-camera noise and brightness, accelerates
learning for whole-scene illumination changes, protects moving pixels from
immediate absorption, and penalizes tiny erratic edge motion. The prior
production frame-difference qualification graph remains available as the
fixed-threshold modular rollback preset and uses independently
registered preprocessing, frame-difference, threshold, morphology, contour
extraction, minimum-area filtering, dominant-centroid tracking, and scoring
stages. The final graph then runs evidence fusion, the persistent event-state
machine, and a transition-only trigger decision so continuous activity does
not create repeated incidents. Each stage consumes and publishes typed context
artifacts without invoking or depending on concrete neighboring stages. The
original all-in-one `legacy_qualifier` and combined `legacy_motion_scorer`
remain registered as parity/reference implementations.

Supporting evidence runs through a separate observation pipeline. The built-in
`onvif_event_evidence` stage normalizes camera motion events into an injected,
bounded `MotionEvidenceRepository`. A later `buffered_evidence_fusion` stage
selects samples by event time and adds aggregated evidence to `MotionContext`.
The repository boundary allows future optical-flow or AI sources to run
independently and join the same fusion stage without calling each other.

New motion implementations are registered explicitly with a
`MotionStageRegistry`; there is no mutable module-level plugin registry. The
factory validates stage IDs and required context artifacts before a camera
pipeline starts. Object inference, recording lookup, event persistence, MQTT,
and SSE remain outside the motion pipeline.

Process-wide camera admission and teardown are owned by
`CameraFleetLifecycle`. It snapshots startup detection preferences, reads the
current recording preference when a queued camera is admitted and starts its recorders,
tracks live camera power changes during progressive admission, releases ONVIF
subscriptions early, and broadcasts nonblocking stop requests before waiting
for the fleet against one absolute deadline. Cameras retain ownership of their
workers; residual camera IDs remain observable and shared inference/recording
services stay alive until every camera reports stopped. Resource `close()` then
runs only for inactive cameras and is not represented as an I/O timeout.
`AppManager` sequences this boundary with inference, recorder, MQTT, and
auxiliary-service lifecycles without detached per-camera shutdown threads.

`InferenceLifecycle` owns the process inference generation as one transaction:
the object detector supervisor, face queue, person/vehicle ReID adapters,
per-camera tracking-session factory and limiter, deferred appearance backfill,
and semantic search. Camera workers are bound once after construction. Runtime
model or tracking changes are prepared before cutover, roll back every swapped
session on failure, and retire old search/backfill generations only after the
replacement is committed. Failed retirement is retained for shutdown retry so
configuration state never points back at a partially closed generation.

`MotionDecisionHandler` owns downstream event persistence and notifications.
It receives object evidence from an injected `RecordedMotionObjectDetector`,
which owns recorded-frame selection, decode, zone-aware inference, and live
fallback. `MotionRuntimeService` is the single lifecycle boundary for the
analysis and decision workers, their queues, retry state, evidence repository,
and pipeline generation. `CameraWorker` only assembles these typed
collaborators and exposes the camera-facing API; it owns no motion policy or
motion worker lifecycle.

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

RTSP streams are normally provided through the configured go2rtc restream. Camera
URLs are normalized by configuration before workers start (`video_backend` is
always URL-backed).

Process startup uses a camera-level admission coordinator instead of launching
every OpenCV capture, ONVIF listener, and recorder at once. By default, two
cameras enter startup concurrently. Each camera gets a bounded first-frame
window; a camera that misses it is marked degraded and keeps reconnecting in
its capture worker while the coordinator advances to the next camera. Capture,
ONVIF, and configured recorders start within the admitted camera group; that
group retains its slot until a first frame arrives (or the bounded window ends),
with a short minimum spacing between groups. Starting the recorders inside the
group also lets their go2rtc attachment warm the upstream stream. This limits
connection and decoder bursts without turning one unavailable camera into a
fleet-wide startup blocker. The HTTP application and MQTT lifecycle expose the
progressive startup state. Configuration-manager cutovers remain transactional
through core service construction and persistence, then expose the same
progressive camera admission rather than blocking the configuration request on
unavailable feeds.

The configured camera concurrency is also the shared native OpenCV-open limit,
so a simultaneous reconnect after a network interruption cannot bypass startup
admission and create a second connection storm. Before admission begins,
orphaned recorder processes are terminated concurrently within one cleanup
window. Recording-index mount migrations run in bounded database batches; a
normal restart trusts the persisted root marker and does not enumerate indexed
paths.

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

`CameraCaptureService` keeps the live source open through OpenCV/FFmpeg. A successful
capture read transfers exclusive ownership of its NumPy buffer to the capture
service, which publishes it as an immutable shared frame. Snapshot, MJPEG, and
motion observers share that buffer; only consumers that explicitly request a
writable latest frame receive a copy. Those copies occur outside the capture
state lock and are counted separately. Motion sampling produces immutable color,
grayscale, and Gaussian-preprocessed derivatives once per admitted frame. The
motion ring shares those derivatives across overlapping continuous-analysis
windows, avoiding repeated frame copies and repeated preprocessing. Cached
derivatives carry their configured preprocessor provenance and are ignored if
the selected pluggable preprocessor changes.

Each camera's lightweight preprocessing worker continues filling its temporal
ring at the configured sample rate even when expensive qualification is busy.
Only qualification competes for the fleet-wide fair analysis limit
(`motion_qualification.max_concurrent_analysis`, default 2). A camera
that cannot immediately obtain a slot retains one fair pending request and
keeps ingesting newer temporal samples. Slot release wakes the next fair camera
without polling or blocking preprocessing; its grant evaluates the latest
coherent window. Capture sequence provides processing order, while wall-clock
time remains event metadata. A backward camera-clock discontinuity resets the
affected temporal runtime instead of suspending motion analysis. Main-source
OpenCV capture is demand-driven and stops after an idle
period; continuous main recording is handled by its own FFmpeg process.

Live capture reconnects with exponential backoff when a stream read or open
fails. After the first failure on a live source, the open deadline escalates to
a longer reconnect timeout before resetting once frames resume. Camera status
reports `capture_connectivity` as `healthy`, `reconnecting`, `offline`, or
`paused` so Live and Admin can distinguish a recovering stream from a powered-off
camera. `capture_reconnects` counts successful live recoveries since the current
capture generation started.

Browser live-view modes currently include:

- Automatic motion mode: snapshot while idle, WebRTC while camera motion is
  active.
- MJPEG: frames served by SurvNG from the latest OpenCV capture.
- WebRTC: SurvNG relays go2rtc signaling; media remains a shared go2rtc stream.
- MSE: if WebRTC fails, SurvNG relays go2rtc fragmented MP4 over the existing
  WebSocket connection before falling back to MJPEG. This keeps go2rtc's API
  private and works through HTTP proxies that support WebSocket upgrades.

SurvNG relays the selected go2rtc stream in its native codec and never creates
or writes transcoding aliases into go2rtc. The live-view fallback order is
native WebRTC, native MSE, then MJPEG. If a selected H265 main stream cannot be
decoded through either native browser transport, SurvNG tries the camera's
native live/substream through WebRTC and MSE before using MJPEG. The viewer
labels that source fallback explicitly. Any intentionally transcoded stream
must be configured and owned in go2rtc rather than synthesized by SurvNG.

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

Topics or messages containing configured motion terms enter the camera's
`MotionRuntimeService` through `CameraWorker.handle_motion_event`. The ingress
path is intentionally lightweight:

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

The existing live/substream capture publishes its stable stored-frame reference
to a one-slot, latest-only motion mailbox at
`motion_qualification.sample_fps`. The capture callback performs no resize or
color conversion. The camera's motion-analysis worker acquires the shared CPU
analysis slot, downscales the newest frame to
`motion_qualification.frame_width`, converts it to grayscale, and appends both
compact derivatives to their bounded rings. If analysis falls behind, a pending
raw frame is replaced rather than building stale work. The global width default
is 320 pixels; cameras with small or distant objects can override it
independently. This does not open another camera connection. The ring is sized
from the configured sample rate, analysis window, and post-trigger horizon with
additional history for timestamp jitter.

### ONVIF event evidence

Each accepted ONVIF motion notification also traverses the observation graph.
The `onvif_event_evidence` stage records the normalized topic, bounded message,
camera event timestamp, local receipt timestamp, source type, and score. Basic
motion events default to `0.55`; semantic topics containing configured person,
vehicle, animal, face, or manual keywords default to `0.95`. These values and
keywords are stage options, not hard-coded fusion policy.

Frame and event observations can arrive concurrently. Irrelevant stages no-op
based on the observation kind. The shared repository is thread-safe and
bounded. Evidence-stage failure is logged but never blocks the established
trigger queue.

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

### Trigger modes

- `camera`: only ONVIF and manual notices enter the event queue. EMA validation
  is optional and never creates an event.
- `adaptive`: accepted EMA motion is the only automatic trigger. Ordinary
  ONVIF notices remain diagnostic evidence only.

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
percentage of rejected validation decisions is saved under:

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
the low-resolution qualification frames. By default SurvNG samples five target
times around the ONVIF event:

```text
-1.0s, -0.5s, event time, +0.5s, +1.0s
```

`detector.event_refinement_stages` can narrow or extend that window. Within each
stage, inference stops early once temporal confirmation is already satisfied.

For each available target:

1. Find the indexed main recording containing the target timestamp.
2. Decode one frame with FFmpeg, using configured hardware decoding when
   available and CPU fallback otherwise.
3. Nudge the requested offset when damaged timestamps or keyframe placement
   prevent an exact read.
4. Run the configured OpenVINO/Core ML detector.
5. Apply camera detection zones and the inherited incident-zone policy.

`detector.require_incident_zone` is the global incident eligibility default,
and each camera can inherit or override it with `require_incident_zone`. When
required, a label that has a matching incident zone must enter one of those
zones. When not required, objects meeting the normal detector threshold remain
eligible anywhere, while matching incident zones may still admit objects at a
zone-specific threshold. Ignore zones always take precedence in both modes.

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

### Detection-triggered object tracking

When the selected event frame contains an incident-eligible object and
`detector.tracking.enabled` is true, SurvNG starts a bounded per-camera
tracking session **after recorded refinement has finished** (or after live
evidence if refinement cannot run). Tracking is identification and cover
enrichment; it does not admit the incident. The existing OpenVINO detector
remains responsible for frame-level boxes, classes, and confidence. A
separate ByteTrack-style tracking-by-detection stage associates those boxes
over time and assigns camera-local track IDs.

The initial incident-eligible detections are confirmed immediately, including
per-camera detections admitted by a threshold lower than the global default.
During the session, lower-confidence boxes may preserve an existing track but
cannot create a new track until a normal detector-confidence observation is
seen. A short lost timeout preserves IDs through missed detections and brief
occlusion. Tracking ends when all tracks expire or `max_session_seconds` is
reached. IDs are local to the event and camera; they are not identities and do
not carry across a service restart or between cameras.

Association first uses predicted overlap, then a conservative center-distance
and containment fallback for rapidly changing boxes. Optional person and
vehicle ReID can reconnect a recently lost track using whole-object appearance
embeddings. ReID runs in its own isolated inference worker and is disabled
unless the corresponding feature and compatible OpenVINO model are configured.
Face-recognition embedding models are not interchangeable with person ReID.

The default SurvNG Hybrid tracker computes embeddings on demand: once when a
track is created, when geometry cannot associate a detection, and periodically
to refresh a matched track's appearance. Ordinary geometry matches between
refreshes do not invoke the ReID worker. `reid_refresh_interval_frames` controls
the refresh cadence; `reid_max_embeddings_per_frame` bounds candidate work when
detector output is unusually crowded or noisy. Per-camera attempts, latency,
failures, object-label counts, attempt reasons, avoided checks, and successful
recoveries appear in Telemetry. Each stored recovery also includes its capture
time, similarity, resumed-track state, and box. Incident replay highlights that
box in amber so an operator can verify the handoff against the recording.
Offline comparisons remain eager so both engines receive identical appearance
inputs and comparison runs are reproducible.

The object-tracking implementation is internally pluggable, but production
sessions intentionally use `survng_hybrid`; historical `bytetrack`, BoT-SORT,
and Deep OC-SORT configuration values normalize to Hybrid. Deep OC-SORT is an
optional, offline comparison adapter rather than a production tracker. It
retains one tracker instance per comparison, prevents association across object
classes, disables global camera-motion compensation for fixed cameras, and can
consume the same isolated OpenVINO appearance embeddings used by Hybrid.
Install `requirements-ultralytics-tracking.txt` to enable Compare. This optional
runtime adds a large PyTorch dependency. The requirements pin the reviewed
Ultralytics version for reproducible installs; SurvNG also accepts compatible
patches on the reviewed 8.4.x API line. Adapter regression tests exercise the
private APIs SurvNG uses. Ultralytics is distributed under AGPL-3.0;
deployments that redistribute SurvNG with this optional dependency should
review the applicable license terms.

Tracking runs only after an eligible object is found and recorded confirmation
has completed (or been dropped). It samples the main camera source at
`sample_fps`, and `max_active_cameras` bounds simultaneous sessions so a burst
of camera activity cannot create unbounded inference and decode load. Track
summaries, trajectories, zones, observation counts, and first/last-seen
timestamps are stored with the originating incident.
`max_tracks_per_session` additionally bounds association work and persisted
metadata if a detector produces an abnormal number of boxes.

The Incidents workspace visualizes that persisted metadata without rerunning
inference. Expanding a tracked incident or opening its viewer draws each track
in a stable color, labels its last stored box with `#<track_id>`, and connects
the sampled centers as a path. New tracking sessions also store a bounded box
history. During incident-video playback, the viewer interpolates those sampled
boxes against the recording clock and progressively draws each path. The
viewer’s **Tracks** control toggles both layers, while the inspector reports the
implementation, sampling rate, observation count, duration, zones, and ReID
recoveries. Historical incidents without box history retain snapshot trails
but cannot show synchronized video boxes. Snapshot boxes represent the last
stored position and can therefore be later than the representative snapshot.

The incident viewer’s **Compare** action is an explicit, offline diagnostic.
It decodes a bounded 30-second recording window, runs OpenVINO detection and optional
person/vehicle appearance extraction once per sampled frame, then gives that identical
detection sequence to SurvNG Hybrid and Deep OC-SORT. The result shows
side-by-side paths, track and observation counts, extra-track-ID and processing
time signals, and lets either result replay over the same recording. Detection
and appearance costs are reported separately because they are shared inputs,
not tracker costs. Extra track IDs are only a fragmentation proxy: without
labeled ground truth, the comparison cannot claim a true ID-switch count or
choose a winner automatically. After watching the replays, an operator can
mark Hybrid, Deep OC-SORT, or no clear winner. SurvNG retains only compact evidence
for the latest 100 compared incidents per camera (not the replay trajectories),
shows the reviewed totals in the incident viewer, and resets an incident's
verdict when that incident is compared again. This evidence never changes the
configured tracker automatically. A single-job limiter prevents concurrent
reviews from competing with live inference.

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
passes, suppressions, priority bypasses, insufficient-frame decisions,
validator fail-opens, dropped triggers, queue depth, ring depth, and the last
decision details.

Camera dataflow telemetry also measures the costs surrounding inference rather
than treating model time as the whole pipeline. Per-source capture status
reports stored-frame copy volume and synchronous observer latency. Motion
analysis reports capture-to-analysis handoff latency, preprocessing latency,
analysis-cycle latency, derived-frame allocation, and explicit frame-copy
volume by reason. It also reports accepted raw-frame handoffs and latest-frame
mailbox replacements. The semantic motion-event coordinator reports queue and retry
high-water marks, evictions, rejections, and exhausted retries. Lifecycle status
lists the workers that remain active for each camera. Bounded latency samples
are exposed live; selected p95/p99 latency and counter deltas are persisted for
the two-hour and seven-day Telemetry charts. These measurements provide the
baseline used to evaluate lifecycle, frame-ownership, and hot-path changes.

Rejected decisions are also indexed in the `motion_audits` table inside
`survng.sqlite3` and displayed under Config > Motion Audit. Rejected entries
report that detection was skipped and use the configured rejected-frame
sampling rate to attach an image. Borderline rescues also report whether
OpenVINO subsequently found an eligible object. The viewer is
paginated and filterable by camera and detector outcome. Its card grid uses the
remaining browser height; selecting a thumbnail opens a full-viewport image
overlay with decision details and keyboard previous/next navigation.

## 11. Lifecycle And Failure Behavior

SurvNG runs camera capture, motion qualification, ONVIF, recording, indexing,
and maintenance as managed workers. Shutdown order stops command intake,
camera/ONVIF workers, face recognition, the isolated inference processes, and
recorder processes before process exit. Systemd then provides boot startup and
crash recovery.

Each camera exposes an explicit `stopped`, `starting`, `running`, `stopping`,
`failed`, or `closed` lifecycle phase and a runtime-generation counter. A short
state lock protects only those in-memory transitions. A separate operation lock
serializes start, stop, and close commands while blocking capture, ONVIF, and
worker joins run without holding the state lock. Status and health reads
therefore remain available while a camera connection is slow to open or close,
and failures retain their last lifecycle error for diagnosis.

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
    "mode": "camera",
    "sensitivity": "balanced",
    "frame_width": 320,
    "sample_fps": 5.0,
    "window_seconds": 1.6,
    "post_trigger_seconds": 2.5,
    "burst_quiet_seconds": 0.5,
    "rejected_sample_rate": 0.05,
    "borderline_rescue_enabled": true,
    "borderline_margin": 0.03,
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

The built-in final graph uses adaptive validation, starts after one accepted
decision, releases after three rejected decisions, applies a five-second
cooldown, and expires state after 10 seconds of inactivity. Custom final graphs
may select registered evidence sources through the generic fusion interface.

Fusion policies are `bypass`, `audit` (adaptive only), `any`, `all`, and
`weighted`. Unavailable or unwarmed selected validators fail open when
`fail_open` is enabled. Generic
evidence producers only need to write
`score` and optional `warmed` values to the per-camera repository. The final
graph must provide a trigger `decision`; preflight validation rejects partial
graphs before configuration is persisted.

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

Start with camera-triggered mode and EMA validation. Review representative
daytime, nighttime, weather, insect, and headlight events before switching a
camera with unreliable ONVIF to visual-triggered mode.

## 13. Verification

Relevant automated coverage includes:

- Motion scoring for coherent movement, erratic edge movement, global
  brightness changes, and insufficient-frame fail-open behavior.
- EMA scene learning, persistent motion tracking, evidence aggregation, and
  fail-open validator behavior.
- Concurrent ONVIF event normalization, semantic priority scoring, bounded
  storage, and event-window aggregation.
- Event-state activation/release hysteresis, cooldown, timeout reset, and
  per-camera isolation.
- Bypass, adaptive-only, any-source, all-source, weighted, and source-only
  evidence-fusion policies.
- Non-blocking event enqueue and ONVIF burst coalescing.
- Recorder ownership, failure isolation, indexing, media validation, recording
  range selection, and playback continuity helpers.
- Incident grouping and event clip behavior.

Run backend tests and build the production UI with:

```bash
scripts/run-tests.sh -q
npm --prefix frontend run build
```

Long-running recording playback should also be tested in current Safari and
Chrome across repeated seeks and many segment boundaries.

## 14. Current Boundaries And Future Work

The repository and fusion stage accept multiple independently produced motion
sources, but SurvNG does not yet calculate dense/sparse optical flow, maintain
semantic object tracks, use a KNN background model, or consume vendor motion
bounding boxes beyond normalized ONVIF event signals. Those remain possible
source stages if audit data shows that EMA cannot separate scene motion from
insects reliably enough.

The next evidence-driven progression is:

1. Run camera-triggered / balanced with adaptive validation.
2. Review rejected samples and borderline rescues that produced objects.
3. Tune per-camera sensitivity before changing global behavior.
4. Use visual-triggered mode only for cameras with unreliable ONVIF.
5. Add optical flow only where measured failures justify the additional CPU
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
