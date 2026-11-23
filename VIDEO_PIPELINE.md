# SurvNG Video Processing Pipeline

This is the living technical reference for SurvNG's video-processing path. It
describes the implementation currently in the repository, not an aspirational
design. Update it whenever ingest, recording, motion qualification, inference,
incident generation, media storage, or browser playback behavior changes.

Last reviewed: 2026-11-22 (motion qualification, detection sampling and
tracking sections rewritten from code; other sections not re-verified)

## Motion Pipeline Structure

Each camera gets its own `MotionPipeline` instances, assembled by `AppManager`
from an instance-scoped `MotionStageRegistry`
(`survng/app/motion_pipeline`). A pipeline owns per-camera runtime state, runs
its stages in order, and records call, failure, last, average and maximum
timing per stage. The factory validates stage IDs, options and required
artifacts before a camera starts.

Three graphs exist per camera:

- **Qualification** — Enhanced Motion Analysis (EMA), the only qualification
  graph: `gray_blur` preprocessing, `adaptive_ema_background`,
  `adaptive_statistical_threshold`, `ema_zone_exclusion`, `open_close`
  morphology, `connected_component_blobs`, `adaptive_blob_filter`,
  `persistent_centroid_tracker`, `adaptive_motion_score` and
  `illumination_change_filter`. The former fixed-threshold frame-difference
  graph is retired; a configuration that names its stages
  (`frame_difference`, `fixed_threshold`, `contour_blobs`, `minimum_area`,
  `dominant_centroid`, `default_motion_score`) is rejected with an explicit
  error.
- **Observation** — `onvif_event_evidence` records each normalized camera
  notice in a bounded per-camera `MotionEvidenceRepository`. It is diagnostic
  only: camera status shows it, but no decision consumes it.
- **Fusion** — `buffered_evidence_fusion` is the EMA validation switch. Policy
  `audit` keeps the EMA decision; `bypass` disables EMA validation (camera
  mode). No evidence source can be selected, and other policies are rejected.

Downstream of qualification:

- `MotionRuntimeService` owns the analysis and decision workers, their queues,
  retry state, the evidence repository and the pipeline generation.
- `MotionEpisodeController` (`ema_v2.py`) merges camera notices and qualified
  EMA edges into per-camera motion episodes and is the only owner of detector
  reservation, admission, follow-up limits and incident linkage.
- `MotionDecisionOrchestrator` coalesces triggers into a batch, qualifies it and
  hands accepted work to `MotionIncidentService`, which persists the incident
  and queues durable recorded refinement.
- `MotionDecisionHandler` runs object detection through
  `RecordedMotionObjectDetector` (fast live check and recorded main-stream
  refinement) and owns event persistence and notifications.
- `CameraWorker` assembles these collaborators and exposes the camera-facing
  API; it owns no motion policy.

Process-wide camera admission and teardown are owned by
`CameraFleetLifecycle`, and `InferenceLifecycle` owns the inference generation
(object detector supervisor, face queue, ReID adapters, tracking factory and
limiter, deferred appearance backfill and semantic search) as one transaction.

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
  +-- live/substream --> FFmpeg capture --> grayscale ring    |
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
every FFmpeg capture, ONVIF listener, and recorder at once. By default, two
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

The configured camera concurrency is also the shared external FFmpeg-open limit,
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

`CameraCaptureService` keeps the live source open through external FFmpeg. A successful
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
FFmpeg capture is demand-driven and stops after an idle
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
- MJPEG: frames served by SurvNG from the latest FFmpeg capture.
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
3. Offer the notice to the camera's motion episode controller, which decides
   whether it reserves a detection request or joins an existing episode.
4. Enqueue a trigger for a reserved request and return immediately, so ONVIF
   polling is never blocked by inference.

The in-memory trigger queue holds 32 triggers. Admitted triggers are also
written to a durable queue in the event store; when the in-memory queue is
full, the trigger stays in that durable FIFO and is delivered later rather
than dropped. Triggers are retried at most twice.

## 5. Motion Qualification

Motion qualification decides when the high-resolution object detector should
run. It does not classify objects.

### Trigger modes

`motion_qualification.mode` (cameras can `inherit` or override it):

| Mode | Starts object detection | Role of EMA |
| --- | --- | --- |
| `camera_rescue` (default) | Camera ONVIF notices, manual tests, speech, and qualified EMA backup motion | Continuous backup trigger; validates generic camera motion notices, never semantic ones |
| `camera` | Camera ONVIF notices, manual tests and speech | Optional validator of camera notices (fusion `audit`); `bypass` turns validation off |
| `adaptive` | Accepted EMA motion, manual tests and speech | The only automatic trigger; ONVIF notices are diagnostic only |

Each trigger is classified once (`classify_motion_trigger`) as manual, speech,
semantic camera, generic camera or adaptive. A notice is semantic only when its
raw topic and payload affirm a person, vehicle, animal or face; cleared,
ambiguous, malformed and unknown payloads count as generic.

In `camera_rescue`, manual, speech and semantic notices are processed
immediately with no burst wait and no visual validation
(`camera_primary_fast_path`). Generic camera motion notices are coalesced and
validated against EMA like camera mode. In `camera` mode a notice is validated
unless validation is bypassed; semantic notices, manual tests and speech
always bypass validation.

The notice policy is persisted with the decision
(`qualification.camera_notice_admission`). Establishment then requires temporal
object consensus for a generic notice; a semantic notice can stand in for
consensus only for a compatible, otherwise eligible detection near the notice.

### Frame sampling and rings

The live/substream capture publishes each frame to a latest-only mailbox. The
camera's analysis worker admits frames at `sample_fps` (default 5), downscales
to `frame_width` (default 320 px on the longer side), converts to grayscale and
applies a 5x5 Gaussian blur once per frame. Retained per camera:

- a grayscale ring of `motion_ring_size()` frames:
  `max(12, round(sample_fps * (window_seconds + post_trigger_seconds + 3)))`,
  36 with defaults;
- a 4-frame color and blurred window for continuous analysis;
- an evidence ring (12–16 frames: the motion ring size, capped at 16) holding
  the analysis frame plus a reference to the read-only full-resolution live
  capture it came from. Those references share the capture buffers and are
  bounded per camera by `EVIDENCE_SOURCE_FRAME_BUDGET_BYTES` (64 MiB); older
  entries keep only the analysis frame. A live frame larger than the inference
  worker accepts (`MAX_INFERENCE_FRAME_BYTES`, 64 MiB, about 6300 px wide at
  16:9) is never kept, so the fast check uses its analysis-width copy.

A backward wall-clock step clears the rings and resets the EMA runtime; an
equal timestamp is treated as a duplicate sample.

### Continuous EMA analysis

EMA runs continuously whenever EMA validation is in use or the mode is
`camera_rescue` or `adaptive`: every admitted sample in `adaptive`, otherwise
at `min(sample_fps, camera_mode_background_fps)` (default 2 per second). A
fleet-wide fair limiter allows `max_concurrent_analysis` cameras (default 2)
to run qualification at once; a waiting camera keeps one pending request and
keeps filling its rings.

The EMA stages are stateful and process each frame exactly once. Each cycle
passes the newest frame already processed (as the pair anchor) plus newer
frames; after a runtime reset the retained window is replayed once. The
illumination filter keeps each frame pair's motion mask from when the newer
frame was analysed. Before incremental analysis it recomputed that mask on
every cycle against a background that had already advanced, so lighting-scene
scores can differ from older builds in the fourth decimal; accept/reject
changes only for a score that close to its threshold.

- **Background:** a selective EMA background learns quiet pixels at
  `learning_rate`, learns quickly on whole-scene changes
  (`global_change_ratio`), and absorbs pixels that stay changed for the
  stationary-learning period slowly. It tracks scene noise and brightness.
- **Threshold:** a smoothed threshold from the median/MAD of each difference
  image, bounded by `minimum` and `maximum`.
- **Zones:** detection zones with **Exclude from EMA** are masked out after
  thresholding.
- **Blobs and tracking:** open/close morphology, connected components, and a
  noise-scaled filter on area, fill, aspect ratio and edge contact. A
  persistent centroid tracker keeps multi-region tracks across samples.
- **Scoring:** each track is scored on persistence, translation, size,
  direction and speed consistency, fill, interior position and zone weight,
  minus an insect-like penalty. The acceptance threshold comes from
  sensitivity, raised for noisy scenes (up to +0.12) and at night (+0.025),
  capped at 0.78. Stationary-motion policy (`stationary_object_tolerance`)
  rejects confined shimmer, oscillation and persistent scene motion.
- **Illumination filter** (`illumination_filter_enabled`, default off):
  rejects accepted motion whose color, luminance-polarity and edge-structure
  evidence indicates a lighting change. Evidence is computed once per frame
  pair.

| Sensitivity | Base threshold |
| --- | ---: |
| `high` | 0.36 |
| `balanced` | 0.48 |
| `low` | 0.60 |

### Camera-notice validation (camera mode)

Notices are coalesced until `burst_quiet_seconds` (default 0.5) pass without
another trigger, bounded by a hard deadline of `max(2 s, 4 x quiet)`. The
decision then reads the continuous EMA results produced since the notice
arrived, for up to `post_trigger_seconds` (default 2.5), and accepts on the
first accepted result; otherwise it keeps the strongest. If no usable result
exists the burst fails open (`insufficient_frames`); an all-zero score also
fails open (`no_temporal_signal`).

A rejected score within `borderline_margin` (default 0.03) of the threshold is
still sent to the detector when borderline rescue is enabled, and accepted
only if an eligible object is found. A sampled fraction of other rejections
(`suppression_verification_rate`, default 0.05) is also verified by the
detector.

### EMA backup (camera_rescue)

An EMA signal conditioner turns continuous results into a single qualified
edge:

- After a learning period (`visual_backup_warmup_seconds`, default 10), a
  sample is strong when accepted, at or above the required score
  (`visual_backup_min_score` 0.70 or threshold plus
  `visual_backup_score_margin` 0.15) and not a nuisance reason
  (illumination, insect-like, persistent-scene, stationary motion).
- An edge qualifies after `visual_backup_min_consecutive` (3) strong samples
  spanning `visual_backup_grace_seconds` (1.5 s).
- A slower verification lane accepts persistent motion below the rescue score
  with two more samples and one more second of persistence. A confirmed
  upstream route watch needs one sample; proven ONVIF degradation or an
  unavailable ONVIF transport uses the normal persistence.

The episode controller then applies `visual_backup_cooldown_seconds` (20) and
`visual_backup_max_triggers_5m` (3), except for a one-shot route watch. A
backup invocation never creates an empty incident: the recorded refinement must
find an incident-eligible object that moved or explains the EMA motion region.

In `adaptive` mode every accepted EMA sample is a qualified edge for the
episode controller.

### Motion audit

Rejected and backup decisions are stored in the `motion_audits` table and shown
under Admin > Motion Audit. `rejected_sample_rate` (default 1.0) controls how
often a rejected decision keeps an image; samples are capped at 100 per camera
under `STORAGE_DIR/motion_samples/CAMERA_ID/`.

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

API keys are stored in the encrypted secrets sidecar, not in `config.json`.

## 6. High-Resolution Object Detection

An accepted batch gets two object checks:

1. **Fast check** on a live frame, so the initial decision does not wait for a
   recording segment to close. EMA-triggered work uses the exact frame EMA
   analysed, at full live resolution while its reference is retained
   (otherwise the analysis-width copy). Camera notices use the latest live
   frame. Either frame must be at most 1 s old, or the fast check reports
   `fast_frame_stale` and relies on refinement. Results are provisional; if the live and
   main geometry is not reliably aligned, zone-based eligibility waits for
   refinement.
2. **Recorded refinement**, queued as a durable detection job, decodes the
   main recording around the event in stages
   (`detector.event_refinement_stages`):

```text
[-1.0, -0.5, 0.0, +0.5, +1.0]  [1.5 ... 3.0]  [3.5 ... 4.5]  [8.0, 8.5]  [12.0, 12.5]
```

A confirmed route watch uses the denser `event_route_refinement_stages`. Within
each stage, inference stops early once temporal confirmation is satisfied.
Recordings are plain MP4 segments that can only be read once closed, so
refinement waits for the segment to be indexed (predicted from the newest
indexed segment), plus `event_refinement_settle_seconds` (0.75). It retries
for up to `event_refinement_retry_seconds` (24) at
`event_refinement_retry_interval_seconds` (1.0) intervals.
A segment that has just closed is usually not indexed yet, because the indexer
polls every 10 s. The first lookup that misses asks the indexer for it, and
the worker waits for that one discovery to finish instead of a whole retry
interval. Later misses in the same stage mean the segment is still open and
use the plain interval.

For each available target:

1. Find the indexed main recording containing the target timestamp.
2. Decode one frame with FFmpeg, using configured hardware decoding when
   available and CPU fallback otherwise. Samples that share a segment decode
   in one process, and hardware surfaces are downloaded only after frame
   selection. The batch seeks to the whole second before the first sample,
   including under `qsv`. Recorded reads use `-copyts -start_at_zero` to retain
   the image/timestamp pairing through QSV seeking on B-frame streams while
   normalizing container start offsets to the segment timeline.
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

`RecordedMotionObjectDetector` owns frame sampling, decode admission, and retries.
Every FFmpeg process that reads frames from a recording (refinement single and
batch reads, the tracking sampler and exact-frame read, and the stream-alignment
and face-dataset still) is built and run by `recorded_decoder`: one set of
hardware-then-software plans, one command builder that always applies the
thread limits, one process runner, and one classification of failures
(`unfinalized_input`, `missing_input`, `unreadable_input`, `hardware_init`,
`bitstream_corruption`, `timestamp`, `timeout`, `stalled`, `spawn`). A
streaming decode that produces nothing for 10 s while its reader waits (60 s
before its first frame) is killed. Codec errors and failed
decodes are logged at most once a minute per consumer and recording directory.
It passes collected samples to `motion_pipeline.recorded_evidence` for temporal
association, confirmation, and representative frame selection.

The representative frame is the sample that best shows the active subject:
one that contains it, in an incident zone, fully inside the frame, and with
the most resolved detail. Detail combines subject size with the sharpness of
the subject crop, so a blurred larger view does not beat a clear, slightly
smaller one. When no object reaches its confirmation count, the best-supported
zone-admitted detection still anchors the frame, so a subject that scene
establishment accepts is never represented by an empty scene. That choice is
presentation only and does not change which objects are incident-eligible.
Only when no sampled frame holds a zone-admitted detection does overall image
quality, then proximity to the event time, decide. SurvNG retries briefly while newly written recording segments are
being finalized. If no recorded frame becomes readable, it fails over to the
latest live frame and marks that source in event metadata.

### Monocular depth enrichment

When `detector.depth.enabled` and a model path are configured, the object worker
runs depth estimation on the representative recorded frame after object
detection. It records each object's median and bounded depth statistics in
meters. The global `max_incident_distance_m` policy and per-zone
`min_depth_m`/`max_depth_m` policies can mark recorded objects ineligible;
ordinary confidence, temporal, and zone checks still apply.

Depth heatmaps are optional. With `store_heatmap=true`, SurvNG stores a bounded
encoded heatmap alongside the incident evidence; otherwise it retains only the
per-object statistics. Decision-scoped depth-shadow telemetry measures how a
near-depth rule would affect EMA/object correlation. Shadow results are
informational and never alter admission.

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

Two tracker implementations are available through
`detector.tracking.implementation`: `survng_sparse_identity` (the default) and
`survng_hybrid`. Sparse Identity extends Hybrid with cascaded association,
buffered IoU fallback, quality-gated ReID galleries and entity relink; see
[README.tracking.md](README.tracking.md). Both run on the existing NumPy/OpenCV
stack. The former tracker comparison UI, APIs, replay command and optional
Ultralytics backends have been removed.

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

An event's `created_at` is its trigger time. Its cover frame is chosen from
samples around the trigger, so it can be captured seconds earlier or later.
Event payloads therefore also report `snapshot_captured_at`, the cover
frame's own time, and the UI labels and orders cover images by it. The value
comes from the objects drawn on the cover, or, for a cover with no object,
from the snapshot file name, which every writer stamps with the capture time.
It is omitted when neither is available, and clients fall back to
`created_at`.

Each incident can also retain up to four additional evidence images beside
its cover. They come from two sources and never change the cover, admission,
or alerts:

- Recorded detection contributes at most two per incident, chosen from
  frames it already decoded. The other two slots are reserved for tracking,
  which runs later, so detection passes on several events cannot fill the
  gallery first. Tracking may also use slots detection left empty.
- Object tracking contributes when a session completes. While it follows a
  subject it nominates the best view per subject every two seconds, then
  re-reads the chosen moments from the main recording at full detail. Views
  that repeat the cover or an image already kept are skipped: the same place
  at a similar size counts as the same picture, however far apart in time.
  With `tracking.analysis_mode: on_demand` this happens when the incident's
  recorded analysis is requested, not when the incident is created.

Tracking status reports `evidence_images_saved`, `evidence_images_nominated`,
and `evidence_images_result` for the session.

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
    "mode": "camera_rescue",
    "sensitivity": "balanced",
    "stationary_object_tolerance": "balanced",
    "illumination_filter_enabled": false,
    "frame_width": 320,
    "sample_fps": 5.0,
    "camera_mode_background_fps": 2.0,
    "window_seconds": 1.6,
    "post_trigger_seconds": 2.5,
    "burst_quiet_seconds": 0.5,
    "max_concurrent_analysis": 2,
    "rejected_sample_rate": 1.0,
    "suppression_verification_rate": 0.05,
    "borderline_rescue_enabled": true,
    "borderline_margin": 0.03,
    "visual_backup_warmup_seconds": 10.0,
    "visual_backup_grace_seconds": 1.5,
    "visual_backup_min_score": 0.7,
    "visual_backup_score_margin": 0.15,
    "visual_backup_min_consecutive": 3,
    "visual_backup_cooldown_seconds": 20.0,
    "visual_backup_max_triggers_5m": 3,
    "pipeline": {
      "qualification": [],
      "observation": [],
      "fusion": []
    }
  }
}
```

Object inference and optional depth enrichment:

```json
{
  "detector": {
    "object_worker_count": 2,
    "depth": {
      "enabled": false,
      "model_path": "",
      "device": "AUTO",
      "input_size": 768,
      "min_distance_m": 0.05,
      "max_distance_m": 150.0,
      "max_incident_distance_m": null,
      "store_heatmap": false,
      "heatmap_max_width": 192
    }
  }
}
```

OpenVINO uses one to four object workers and defaults to two. When tracking is
enabled, SurvNG floors `object_worker_count` at two so incident admission keeps
a detector lane. Core ML uses one worker.

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
replaces that graph for only that camera. Each item has a stable stage ID, a
registered implementation and its options. Only EMA stages are registered for
qualification, so a custom qualification graph is a re-tuned EMA graph, for
example a camera with a less aggressive blob filter:

```json
{
  "stage_id": "blob_filter",
  "implementation": "adaptive_blob_filter",
  "options": {"minimum_area_ratio": 0.0005}
}
```

Configured graphs are checked for non-empty names, duplicate IDs, registered
implementations, known options and artifact ordering before configuration is
saved or workers are stopped. Retired implementations (MOG2 and the
fixed-threshold stages) are rejected with an explicit message. Camera status
reports the resolved implementation, options and whether each graph came from
built-in defaults, global config or a camera override.

The fusion graph accepts only `buffered_evidence_fusion` with policy `audit`
or `bypass` and no sources. Configurations saved by the UI may still contain
`source_thresholds`, `source_weights`, `weighted_threshold`,
`minimum_sources`, `require_warmed` or `fail_open`; they load and are
ignored. The UI records "EMA validation off" as policy `bypass` with
`include_primary: false`. Only policy `bypass` turns EMA validation off:
`include_primary: false` with policy `audit` rejects every generic notice
(`adaptive_validation_disabled`), so a hand-edited config must set both.

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

- EMA scene learning, thresholding, persistent tracking, scoring, stationary
  policy and the illumination filter (`tests/test_adaptive_motion.py`,
  `tests/test_motion_pipeline.py`, `tests/test_stationary_policy.py`).
- Incremental continuous analysis matching a full-window replay for moving,
  lighting-ramp and static scenes
  (`tests/test_motion_incremental_qualification.py`).
- EMA signal conditioning and the motion episode controller
  (`tests/test_ema_v2.py`, `tests/test_motion_analysis_service.py`).
- Fusion validation: audit/bypass only, rejected policies and sources, saved
  UI options still loading (`tests/test_motion_configuration.py`,
  `tests/test_motion_evidence.py`).
- ONVIF event normalization, semantic priority, burst coalescing and durable
  trigger delivery (`tests/test_motion_ingress.py`,
  `tests/test_motion_events.py`, `tests/test_motion_decisions.py`).
- Fast live checks, recorded refinement, temporal consensus and full-resolution
  evidence frames (`tests/test_recorded_object_detection.py`,
  `tests/test_camera_worker.py`).
- Recorder ownership, failure isolation, indexing, media validation, recording
  range selection and playback continuity helpers.
- Incident grouping and event clip behavior.

Run backend tests and build the production UI with:

```bash
scripts/run-tests.sh -q
npm --prefix frontend run build
```

Long-running recording playback should also be tested in current Safari and
Chrome across repeated seeks and many segment boundaries.

## 14. Current Boundaries And Future Work

EMA is the only visual motion analysis. SurvNG does not calculate optical flow,
use a KNN or MOG background model, or consume vendor motion bounding boxes
beyond normalized ONVIF event signals. Optical flow is on the roadmap as an
optional evidence source (`docs/motion-roadmap.md`); it would need a fusion
source contract again, which the current audit/bypass fusion stage does not
provide.

Recorded refinement cannot start until a segment closes, which adds up to one
segment duration (10 s by default) of latency. See
`docs/motion-pipeline-review-2026-11-22.md` for this and other open items.

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
