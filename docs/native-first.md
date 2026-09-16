# Native-first live pipeline

This experimental branch replaces the application camera execution path. It is
independent of PR #206 and does not provide a legacy/fallback pipeline switch.
No application Python object-inference pool, Hybrid/ByteTrack tracking session,
EMA admission, ONVIF event subscription, ReID, face recognition, depth inference,
legacy recorded-frame refinement, or semantic inference worker runs in this design.
A separate bounded native CPU verifier checks shortlisted main-recording images
for high-resolution incident covers; it is not continuous main-stream detection.
Historical events, recordings, identity records, and existing media APIs remain
readable. Some legacy library/configuration definitions remain for historical
features and standalone tests; they are not an alternative camera runtime.

## Execution and ownership

```text
Live/substream RTSP
  → hardware decode when available
  → shared VA surfaces
      ├─ sampled color frames / JPEG preview → preview and exact-PTS snapshots
      └─ drop-only sampling (target 5 FPS)
          → gvadetect: OpenVINO, configurable shared batch and interval (both default 1)
          → selected-class metadata filter (no pixel mapping)
          → gvatrack: short-term-imageless
          → gvaanalytics: live-coordinate zone membership
          → bounded metadata delivery
          → session-qualified native observation consumer
              → class confidence + zone admission policy
              → fresh-observation confirmation
              → presence episode → event database, incident updates, MQTT/SSE

Main stream → existing continuous recorder → incident video/playback
```

The compiled model is shared across camera graphs with the same inference-region
mode (full-frame or ROI). The initial throughput
profile uses four inference requests and two CPU/GPU inference streams. GPU
compilation remains serialized to avoid the known driver compiler issue.
GPU automatic batching is explicitly disabled: the throughput hint can otherwise
activate an OpenVINO batching wrapper despite `gvadetect` batch size 1, which
failed to initialize VA surface inputs with the deployed model. Four requests
and two inference streams remain enabled. Queues
shed old frames rather than accumulating an unbounded backlog. Tracking runs
before the leaky metadata queue so delivery drops do not skip tracker updates.
OpenVINO remains the inference engine inside `gvadetect`.

`NativeCameraWorker` owns capture and one metadata consumer. `NativeActivity`
owns confirmation, episode state and persistence; it never calls a model or a
second association algorithm. Metadata processing does not wait for a matching
Python pixel frame. Only snapshot creation requires an exact session/PTS match.
There is no on-demand detection when metadata is missing.

## Presence semantics

- Default confirmation: two consecutive fresh, eligible observations of the same
  native ID and class. Empty fresh results break tentative confirmation.
- Native IDs are scoped to camera + stream session + geometry generation. Reconnects,
  resolution changes and graph rebuilds
  start new identities; the application does not claim cross-camera identity or
  re-identification after disappearance.
- `gvatrack` can append predicted ROIs even to a fresh detector frame. Exact
  pre-tracker ROI evidence distinguishes these from fresh observations. Predicted
  confidence never admits an object, increments observations, or extends presence.
- Default completion: five seconds without fresh eligible presence. Missing or
  invalid metadata ends activity with a coverage-loss reason, not evidence that
  the scene was empty. Persistent metadata stalls rebuild the native stream after
  15 seconds; failed graph shutdown is reported rather than starting a second graph.
- **Stationary vehicles remain tracked but do not create or extend incidents.**
  New vehicles start uncertain and must demonstrate movement. After eight seconds
  of stable evidence, moving vehicles become stationary; the normal five-second
  activity timeout then completes the incident when no other object is active.
  People and other unlisted classes retain presence-based admission.
- Motion uses fresh native boxes and elapsed stream time, a bounded two-second
  window, trimmed coordinate spread and accumulated drift relative to box size.
  Movement requires 0.15 box-width/height displacement; the stability threshold
  is 0.05. At least five observations spanning 0.4 seconds are required; at low FPS the
  window retains five observations even when they span more than two seconds. This
  hysteresis tolerates jitter; it is image-space motion, not calibrated speed.
- Default vehicle labels: car, truck, bus, van, suv, motorcycle. Configure
  `detector.native.stationary` with `enabled`, `labels`, `stationary_seconds`,
  `window_seconds`, `moving_threshold`, and `stationary_threshold`. Disabling it
  restores presence-based admission for every class. All detector/tracker work
  continues; suppression saves incident work, not inference work.
- Incident completion preserves live stationary context. Track disappearance,
  long observation gaps, native ID changes, reconnects and resolution changes
  require new evidence; there is no cross-ID appearance matching. A vehicle
  already parked at startup does not alert merely because it received a new ID.
- Track history is bounded to 150 observations per track and 128 tracks per
  live camera and 128 archived tracks per episode by default. Expired live tracks
  are evicted independently of incident history. Capacity drops are counted.
- Updates persist at most once per second after admission, plus completion.
  Each episode is one incident, even when another starts within the old gap window.
  Native completion explicitly settles incident notifications. Native track times
  extend incident duration and recording windows.

## Test configuration

Keep the existing valid camera/recording configuration and model path. The only
activation flag is `detector.enabled`. A minimal detector section is:

```json
{
  "detector": {
    "enabled": true,
    "model_path": "/absolute/path/to/model.xml",
    "device": "GPU",
    "live_sample_fps": 5,
    "event_confirmation_frames": 2,
    "native": {
      "activity_timeout_seconds": 5,
      "maximum_observation_age_seconds": 2,
      "metadata_restart_seconds": 15,
      "inference_interval": 1,
      "inference_requests": 4,
      "inference_streams": 2,
      "maximum_tracks": 128
    }
  }
}
```

Replace the model placeholder with your existing OpenVINO IR model. Existing
labels, model-proc, confidence and zone settings still apply. An enabled detector
with no model path is rejected. `live_sample_fps` is a target per camera, not a
throughput guarantee: 13 enabled cameras at 5 FPS and interval 1 request 65
inferences per second. `native.inference_interval` accepts integers 1–5 and is
available in Admin → Detection → Inference interval. Interval 2 at 5 FPS targets
2.5 fresh detections/sec per camera; tracker predictions fill intervening pipeline
frames but cannot admit or extend incidents. Confirmation and stationary decisions
take longer with fewer fresh observations. Saving an interval change reloads native
capture. Requests/streams configure the shared model, not separate per-camera pools.

Do not add PR #206's `live_pipeline_inference_enabled`,
`live_pipeline_inference_interval`, or `live_pipeline_tracking` settings. This
branch uses `detector.native.inference_interval` and native tracking. The old `detector.tracking.enabled`,
EMA, enrichment, and main-evidence buffering settings do not select runtime paths.

Switching to this branch with `detector.enabled: true` activates native inference
on camera startup. Restart the application after switching builds. To keep test
incidents separate, use separate test storage/database/index directories.
The PR does not deploy the branch or change a running service.

Per-camera detection controls rebuild that camera's native graph. Disabling
inference retains preview; enabling it starts a fresh stream session. Global
model/device/cadence/native-pool changes use the existing manager reload boundary.
Lowering a zone threshold below the native graph floor uses that same transactional
reload so the detector can actually supply the newly eligible objects.

Zones use **live/substream coordinates**, matching the zone editor. Native replay
uses the main recording when live and main share a URL, or when the camera has
`native_same_field_of_view: true`. This confirmation also applies to historical
native tracks at response time, without changing their archived coordinates.
For confirmed cameras, Clean and Tracks use the same high-resolution recording;
Tracks adds scaled boxes. An old false compatibility flag means unverified, not
an observed FOV mismatch. Without confirmation, Tracks uses recorded substream
video (complete MP4 on Safari); Clean uses main recordings.

The existing alignment estimator is shared with offline cover selection. That
bounded process verifies full-resolution main frames before promoting a cover;
it does not automatically change camera replay geometry settings. Exact-PTS
snapshot misses remain visible in counters.

## What to measure

Camera status and `GET /api/cameras/{id}/native` expose health, fresh-result age,
actual consumed fresh FPS, presence state, and bounded counters. `survngctl status`
allowlists native health/counters and native timing. The camera information panel
shows the same native metrics.

`native_detector_average_ms` and `native_detector_p95_ms` measure gvadetect sink-to-
source time, including its request scheduling. They exclude camera transport,
decode and application persistence. They are not camera-to-notification latency
and are not directly comparable to the old isolated worker's model-only timer.
`effective_fresh_fps` measures consumed fresh observations; the pipeline's
`effective_inference_fps` remains the configured target. Watch both freshness and
actual throughput, plus metadata restarts, invalid evidence, missing native IDs,
track capacity drops and missing snapshot frames. A higher stage latency can be
acceptable when throughput and notification delay remain acceptable; this PR
makes no claim about the capacity of the deployed camera fleet.

## Validation

- Python regression suite and frontend unit/build checks.
- Native activity tests cover confirmation, duplicates, empty versus missing
  evidence, predicted ROIs, stale metadata, session reset, persistence and duration.
- A real synthetic video/model test exercises the application manager, native
  graph, IPC, capture, event store, incident completion, and disabling inference
  while preview stays connected, plus native-process death and session recovery.
  It uses temporary storage and no real camera URLs.
- `python3 scripts/gstreamer-smoke.py` exercises real Intel GStreamer metadata,
  positive/empty outputs, native IDs, interval provenance, multistream handling,
  model formats and frames-only preview. The native CI job runs in the pinned
  Intel DL Streamer image.

Real camera recognition quality, ID continuity through occlusion, fleet GPU
capacity and long-running stationary-object behavior still require field testing.

## Shared batching and recorded evidence

Admin → Detection → Shared inference batch size configures
`detector.native.batch_size` (integer 1–4). **1 is the default and means no
batching**; 2–4 enable explicit DL Streamer batching across the shared model
instance. The existing shared requests and streams remain independently
configurable. Saving rebuilds native capture. OpenVINO automatic batching stays
disabled for VA input compatibility. Larger batches wait for more frames and
can increase latency; there is no supported VA batch-timeout control. Start at 2
and measure actual camera FPS and latency before increasing it. Configuration
validation rejects nominal batch-fill times at or above the maximum result age,
including the case where only one camera remains connected. Lower actual source
FPS can still increase waiting beyond that nominal estimate.

Terminal events queue a full recorded-history cover selection even if preview
selection is already running. Late track history refreshes the requested replay
window in both incident views; the selected tracking event contributes its full
bounds even when the incident summary contains no tracks. Paused replay initializes
its overlay immediately.


## Recorded-track timing alignment

Completed native incidents on cameras with confirmed matching fields of view can
receive an episode-specific main-recording timing correction. The existing
bounded cover worker reuses its native CPU detections (at most 12 candidate
frames) and compares moving trajectories across offsets within ±3 seconds.
It requires at least five detected frames over three seconds, meaningful
within-track motion, a distinct score peak, and agreement between frame votes.
Stationary fragments, sparse evidence, and competing offsets remain uncorrected.

Verified `object_tracking.recording_alignment` metadata changes only the saved
track lookup clock for the matching replay source; it never seeks or speeds up
the video. Original track coordinates/timestamps and cover imagery are retained.
It is not a camera-wide offset or a claim of shared hardware timestamps between
the independent capture and recording paths. Existing incidents require recorded
verification to gain a correction; they do not inherit another incident's result.


## Selecting tracked classes

Admin → Native detection and tracking → **Tracked classes** is a checkbox
selection dropdown populated from model labels. Save applies it by rebuilding
native capture. `detector.native.tracking_classes` is `null` by default (all
classes), `[]` disables tracking/admission for every class, and an explicit
array such as `["person", "car", "cat", "dog", "robot_lawnmower"]` excludes
face detections from this model. Labels are trimmed, lowercased and deduplicated.
Changing the model does not automatically expand an explicit selection.

Filtering runs after gvadetect and before both native evidence capture and
gvatrack. It removes excluded detections from both GstVideo ROI and GstAnalytics
relation metadata; removing only the former does not filter this DL Streamer
version's tracker input. Selected bounding boxes, confidence and class IDs are
retained without mapping/copying pixel memory. Model inference still evaluates
all output classes. Existing stored tracks are not rewritten. ReID remains off.

Validation with the installed native runtime:
`/usr/bin/python3 scripts/check-native-class-filter.py` exercises all/none/subset
selection through real gvatrack and JSON conversion, retained IDs, and unchanged
pixel-memory references.

## Native zones and optional inference regions

Each camera sends its existing normalized zone configuration to its native graph.
`gvaanalytics` runs after `gvatrack`, evaluating bottom-center points in uncropped
live-stream coordinates. Python consumes native zone IDs and retains class and
confidence rules, ignore precedence, confirmation, and notification policy. The
2026.2 plugin rounds geometry to integer pixels and does not include every polygon
edge; a narrow two-pixel boundary compatibility check preserves SurvNG's existing
inclusive, normalized-coordinate behavior. This is an architectural change, not
an inference-rate optimization.

Zone metadata includes a configuration revision. Missing/mismatched revisions,
including on empty observations, cannot supply incident evidence. Zone edits stop
and recreate that camera's capture session before accepting the new configuration.
The plugin reads polygons at startup, so the graph defers analytics startup until
input dimensions are known. A later resolution change stops admission and uses the
existing metadata watchdog to rebuild the stream, rather than evaluating new
coordinates against old polygons. Cameras whose main/live fields of view differ
must author zones for the live image; no new main-to-live projection is introduced.

Admin → Cameras → Detection → Detection region exposes `camera.native_roi`:

- `enabled`: false by default; opt in per camera.
- `zone_names`: empty means all enabled incident zones; otherwise selects names.
- `padding`: fraction of frame width/height around their enclosing rectangle;
  default 0.15, range 0–0.5. Include whole people/vehicles above floor polygons.
- `full_frame_interval`: every N actual inference inputs is full-frame, starting
  with the first; default 5, range 1–30. 1 always covers the full frame.

The initial implementation uses **one enclosing rectangle**, bounding each input
to one inference rather than multiplying work across overlapping crops. No matching
valid incident zones means full-frame coverage. A metadata-only `gvapython` adapter
attaches the rectangle using a writable buffer header, retaining VA pixel memory;
`gvadetect inference-region=roi-list` restores detections to full-frame coordinates.
Inference-region markers are removed before fresh evidence capture and tracking.
Fresh detections, empty results, and predictions retain distinct meanings.

With adaptive inference disabled, detection FPS and inference interval remain unchanged. Objects outside the crop
receive only periodic coverage and may not satisfy consecutive-frame confirmation;
full-frame sweeps do not guarantee detection of brief appearances. Crop changes can
also change boxes/track IDs. ROI selection is therefore an opt-in accuracy tradeoff,
not a promise of faster inference or unchanged recall.

ROI streams use a separate shared model-instance pool (`-roi` suffix). Native
mixed-mode batch tests found incorrect crop coordinates when full-frame and ROI
streams shared the same compiled preprocessing. ROI cameras share with each other;
turning ROI on can allocate another compiled model and inference pool. Shared
request/stream settings apply to each pool. Adaptive cameras also use the ROI pool
to preserve explicit full-frame idle inputs.

Validation: `scripts/gstreamer-spatial-check.py` uses a synthetic batch-aware model
to check mixed full-frame/ROI streams, batch sizes 1/2, inference intervals 1/3,
coordinates, zone IDs, tracking provenance, and fresh empty results. `--va` checks
12 results using GPU inference and VA-surface-sharing with VAMemory retained.
`tests/test_native_spatial.py` compares native polygon output against SurvNG policy
for 240 cases at three resolutions when the installed runtime is available. These
checks verify plumbing; scene recall, GPU memory growth, and performance still need
camera-specific measurement before broadly enabling ROI.


### Adaptive per-camera inference budget

Admin → Detection → Adaptive inference budget sets global defaults. Admin →
Cameras → Detection provides an inherit/on/off switch and individual overrides;
blank values inherit. The feature defaults off. Saving changed budget or motion
settings rebuilds the affected native capture configuration.

Defaults are 1 FPS idle, 5 FPS active, a 5-second cooldown, and an approach margin
of 10% of frame width/height around incident polygons. Rates count fresh inference
inputs: adaptive mode replaces the fixed inference interval with an upstream gate.
Fresh, sufficiently confident objects of relevant classes near incident zones, or
zone-overlapping native motion, extend the active period. Confirmation supplies a
minimum hold even if cooldown is zero. Without incident zones, the whole frame is
relevant. Ignore zones still suppress incidents without masking discovery.

Every idle inference covers the full frame, even when optional ROI inference is
enabled. Active ROI inference retains periodic full-frame sweeps. Fresh detections
of stationary objects can keep a camera active. Tracker predictions cannot wake it;
motion alone cannot confirm or extend incidents. Skipped inputs produce no synthetic
empty detection results. Health checks allow intentional idle gaps; configurations
whose idle batch fill exceeds freshness/recovery limits are rejected.

The motion subsection controls `gvamotiondetect` before inference: `block-size`,
`motion-threshold`, `min-persistence`, `max-miss`, `iou-threshold`, `smooth-alpha`,
`confirm-frames`, `pixel-diff-threshold`, and `min-rel-area`. Each is also overridable
per camera. Motion wake-up can be disabled independently of the budget. The element
receives NV12 and retains VAMemory on the VA path; Python handles metadata only.
Motion metadata is removed before object inference/tracking so it cannot become
object evidence. Foliage and lighting may require higher thresholds/persistence.

Zone settings provide **Exclude from motion wake-up**, independent of object
behavior. Enabled exclusion polygons subtract from motion eligibility after
incident-zone approach padding; they cannot be overridden by that padding.
Only remaining eligible area can wake inference, including when a motion rectangle
crosses an exclusion boundary. Existing `exclude_from_ema` values are retained as
the compatible persisted key. This filters motion metadata, not video pixels;
recording, object inference, and object-based wake-ups are unchanged. A detector
rectangle can extend beyond the actual moving pixels, so tight exclusions may
need a margin.

The owner-only status snapshot reports each camera's budget mode, target rates,
cooldown, admitted/skipped inputs, and wake counters. `excluded_motion_regions`
counts otherwise-relevant motion rectangles fully suppressed by exclusions, once
per rectangle per sampled frame; partial exclusions that still permit a wake do
not increment it. Counters reset when the native pipeline is rebuilt.
Health camera tiles display budget mode, target/idle/active rates, active hold,
admitted/skipped/sampled inputs and skipped percentage, motion wake-up enablement,
motion/object wake requests, excluded motion rectangles, and mode transitions.
They distinguish disabled detection, fixed-rate operation, disconnected cameras,
and pending telemetry. Wake requests include extensions of an existing active
hold; skipped-input percentage is not measured GPU savings.
`gstreamer-spatial-check.py
--budget` verifies native admission, idle full-frame coordinates, motion wake-up,
empty-result provenance, and mixed adaptive/fixed streams. `--va` also exercises
native motion followed by GPU ROI inference while retaining VAMemory.

Reduced input counts are not measured GPU savings. Idle sampling trades entry
latency and brief-appearance recall for fewer inferences; validate these on each
scene before enabling broadly. Motion still processes frames at the active rate,
and shared compiled pools and batching affect realized compute and latency.

### Main-recording incident confirmation

`detector.native.verification_enabled` defaults true and is exposed under Admin →
Detection → High-resolution incident confirmation. Low-resolution fresh detections
nominate candidates after normal class, zone, confidence, movement and consecutive
frame checks. They do not create events or publish incident/object notifications
until a main-recording crop confirms the same object. Turning the option off restores
immediate native presence admission.

A bounded worker collects up to three nominations spaced at least 0.4 seconds
apart. It waits initially 15 seconds for recorded segments, aligns each original
preview to the matching main frame, and checks a contextual crop at its native
resolution with the existing isolated CPU `gvadetect` verifier. Positive detections
must match class, location and extent (IoU at least 0.3, with half of the detected
box overlapping the nominated object) and satisfy the configured class/zone threshold.
A same-class fragment inside the original box is ambiguous, not confirmation or a
clear negative.
One clear positive confirms; three clear negatives reject. Missing recordings,
failed registration, poor image quality, weak detections and verifier failures do
not count as negatives. Unavailable evidence is retried for up to 90 seconds;
remaining candidates are unverified and do not alert. Cameras without usable main
recordings therefore cannot confirm incidents with this option enabled.

Detection and tracking continue during this delay. Candidate histories are bounded
and survive a track leaving the scene while verification runs. Accepted events retain
the original observation timeline and start with a genuine main-resolution cover;
cover objects and live tracking histories keep their respective coordinate geometry.
Stream, geometry, policy and stop transitions cancel pending candidates and discard
late results. Rejected tracks can be nominated again after substantial displacement;
unverified tracks can retry after 30 seconds. No permanent detection mask is created.

There are at most 32 pending jobs globally and 32 candidate histories per camera.
Overload expires as unverified, never as rejection or an automatic alert. Camera
status and the owner-only observability snapshot expose pending counts, confirmed/
rejected/unverified counters and bounded recent outcomes. Continuous recordings and
historical incidents are retained. Alert latency includes segment availability and
CPU verification; this is a recall/latency tradeoff, not a guarantee against false
positives. Failed cover-only verification remains separate from admission decisions.

Validation includes delayed confirmation after departure, no notifications from
pending/rejected/unverified candidates, cancellation of in-flight work, queue limits,
spatial and confidence checks, and three distinct temporal samples. Native crop tests
exercise varying odd-width BGR buffers (which require padded GStreamer row strides).
Read-only replays of Back-Middle events 71408 and 71404 returned three negative crop
checks each. Real person examples are also checked for positive confirmation; unknown
alignment is explicitly not treated as evidence of absence.

A post-deployment example (Back-Middle 71416) exposed a small flower fragment being
accepted inside a much larger nominated box. Verification now checks object extent,
not just containment. Its replay remains unverified rather than creating an alert;
the two real-person examples still confirm. This does not establish universal recall
or eliminate the model's underlying flower/dog confusion.


### Native-resolution live evidence frames

The application requests `frame_width=0` for the live/substream evidence branch:
retain negotiated camera dimensions rather than resize to 640 pixels. Frames are
sampled at the configured live rate (normally 5 FPS) before downloading VA memory.
The native detector branch and separate JPEG preview branch retain their own rate
and format settings. Explicit legacy resized consumers still support 240–960 pixels;
main-stream display capture remains separately sized.

Initial snapshots retain the original live pixels and must match detection geometry;
they are no longer enlarged to make their coordinates fit. Main-recording confirmation
and recorded-live candidate extraction now use native-resolution live frames too.
Global registration may use a smaller working image to locate features, but the
original images remain available for local matching and contextual main-stream crops.
The latest confirmation rule still requires agreement in class, position and extent.

Recent live evidence retains at most 32 frames and 64 MiB per camera (or one frame
if a single image exceeds that limit); old frames are evicted without resizing new
ones. Native frames increase CPU transfer and memory use, not detector inference
resolution or inference count. Runtime status exposes actual `evidence_width`,
`evidence_height`, and configured `evidence_sample_fps`.

Checks cover native and resized software/VA graph construction, a real native
capture/metadata path with matching image dimensions, the frame-byte bound, and
native-resolution recorded replays. Foyer 71392 and Downstairs 71307 still confirm;
Back-Middle 71408 and 71416 remain unverified without alerts. The recording-based
confirmation delay is unchanged by this image-resolution change.

Native-size VA downloads must force `vapostproc disable-passthrough=true` so Python
maps a distinct surface, not the decoder surface still used by `gvadetect`. Without
this separation, live testing produced VA rendering failures. The GPU `--va` spatial
check now maps native-size BGR evidence concurrently with VA-surface-sharing inference
and verifies twelve results on both branches.
