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

Admin → Cameras → Settings → Detection region exposes `camera.native_roi`:

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

Detection FPS and inference interval remain unchanged. Objects outside the crop
receive only periodic coverage and may not satisfy consecutive-frame confirmation;
full-frame sweeps do not guarantee detection of brief appearances. Crop changes can
also change boxes/track IDs. ROI selection is therefore an opt-in accuracy tradeoff,
not a promise of faster inference or unchanged recall.

ROI streams use a separate shared model-instance pool (`-roi` suffix). Native
mixed-mode batch tests found incorrect crop coordinates when full-frame and ROI
streams shared the same compiled preprocessing. ROI cameras share with each other;
turning ROI on can allocate another compiled model and inference pool. Shared
request/stream settings apply to each pool. There is no automatic cadence reduction,
motion gating, or new secondary detector.

Validation: `scripts/gstreamer-spatial-check.py` uses a synthetic batch-aware model
to check mixed full-frame/ROI streams, batch sizes 1/2, inference intervals 1/3,
coordinates, zone IDs, tracking provenance, and fresh empty results. `--va` checks
12 results using GPU inference and VA-surface-sharing with VAMemory retained.
`tests/test_native_spatial.py` compares native polygon output against SurvNG policy
for 240 cases at three resolutions when the installed runtime is available. These
checks verify plumbing; scene recall, GPU memory growth, and performance still need
camera-specific measurement before broadly enabling ROI.
