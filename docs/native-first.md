# Native-first live pipeline

This experimental branch replaces the application camera execution path. It is
independent of PR #206 and does not provide a legacy/fallback pipeline switch.
No application Python object-inference pool, Hybrid/ByteTrack tracking session,
EMA admission, ONVIF event subscription, ReID, face recognition, depth inference,
recorded-frame refinement, or semantic inference worker runs in this design.
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
          → gvadetect: OpenVINO, batch 1, interval 1
          → gvatrack: short-term-imageless
          → bounded metadata delivery
          → session-qualified native observation consumer
              → class confidence + live-coordinate zones
              → fresh-observation confirmation
              → presence episode → event database, incident updates, MQTT/SSE

Main stream → existing continuous recorder → incident video/playback
```

The compiled model is shared across camera graphs. The initial throughput
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
- **A stationary visible object keeps its presence episode active.** This is an
  object-presence experiment, not EMA motion qualification. Parked vehicles can
  therefore produce long episodes. There is no stationary-scene suppression.
- Track history is bounded to 150 observations per track and 128 tracks per
  episode by default. Capacity drops are counted. This limit is especially
  relevant when a stationary object holds an episode open for a long time.
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
throughput guarantee: 13 cameras at 5 FPS request 65 inferences per second.

Do not add PR #206's `live_pipeline_inference_enabled`,
`live_pipeline_inference_interval`, or `live_pipeline_tracking` settings. This
branch fixes interval 1 and native tracking. The old `detector.tracking.enabled`,
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

Zones use **live/substream coordinates**, matching the zone editor. Main recordings
can have a different field of view. Native replay overlays are automatically
allowed when the same URL supplies live and main; otherwise explicitly set
`native_same_field_of_view: true` on the camera only after verifying the views.
Snapshot annotations remain valid without that setting. No automatic image
registration or high-resolution main-stream refinement runs. An exact snapshot
frame miss can leave an event without a cover; its counter remains visible.

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
