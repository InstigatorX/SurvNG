# Incident evidence data path

SurvNG deliberately separates **fast detection**, **incident admission**,
**causal motion correlation**, and **representative-image selection**. They use
some of the same frames, but they answer different questions and must not be
treated as one decision.

This document is the authoritative contract for the image and object-evidence
path after a camera notice or qualified EMA episode requests object detection.
For the upstream trigger rules, see [Motion triggers and
validation](adaptive-motion.md). For arbitrary recording-frame extraction, see
[Recording frame API](recording-frame-api.md).

## End-to-end flow

```text
ONVIF/manual notice or qualified EMA episode
  -> generation-tagged episode admission
  -> durable detection/refinement work
  -> immediate fresh live-frame check
       source: live capture, usually the configured substream
       purpose: low-latency provisional evidence
  -> provisional incident when trigger policy permits it
  -> delayed main-recording temporal analysis
       source: finalized high-resolution main recordings
       purpose: label confirmation, activity, zones, correlation, faces,
                and representative-frame selection
  -> one of:
       create incident
       refine the existing provisional incident
       preserve existing evidence when refinement is unavailable
       reject the refined object as non-causal
  -> independently select/promote the best compatible incident cover
  -> optionally promote a later identity-verified tracking cover
```

The live check reduces time to first inference. The recorded pass supplies the
stronger evidence. A fast negative, missing frame, stale frame, or invalid frame
never cancels the recorded pass.

## Why the live/substream frame is used

At the live edge, the applicable main recording segment may still be open and
cannot yet be decoded reliably. The live capture frame is already available in
memory, so SurvNG can run an initial inference without waiting for segment
finalization.

The source is the camera's live capture:

1. `live_stream_url`, normally the lower-resolution substream, when configured.
2. `stream_url`, the main stream, when there is no separate live stream.

This frame is accepted for the fast path only when its provenance is valid and
its receipt timestamp is no more than one second old (with a small future-clock
tolerance). Camera generation, capture generation, and frame sequence prevent a
frame from an old camera runtime from becoming current evidence.

The fast frame is tagged with:

- `frame_source=live_fast_path`
- `provisional_detection=true`
- capture receipt time and frame age
- camera and capture generation
- frame sequence
- whether live/main geometry is trusted
- `frame_timestamp_exact=false`

Receipt time establishes freshness; it is not a decoded camera PTS. The image
can therefore be useful immediately without claiming exact recording time.

## Main-recording refinement

The refinement worker analyzes finalized main-stream recordings around the
event. Its default sampling stages are:

| Stage | Requested offsets from the event |
| --- | --- |
| Initial temporal window | -1.0, -0.5, 0.0, +0.5, +1.0 seconds |
| Early bridge 1 | +1.5, +2.0, +2.5, +3.0 seconds |
| Early bridge 2 | +3.5, +4.0, +4.5 seconds |
| Delayed discovery 1 | +8.0, +8.5 seconds |
| Delayed discovery 2 | +12.0, +12.5 seconds |

Operators can tighten `detector.event_refinement_stages` and
`detector.event_refinement_retry_seconds` when detector occupancy matters more
than the widest delayed-discovery window. Later stages are used only as needed.
They allow an object that was distant, occluded, or not yet in view at the
trigger instant to be discovered without holding the initial response open.

Within a stage, SurvNG stops requesting additional detector inferences once the
configured confirmation count is already met. That early exit keeps the live →
recorded paradigm while reducing how long refinement holds the shared
accelerator.

SurvNG associates detections across distinct timestamps, votes on labels, uses
median confidence rather than the highest outlier, and applies the configured
global or per-class confirmation count. Every frame counted toward confirmation
must independently meet the applicable class confidence threshold; a weak
association alone does not confirm an object. The selected frame is chosen from
the confirmed temporal evidence, not simply the first or highest-confidence
frame. If that representative clips or poorly shows the subject, SurvNG may
inspect a bounded later stage for a better cover.

Recorded results carry:

- the actual selected frame time when the decoder can establish it;
- `frame_source=recorded_main`;
- whether the timestamp is exact;
- the recording path and detection-frame dimensions;
- temporal, quality, and activity evidence for each object.

Snapshot filenames and stored object provenance use the selected frame's actual
time, not merely the trigger time. Exact timestamps remain distinct from
estimated timestamps.

## Optional depth enrichment

When `detector.depth.enabled` and a resolvable model path are configured, the
isolated inference worker runs monocular depth on the selected recorded frame
and adds bounded distance statistics to each object. A configured
`detector.depth.max_incident_distance_m`, or a matching zone's
`min_depth_m`/`max_depth_m`, can make a recorded object ineligible. Depth does
not replace the trigger, object detector, temporal confirmation, or spatial-zone
checks.

`store_heatmap` optionally persists a small encoded heatmap with the incident;
otherwise only object distance statistics are retained. Motion-attribution
**depth shadow** results are separate, decision-scoped diagnostics. Shadow mode
reports what depth would have changed but deliberately does not change
admission.

## Four independent decisions

### 1. Trigger and work admission

The per-camera episode controller decides whether an ONVIF/manual/EMA
observation creates or joins a detection request. It owns episode identity,
deduplication, reservation, cooldown, and lifecycle generation. Camera + EMA
Backup treats the camera notice as primary and merges credible EMA evidence
without silently losing an admitted request.

Episode and intent IDs include a process-unique controller incarnation as well
as camera, lifecycle generation, and episode sequence. The full intent ID is the
durable refinement key and event idempotency key. A restart may repeat a local
generation or sequence number, but it cannot collide with historical work.
Exact redelivery of one intent coalesces; a different occurrence presenting the
same durable identity raises a terminal error rather than returning an old job
or incident.

The fast EMA check carries the qualifying frame's epoch, capture sequence,
capture generation, and camera lifecycle generation. It selects that exact
bounded-ring frame, or a nearby frame from the same generations, instead of
asking for whatever substream image happens to be newest after queueing. If the
token is stale or unavailable, the fast result remains nonterminal and durable
main-recording refinement still runs.

Accepted EMA below the normal rescue score enters a longer persistence lane
instead of becoming a permanent drop. A confirmed upstream object on a
configured camera route, or measured ONVIF semantic degradation, shortens that
extra persistence. Route windows are bounded and directional. Matching remains
non-consuming while evidence is evaluated; a watch is consumed only after its
route-specific trigger is durably admitted, and that consumption is persisted
across restarts. Only a newly confirmed downstream incident opens the next leg.
Watches authorize analysis, not admission, and therefore cannot propagate a
false incident by themselves.

### 2. Object and incident admission

Confidence, temporal confirmation, configured zones, stationary-scene context,
and the selected motion mode decide whether an object is incident eligible.
The substream is not allowed to bypass these policies merely because it was
available first.

For cameras with incident/ignore zones, a live-frame object cannot receive
zone-based provisional eligibility when live/main geometry is untrusted. It is
retained as provisional evidence while the main-recording pass makes the
geometry-dependent decision.

### 3. Causal motion correlation

EMA and object detection answer different questions: EMA identifies changed
image regions; object detection identifies semantic objects. For EMA backup and
other policies that require correlation, a main-stream object must credibly
explain the motion through aligned overlap or temporal movement.

When substream and main-stream crops/FOV differ and no calibration is trusted,
SurvNG does not pretend their coordinates align. A real object may therefore be
semantically valid yet fail `object_not_motion_correlated`. That rejection must
not erase an already valid camera-primary incident.

### 4. Representative cover selection

Cover selection is presentation, not admission. A cover can improve after an
incident already exists without changing why the incident was admitted.

SurvNG has three bounded cover opportunities:

1. the immediate live frame, which may become the provisional cover;
2. the representative temporally confirmed main frame;
3. a later identity-verified tracking frame.

The second step can promote a main frame even when that object does not explain
an untrusted EMA region, but only when it is safely compatible with the already
admitted subject.

## Guarded main-cover promotion

Main-cover promotion is intentionally conservative. It requires:

- an existing incident containing exactly one admitted provisional subject;
- exactly one temporally confirmed main-stream candidate with the same label;
- no ambiguity from a second candidate of that same label;
- a selected main frame within 15 seconds of the provisional frame;
- a larger frame and at least 1.5 times as many subject pixels;
- enough clearance that the subject is not clipped at the image edge;
- eligible confidence and zone evidence on the main candidate.

Other object classes may be present. Auxiliary face detections do not create
same-label ambiguity. Label and time compatibility are not identity proof; a
scene with multiple same-label candidates is left unchanged so tracked identity
can make the later, stronger decision.

Promotion atomically changes only presentation data:

- event snapshot and associated recording path;
- the visible object's display box and detection-frame dimensions;
- snapshot source, selected timestamp, exactness, and quality metadata.

It preserves the original incident confidence and eligibility and records a
`cover_promotion` status with `admission_preserved=true`. The displaced snapshot
is deleted only after confirming that no other database row references it.

Cover-promotion failure is nonterminal. It cannot retry or fail completed
security work, and it cannot create a second incident or duplicate MQTT object
notification. A successful promotion sends an internal `incident_update` so
connected incident lists refresh and semantic indexing sees the new evidence.

## When a substream cover can legitimately remain

A live/substream cover remains when no safe, materially better replacement is
available. Common reasons include:

- the main segment has not become readable or refinement failed;
- main detection did not reach temporal confirmation;
- no same-label main candidate exists;
- multiple same-label candidates make identity ambiguous;
- the main candidate is too far from the provisional time;
- the main image or subject is not materially larger/better;
- the candidate is clipped near an edge;
- confidence or zone policy rejects the main candidate.

An unavailable/error refinement is additive: SurvNG preserves known-good
provisional evidence instead of replacing it with an empty image or status-only
payload. Existing incidents are not retroactively reprocessed when this policy
changes.

## Snapshot and annotation geometry

Object boxes are stored with the dimensions of the frame on which inference ran.
When a cover is promoted, its display box and `detection_frame_width` /
`detection_frame_height` are replaced together. The UI must map the box against
those stored dimensions, not against the trigger stream or a derivative image.

The displayed incident image may be a WebP/JPEG encoding of a decoded video
frame. It is not necessarily the camera's snapshot JPEG or an encoded I-frame.
Responsive preview derivatives affect transport/display only; zoom must promote
to the original stored evidence image.

## Durable and optional work

Delayed object discovery is mandatory security work and is stored in the local
detection-job ledger before optional tracking prewarm. It survives process
restart and is retried according to its lease/attempt policy. Cover promotion,
face enrichment, and tracking presentation are optional enrichment: their
failure cannot discard or downgrade admitted evidence.

Tracking starts after recorded confirmation finishes, or immediately from live
evidence when refinement cannot run. Handoff is idempotent per event, so later
refinement cannot start a duplicate tracking session.

## Operator-visible diagnostics

Relevant stored fields and statuses include:

- `live_fast_path`, `recorded_main`, and exact/estimated timestamp metadata;
- `provisional_detection` and `refinement_pending`;
- object-detection phase timings and decision queue wait;
- temporal confirmation, activity attribution, and motion correlation;
- `refinement_unavailable_preserved`;
- `object_not_motion_correlated`;
- `cover_promotion` and its reason;
- motion-audit `cover_promotion.promoted/reason` for EMA backup attempts.

These distinctions are intentional. A message saying that the main object did
not correlate with EMA does not mean the main image was unusable as a cover, and
a substream cover does not mean SurvNG skipped main-stream refinement.

## Implementation map

| Responsibility | Implementation |
| --- | --- |
| Timestamped live-frame provenance | `survng/app/camera.py`, `TimestampedLiveFrame` in `survng/app/motion_pipeline/object_detection.py` |
| Fast live inference and freshness/geometry gates | `RecordedMotionObjectDetector.detect_initial()` |
| Recorded temporal sampling and representative selection | `RecordedMotionObjectDetector.detect()` / `_detect()` |
| Admission, activity, correlation, snapshot writes | `MotionDecisionHandler` in `survng/app/motion_pipeline/decision_handler.py` |
| Durable delayed-refinement orchestration | `MotionIncidentService` in `survng/app/motion_incidents.py` |
| Guarded atomic main-cover promotion | `EventStore.promote_refinement_cover()` in `survng/app/events.py` |
| Tracking-based cover verification/promotion | `survng/app/object_tracking.py` |
| Live client refresh without duplicate notification | `survng/app/manager.py` |
| EMA audit outcome | `survng/app/motion_decisions.py` |
