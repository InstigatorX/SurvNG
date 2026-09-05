# Motion triggers and validation

SurvNG separates the signal that **starts** a motion event from the visual analysis that can **validate** it. Motion processing decides when the more expensive object detector should run; it does not classify people, vehicles, or animals.

## Camera-triggered mode

```text
ONVIF camera notice
  → optional adaptive validation
  → fresh live-frame object check
  → durable high-resolution recorded-frame refinement
  → incident when an eligible object is found
```

Only ONVIF camera notices and manual tests can start object detection. EMA is a validator, never an independent trigger in this mode.

EMA validation may be enabled or disabled so every camera notice runs object detection.

Semantic ONVIF notices naming a person, vehicle, animal, or face bypass ordinary visual validation. If a configured validator is unavailable or still warming up, SurvNG fails open and runs object detection rather than risking a missed event.

## Camera + EMA backup mode (default)

```text
ONVIF camera notice ───────────────────────────────┐
                                                   ├→ object detection
strong persistent EMA motion                       │
  → conservative score and persistence safeguards │
accepted lower-score persistent EMA                │
  → slower security-verification lane              ┘
```

This default mode keeps ONVIF as the primary trigger but covers a camera that occasionally fails to send a notice. After a startup learning period lets the adaptive background stabilize, strong accepted motion uses the normal score and persistence safeguards. Accepted motion below that rescue score is not permanently discarded: a second, slower conditioner requests one security-verification pass when it persists longer. Known illumination, insect, persistent-scene, and stationary-foreground rejection reasons remain ineligible for either path.

A confirmed eligible object can open a short verification window on the next camera in a configured camera-transition route. During that bounded window, one accepted non-nuisance EMA sample is enough to request analysis because the upstream object supplies the temporal prior. After ONVIF transport has repeatedly failed to correlate with qualified EMA evidence, the slower lane instead uses the normal persistence duration. These signals authorize object analysis only; they never create an incident or bypass object confidence, temporal confirmation, zones, activity attribution, or EMA/object correlation.

Camera notices and qualified EMA evidence enter one per-camera motion episode. An ONVIF notice suppresses duplicate EMA work only after an object-detection request for that episode was actually admitted; an observed notice whose work could not be queued cannot hide a later EMA rescue. A backup invocation does not create an empty incident: durable main-recording refinement must find an incident-eligible object that passes the required causal checks. Every completed backup attempt is stored under **Admin > Motion Audit > EMA backup**, including attempts where no object was found.

This is the recommended general-purpose behavior when ONVIF is normally reliable but an occasional missing camera notice is unacceptable. It costs more CPU than Camera-triggered mode because EMA must continuously inspect the live/substream feed. It remains more conservative than EMA-only mode.

## EMA-only mode

```text
live/substream video
  → EMA visual trigger
  → event state and cooldown
  → fresh live-frame object check
  → durable high-resolution recorded-frame refinement
```

EMA visual motion is the only automatic trigger. Ordinary ONVIF notices are retained as diagnostic evidence but cannot start object detection. Manual tests remain available.

Use this mode only when ONVIF notifications are unavailable or unreliable. If the live analysis feed fails, automatic motion triggering is unavailable until that feed recovers.

## Which video feed is analyzed?

Adaptive analysis reads the camera's live capture source:

1. `live_stream_url` when a live/substream is configured.
2. `stream_url` when no separate live stream exists.

The frame is downscaled to `motion_qualification.frame_width` and converted to
grayscale. The default is 320 pixels wide at 5 samples per second. Continuous
recording remains stream-copy based. After a trigger, object detection performs
a strictly fresh live-frame check for low latency and always schedules the
high-resolution main-recording temporal pass. The fast frame is provisional;
main evidence supplies temporal confirmation and normally supplies the durable
representative cover. See [Incident evidence data
path](incident-evidence-data-path.md) for the complete frame provenance,
geometry, correlation, refinement, and cover-selection contract.

## EMA exclusion zones

Each detection zone can independently enable **Exclude from EMA**. Motion inside that polygon is removed after scene learning and adaptive thresholding but before morphology, connected-component extraction, tracking, and scoring. Background learning still covers the complete frame, so resizing or disabling the exclusion does not reveal a stale scene model.

EMA exclusion does not change the zone's object behavior. An Incident zone can exclude nuisance motion while continuing to admit matching objects, and an Ignore zone suppresses matching object incidents without excluding EMA unless both settings are selected. For a dedicated EMA-only polygon, select **No object effect** and enable **Exclude from EMA**. Motion crossing out of an excluded polygon becomes eligible at the boundary. The rasterized mask is cached per camera and resolution and rebuilt automatically when zone geometry changes.

## Stationary objects and scene context

SurvNG deliberately keeps two complementary controls separate:

- **EMA stationary-motion filtering** runs on the low-resolution visual-analysis feed before object detection. It rejects confined outline shimmer, centroid oscillation, and persistent scene motion. Light, Standard, and Strong change how much movement may still be considered stationary; Strong can also reject unusually slow or distant travel.
- **Repeated scene context** runs after high-resolution temporal object detection. It distinguishes an object that moved, appeared, or entered a zone from an object repeatedly observed at the same location across incidents. In enforcement mode, proven scene context remains stored as evidence but does not label the incident. Uncertain evidence fails open.

The global controls live together under **Admin > General > Object Detection > Stationary objects & scene context**. Cameras can inherit or override both policies. Fixed zones remain explicit and independent: **Ignore** affects matching object classes, while **Exclude from EMA** affects all visual motion in the polygon because EMA has no object-class information.

Object confidence and confirmation are separate from activity. They establish that a label is credible and repeatable; they do not establish that the object caused the event. Visual-backup correlation adds one further causal check by requiring an admitted object to move across detector samples or explain the EMA motion region.

## CPU and queue behavior

Only enabled visual processors consume analysis time. Camera + EMA backup and EMA-only modes continuously run EMA qualification. A shared fair limiter permits `motion_qualification.max_concurrent_analysis` cameras (default 2) to execute visual analysis at the same instant. Raise that ceiling on a larger NVR if backup coverage is falling behind. Cameras waiting for a slot retain recent frames, while their bounded latest-frame mailboxes replace stale pending requests with the newest request.

The former `motion_qualification.temporal_filter_threshold` shortcut is retired. Existing configuration files containing it remain valid, but the value is ignored and its control no longer appears in Admin. Every cadence-due sample now reaches the scene-aware pipeline so quiet frames and small motion can update the adaptive background. Sampling rate, latest-frame mailboxes, and the shared concurrency limit still bound the work, though installations that previously skipped many quiet frames should expect higher EMA CPU use.

The classic modular scorer now waits for four frames before it can produce a continuous trigger. Camera-event validation remains fail-open while a frame window is incomplete or has no temporal signal. In custom adaptive graphs, explicit `stationary_displacement_ratio` and `stationary_path_ratio` stage options override the named Light, Standard, or Strong stationary policy; omit those options to follow the named policy.

## Episode admission

EMA scene learning and score persistence produce a single qualified edge rather than detector work for every sampled frame. A generation-tagged episode controller then merges that edge with camera notices and is the sole owner of detector reservation, admission, follow-up limits, completion, and incident linkage. Every controller incarnation has a random durable namespace, so restarting SurvNG or replacing a camera runtime cannot reuse an old episode, refinement-job, or event idempotency key. Exact retries still coalesce; a conflicting reuse is an explicit error instead of a silent drop.

A qualified edge is never passed through a second motion state machine. Queue pressure durably defers admitted work rather than discarding it. Ordinary and persistent nuisance protection still use cooldown and request budgets. Only a one-shot, confirmed route watch receives a stable route-specific intent that bypasses those two pre-detector limits; downstream eligibility remains unchanged. Compact accepted EMA route candidates are retained for ten minutes in a dedicated, bounded replay-cache database so a restart during delayed upstream confirmation does not erase downstream evidence or contend with incident writes. Detector retry exhaustion is recorded separately from nuisance rejection, and an old lifecycle generation cannot mutate the replacement camera runtime.

Capture, recording, and live view are not limited to two cameras. Runtime camera status reports analyzed frames, accepted candidates, dropped scheduling requests, delivered triggers, pipeline failures, and per-stage timing.
