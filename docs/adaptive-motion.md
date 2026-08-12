# Motion triggers and validation

SurvNG separates the signal that **starts** a motion event from the visual analysis that can **validate** it. Motion processing decides when the more expensive object detector should run; it does not classify people, vehicles, or animals.

## Camera-triggered mode

```text
ONVIF camera notice
  → optional adaptive validation
  → high-resolution recorded-frame extraction
  → object detection
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
  → wait briefly for ONVIF                         │
  → conservative score and persistence safeguards ┘
```

This default mode keeps ONVIF as the primary trigger but covers a camera that occasionally fails to send a notice. After a startup learning period lets the adaptive background stabilize, an EMA backup is considered only when motion is accepted, exceeds both an absolute confidence floor and a margin above the learned scene threshold, persists for multiple samples, and does not match known illumination, insect, persistent-scene, or stationary-foreground rejection reasons.

Camera notices and qualified EMA evidence enter one per-camera motion episode. An ONVIF notice suppresses duplicate EMA work only after an object-detection request for that episode was actually admitted; an observed notice whose work could not be queued cannot hide a later EMA rescue. A backup invocation does not create an empty incident: high-resolution object detection must find an incident-eligible object. Every completed backup attempt is stored under **Admin > Motion Audit > EMA backup**, including attempts where no object was found.

This is the recommended general-purpose behavior when ONVIF is normally reliable but an occasional missing camera notice is unacceptable. It costs more CPU than Camera-triggered mode because EMA must continuously inspect the live/substream feed. It remains more conservative than EMA-only mode.

## EMA-only mode

```text
live/substream video
  → EMA visual trigger
  → event state and cooldown
  → high-resolution recorded-frame extraction
  → object detection
```

EMA visual motion is the only automatic trigger. Ordinary ONVIF notices are retained as diagnostic evidence but cannot start object detection. Manual tests remain available.

Use this mode only when ONVIF notifications are unavailable or unreliable. If the live analysis feed fails, automatic motion triggering is unavailable until that feed recovers.

## Which video feed is analyzed?

Adaptive analysis reads the camera's live capture source:

1. `live_stream_url` when a live/substream is configured.
2. `stream_url` when no separate live stream exists.

The frame is downscaled to `motion_qualification.frame_width` and converted to grayscale. The default is 320 pixels wide at 5 samples per second. Continuous recording remains stream-copy based and object detection uses high-resolution frames from the main recording after a trigger.

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

Only enabled visual processors consume analysis time. Camera + EMA backup and EMA-only modes continuously run EMA qualification. A shared semaphore permits no more than two cameras to execute visual analysis at the same instant. Cameras waiting for a slot retain recent frames, while their bounded latest-frame mailboxes replace stale pending requests with the newest request.

## Episode admission

EMA scene learning and score persistence produce a single qualified edge rather than detector work for every sampled frame. A generation-tagged episode controller then merges that edge with camera notices and is the sole owner of detector reservation, admission, follow-up limits, completion, and incident linkage. A qualified edge is never passed through a second motion state machine. Queue rejection aborts the reservation without starting cooldown, detector retry exhaustion is recorded separately from nuisance rejection, and an old lifecycle generation cannot mutate the replacement camera runtime.

Capture, recording, and live view are not limited to two cameras. Runtime camera status reports analyzed frames, accepted candidates, dropped scheduling requests, delivered triggers, pipeline failures, and per-stage timing.
