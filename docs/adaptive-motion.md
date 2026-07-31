# Motion triggers and validation

SurvNG separates the signal that **starts** a motion event from the visual analysis that can **validate** it. Motion processing decides when the more expensive object detector should run; it does not classify people, vehicles, or animals.

## Camera-triggered mode (recommended)

```text
ONVIF camera notice
  → optional adaptive validation
  → optional MOG2 confirmation
  → high-resolution recorded-frame extraction
  → object detection
  → incident when an eligible object is found
```

Only ONVIF camera notices and manual tests can start object detection. Adaptive analysis and MOG2 are validators, never independent triggers in this mode.

Validation choices are:

- neither validator: every camera notice runs object detection;
- adaptive only: the recommended default;
- MOG2 only: retained for compatibility and comparison; or
- adaptive plus MOG2: require both for fewer false triggers, or allow either for greater sensitivity.

Semantic ONVIF notices naming a person, vehicle, animal, or face bypass ordinary visual validation. If a configured validator is unavailable or still warming up, SurvNG fails open and runs object detection rather than risking a missed event.

## Camera + visual backup mode

```text
ONVIF camera notice ───────────────────────────────┐
                                                   ├→ object detection
strong persistent adaptive motion                  │
  → wait briefly for ONVIF                         │
  → conservative score and persistence safeguards ┘
```

This mode keeps ONVIF as the primary trigger but covers a camera that occasionally fails to send a notice. After a startup learning period lets the adaptive background stabilize, a visual backup is considered only when adaptive motion is accepted, exceeds both an absolute confidence floor and a margin above the learned scene threshold, persists for multiple samples, and does not match known illumination, insect, persistent-scene, or stationary-foreground rejection reasons.

SurvNG waits briefly for the camera notice before taking the backup path. A recent ONVIF notice suppresses the backup, and per-camera cooldown plus a five-minute rate limit bounds detector work during a noisy scene. A backup invocation does not create an empty incident: high-resolution object detection must find an incident-eligible object. Every completed backup attempt is stored under **Admin > Motion Audit > Visual backup**, including attempts where no object was found.

Use this mode when ONVIF is normally reliable but an occasional missing camera notice is unacceptable. It costs more CPU than Camera-triggered mode because adaptive analysis must continuously inspect the live/substream feed. It remains more conservative than Visual-triggered mode.

## Visual-triggered mode

```text
live/substream video
  → adaptive visual trigger
  → optional MOG2 confirmation
  → event state and cooldown
  → high-resolution recorded-frame extraction
  → object detection
```

Adaptive visual motion is the only automatic trigger. MOG2 can optionally corroborate it. Ordinary ONVIF notices are retained as diagnostic evidence but cannot start object detection. Manual tests remain available.

Use this mode only when ONVIF notifications are unavailable or unreliable. If the live analysis feed fails, automatic motion triggering is unavailable until that feed recovers.

## Which video feed is analyzed?

Adaptive analysis reads the camera's live capture source:

1. `live_stream_url` when a live/substream is configured.
2. `stream_url` when no separate live stream exists.

The frame is downscaled to `motion_qualification.frame_width` and converted to grayscale. The default is 320 pixels wide at 5 samples per second. Continuous recording remains stream-copy based and object detection uses high-resolution frames from the main recording after a trigger.

## CPU and queue behavior

Only enabled visual processors consume analysis time. Camera + visual backup and Visual-triggered modes continuously run the selected qualification pipeline; the adaptive processor remains the recommended default. A shared semaphore permits no more than two cameras to execute visual analysis at the same instant. Cameras waiting for a slot retain recent frames, while their bounded scheduling queues replace stale pending requests with the newest request.

Capture, recording, and live view are not limited to two cameras. Enabling MOG2 adds a second background-analysis algorithm and therefore increases CPU usage. Runtime camera status reports analyzed frames, accepted candidates, dropped scheduling requests, delivered triggers, pipeline failures, and per-stage timing.
