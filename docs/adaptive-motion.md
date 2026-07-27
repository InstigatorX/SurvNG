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

Only enabled visual processors consume analysis time. A shared semaphore permits no more than two cameras to execute visual analysis at the same instant. Cameras waiting for a slot retain recent frames, while their bounded scheduling queues replace stale pending requests with the newest request.

Capture, recording, and live view are not limited to two cameras. Enabling MOG2 adds a second background-analysis algorithm and therefore increases CPU usage. Runtime camera status reports analyzed frames, accepted candidates, dropped scheduling requests, delivered triggers, pipeline failures, and per-stage timing.
