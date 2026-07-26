# Adaptive motion triggers

SurvNG separates continuous video analysis from object detection. Adaptive motion decides **when** the more expensive object detector should run; it does not classify people, vehicles, or animals itself.

## Trigger modes

### SurvNG smart motion (`enforce`)

This is the recommended mode when ONVIF motion notifications are absent or unreliable.

```text
live/substream video
  → adaptive visual analysis
  → credible motion decision
  → high-resolution recorded-frame extraction
  → object detection
  → incident when an eligible object is found
```

SurvNG visual motion can initiate detection independently. ONVIF remains an optional supporting source. Routine camera motion notices are qualified, while semantic person, vehicle, animal, face, and manual notices bypass image qualification.

### Camera alerts + decision preview (`audit`)

```text
ONVIF/manual notice
  → object detection

live/substream video
  → adaptive analysis and telemetry only
```

Camera notices are never filtered. Adaptive analysis remains warm and reports what it would decide, but it cannot create an event by itself. If the camera sends no ONVIF motion notice, object detection does not run.

### Camera alerts without filtering (`off`)

Only ONVIF or manual notices start object detection, and every notice is passed through. Scene learning remains warm so switching modes does not require a cold start.

## Which video feed is analyzed?

Adaptive analysis reads the camera's live capture source:

1. `live_stream_url` when a live/substream is configured.
2. `stream_url` when no separate live stream exists.

The frame is downscaled to `motion_qualification.frame_width` and converted to grayscale. The default is 320 pixels wide at 5 samples per second. Continuous recording remains stream-copy based and object detection uses high-resolution frames from the main recording after a trigger.

## CPU and queue behavior

Every enabled camera samples frames continuously. A shared semaphore permits no more than two cameras to execute adaptive analysis at the same instant. Cameras waiting for a slot retain recent frames, while their bounded scheduling queues replace stale pending requests with the newest request.

This design means:

- motion detection remains enabled for every camera;
- capture, recording, and live view are not limited to two cameras;
- the system avoids a burst where every camera runs OpenCV stages simultaneously;
- delayed analysis does not become an ever-growing stale backlog; and
- switching from preview to smart motion does not add a second continuous analyzer, though accepted triggers add recorded-frame extraction and object-detection work.

Runtime camera status reports visual frames analyzed, accepted analysis frames, dropped scheduling requests, delivered triggers, pipeline failures, and per-stage timing. Accepted analysis frames are not incident counts: event-state hysteresis and cooldown consolidate continuous activity into bounded trigger transitions.

## Choosing a mode

Use **SurvNG smart motion** when the camera misses notifications, when ONVIF is unavailable, or when SurvNG should be the primary motion trigger. Use **Camera alerts + decision preview** only while evaluating SurvNG decisions on cameras whose ONVIF motion notices are known to work. Use **Camera alerts without filtering** when the camera's own alerting is authoritative and every notice should run object detection.
