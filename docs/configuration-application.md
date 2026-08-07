# Configuration application boundaries

SurvNG applies saved configuration at the narrowest runtime boundary that owns
the changed setting. The `PUT /api/config` response reports `apply_mode`,
`camera_workers_restarted`, `subsystems_restarted`, and `hot_updated` so the UI
can explain what was interrupted.

## Hot-applied settings

These settings do not restart cameras, ONVIF subscriptions, capture, or
recording processes:

- web base path and incident-thumbnail display;
- event-clip before/after windows;
- playback-cache size, age, and finalized-recording prewarming;
- AI motion-review provider settings; and
- recording-retention policy and per-camera retention overrides.

Retention changes wake the index-driven planner. They do not reconstruct the
recorder or camera workers.

## Targeted subsystem restarts

MQTT changes replace only the MQTT client, network loop, and command worker.
Recorder engine changes (`ffmpeg_path`, hardware acceleration, and segment
duration) fence the recorder watchdog and restart only FFmpeg recorder
processes. Camera capture and ONVIF workers remain active.

A dedicated `RecordingLifecycle` owns stale-process cleanup, indexer/watchdog
startup, terminal teardown, and recorder-engine cutover. A failed targeted
cutover restores the prior FFmpeg settings and desired per-camera recorder
state before surfacing the error, so configuration persistence cannot leave a
partially replaced recording generation behind.

## Full manager reload

A transactional manager reload remains necessary when a setting changes object
detection/tracking, motion processing, camera streams, ONVIF, recording source
selection, camera membership, or the media/database/index storage locations.
These dependencies are constructed into camera workers or shared services and
cannot be safely replaced as a standalone value.

The replacement manager constructs a dedicated `CameraFleetLifecycle`, which
owns bounded camera admission, live power-state changes, early ONVIF release,
and a two-phase fleet stop: broadcast every stop request, then wait against one
absolute deadline. A timed-out camera remains an observable fleet residual;
SurvNG will not start a replacement manager or close shared inference/recording
dependencies beneath it. The process supervisor owns the hard termination
deadline. The web process starts serving during normal
progressive startup. A configuration cutover commits after core service
construction and atomic persistence succeed, then exposes progressive camera
admission instead of blocking the save on unavailable feeds. MQTT replacement
also updates the fleet's typed state-publisher dependency, so an admission task
never publishes through a retired MQTT generation.

Inference-only changes do not rebuild the camera fleet. A dedicated
`InferenceLifecycle` transaction owns detector-role replacement, face queue
coordination, tracking-session cutover, ReID/backfill dependencies, and semantic
search generations. It prepares replacements before cutover, restores prior
tracking and detector state on failure, and exposes any old generation whose
cleanup must be retried. This keeps configuration persistence aligned with the
generation actually serving requests.

The configuration file is written atomically before a targeted runtime update.
If runtime application fails, SurvNG restores both the previous runtime settings
and the previous persisted configuration. Full manager reloads retain their
existing replacement-manager rollback behavior.
