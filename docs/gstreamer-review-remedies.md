# GStreamer integration review remedies

These changes target `v1.3-gstreamer` after PR #195. They do not change v1.2,
recording format, event storage, detector cadence, or the VA-memory inference
path. No new dependency is required.

## Tracking continuity without false color evidence

The camera now retains a bounded history of actual live qualifier frames when
live/main geometry is trusted and near identity. Each sample keeps its capture
generation, session, PTS, and receipt timestamp. Asynchronous sidecars are
attached as they arrive; a later exact empty result replaces an older positive
match. Retained snapshots survive eviction from the capture metadata history.

Tracking merges this history with recorded/main samples. Buffered live samples
use their matched detections, never a new OpenVINO request. Duplicate results
are consumed once; missing results are neither empty detections nor deferred
OpenVINO jobs. The existing continuity-gap limit remains in force. Reconnects
create a boundary while preserving the readable old-session prefix.

Luma samples do not enter the color refinement iterator or feed tracking depth,
ReID, or cover selection. Cropped/untrusted views still require main evidence;
this change does not invent a cross-view tracking transform.

## Native worker ownership

Stream removal retains the worker until its thread exits. If the bounded join
times out, the supervisor stops admitting commands and emits a fatal failure
through the existing protocol. The parent invalidates affected capture inboxes
and closes/kills the failed process before creating its replacement. Global
shutdown uses one shared deadline rather than a timeout multiplied by cameras.

This addresses an ownership defect. It does **not** establish or fix the root
cause of the previously observed VA-API `vaSyncSurface` failure. A GPU soak and
reconnect test is still required before claiming runtime stability.

## Per-camera EMA resolution

Each capture handle receives its camera's effective EMA width; the shared
supervisor applies that width to the stream's qualifier branch. Main capture
remains 640-wide BGR. Existing configuration reload logic replaces only the
camera whose width override changed. Global width remains the default.

## Native NMS policy

`detector.nms_threshold` now reaches the native child. Intel 2026.2 reads
`iou_threshold` from [model metadata or model-proc](https://docs.openedgeplatform.intel.com/2026.2/edge-ai-libraries/dlstreamer/dev_guide/model_info_xml.html),
not a `gvadetect` element property. A process-private temporary XML/model-proc
applies the policy without modifying installed models or copying weights.
Adjacent exporter metadata and the original preprocessing/converter are retained.

Intel's [YOLO10/26 converter forces an NMS pass](https://github.com/open-edge-platform/dlstreamer/blob/v2026.2.0/src/monolithic/gst/inference_elements/common/post_processor/converters/to_roi/yolo_v10.h),
including for end-to-end outputs. For recognized final-output YOLO converters
and generic YOLO IR with a final `[batch, proposals, 6]` head, the native IoU
threshold is set to 1.0 so it does not suppress final detections. Raw-output
models use the configured threshold. There is no additional Python NMS pass.
Capture status reports the **requested** threshold; final-output policy takes
precedence. Direct native CLI use without `--nms-threshold` keeps model policy.

## Regression coverage

`tests/test_gstreamer_review_contracts.py` covers the camera-to-history-to-tracker
handoff, delayed/empty sidecars, geometry rejection, timed-out worker removal,
per-camera width/reload, and non-mutating NMS configuration.

The CI native smoke uses Intel's pinned 2026.2 CPU runtime and synthetic models:

- Two shared live streams at 320 and 960 pixels plus 640-wide BGR main capture.
- Independent 5 FPS EMA / 2.5 FPS detection, including completed empty results.
- Overlapping raw YOLO boxes: NMS 0.1 emits one box, 0.95 emits two, through both
  IR metadata and model-proc configuration paths.
- Final YOLO26 boxes remain unsuppressed at both requested thresholds, with
  adjacent Ultralytics metadata still resolved.

These are plumbing and policy checks, not a camera-accuracy benchmark or GPU
stability test. Deployment remains an explicit follow-up, not part of this PR.
