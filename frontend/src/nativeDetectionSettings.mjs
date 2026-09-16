// Numeric controls mirror the native runtime's Pydantic configuration bounds.
export const NATIVE_DETECTION_FIELDS = [
  { path: "live_sample_fps", label: "Detection frames per second", initial: 5, min: 0.5, max: 10, step: 0.5, group: "detection", help: "Live/substream frames entering the native pipeline. Fresh inference rate is this value divided by the inference interval." },
  { path: "native.batch_size", label: "Shared inference batch size", initial: 1, min: 1, max: 4, step: 1, integer: true, group: "detection", help: "1 disables batching (default). 2–4 pool frames across cameras in the shared native inference worker. Larger batches can increase waiting time, especially with fewer cameras or higher inference intervals. Saving restarts native capture." },
  { path: "native.inference_interval", label: "Inference interval", initial: 1, min: 1, max: 5, step: 1, integer: true, group: "detection", help: "Run gvadetect every Nth sampled frame. At 5 FPS, interval 2 targets 2.5 fresh detections/sec; skipped frames use tracker predictions and cannot create or extend incidents." },
  { path: "confidence_threshold", label: "Confidence", initial: 0.45, min: 0.01, max: 0.99, step: 0.01, group: "detection" },
  { path: "event_confirmation_frames", label: "Confirmation frames", initial: 2, min: 1, max: 5, step: 1, group: "detection", integer: true },
  { path: "native.activity_timeout_seconds", label: "Presence timeout (seconds)", initial: 5, min: 1, max: 60, step: 0.5, group: "detection", help: "Time without fresh eligible activity before the episode ends." },
  { path: "native.stationary.stationary_seconds", label: "Stationary after (seconds)", initial: 8, min: 1, max: 120, step: 1, group: "stationary" },
  { path: "native.stationary.window_seconds", label: "Motion window (seconds)", initial: 2, min: 0.5, max: 10, step: 0.5, group: "stationary", help: "At least five observations are retained. Low frame rates take longer to establish movement." },
  { path: "native.stationary.moving_threshold", label: "Movement threshold", initial: 0.15, min: 0.01, max: 2, step: 0.01, group: "stationary", help: "Movement relative to box size: 0.15 is 15% of object width or height." },
  { path: "native.stationary.stationary_threshold", label: "Stationary jitter threshold", initial: 0.05, min: 0.001, max: 1, step: 0.001, group: "stationary", help: "Allowed variation relative to box size. Must be below the movement threshold." },
  { path: "nms_threshold", label: "Overlapping detection threshold (NMS)", initial: 0.45, min: 0.01, max: 0.99, step: 0.01, group: "advanced", help: "Applies when the model postprocessor performs overlap suppression." },
  { path: "native.maximum_observation_age_seconds", label: "Maximum result age (seconds)", initial: 2, min: 0.2, max: 10, step: 0.1, group: "advanced", help: "Older arriving results cannot create or extend incidents." },
  { path: "native.maximum_tracks", label: "Track capacity per camera", initial: 128, min: 1, max: 1024, step: 1, integer: true, group: "advanced", help: "Separate limits for live context and the active episode archive." },
  { path: "native.metadata_restart_seconds", label: "Metadata recovery timeout (seconds)", initial: 15, min: 5, max: 120, step: 1, group: "advanced", help: "Restart capture when frames arrive but fresh detection metadata remains absent." },
  { path: "native.inference_requests", label: "Shared inference requests", initial: 4, min: 1, max: 16, step: 1, integer: true, group: "advanced", help: "Request pool for the model shared by native camera pipelines. Higher values use more resources." },
  { path: "native.inference_streams", label: "Shared inference streams", initial: 2, min: 1, max: 8, step: 1, integer: true, group: "advanced", help: "OpenVINO GPU/CPU execution streams for the shared native model." },
];
export const DEFAULT_STATIONARY_LABELS = ["car", "truck", "bus", "van", "suv", "motorcycle"];
export function detectionFieldValue(detector, field) {
  return field.path.split(".").reduce((value, key) => value?.[key], detector) ?? field.initial;
}
export function nativeDetectionError(detector = {}) {
  for (const field of NATIVE_DETECTION_FIELDS) {
    const value = detectionFieldValue(detector, field);
    if (typeof value !== "number" || !Number.isFinite(value) || value < field.min || value > field.max || (field.integer && !Number.isInteger(value))) {
      return `${field.label} must be ${field.integer ? "a whole number " : ""}between ${field.min} and ${field.max}.`;
    }
  }
  const native = detector.native || {};
  const tracked = native.tracking_classes;
  if (tracked != null && (!Array.isArray(tracked) || tracked.length > 256 || tracked.some((label) => typeof label !== "string" || !label.trim() || label.length > 128))) return "Tracked classes must contain at most 256 nonempty model labels.";
  const batchSize = native.batch_size ?? 1;
  const batchWait = (batchSize - 1) * (native.inference_interval ?? 1) / (detector.live_sample_fps ?? 5);
  if (batchSize > 1 && batchWait >= (native.maximum_observation_age_seconds ?? 2)) return "Batch waiting time must be below maximum result age when only one camera is connected. Reduce batch size or inference interval, increase detection FPS, or increase maximum result age.";
  const stationary = detector.native?.stationary || {};
  if ((stationary.stationary_threshold ?? 0.05) >= (stationary.moving_threshold ?? 0.15)) return "Stationary jitter threshold must be below the movement threshold.";
  if ((stationary.labels || []).length > 64) return "Stationary filtering supports at most 64 object classes.";
  for (const [key, min, max, integer] of [["event_class_confirmation_frames", 1, 5, true], ["event_class_confidence_thresholds", 0.01, 0.99, false]]) {
    for (const [label, value] of Object.entries(detector[key] || {})) {
      if (!Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) return `${label}: ${integer ? "confirmation frames" : "confidence"} must be between ${min} and ${max}${integer ? " (whole number)" : ""}.`;
    }
  }
  return "";
}
