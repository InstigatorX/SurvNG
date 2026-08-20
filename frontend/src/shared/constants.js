export const DEFAULT_TIME_ZONE = "America/New_York";
export const MEDIA_STORAGE_ROLES = [
  ["recordings", "Recordings"],
  ["snapshots", "Snapshots"],
  ["motion_audits", "Motion audits"],
  ["clips", "Clips"],
  ["exports", "Exports"],
];
export const CAMERA_ADMIN_SECTIONS = ["settings", "motion", "zones", "info"];
export const TELEMETRY_ADMIN_SECTIONS = ["overview", "cameras", "diagnostics"];
export const GENERAL_ADMIN_SECTIONS = ["general", "storage", "mqtt", "detection", "motion-review"];
export const LEGACY_INCIDENT_FILTER_KEYS = [
  "survng.liveEventFilter.v2",
  "survng.incidentDay.v1",
  "survng.incidentCameraFilter.v1",
  "survng.incidentObjectFilter.v1",
  "survng.incidentZoneFilter.v1",
];
export const US_TIME_ZONES = [
  ["America/New_York", "Eastern"],
  ["America/Chicago", "Central"],
  ["America/Denver", "Mountain"],
  ["America/Phoenix", "Arizona"],
  ["America/Los_Angeles", "Pacific"],
  ["America/Anchorage", "Alaska"],
  ["Pacific/Honolulu", "Hawaii"],
];
export const THEMES = ["auto", "light", "dark"];
export const SECRET_PLACEHOLDER = "__SURVNG_SECRET_SET__";
export const PREFER_NATIVE_HLS = /iPad|iPhone|iPod/.test(navigator.userAgent)
  || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
export const APP_EVENT_TYPES = ["camera_state", "cameras_state", "motion", "object", "incident", "system_state"];
export const INCIDENT_REFRESH_FALLBACK_MS = 15_000;
export const STREAM_MODES = ["motion", "mjpeg", "webrtc"];
export const STREAM_LABELS = {
  motion: "Auto",
  mjpeg: "MJPEG",
  webrtc: "WebRTC",
};
export const MOTION_WEBRTC_HOLD_MS = 30_000;
export const LIVE_TRANSPORT_LABELS = {
  webrtc: "WebRTC",
  mse: "MSE",
  mjpeg: "MJPEG",
  recording: "Recording",
  snapshot: "Snapshot",
};
export const ALL_RECORDING_CAMERAS_ID = "all";
