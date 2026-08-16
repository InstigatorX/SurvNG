export const ADMIN_WORKSPACES = Object.freeze([
  { id: "general", label: "Settings" },
  { id: "cameras", label: "Cameras" },
  { id: "audit", label: "Motion Audit" },
  { id: "calibration", label: "Detection Tune-Up" },
  { id: "telemetry", label: "Telemetry" },
  { id: "maintenance", label: "Maintenance" },
  { id: "logs", label: "Logs" },
]);

export const GENERAL_SECTION_LABELS = Object.freeze({
  general: "General",
  storage: "Storage & Retention",
  mqtt: "API & MQTT",
  detection: "Object Detection",
  "motion-review": "Camera Advisor",
});

export function nextTabId(ids, selected, key) {
  const values = Array.isArray(ids) ? ids.filter(Boolean) : [];
  if (!values.length || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(key)) return null;
  const currentIndex = Math.max(0, values.indexOf(selected));
  if (key === "Home") return values[0];
  if (key === "End") return values.at(-1);
  const offset = key === "ArrowLeft" ? -1 : 1;
  return values[(currentIndex + offset + values.length) % values.length];
}

export function adminWorkspaceId(value, fallback = "general") {
  const candidate = String(value || "");
  return ADMIN_WORKSPACES.some((item) => item.id === candidate) ? candidate : fallback;
}

export function readAdminWorkspace(search = "", stored = "general") {
  const params = new URLSearchParams(String(search || "").replace(/^\?/, ""));
  return adminWorkspaceId(params.get("section"), adminWorkspaceId(stored));
}

export function adminWorkspaceSearch(section, currentSearch = "", { subsection = "", camera = "" } = {}) {
  const id = adminWorkspaceId(section);
  const current = new URLSearchParams(String(currentSearch || "").replace(/^\?/, ""));
  const next = new URLSearchParams();
  if (id !== "general" || subsection || camera) next.set("section", id);
  if (subsection) next.set("subsection", String(subsection));
  if (camera) next.set("camera", String(camera));
  if (id === "audit" && current.get("audit_id")) next.set("audit_id", current.get("audit_id"));
  const query = next.toString();
  return query ? `?${query}` : "";
}

export function readAdminSubsection(search = "", allowed = [], fallback = "") {
  const params = new URLSearchParams(String(search || "").replace(/^\?/, ""));
  const candidate = String(params.get("subsection") || "");
  return allowed.includes(candidate) ? candidate : fallback;
}

export function comparableSystemConfig(config) {
  if (!config || typeof config !== "object") return null;
  const { cameras: _cameras, ...system } = config;
  return system;
}

export function configValuesEqual(left, right) {
  return JSON.stringify(left ?? null) === JSON.stringify(right ?? null);
}

export function comparableCameraSettings(camera) {
  if (!camera || typeof camera !== "object") return null;
  const { zones: _zones, ...settings } = camera;
  return settings;
}

export function cameraConfigDirtyState(cameras = [], baselineCameras = []) {
  const baseline = new Map((baselineCameras || []).map((camera) => [camera.id, camera]));
  const settings = (cameras || []).some((camera) => !configValuesEqual(
    comparableCameraSettings(camera),
    comparableCameraSettings(baseline.get(camera.id)),
  ));
  const zones = (cameras || []).some((camera) => !configValuesEqual(
    camera.zones || [],
    baseline.get(camera.id)?.zones || [],
  ));
  return { settings, zones };
}

export function preferredStoredValue(initialValue, storedValue, preferInitial = false) {
  return preferInitial ? initialValue : storedValue;
}
