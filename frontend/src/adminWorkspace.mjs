export const ADMIN_WORKSPACES = Object.freeze([
  { id: "home", label: "Configure Home" },
  { id: "general", label: "Settings" },
  { id: "cameras", label: "Cameras" },
  { id: "reid", label: "ReID Training" },
  { id: "audit", label: "Motion Audit" },
  { id: "calibration", label: "Detection Tune-Up" },
  { id: "telemetry", label: "Telemetry" },
  { id: "maintenance", label: "Maintenance" },
  { id: "logs", label: "Logs" },
]);

export const ADMIN_RESPONSIBILITY_GROUPS = Object.freeze([
  {
    id: "configure",
    label: "Configure",
    items: [
      { id: "cameras", label: "Cameras", workspace: "cameras" },
      { id: "detection", label: "Detection", workspace: "general", subsection: "detection" },
      { id: "storage", label: "Storage", workspace: "general", subsection: "storage" },
      { id: "integrations", label: "Integrations", workspace: "general", subsection: "mqtt" },
      { id: "access", label: "Access", workspace: "general", subsection: "access" },
      { id: "server", label: "Server", workspace: "general", subsection: "general", secondary: true },
    ],
  },
  {
    id: "observe",
    label: "Observe",
    items: [
      { id: "health", label: "Health", workspace: "telemetry" },
      { id: "audit", label: "Audit", workspace: "audit" },
      { id: "logs", label: "Logs", workspace: "logs" },
    ],
  },
  {
    id: "act",
    label: "Act",
    items: [
      { id: "tuneup", label: "Tune-Up", workspace: "calibration" },
      { id: "reid", label: "ReID Training", workspace: "reid" },
      { id: "diagnostics", label: "Diagnostics", workspace: "telemetry", subsection: "diagnostics" },
      { id: "maintenance", label: "Maintenance", workspace: "maintenance" },
      { id: "advisor", label: "Camera Advisor", workspace: "general", subsection: "motion-review", secondary: true },
    ],
  },
]);

// Operator-facing navigation. Keep the older responsibility map above as the
// compatibility source for deep links and destination resolution, while this
// map groups destinations by the job an administrator is trying to accomplish.
export const ADMIN_NAV_GROUPS = Object.freeze([
  { id: "configure", label: "Configure", items: [
    { id: "home", label: "Configure Home", workspace: "home", description: "See what needs attention and jump into setup." },
    { id: "cameras", label: "Cameras", workspace: "cameras", description: "Add cameras, tune video, motion, and zones." },
  ] },
  { id: "intelligence", label: "Intelligence", items: [
    { id: "detection", label: "Detection", workspace: "general", subsection: "detection", description: "Models, confidence, and object recognition." },
    { id: "tuneup", label: "Detection Tune-Up", workspace: "calibration", description: "Review evidence and apply bounded improvements." },
    { id: "reid", label: "ReID Training", workspace: "reid", description: "Review person crops and cross-camera pairs for domain fine-tuning." },
    { id: "advisor", label: "Camera Advisor", workspace: "general", subsection: "motion-review", description: "Get camera-specific recommendations." },
  ] },
  { id: "data", label: "Data & Retention", items: [
    { id: "storage", label: "Storage & Retention", workspace: "general", subsection: "storage", description: "Locations, retention plans, and cleanup." },
    { id: "maintenance", label: "Storage Maintenance", workspace: "maintenance", description: "Reconcile and repair the media index." },
  ] },
  { id: "integrations", label: "Integrations", items: [
    { id: "integrations", label: "API & MQTT", workspace: "general", subsection: "mqtt", description: "Connect SurvNG to other services." },
  ] },
  { id: "security", label: "Security", items: [
    { id: "access", label: "Users & Access", workspace: "general", subsection: "access", description: "Accounts, sessions, and API tokens." },
  ] },
  { id: "system", label: "System", items: [
    { id: "server", label: "Server Preferences", workspace: "general", subsection: "general", description: "Timezone, appearance, and server behavior." },
  ] },
  { id: "observe", label: "Observe", items: [
    { id: "health", label: "Health", workspace: "telemetry", description: "Runtime health across cameras and services." },
    { id: "audit", label: "Motion Audit", workspace: "audit", description: "Inspect motion decisions and outcomes." },
    { id: "diagnostics", label: "Diagnostics", workspace: "telemetry", subsection: "diagnostics", description: "Capture bounded troubleshooting data." },
    { id: "logs", label: "Logs", workspace: "logs", description: "Review server activity and errors." },
  ] },
]);

export const ADMIN_HOME_DESTINATION_IDS = Object.freeze([
  "cameras",
  "detection",
  "storage",
  "integrations",
  "access",
  "server",
]);

export function adminHomeDestinations() {
  const destinations = new Map(ADMIN_NAV_GROUPS.flatMap((group) => group.items).map((item) => [item.id, item]));
  return ADMIN_HOME_DESTINATION_IDS.map((id) => destinations.get(id)).filter(Boolean);
}

export function normalizeTelemetrySection(value = "") {
  const candidate = String(value || "");
  return candidate === "diagnostics" ? "diagnostics" : "health";
}

export function adminDestination(workspace, { generalSection = "general", telemetrySection = "health" } = {}) {
  if (workspace === "home") return ADMIN_NAV_GROUPS[0].items[0];
  return ADMIN_RESPONSIBILITY_GROUPS
    .flatMap((group) => group.items)
    .find((item) => item.workspace === workspace && (
      workspace === "general"
        ? (item.subsection || "general") === generalSection
        : workspace === "telemetry"
          ? normalizeTelemetrySection(item.subsection || "health") === normalizeTelemetrySection(telemetrySection)
          : true
    )) || ADMIN_RESPONSIBILITY_GROUPS[0].items[0];
}

export const GENERAL_SECTION_LABELS = Object.freeze({
  general: "Server",
  storage: "Storage & Retention",
  mqtt: "Integrations",
  access: "Access",
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

export function perCameraDirtyState(cameras = [], baselineCameras = []) {
  const baseline = new Map((baselineCameras || []).map((camera) => [camera.id, camera]));
  const dirty = {};
  for (const camera of cameras || []) {
    const baselineCamera = baseline.get(camera.id);
    dirty[camera.id] = {
      settings: !configValuesEqual(
        comparableCameraSettings(camera),
        comparableCameraSettings(baselineCamera),
      ),
      zones: !configValuesEqual(camera.zones || [], baselineCamera?.zones || []),
    };
  }
  return dirty;
}

export function dirtyCameraCount(perCameraDirty = {}) {
  return Object.values(perCameraDirty).filter((item) => item?.settings || item?.zones).length;
}

export function preferredStoredValue(initialValue, storedValue, preferInitial = false) {
  return preferInitial ? initialValue : storedValue;
}
