// Admin Overview model: turns runtime health, detector, retention, telemetry
// and System review state into the "Needs attention" queue and the camera
// table. Pure functions so ordering and wording are unit tested.

const SEVERITY_ORDER = { critical: 0, warning: 1, info: 2 };

function finite(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatOverviewBytes(bytes) {
  if (!finite(bytes) || bytes < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 || unit === 0 ? Math.round(value) : value.toFixed(value >= 10 ? 1 : 2)} ${units[unit]}`;
}

export function formatAgeShort(seconds) {
  if (!finite(seconds) || seconds < 0) return "";
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} s`;
  if (seconds < 90 * 60) return `${Math.round(seconds / 60)} min`;
  if (seconds < 36 * 3600) return `${Math.round(seconds / 3600)} h`;
  return `${Math.round(seconds / 86400)} d`;
}

function cameraOffline(runtime) {
  if (!runtime) return false;
  return runtime.connected === false
    || runtime.capture_connectivity === false
    || ["offline", "disconnected", "error"].includes(runtime.capture_connectivity);
}

/** Pending System review suggestions in the latest briefing. */
export function pendingReviewSuggestions(review) {
  const recommendations = review?.briefing?.result?.recommendations;
  if (!Array.isArray(recommendations)) return 0;
  return recommendations.filter((item) => item && !item.dismissed && !item.applied).length;
}

/**
 * Items that need an operator decision, most severe first.
 * Each action id is resolved by the page: open-camera, diagnostics, storage,
 * detection, review, logs, health.
 */
export function overviewAttention({ health, detector = null, retention = null, reviewSuggestions = 0, interruptions = null } = {}) {
  const items = [];
  if (health && !health.cameraDataKnown) {
    items.push({
      id: "camera-status",
      severity: "warning",
      title: "Camera status is unavailable",
      detail: "SurvNG could not confirm which cameras are recording. Check that the service is running.",
      actions: [{ id: "health", label: "Open Health", primary: true }],
    });
  }
  for (const row of health?.cameraDataKnown ? health.rows : []) {
    if (row.state !== "attention" && row.state !== "unavailable") continue;
    const runtime = row.runtime;
    const offline = cameraOffline(runtime);
    const age = formatAgeShort(runtime?.last_frame_age_seconds);
    const reconnects = Number(runtime?.capture_reconnects || 0);
    const parts = [];
    if (row.state === "unavailable") parts.push("The camera process is not reporting status.");
    else if (row.missing.length) parts.push(`Missing ${row.missing.join(" and ")}.`);
    if (offline && age) parts.push(`Last frame ${age} ago.`);
    if (reconnects) parts.push(`${reconnects.toLocaleString()} reconnect${reconnects === 1 ? "" : "s"} since restart.`);
    items.push({
      id: `camera-${row.id}`,
      severity: "critical",
      cameraId: row.id,
      title: `${row.name} is not recording`,
      detail: parts.join(" "),
      actions: [
        { id: "diagnostics", label: "Run diagnostics", cameraId: row.id },
        { id: "open-camera", label: "Open camera", primary: true, cameraId: row.id },
      ],
    });
  }
  const storage = health?.storage;
  if (storage && (storage.state === "critical" || storage.state === "warning")) {
    const free = formatOverviewBytes(storage.free_bytes);
    const percent = finite(storage.freePercent) ? ` (${storage.freePercent.toFixed(1)}%)` : "";
    const threshold = storage.state === "critical" ? storage.emergencyPercent : storage.warningPercent;
    items.push({
      id: "storage",
      severity: storage.state === "critical" ? "critical" : "warning",
      title: storage.state === "critical" ? "Storage is almost full" : "Storage is below the cleanup threshold",
      detail: `${free} free${percent}. Retention removes the oldest recordings below ${threshold}% free.`,
      actions: [{ id: "storage", label: "Adjust retention", primary: true }],
    });
  }
  if (retention?.state === "error") {
    items.push({
      id: "retention",
      severity: "warning",
      title: "Retention cleanup failed",
      detail: retention.error ? String(retention.error) : "The last cleanup run stopped with an error.",
      actions: [{ id: "storage", label: "Open Storage & Retention", primary: true }],
    });
  }
  if (detector && detector.enabled !== false && !detector.loaded_backend) {
    items.push({
      id: "detector",
      severity: "critical",
      title: "Object detection is not running",
      detail: detector.error ? String(detector.error) : "Detection is enabled but no model backend is loaded, so motion is not checked for objects.",
      actions: [{ id: "detection", label: "Open Detection", primary: true }],
    });
  }
  const unexpected = Number(interruptions?.unexpected || 0);
  if (unexpected > 0) {
    items.push({
      id: "restarts",
      severity: "warning",
      title: `SurvNG restarted unexpectedly ${unexpected === 1 ? "once" : `${unexpected} times`} in the last 24 h`,
      detail: "Recording and detection stop while the service restarts. The logs show what happened before each restart.",
      actions: [{ id: "logs", label: "View logs", primary: true }],
    });
  }
  const suggestions = Number(reviewSuggestions || 0);
  if (suggestions > 0) {
    items.push({
      id: "review",
      severity: "info",
      title: `System review has ${suggestions} suggestion${suggestions === 1 ? "" : "s"}`,
      detail: "Review the proposed setting changes and apply the ones you want.",
      actions: [{ id: "review", label: "Review suggestions", primary: true }],
    });
  }
  return items
    .map((item, index) => ({ item, index }))
    .sort((left, right) => (SEVERITY_ORDER[left.item.severity] - SEVERITY_ORDER[right.item.severity]) || left.index - right.index)
    .map(({ item }) => item);
}

/** One uptime slot per hour: tone for the bar and a readable title. */
export function uptimeSlots(slots = []) {
  return (Array.isArray(slots) ? slots : []).map((slot) => {
    const value = slot?.availability_percent;
    const tone = !finite(value) ? "none" : value >= 99 ? "good" : value >= 50 ? "degraded" : "offline";
    const label = !finite(value) ? "No data" : `${value.toFixed(value === 100 ? 0 : 1)}% available`;
    return { startedAt: slot?.started_at || "", tone, label };
  });
}

export function uptimePercent(slots = []) {
  const known = (Array.isArray(slots) ? slots : []).map((slot) => slot?.availability_percent).filter(finite);
  if (!known.length) return null;
  return known.reduce((sum, value) => sum + value, 0) / known.length;
}

function streamSummary(runtime) {
  const dimensions = runtime?.stream_dimensions?.main || runtime?.stream_dimensions?.live;
  const fps = Number(runtime?.capture_stats?.main?.fps || runtime?.capture_stats?.live?.fps || 0);
  const size = finite(dimensions?.width) && finite(dimensions?.height) && dimensions.width > 0
    ? `${dimensions.width}×${dimensions.height}` : "";
  const rate = fps > 0 ? `${fps >= 10 ? Math.round(fps) : fps.toFixed(1)} fps` : "";
  return [size, rate].filter(Boolean).join(" · ") || "—";
}

function cameraStatus(row) {
  if (row.state === "healthy") return { label: "Recording", tone: "good" };
  if (row.state === "paused") return { label: "Not recording by choice", tone: "muted" };
  if (row.state === "disabled") return { label: "Disabled", tone: "muted" };
  if (row.state === "stale") return { label: "Status unknown", tone: "muted" };
  if (row.state === "unavailable") return { label: "No status", tone: "bad" };
  const age = formatAgeShort(row.runtime?.last_frame_age_seconds);
  if (cameraOffline(row.runtime)) return { label: age ? `Offline ${age}` : "Offline", tone: "bad" };
  return { label: "Not recording", tone: "bad" };
}

/** Rows for the Overview camera table, in configured camera order. */
export function overviewCameraRows({ health, cameras = [], availability = {} } = {}) {
  const healthById = new Map((health?.rows || []).map((row) => [row.id, row]));
  const ordered = cameras.length ? cameras.map((camera) => camera.id) : [...healthById.keys()];
  return ordered.filter((id) => healthById.has(id)).map((id) => {
    const row = healthById.get(id);
    const config = cameras.find((camera) => camera.id === id) || {};
    const slots = availability?.[id];
    return {
      id,
      name: row.name,
      status: cameraStatus(row),
      stream: streamSummary(row.runtime),
      detection: row.runtime?.detection_enabled === false || config.detection_enabled === false ? "Off" : "On",
      homekit: config.homekit?.enabled ? "On" : "Off",
      uptime: uptimeSlots(slots),
      uptimePercent: uptimePercent(slots),
    };
  });
}
