const CANONICAL_PAGES = new Set(["live", "incidents", "timeline", "search", "people", "admin", "exports"]);

const LEGACY_PAGE_MAP = Object.freeze({
  recordings: "timeline",
  faces: "people",
  config: "admin",
});

const PAGE_LABELS = Object.freeze({
  live: "Live",
  incidents: "Incidents",
  timeline: "Timeline",
  search: "Search",
  people: "People",
  admin: "Admin",
  exports: "Export Center",
});

export function canonicalAssistantPage(value) {
  const page = String(value || "").trim().toLowerCase();
  const canonical = LEGACY_PAGE_MAP[page] || page;
  return CANONICAL_PAGES.has(canonical) ? canonical : "live";
}

export function snapshotAssistantContext(value = {}, timeZone = "America/New_York") {
  const filters = Object.freeze(Object.fromEntries(Object.entries(value?.filters || {})
    .filter(([key, item]) => String(key).trim() && item != null)
    .slice(0, 16)
    .map(([key, item]) => [String(key).slice(0, 64), String(item).slice(0, 256)])));
  const incidentId = Number(value?.incident_event_id);
  const epoch = Number(value?.recording_epoch);
  return Object.freeze({
    page: canonicalAssistantPage(value?.page),
    camera_id: String(value?.camera_id || "").trim().slice(0, 128),
    incident_event_id: Number.isInteger(incidentId) && incidentId > 0 ? incidentId : null,
    recording_epoch: Number.isFinite(epoch) && epoch >= 0 ? epoch : null,
    export_id: String(value?.export_id || "").trim().slice(0, 128),
    filters,
    time_zone: String(timeZone || value?.time_zone || "America/New_York").slice(0, 128),
  });
}

function cameraLabel(context) {
  return context?.filters?.camera_name || context?.camera_id || "";
}

export function assistantContextLabel(value = {}, formatter = null) {
  const context = snapshotAssistantContext(value, value?.time_zone);
  const pieces = [PAGE_LABELS[context.page]];
  const camera = cameraLabel(context);
  if (camera) pieces.push(camera);
  if (context.incident_event_id) pieces.push(`Event #${context.incident_event_id}`);
  else if (context.export_id) pieces.push(`Export ${context.export_id}`);
  else if (context.recording_epoch != null) {
    const formatted = typeof formatter === "function" ? formatter(context.recording_epoch) : "Selected time";
    pieces.push(formatted || "Selected time");
  } else if (context.page === "search" && (context.filters.query || context.filters.semantic_query)) pieces.push(`“${context.filters.query || context.filters.semantic_query}”`);
  else if (context.page === "admin" && context.filters.section) pieces.push(context.filters.section.replaceAll("-", " "));
  return pieces.filter(Boolean).join(" · ");
}

export function assistantContextPrompts(value = {}) {
  const context = snapshotAssistantContext(value, value?.time_zone);
  const camera = cameraLabel(context) || "this camera";
  if (context.page === "incidents" && context.incident_event_id) return ["Analyze this incident", "Trace this incident across cameras", "Open this incident in Timeline"];
  if (context.page === "timeline") return ["Summarize activity around this time", `Create a timelapse for ${camera}`, "Find related incidents"];
  if (context.page === "search") return ["Explain these search results", "Refine this search", "Find the strongest matching incident"];
  if (context.page === "people") return ["Summarize recent person activity", "Is everything healthy?", "Find person incidents from the last 24 hours"];
  if (context.page === "admin") return ["Explain these settings", "Is everything healthy?", "What needs attention?"];
  if (context.page === "exports") return ["Explain the active exports", "Create a recording export", "Is export processing healthy?"];
  if (context.page === "live" && context.camera_id) return [`Is ${camera} healthy?`, `Summarize recent activity for ${camera}`, `Create a timelapse for ${camera}`];
  return ["Is everything healthy?", "Find person incidents from the last 24 hours", "Summarize recent activity"];
}
