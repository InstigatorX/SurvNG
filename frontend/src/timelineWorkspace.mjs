export const TIMELINE_PLAYBACK_RATES = Object.freeze([0.5, 1, 2, 4]);

export function normalizedTimelinePlaybackRate(value) {
  const numeric = Number(value);
  return TIMELINE_PLAYBACK_RATES.includes(numeric) ? numeric : 1;
}

export function filteredTimelineCameras(cameras, query) {
  const normalizedQuery = String(query || "").trim().toLocaleLowerCase();
  if (!normalizedQuery) return Array.isArray(cameras) ? cameras : [];
  return (Array.isArray(cameras) ? cameras : []).filter((camera) => {
    const searchable = `${camera?.name || ""} ${camera?.id || ""}`.toLocaleLowerCase();
    return searchable.includes(normalizedQuery);
  });
}

export function timelineStageCameras(cameras, primaryCameraId, limit = 7) {
  const available = Array.isArray(cameras) ? cameras : [];
  const primary = available.find((camera) => camera?.id === primaryCameraId) || available[0];
  if (!primary) return [];
  return [primary, ...available.filter((camera) => camera?.id !== primary.id)].slice(0, Math.max(1, limit));
}

export function expectedTimelineCameras(cameras, routes, activeCameraId, limit = 6) {
  const available = Array.isArray(cameras) ? cameras : [];
  const availableById = new Map(available.map((camera) => [String(camera?.id || ""), camera]));
  const selectedId = String(activeCameraId || "");
  const seen = new Set();
  const expected = [];
  for (const route of Array.isArray(routes) ? routes : []) {
    if (!route || route.enabled === false) continue;
    const fromCamera = String(route.from_camera || "");
    const toCamera = String(route.to_camera || "");
    let expectedId = "";
    if (fromCamera === selectedId) expectedId = toCamera;
    else if (route.bidirectional && toCamera === selectedId) expectedId = fromCamera;
    if (!expectedId || expectedId === selectedId || seen.has(expectedId) || !availableById.has(expectedId)) continue;
    seen.add(expectedId);
    expected.push(availableById.get(expectedId));
    if (expected.length >= Math.max(1, Number(limit) || 6)) break;
  }
  return expected;
}

export function timelineCompanionGrid(count) {
  const available = Math.max(0, Math.min(6, Math.trunc(Number(count) || 0)));
  if (available <= 1) return { columns: 1, rows: 1 };
  if (available <= 3) return { columns: 1, rows: available };
  if (available === 4) return { columns: 2, rows: 2 };
  return { columns: 2, rows: 3 };
}

export function timelineStagePage(cameras, page = 0, pageSize = 7) {
  const available = Array.isArray(cameras) ? cameras : [];
  const size = Math.max(1, Number(pageSize) || 7);
  const pages = Math.max(1, Math.ceil(available.length / size));
  const normalizedPage = Math.min(pages - 1, Math.max(0, Number(page) || 0));
  return { cameras: available.slice(normalizedPage * size, (normalizedPage + 1) * size), page: normalizedPage, pages };
}

export function timelineEventMatchesFilter(event, filter) {
  if (filter === "all") return true;
  if (filter === "object") return Boolean(event?.has_objects);
  if (filter === "motion") return !event?.has_objects;
  const labels = (Array.isArray(event?.labels) ? event.labels : []).map((label) => String(label).toLocaleLowerCase());
  if (filter === "people") return labels.some((label) => label === "person" || label === "people");
  if (filter === "vehicles") return labels.some((label) => ["vehicle", "car", "truck", "bus", "motorcycle", "bicycle"].includes(label));
  return true;
}

export function timelineEvidenceWindow(events, playhead, limit = 12) {
  const available = (Array.isArray(events) ? events : []).filter((event) => Number.isFinite(event?.incident_epoch));
  if (!available.length) return [];
  const target = Number.isFinite(playhead) ? playhead : available[0].incident_epoch;
  return [...available]
    .sort((left, right) => Math.abs(left.incident_epoch - target) - Math.abs(right.incident_epoch - target))
    .slice(0, Math.max(1, limit))
    .sort((left, right) => left.incident_epoch - right.incident_epoch);
}

export function timelineViewport(startEpoch, endEpoch, anchorEpoch, windowHours) {
  const start = Number(startEpoch);
  const end = Number(endEpoch);
  const duration = Math.max(1, end - start);
  const hours = Number(windowHours);
  if (!Number.isFinite(hours) || hours >= 24 || hours * 3600 >= duration) {
    return { startEpoch: start, endEpoch: end };
  }
  const span = Math.max(1, hours * 3600);
  const anchor = Math.max(start, Math.min(end, Number(anchorEpoch) || start));
  const viewportStart = Math.max(start, Math.min(end - span, anchor - span / 2));
  return { startEpoch: viewportStart, endEpoch: viewportStart + span };
}

export function timelinePanViewport(dayStartEpoch, dayEndEpoch, viewport, deltaSeconds) {
  const start = Number(dayStartEpoch);
  const end = Number(dayEndEpoch);
  const currentStart = Number(viewport?.startEpoch);
  const currentEnd = Number(viewport?.endEpoch);
  const span = Math.max(1, currentEnd - currentStart);
  const maximumStart = Math.max(start, end - span);
  const nextStart = Math.max(start, Math.min(maximumStart, currentStart + (Number(deltaSeconds) || 0)));
  return { startEpoch: nextStart, endEpoch: nextStart + span };
}

export function timelineViewportPage(startEpoch, endEpoch, viewport, direction) {
  const currentStart = Number(viewport?.startEpoch);
  const currentEnd = Number(viewport?.endEpoch);
  const span = Math.max(1, currentEnd - currentStart);
  return timelinePanViewport(startEpoch, endEpoch, viewport, direction < 0 ? -span / 2 : span / 2);
}

export function timelinePlayheadInComfortZone(viewport, playhead, edgeRatio = 0.2) {
  const start = Number(viewport?.startEpoch);
  const end = Number(viewport?.endEpoch);
  const epoch = Number(playhead);
  if (![start, end, epoch].every(Number.isFinite) || end <= start) return false;
  const margin = (end - start) * Math.max(0, Math.min(0.45, Number(edgeRatio) || 0));
  return epoch >= start + margin && epoch <= end - margin;
}

export function resolveTimelineHeroCameraId(cameras, requestedId) {
  const available = Array.isArray(cameras) ? cameras : [];
  const requested = String(requestedId || "");
  if (requested && requested !== "all" && available.some((camera) => camera?.id === requested)) return requested;
  return available[0]?.id || "";
}

export function timelineTickIntervalSeconds(windowHours) {
  const hours = Number(windowHours);
  if (hours >= 24) return 3600;
  if (hours >= 8) return 30 * 60;
  if (hours >= 4) return 15 * 60;
  return 5 * 60;
}

export function parseTimelineView(search, today = "") {
  const params = search instanceof URLSearchParams ? search : new URLSearchParams(search || "");
  const rawDate = params.get("date") || today;
  const date = /^\d{4}-\d{2}-\d{2}$/.test(rawDate) && (!today || rawDate <= today) ? rawDate : today;
  const at = Number(params.get("at"));
  const rawEvent = params.get("event");
  return {
    cameraId: params.get("camera") || "all",
    date,
    source: ["main", "live"].includes(params.get("source")) ? params.get("source") : null,
    at: Number.isFinite(at) && at > 0 ? at : null,
    eventFilter: ["all", "object", "motion", "people", "vehicles"].includes(params.get("filter")) ? params.get("filter") : "all",
    eventId: rawEvent && Number.isFinite(Number(rawEvent)) ? Number(rawEvent) : rawEvent || null,
    inspector: ["details", "ai", "related"].includes(params.get("inspector")) ? params.get("inspector") : "details",
    windowHours: [1, 2, 4, 8, 12, 24].includes(Number(params.get("window"))) ? Number(params.get("window")) : 1,
    lanes: { object: params.get("objects") !== "0", motion: params.get("motion") !== "0" },
    thumbnails: params.get("thumbs") !== "0",
    speed: normalizedTimelinePlaybackRate(params.get("speed")),
  };
}
