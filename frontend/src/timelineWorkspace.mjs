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
