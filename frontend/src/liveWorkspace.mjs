export function focusedLiveCameraId(cameras, storedCameraId) {
  const ids = (cameras || []).map((camera) => String(camera?.id || "")).filter(Boolean);
  const stored = String(storedCameraId || "");
  return ids.includes(stored) ? stored : ids[0] || "";
}

export function orderedLiveCamerasForFocus(cameras, focusedCameraId, mobile) {
  const items = [...(cameras || [])];
  if (!mobile || !focusedCameraId) return items;
  return [
    ...items.filter((camera) => String(camera?.id) === String(focusedCameraId)),
    ...items.filter((camera) => String(camera?.id) !== String(focusedCameraId)),
  ];
}

export const LIVE_DENSITY_OPTIONS = Object.freeze(["fit", "4", "6", "9", "16", "25"]);

export function normalizedLiveDensity(value) {
  const normalized = String(value || "fit").toLowerCase();
  return LIVE_DENSITY_OPTIONS.includes(normalized) ? normalized : "fit";
}

export function liveDensityPage(cameras, density, page = 0) {
  const items = [...(cameras || [])];
  const normalized = normalizedLiveDensity(density);
  if (normalized === "fit") return { cameras: items, page: 0, pageCount: 1 };
  const limit = Number(normalized);
  const pageCount = Math.max(1, Math.ceil(items.length / limit));
  const currentPage = Math.max(0, Math.min(pageCount - 1, Math.floor(Number(page) || 0)));
  return {
    cameras: items.slice(currentPage * limit, currentPage * limit + limit),
    page: currentPage,
    pageCount,
  };
}

export function liveActivityQuickFilter(eventType, objectFilter) {
  const type = String(eventType || "all");
  const object = String(objectFilter || "all").toLowerCase();
  if (type === "object" && object === "all") return "object";
  if (type === "motion" && object === "all") return "motion";
  return "custom";
}

export function liveActivityQuickSelection(mode) {
  if (mode === "motion") return { eventType: "motion", objectFilter: "all" };
  return { eventType: "object", objectFilter: "all" };
}

export function liveActivityEventId(incident) {
  const value = Number(
    incident?.representative_event_id
      ?? incident?.events?.[0]?.id
      ?? incident?.id,
  );
  return Number.isFinite(value) && value > 0 ? value : null;
}

export function liveActivityIncidentHref(incident) {
  const eventId = liveActivityEventId(incident);
  return eventId ? `/incidents?event_ids=${encodeURIComponent(eventId)}` : "/incidents";
}
