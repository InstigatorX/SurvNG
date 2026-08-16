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

export function uniformLiveGridLayout(cameras, width, height, gap = 4, aspect = 16 / 9) {
  const items = [...(cameras || [])];
  const availableWidth = Number(width);
  const availableHeight = Number(height);
  const tileAspect = Number(aspect);
  if (!items.length || !(availableWidth > 0) || !(availableHeight > 0) || !(tileAspect > 0)) return [];

  let best = null;
  for (let columns = 1; columns <= items.length; columns += 1) {
    const rows = Math.ceil(items.length / columns);
    const widthPerTile = (availableWidth - gap * (columns - 1)) / columns;
    const heightPerTile = (availableHeight - gap * (rows - 1)) / rows;
    if (!(widthPerTile > 0) || !(heightPerTile > 0)) continue;
    const tileWidth = Math.min(widthPerTile, heightPerTile * tileAspect);
    const tileHeight = tileWidth / tileAspect;
    const area = tileWidth * tileHeight;
    const emptyCells = columns * rows - items.length;
    if (!best || area > best.area + 0.001 || (Math.abs(area - best.area) <= 0.001 && emptyCells < best.emptyCells)) {
      best = { columns, rows, tileWidth, tileHeight, area, emptyCells };
    }
  }
  if (!best) return [];

  const gridWidth = best.columns * best.tileWidth + gap * (best.columns - 1);
  const offsetX = Math.max(0, (availableWidth - gridWidth) / 2);
  return items.map((camera, index) => ({
    camera,
    x: offsetX + (index % best.columns) * (best.tileWidth + gap),
    y: Math.floor(index / best.columns) * (best.tileHeight + gap),
    width: best.tileWidth,
    height: best.tileHeight,
  }));
}

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
