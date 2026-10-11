export const HIGHLIGHT_FADE_MS = 90_000;
export const HIGHLIGHT_PEAK_OPACITY = 0.72;
export const PROPERTY_ZOOM_MIN = 1;
export const PROPERTY_ZOOM_MAX = 8;
export const PROPERTY_ZOOM_FACTOR = 1.25;

export function propertyViewAtRest() {
  return { scale: 1, x: 0, y: 0 };
}

export function clampPropertyView(view, frame) {
  const scale = Math.max(PROPERTY_ZOOM_MIN, Math.min(PROPERTY_ZOOM_MAX, Number(view?.scale) || 1));
  if (scale <= 1) return propertyViewAtRest();
  const limitX = (Number(frame?.width) || 0) * (scale - 1) / 2;
  const limitY = (Number(frame?.height) || 0) * (scale - 1) / 2;
  return {
    scale,
    x: Math.max(-limitX, Math.min(limitX, Number(view?.x) || 0)),
    y: Math.max(-limitY, Math.min(limitY, Number(view?.y) || 0)),
  };
}

export function zoomPropertyView(view, frame, anchor, factor) {
  const currentScale = Number(view?.scale) > 0 ? Number(view.scale) : 1;
  const nextScale = Math.max(PROPERTY_ZOOM_MIN, Math.min(PROPERTY_ZOOM_MAX, currentScale * factor));
  if (nextScale <= 1) return propertyViewAtRest();
  const scaleRatio = nextScale / currentScale;
  const anchorX = Number(anchor?.x) || 0;
  const anchorY = Number(anchor?.y) || 0;
  return clampPropertyView({
    scale: nextScale,
    x: anchorX - (anchorX - (Number(view?.x) || 0)) * scaleRatio,
    y: anchorY - (anchorY - (Number(view?.y) || 0)) * scaleRatio,
  }, frame);
}

export function panPropertyView(view, frame, dx, dy) {
  return clampPropertyView({
    scale: view?.scale,
    x: (Number(view?.x) || 0) + dx,
    y: (Number(view?.y) || 0) + dy,
  }, frame);
}

const PERSON = new Set(["person", "people", "human", "pedestrian"]);
const VEHICLE = new Set(["car", "vehicle", "truck", "bus", "suv", "van", "motorcycle", "motorbike"]);
const ANIMAL = new Set(["dog", "cat", "animal", "bird", "deer", "bear"]);
const BICYCLE = new Set(["bicycle", "bike", "cyclist"]);
const PACKAGE = new Set(["package", "box", "parcel"]);

export function detectionIcon(label) {
  const value = String(label || "").trim().toLowerCase();
  if (PERSON.has(value)) return "person";
  if (VEHICLE.has(value)) return "vehicle";
  if (ANIMAL.has(value)) return "animal";
  if (BICYCLE.has(value)) return "bicycle";
  if (PACKAGE.has(value)) return "package";
  return "generic";
}

export function highlightOpacity(detectedAtMs, nowMs, fadeMs = HIGHLIGHT_FADE_MS) {
  if (!Number.isFinite(detectedAtMs) || !Number.isFinite(nowMs)) return 0;
  const age = nowMs - detectedAtMs;
  if (age < 0) return HIGHLIGHT_PEAK_OPACITY;
  if (age >= fadeMs) return 0;
  return HIGHLIGHT_PEAK_OPACITY * (1 - age / fadeMs);
}

export function zoneActivityKey(cameraId, zoneName) {
  return `${cameraId}\u0000${zoneName}`;
}

export function applyDetection(highlights, event, nowMs) {
  const detectedAt = Date.parse(event?.timestamp);
  if (!event?.camera_id || !Number.isFinite(detectedAt) || highlightOpacity(detectedAt, nowMs) <= 0) {
    return highlights;
  }
  const zones = Array.isArray(event.zones) ? event.zones.filter(Boolean) : [];
  const classes = Array.isArray(event.classes) ? event.classes.filter(Boolean).map(String) : [];
  if (!zones.length) return highlights;
  let next = highlights;
  for (const zone of zones) {
    const key = zoneActivityKey(event.camera_id, zone);
    const previous = next[key];
    if (previous && previous.detectedAt > detectedAt) continue;
    if (next === highlights) next = { ...highlights };
    next[key] = { detectedAt, classes };
  }
  return next;
}

export function pruneHighlights(highlights, nowMs) {
  const next = {};
  for (const [key, value] of Object.entries(highlights || {})) {
    if (highlightOpacity(value.detectedAt, nowMs) > 0) next[key] = value;
  }
  return next;
}

export function configuredZoneRows(appConfig) {
  return (appConfig?.cameras || []).flatMap((camera) => (
    (camera.zones || []).filter((zone) => zone?.name).map((zone) => ({
      cameraId: camera.id,
      cameraName: camera.name || camera.id,
      zoneName: zone.name,
      color: zone.color || "#22c55e",
      notificationsEnabled: zone.notifications_enabled !== false,
    }))
  ));
}

export function overlayPresentation(overlay, zones) {
  const match = (zones || []).find((zone) => zone.cameraId === overlay.camera_id && zone.zoneName === overlay.zone_name);
  return {
    stale: !match,
    color: match?.color || "#94a3b8",
    label: match ? `${match.cameraName} · ${match.zoneName}` : `${overlay.camera_id} · ${overlay.zone_name}`,
  };
}

export function offsetPoint(x, y, headingDeg, rangeM, widthM, heightM) {
  const rad = (headingDeg * Math.PI) / 180;
  return {
    x: x + Math.sin(rad) * (rangeM / widthM),
    y: y - Math.cos(rad) * (rangeM / heightM),
  };
}

export function conePolygon(placement, widthM, heightM, steps = 12) {
  const points = [{ x: placement.x, y: placement.y }];
  const half = Number(placement.hfov_deg) / 2;
  const span = Number(placement.hfov_deg);
  for (let index = 0; index <= steps; index += 1) {
    const heading = Number(placement.heading_deg) - half + (span * index) / steps;
    points.push(offsetPoint(placement.x, placement.y, heading, placement.range_m, widthM, heightM));
  }
  return points;
}

export function aimFromPointer(origin, point, widthM, heightM) {
  const east = (point.x - origin.x) * widthM;
  const north = -(point.y - origin.y) * heightM;
  const heading = (Math.atan2(east, north) * 180) / Math.PI;
  return {
    heading_deg: (heading + 360) % 360,
    range_m: Math.min(500, Math.max(1, Math.hypot(east, north))),
  };
}

export function rotationFromPointer(origin, point) {
  const heading = (Math.atan2(point.x - origin.x, -(point.y - origin.y)) * 180) / Math.PI;
  if (heading > 180) return heading - 360;
  if (heading <= -180) return 180;
  return heading;
}

export function clampUnit(value) {
  return Math.min(1, Math.max(0, value));
}
