export const HIGHLIGHT_FADE_MS = 90_000;
export const HIGHLIGHT_PEAK_OPACITY = 0.72;

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
