function sameEventId(left, right) {
  if (left == null || right == null) return false;
  return String(left) === String(right);
}

export function incidentDetectionFrameSize(event) {
  const detected = (Array.isArray(event?.objects) ? event.objects : []).find((object) => (
    Number(object?.detection_frame_width) > 0
    && Number(object?.detection_frame_height) > 0
  ));
  if (detected) return {
    width: Number(detected.detection_frame_width),
    height: Number(detected.detection_frame_height),
  };
  return incidentTrackingFrameSize(event, false);
}

export function incidentTrackingFrameSize(event, fallbackToDetection = true) {
  const trackedWidth = Number(event?.object_tracking?.frame_width);
  const trackedHeight = Number(event?.object_tracking?.frame_height);
  if (trackedWidth > 0 && trackedHeight > 0) {
    return { width: trackedWidth, height: trackedHeight };
  }
  return fallbackToDetection ? incidentDetectionFrameSize(event) : null;
}

function incidentRecencyEpoch(incident) {
  for (const value of [
    incident?.last_epoch,
    incident?.end_at,
    incident?.start_epoch,
    incident?.created_epoch,
    incident?.created_at,
  ]) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
    const parsed = new Date(value || 0).getTime() / 1000;
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  return 0;
}

export function incidentsNewestFirst(incidents) {
  if (!Array.isArray(incidents)) return [];
  return incidents
    .map((incident, index) => ({ incident, index, epoch: incidentRecencyEpoch(incident) }))
    .sort((left, right) => right.epoch - left.epoch || left.index - right.index)
    .map(({ incident }) => incident);
}

export function retainFocusedIncident(incidents, incidentId, retained = null) {
  if (incidentId == null) return null;
  const selected = Array.isArray(incidents)
    ? incidents.find((incident) => sameEventId(incident?.id, incidentId))
    : null;
  if (selected) return selected;
  return sameEventId(retained?.id, incidentId) ? retained : null;
}

export function createIncidentPageCache(loader) {
  let entries = new Map();
  return {
    load(key) {
      const existing = entries.get(key);
      if (existing) return existing.pending;
      const entry = { pending: null, value: undefined };
      const pending = Promise.resolve()
        .then(() => loader(key))
        .then((value) => {
          entry.value = value;
          return value;
        });
      entry.pending = pending;
      entries.set(key, entry);
      pending.catch(() => {
        if (entries.get(key) === entry) entries.delete(key);
      });
      return pending;
    },
    peek(key) {
      return entries.get(key)?.value;
    },
    retain(keys) {
      const retained = new Set(keys.filter(Boolean));
      for (const key of entries.keys()) {
        if (!retained.has(key)) entries.delete(key);
      }
    },
    invalidate(key) {
      if (key) entries.delete(key);
    },
    clear() {
      entries = new Map();
    },
    size() {
      return entries.size;
    },
  };
}

export function incidentDetailQuery(incident) {
  const eventIds = (incident?.events || [])
    .map((event) => Number(event?.id))
    .filter((eventId) => Number.isInteger(eventId) && eventId > 0);
  if (!eventIds.length) return "";
  return new URLSearchParams({
    event_ids: [...new Set(eventIds)].join(","),
    gap_seconds: "45",
  }).toString();
}

export function incidentMosaicEvents(incident) {
  if (!Array.isArray(incident?.events)) return [];
  return incident.events
    .map((event, index) => {
      const parsedEpoch = new Date(event?.created_at || 0).getTime();
      return { event, index, epoch: Number.isFinite(parsedEpoch) ? parsedEpoch : Number.POSITIVE_INFINITY };
    })
    .sort((left, right) => left.epoch - right.epoch || left.index - right.index)
    .map(({ event }) => event);
}

export function incidentMosaicPage(events, page, pageSize = 6) {
  const items = Array.isArray(events) ? events : [];
  const size = Math.max(1, Math.floor(Number(pageSize) || 6));
  const pageCount = Math.max(1, Math.ceil(items.length / size));
  const pageIndex = Math.max(0, Math.min(pageCount - 1, Math.floor(Number(page) || 0)));
  return {
    items: items.slice(pageIndex * size, (pageIndex + 1) * size),
    page: pageIndex,
    pageCount,
  };
}

export function incidentIndexForEvent(incidents, event) {
  if (!Array.isArray(incidents) || !event) return -1;
  return incidents.findIndex((incident) => (
    sameEventId(incident?.id, event.id)
    || (incident?.events || []).some((child) => sameEventId(child?.id, event.id))
  ));
}

export function adjacentIncident(incidents, event, direction) {
  if (!Array.isArray(incidents) || incidents.length < 2) return null;
  const currentIndex = incidentIndexForEvent(incidents, event);
  if (currentIndex < 0) return null;
  const step = direction < 0 ? -1 : 1;
  return incidents[(currentIndex + step + incidents.length) % incidents.length] || null;
}

export function showIncidentCardAnnotations(expanded, thumbnailAnnotations) {
  return !expanded && Boolean(thumbnailAnnotations);
}

export function incidentProgressiveImageWidth(renderedWidth, devicePixelRatio = 1) {
  const width = Math.max(0, Number(renderedWidth) || 0);
  const ratio = Math.max(1, Math.min(4, Number(devicePixelRatio) || 1));
  if (!width) return 1280;
  const requiredPixels = width * ratio;
  if (requiredPixels <= 1280) return 1280;
  if (requiredPixels <= 1920) return 1920;
  return 2560;
}

export function incidentTriggerLabel(incident) {
  const source = String(incident?.trigger_source || "camera").toLowerCase();
  return ["ema", "adaptive", "visual_backup", "adaptive/visual_backup"].includes(source)
    ? "EMA"
    : "Camera";
}

export function incidentObjectIconName(label) {
  const normalized = String(label || "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
  if (["person", "human"].includes(normalized)) return "person";
  if (normalized === "face") return "face";
  if (["car", "vehicle"].includes(normalized)) return "car";
  if (normalized === "truck") return "truck";
  if (normalized === "bus") return "bus";
  if (["motorcycle", "motorbike", "bicycle", "bike"].includes(normalized)) return "bike";
  if (normalized === "cat") return "cat";
  if (normalized === "dog") return "dog";
  if (["robot_lawnmower", "robot_mower", "lawnmower", "mower"].includes(normalized)) return "mower";
  return "object";
}

export function incidentThumbnailPageSize({ width, height, density, columns: requestedColumns, gap: requestedGap, horizontalPadding: requestedPadding }) {
  const safeWidth = Math.max(0, Number(width) || 0);
  const safeHeight = Math.max(0, Number(height) || 0);
  if (!safeWidth || !safeHeight) return density === "comfortable" ? 10 : 16;
  const compact = density !== "comfortable";
  const columns = Math.max(1, Math.floor(Number(requestedColumns) || (compact ? 2 : 1)));
  const gap = Math.max(0, Number.isFinite(Number(requestedGap)) ? Number(requestedGap) : compact ? 7 : 9);
  const horizontalPadding = Math.max(0, Number.isFinite(Number(requestedPadding)) ? Number(requestedPadding) : 16);
  const usableWidth = Math.max(1, safeWidth - horizontalPadding);
  const cardWidth = Math.max(1, (usableWidth - gap * (columns - 1)) / columns);
  const cardHeight = cardWidth * 10 / 16 + 2;
  const rows = Math.max(1, Math.floor((safeHeight + gap) / (cardHeight + gap)));
  return rows * columns;
}
