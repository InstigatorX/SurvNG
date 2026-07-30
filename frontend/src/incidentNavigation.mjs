function sameEventId(left, right) {
  if (left == null || right == null) return false;
  return String(left) === String(right);
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
    clear() {
      entries = new Map();
    },
    size() {
      return entries.size;
    },
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
