function sameEventId(left, right) {
  if (left == null || right == null) return false;
  return String(left) === String(right);
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

export function incidentThumbnailPageSize({ width, height, density }) {
  const safeWidth = Math.max(0, Number(width) || 0);
  const safeHeight = Math.max(0, Number(height) || 0);
  if (!safeWidth || !safeHeight) return density === "comfortable" ? 10 : 16;
  const compact = density !== "comfortable";
  const columns = compact ? 2 : 1;
  const gap = compact ? 7 : 9;
  const horizontalPadding = 16;
  const usableWidth = Math.max(1, safeWidth - horizontalPadding);
  const cardWidth = Math.max(1, (usableWidth - gap * (columns - 1)) / columns);
  const cardHeight = cardWidth * 10 / 16 + 2;
  const rows = Math.max(1, Math.floor((safeHeight + gap) / (cardHeight + gap)));
  return rows * columns;
}
