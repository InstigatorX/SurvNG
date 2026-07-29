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
