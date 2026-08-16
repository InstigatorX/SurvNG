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
