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
