import { appUrl } from "./api.js";

export function eventSnapshotUrl(event) {
  if (event?.snapshot_url) return appUrl(event.snapshot_url);
  const eventId = Number(event?.representative_event_id || event?.id);
  return Number.isFinite(eventId) ? appUrl(`/api/events/${eventId}/snapshot.jpg`) : "";
}

export function eventSnapshotDownloadUrl(event) {
  const snapshotUrl = eventSnapshotUrl(event);
  return snapshotUrl ? `${snapshotUrl}?download=true` : "";
}

export function eventThumbnailUrl(event, width = 720, quality = 82, options = {}) {
  if (event?.snapshot_url) return appUrl(event.snapshot_url);
  const eventId = Number(event?.representative_event_id || event?.id);
  if (!Number.isFinite(eventId)) return "";
  const params = new URLSearchParams({
    width: String(Math.max(160, Math.min(2560, Math.round(Number(width) || 720)))),
    quality: String(Math.max(50, Math.min(95, Math.round(Number(quality) || 82)))),
  });
  if (options?.objectFocus) {
    params.set("object_focus", "true");
    if (options.incidentEligibleOnly) params.set("incident_eligible_only", "true");
    const zoom = Number(options.zoom);
    if (Number.isFinite(zoom)) params.set("zoom", String(zoom));
    const aspectW = Number(options.aspectWidth);
    const aspectH = Number(options.aspectHeight);
    if (Number.isFinite(aspectW) && Number.isFinite(aspectH) && aspectW > 0 && aspectH > 0) {
      params.set("aspect_w", String(aspectW));
      params.set("aspect_h", String(aspectH));
    }
  }
  return appUrl(`/api/events/${eventId}/thumbnail.jpg?${params.toString()}`);
}
export function eventClipUrl(eventId, before = 5, after = 5, source = "main") {
  const params = new URLSearchParams({ before: before.toFixed(3), after: after.toFixed(3), source });
  return appUrl(`/api/events/${eventId}/clip.mp4?${params.toString()}`);
}

export function eventStreamUrl(eventId, before = 5, after = 5, source = "main") {
  const params = new URLSearchParams({ before: before.toFixed(3), after: after.toFixed(3), source });
  return appUrl(`/api/events/${eventId}/stream.m3u8?${params.toString()}`);
}
export function recordingDayUrl(cameraId, startEpoch, endEpoch, source, includeIdentities = true) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    source,
    include_identities: includeIdentities ? "true" : "false",
  });
  return appUrl(`/api/cameras/${cameraId}/recordings/day?${params.toString()}`);
}

export function recordingWindowUrl(cameraId, startEpoch, endEpoch, source) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    source,
  });
  return appUrl(`/api/cameras/${cameraId}/recordings/window?${params.toString()}`);
}

export function recordingUpdatesUrl(cameraId, startEpoch, endEpoch, afterEpoch, source, includeIdentities = true) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    after_epoch: afterEpoch.toFixed(3),
    source,
    include_identities: includeIdentities ? "true" : "false",
  });
  return appUrl(`/api/cameras/${cameraId}/recordings/updates?${params.toString()}`);
}

export function recordingDayHlsUrl(cameraId, startEpoch, endEpoch, source) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    source,
  });
  return appUrl(`/api/cameras/${cameraId}/recordings/day.m3u8?${params.toString()}`);
}

export function recordingMobileWindowUrl(cameraId, epoch, source) {
  const params = new URLSearchParams({ epoch: epoch.toFixed(3), source });
  return appUrl(`/api/cameras/${cameraId}/recordings/mobile-window.mp4?${params.toString()}`);
}

export function recordingGridDayUrl(startEpoch, endEpoch, source, includeIdentities = true) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    source,
    include_identities: includeIdentities ? "true" : "false",
  });
  return appUrl(`/api/recordings/grid/day?${params.toString()}`);
}

export function recordingGridUpdatesUrl(startEpoch, endEpoch, afterEpoch, source, includeIdentities = true) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    after_epoch: afterEpoch.toFixed(3),
    source,
    include_identities: includeIdentities ? "true" : "false",
  });
  return appUrl(`/api/recordings/grid/updates?${params.toString()}`);
}

export function recordingPreviewUrl(cameraId, epoch, source, options = {}) {
  const params = new URLSearchParams({
    epoch: epoch.toFixed(3),
    source,
  });
  const width = Number(options.width);
  if (Number.isFinite(width)) {
    params.set("width", String(Math.max(320, Math.min(1920, Math.round(width)))));
  }
  if (options.exact) params.set("exact", "true");
  return appUrl(`/api/cameras/${cameraId}/recordings/preview.jpg?${params.toString()}`);
}
