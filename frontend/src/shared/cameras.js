import { browserStorage, removeStoredValue } from "../storage.mjs";
import { MEDIA_STORAGE_ROLES, LEGACY_INCIDENT_FILTER_KEYS, LIVE_TRANSPORT_LABELS } from "./constants.js";
import { isMobileViewport } from "./hooks.js";

export function mediaStorageConfigurationError(mediaStorage) {
  const locations = mediaStorage?.locations || [];
  if (!locations.length) return "";
  const enabledRoles = new Set(
    locations
      .filter((location) => location.enabled !== false)
      .flatMap((location) => location.roles || []),
  );
  const missing = MEDIA_STORAGE_ROLES
    .filter(([role]) => !enabledRoles.has(role))
    .map(([, label]) => label);
  return missing.length
    ? `At least one enabled media location must accept: ${missing.join(", ")}.`
    : "";
}

export function clearLegacyIncidentFilterStorage() {
  const storage = browserStorage(window);
  LEGACY_INCIDENT_FILTER_KEYS.forEach((key) => removeStoredValue(storage, key));
}
export function preferredStreamSource() {
  return isMobileViewport() ? "live" : "main";
}

export function sourceLabel(source) {
  return source === "main" ? "Main" : "Sub";
}
export function liveTransportLabel(stage) {
  return LIVE_TRANSPORT_LABELS[stage] || "Connecting";
}
export function slugify(value) {
  return String(value || "camera")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    || "camera";
}

export function uniqueCameraId(cameras, base) {
  const existing = new Set(cameras.map((camera) => camera.id));
  let candidate = slugify(base);
  let index = 2;
  while (existing.has(candidate)) {
    candidate = `${slugify(base)}-${index}`;
    index += 1;
  }
  return candidate;
}

export function streamUrlDefaults(url) {
  try {
    const parsed = new URL(url);
    if (!parsed.hostname) return {};
    return {
      scheme: parsed.protocol.replace(":", "").toLowerCase(),
      host: parsed.hostname,
      port: parsed.port ? Number(parsed.port) : null,
      username: decodeURIComponent(parsed.username || ""),
      password: decodeURIComponent(parsed.password || ""),
      channel: Number(parsed.searchParams.get("channel") || parsed.searchParams.get("chn") || 0),
    };
  } catch {
    return {};
  }
}

export function inferredBackendLabel(camera) {
  const defaults = streamUrlDefaults(camera.stream_url || camera.live_stream_url || "");
  if (defaults.scheme === "rtsp" || defaults.scheme === "rtsps") return "RTSP";
  if (defaults.scheme) return defaults.scheme.toUpperCase();
  return "URL";
}

export function cameraWithDerivedConnection(camera) {
  const defaults = streamUrlDefaults(camera.stream_url || camera.live_stream_url || "");
  if (!defaults.host) return camera;
  return {
    ...camera,
    video_backend: "url",
    onvif: {
      ...camera.onvif,
      host: camera.onvif?.host || defaults.host,
      username: camera.onvif?.username || defaults.username,
      password: camera.onvif?.password || defaults.password,
    },
  };
}

export function camerasWithGeneratedIds(cameras) {
  const used = new Set();
  return (cameras || []).map((camera) => {
    const base = slugify(camera.name || camera.id || "camera") || "camera";
    let id = base;
    let index = 2;
    while (used.has(id)) {
      id = `${base}-${index}`;
      index += 1;
    }
    used.add(id);
    return { ...cameraWithDerivedConnection(camera), id };
  });
}

