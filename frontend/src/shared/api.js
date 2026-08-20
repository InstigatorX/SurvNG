import { safeMediaUrl } from "../mediaUrl.mjs";
import { timelineHref as timelineWorkspaceHref, workspaceHref } from "../workspaceNavigation.mjs";

export const APP_BASE_PATH = String(window.__SURVNG_BASE_PATH__ || "").replace(/\/+$/, "");
document.documentElement.dataset.embedded = window.self !== window.top ? "true" : "false";

export function appUrl(path = "/") {
  if (typeof path !== "string" || !path.startsWith("/") || path.startsWith("//")) return path;
  if (APP_BASE_PATH && (path === APP_BASE_PATH || path.startsWith(`${APP_BASE_PATH}/`))) return path;
  return `${APP_BASE_PATH}${path}`;
}

export function mediaUrl(value) {
  return safeMediaUrl(value, APP_BASE_PATH, window.location.origin);
}

export function appPathname() {
  const pathname = window.location.pathname;
  if (!APP_BASE_PATH || (!pathname.startsWith(`${APP_BASE_PATH}/`) && pathname !== APP_BASE_PATH)) return pathname;
  return pathname.slice(APP_BASE_PATH.length) || "/";
}

export function incidentRecordingContext(item) {
  if (!item?.camera_id || !item?.created_at) return null;
  const epoch = new Date(item.created_at).getTime() / 1000;
  if (!Number.isFinite(epoch)) return null;
  return { cameraId: item.camera_id, epoch };
}

export function recordingsHref(context) {
  if (!context?.cameraId || !Number.isFinite(context?.epoch)) return appUrl(workspaceHref("timeline"));
  return appUrl(timelineWorkspaceHref({
    cameraId: context.cameraId,
    epoch: Math.round(context.epoch * 1000) / 1000,
    source: context.source,
  }));
}

export const fetch = (resource, options) => window.fetch(
  typeof resource === "string" ? appUrl(resource) : resource,
  options,
);
