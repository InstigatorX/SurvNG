import { liveDensityPage, normalizedLiveDensity } from "./liveWorkspace.mjs";

// All cameras tiles each play one window playlist of this length.
export const ALL_CAMERAS_WINDOW_SECONDS = 15 * 60;
// Today, All cameras starts this far behind the newest recording so tiles have footage ahead
// of them instead of reloading as each new recording file is indexed.
export const ALL_CAMERAS_LIVE_DELAY_SECONDS = 60;

export function allCamerasWindowStart(epoch) {
  const value = Math.max(0, Number(epoch) || 0);
  return Math.floor(value / ALL_CAMERAS_WINDOW_SECONDS) * ALL_CAMERAS_WINDOW_SECONDS;
}

// Cameras with any recording (either stream) in the window around an epoch.
// Decided per window, not per instant, so tiles do not come and go mid-window.
export function allCamerasWithRecording(cameras, ranges, epoch) {
  const start = allCamerasWindowStart(epoch);
  const end = start + ALL_CAMERAS_WINDOW_SECONDS;
  const recorded = new Set();
  for (const range of ranges || []) {
    if (Number(range?.start_epoch) < end && Number(range?.end_epoch) > start) recorded.add(String(range.camera_id || ""));
  }
  return (cameras || []).filter((camera) => recorded.has(camera.id));
}

export const ALL_CAMERAS_LAYOUT_MODES = Object.freeze(["auto", "focus", "custom"]);

// The Custom camera set: stored camera ids that still exist, in camera order.
// Nothing stored (or nothing left) means every camera.
export function allCamerasCustomIds(stored, cameras) {
  let ids = [];
  try {
    const parsed = JSON.parse(stored || "[]");
    if (Array.isArray(parsed)) ids = parsed.map(String);
  } catch {
    ids = [];
  }
  const chosen = (cameras || []).filter((camera) => ids.includes(camera.id)).map((camera) => camera.id);
  return chosen.length ? chosen : (cameras || []).map((camera) => camera.id);
}

/**
 * What the All cameras grid shows, with Live's layout rules: Automatic pages
 * every camera by density, Focus shows every camera around one primary (or
 * the chosen set, when entered from Custom: focusScope "custom"), and Custom
 * shows the chosen set (Focus and Custom at "fit"). Only cameras with recording
 * in the playhead's window get a tile. Without desktop width
 * (layoutsAvailable false), Automatic applies at "fit", as Live's density
 * buttons are hidden there.
 */
export function allCamerasGridSelection({ cameras, ranges, epoch, mode, density, page, customIds, focusCameraId, focusScope = "all", layoutsAvailable = true }) {
  const layoutMode = layoutsAvailable && ALL_CAMERAS_LAYOUT_MODES.includes(mode) ? mode : "auto";
  const all = cameras || [];
  const useCustom = layoutMode === "custom" || (layoutMode === "focus" && focusScope === "custom");
  const chosen = useCustom ? all.filter((camera) => (customIds || []).includes(camera.id)) : all;
  const recorded = allCamerasWithRecording(chosen, ranges, epoch);
  // Live hides its density buttons below desktop width, so there it is "fit".
  const effectiveDensity = layoutMode === "auto" && layoutsAvailable ? normalizedLiveDensity(density) : "fit";
  const paged = liveDensityPage(recorded, effectiveDensity, page);
  const focusId = recorded.some((camera) => camera.id === focusCameraId) ? focusCameraId : recorded[0]?.id || "";
  return {
    layoutMode,
    density: effectiveDensity,
    chosen,
    recorded,
    unrecorded: chosen.filter((camera) => !recorded.includes(camera)),
    shown: paged.cameras,
    page: paged.page,
    pageCount: paged.pageCount,
    focusCameraId: focusId,
  };
}

export function recordingGridBestEpoch(ranges, requestedEpoch, lookbackSeconds = 300) {
  const requested = Number(requestedEpoch);
  if (!Number.isFinite(requested)) return null;
  const windowStart = requested - Math.max(1, Number(lookbackSeconds) || 300);
  const normalized = (ranges || []).map((range) => ({
    cameraId: String(range.camera_id || ""),
    start: Number(range.start_epoch),
    end: Number(range.end_epoch),
  })).filter((range) => (
    range.cameraId && Number.isFinite(range.start) && Number.isFinite(range.end)
    && range.end > range.start && range.end > windowStart && range.start <= requested
  ));
  if (!normalized.length) return null;
  const candidates = new Set([requested]);
  normalized.forEach((range) => {
    candidates.add(Math.max(windowStart, range.start));
    // Stay clear of a lagging camera's indexed edge so playback does not lose
    // that tile immediately after the synchronized grid begins advancing.
    candidates.add(Math.min(requested, Math.max(range.start, range.end - 2)));
  });
  let best = null;
  for (const candidate of candidates) {
    if (!Number.isFinite(candidate) || candidate < windowStart || candidate > requested) continue;
    const cameraCount = new Set(normalized
      .filter((range) => range.start <= candidate && candidate < range.end)
      .map((range) => range.cameraId)).size;
    if (!best || cameraCount > best.cameraCount || (cameraCount === best.cameraCount && candidate > best.epoch)) {
      best = { epoch: candidate, cameraCount };
    }
  }
  return best?.cameraCount ? best.epoch : null;
}
