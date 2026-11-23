import { PREFER_NATIVE_HLS } from "./shared/constants.js";

/**
 * Chromium-based browsers (desktop Chrome/Edge, Android Chrome and WebView)
 * now answer canPlayType for HLS too. iOS browsers are WebKit underneath and
 * identify as CriOS/EdgiOS/FxiOS, not Chrome/, so they stay on native HLS.
 */
export function isChromiumBrowser(nav = typeof navigator === "undefined" ? null : navigator) {
  if (!nav) return false;
  if (nav.userAgentData?.brands?.some(({ brand }) => /Chromium/i.test(brand))) return true;
  return /\b(?:Chrome|Chromium)\/\d/.test(nav.userAgent || "");
}

/**
 * Native HLS is Safari's player. Chromium's built-in HLS player fails seeking
 * backwards across the EXT-X-DISCONTINUITY between recording files
 * (DEMUXER_ERROR_COULD_NOT_PARSE), which pushed Chrome onto the per-file
 * fallback; Chromium keeps using Shaka's MSE path instead.
 */
export function supportsNativeRecordingHls(nav = typeof navigator === "undefined" ? null : navigator) {
  return typeof document !== "undefined"
    && !isChromiumBrowser(nav)
    && Boolean(document.createElement("video").canPlayType("application/vnd.apple.mpegurl"));
}

export const RECORDING_TRANSPORT_STORAGE_KEY = "survng.recordingTransport";
const RECORDING_TRANSPORT_OVERRIDES = new Set(["hls", "original"]);

/**
 * A per-browser test switch for Timeline playback: `?transport=hls` or
 * `?transport=original` forces that transport and is remembered in this
 * browser; `?transport=auto` returns to the automatic choice.
 */
export function recordingTransportOverride(win = typeof window === "undefined" ? null : window) {
  if (!win) return null;
  let requested = null;
  try {
    requested = new URLSearchParams(win.location?.search || "").get("transport");
  } catch {
    requested = null;
  }
  try {
    if (requested === "auto") {
      win.localStorage?.removeItem(RECORDING_TRANSPORT_STORAGE_KEY);
      return null;
    }
    if (RECORDING_TRANSPORT_OVERRIDES.has(requested)) {
      win.localStorage?.setItem(RECORDING_TRANSPORT_STORAGE_KEY, requested);
      return requested;
    }
    const stored = win.localStorage?.getItem(RECORDING_TRANSPORT_STORAGE_KEY);
    return RECORDING_TRANSPORT_OVERRIDES.has(stored) ? stored : null;
  } catch {
    // Storage can be unavailable (private mode); the URL still applies.
    return RECORDING_TRANSPORT_OVERRIDES.has(requested) ? requested : null;
  }
}

export function recordingPlaybackTransport({
  nativeHls,
  rate,
  incompatible = false,
  preferOriginal = false,
  override = null,
} = {}) {
  if (incompatible) return "transcode";
  // A playback failure still falls back to original files, even when forced.
  if (preferOriginal) return "original";
  if (override === "hls" || override === "original") return override;
  // Every recording player (Timeline, incidents) defaults to HLS on every
  // platform. Native HLS uses original files above 2× because of Safari's
  // I-frame-only fast playback.
  if (nativeHls && rate > 2) return "original";
  return "hls";
}

/**
 * Native HLS startup: the playlist's EXT-X-START owns the first position,
 * because a currentTime write while Safari loads can interrupt the seek it is
 * already doing. Each media event advances this check; the caller acts on the
 * result (keep waiting, ask for play, issue the one correction, or settle).
 * `start` is the mutable state from nativePlaylistStart().
 */
export const NATIVE_PLAYLIST_START_TOLERANCE_SECONDS = 0.08;

export function nativePlaylistStart(mediaTime) {
  return { mediaTime: Number(mediaTime), firstWrongTime: null, corrected: false };
}

export function nativePlaylistStartStep(start, video, eventType) {
  if (!video || video.readyState < 2 || video.seeking || eventType === "seeking") {
    start.firstWrongTime = null;
    return { action: "wait" };
  }
  const atTarget = videoReachedSeekTarget(video, start.mediaTime, NATIVE_PLAYLIST_START_TOLERANCE_SECONDS);
  if (!atTarget) {
    const time = video.currentTime;
    if (video.paused || !Number.isFinite(time)) {
      start.firstWrongTime = null;
      // A paused element never advances, so it cannot prove a wrong start.
      return { action: video.paused && video.readyState >= 3 ? "needs-play" : "wait" };
    }
    if (eventType === "seeked") start.firstWrongTime = null;
    if (eventType !== "timeupdate") return { action: "wait" };
    if (start.firstWrongTime === null || time < start.firstWrongTime) {
      start.firstWrongTime = time;
      return { action: "wait" };
    }
    // Give the initial seek time to settle. Correct only after playable media
    // has actually advanced from the wrong position, and only once.
    if (time - start.firstWrongTime < 0.1) return { action: "wait" };
    if (!start.corrected) {
      start.corrected = true;
      start.firstWrongTime = null;
      return { action: "correct" };
    }
  }
  return { action: "settle", atTarget };
}

export function prefersJpegScrubPreview(options = {}) {
  const preferNativeHls = typeof options.preferNativeHls === "boolean"
    ? options.preferNativeHls
    : PREFER_NATIVE_HLS;
  const coarsePointer = typeof options.coarsePointer === "boolean"
    ? options.coarsePointer
    : (typeof window !== "undefined"
      && Boolean(window.matchMedia?.("(pointer: coarse)").matches));
  return preferNativeHls || coarsePointer;
}

export function scrubPreviewDelayMs(options = {}) {
  const coarsePointer = typeof options.coarsePointer === "boolean"
    ? options.coarsePointer
    : prefersJpegScrubPreview(options);
  return coarsePointer ? 450 : 250;
}

export function scrubPreviewBucketSeconds(options = {}) {
  return prefersJpegScrubPreview(options) ? 10 : 5;
}

export function shouldUseFastSeek(options = {}) {
  if (options.allowFastSeek === false) return false;
  return !prefersJpegScrubPreview(options);
}

export function recordingSeekToleranceSeconds(options = {}) {
  return prefersJpegScrubPreview(options) ? 0.35 : 0.08;
}

export function videoReachedSeekTarget(video, mediaTime, tolerance = 0.08) {
  if (!video || !Number.isFinite(mediaTime)) return false;
  const currentTime = Number(video.currentTime);
  if (!Number.isFinite(currentTime)) return false;
  return Math.abs(currentTime - mediaTime) <= tolerance;
}

export function seekVideoToTime(video, mediaTime, options = {}) {
  if (!video || !Number.isFinite(mediaTime)) return;
  if (shouldUseFastSeek(options) && typeof video.fastSeek === "function") {
    video.fastSeek(mediaTime);
    return;
  }
  video.currentTime = mediaTime;
}

export function seekWatchdogDelayMs(options = {}) {
  // A local seek on an open element should recover in under a second.
  // Playlist loads still wait longer for Safari to publish seekable ranges.
  if (options.local) return 400;
  return prefersJpegScrubPreview(options) ? 3000 : 1500;
}

export function ignorePauseAfterSeekMs(options = {}) {
  return prefersJpegScrubPreview(options) ? 900 : 400;
}

export function playbackRowsCoverEpoch(rows, epoch) {
  if (!Number.isFinite(epoch) || !Array.isArray(rows)) return false;
  return rows.some((row) => {
    const start = Number(row?.start_epoch);
    const end = Number(row?.end_epoch);
    return Number.isFinite(start) && Number.isFinite(end) && start <= epoch && epoch < end;
  });
}

export function adjustRecordingExportRange({ range, kind, key, shiftKey = false, startEpoch, endEpoch, minimumGap = 1 }) {
  if (!range || !["start", "end"].includes(kind)) return null;
  const gap = Math.max(1, Number(minimumGap) || 1);
  const step = shiftKey ? 60 : 1;
  let epoch = kind === "start" ? Number(range.start) : Number(range.end);
  if (key === "ArrowLeft" || key === "ArrowDown") epoch -= step;
  else if (key === "ArrowRight" || key === "ArrowUp") epoch += step;
  else if (key === "Home") epoch = kind === "start" ? startEpoch : Number(range.start) + gap;
  else if (key === "End") epoch = kind === "start" ? Number(range.end) - gap : endEpoch;
  else return null;
  return kind === "start"
    ? { ...range, start: Math.max(startEpoch, Math.min(Number(range.end) - gap, epoch)) }
    : { ...range, end: Math.min(endEpoch, Math.max(Number(range.start) + gap, epoch)) };
}

export function playbackMediaTimeForEpoch(rows, epoch, boundaryTolerance = 0) {
  if (!Number.isFinite(epoch) || !Array.isArray(rows)) return null;
  const tolerance = Math.max(0, Number(boundaryTolerance) || 0);
  const exactRow = rows.find((candidate) => {
    const start = Number(candidate?.start_epoch);
    const end = Number(candidate?.end_epoch);
    return Number.isFinite(start) && Number.isFinite(end) && start <= epoch && epoch < end;
  });
  const row = exactRow || rows.find((candidate) => {
    const start = Number(candidate?.start_epoch);
    const end = Number(candidate?.end_epoch);
    return Number.isFinite(start) && Number.isFinite(end)
      && start - tolerance <= epoch && epoch < end + tolerance;
  });
  if (!row) return null;
  const start = Number(row.start_epoch);
  const end = Number(row.end_epoch);
  const mediaStart = Number(row.media_start);
  const mediaEnd = Number(row.media_end);
  if (!Number.isFinite(mediaStart)) return null;
  const maximum = Number.isFinite(mediaEnd)
    ? Math.max(mediaStart, mediaEnd - 0.01)
    : mediaStart + Math.max(0, end - start - 0.01);
  return Math.max(mediaStart, Math.min(maximum, mediaStart + epoch - start));
}

export function gridPlaybackNeedsSeek({ currentTime, targetTime, playing, epochDelta }) {
  if (!Number.isFinite(currentTime) || !Number.isFinite(targetTime)) return false;
  const drift = Math.abs(currentTime - targetTime);
  if (!playing) return drift > 0.08;
  const continuousClock = Number.isFinite(epochDelta) && epochDelta >= 0 && epochDelta <= 2.5;
  // A running clock tolerates normal lag (more at higher rates, which step
  // further per tick) but still pulls back a tile that stalled or fell behind.
  return continuousClock ? drift > Math.max(2, epochDelta * 2) : drift > 0.2;
}

/** Mobile Safari often pauses while seeking; resume only for intentional autoplay seeks. */
export function shouldResumePlaybackAfterSeek({ pendingSeekMode, autoplay }) {
  return Boolean(autoplay) && (pendingSeekMode === "local" || pendingSeekMode === "window-ready");
}

// Where any of several overlapping range lists has recording (All cameras):
// the sorted union, so a rail drawing it sees no duplicates or false gaps.
export function recordingCoverageUnion(ranges) {
  const ordered = (ranges || [])
    .map((range) => ({ start_epoch: Number(range?.start_epoch), end_epoch: Number(range?.end_epoch) }))
    .filter((range) => Number.isFinite(range.start_epoch) && range.end_epoch > range.start_epoch)
    .sort((left, right) => left.start_epoch - right.start_epoch);
  const union = [];
  for (const range of ordered) {
    const previous = union[union.length - 1];
    if (previous && range.start_epoch <= previous.end_epoch) previous.end_epoch = Math.max(previous.end_epoch, range.end_epoch);
    else union.push(range);
  }
  return union;
}

export function mergeRecordingAvailability(current, updates) {
  const groups = new Map();
  for (const item of [...(current || []), ...(updates || [])]) {
    const normalized = {
      ...item,
      start_epoch: Number(item?.start_epoch),
      end_epoch: Number(item?.end_epoch),
    };
    if (!Number.isFinite(normalized.start_epoch) || !Number.isFinite(normalized.end_epoch)) continue;
    const key = `${normalized.camera_id || ""}\u0000${normalized.source || normalized.fallback_source || ""}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(normalized);
  }
  const merged = [];
  for (const ranges of groups.values()) {
    ranges.sort((left, right) => left.start_epoch - right.start_epoch);
    for (const item of ranges) {
      const previous = merged[merged.length - 1];
      const sameGroup = previous
        && (previous.camera_id || "") === (item.camera_id || "")
        && (previous.source || previous.fallback_source || "") === (item.source || item.fallback_source || "");
      if (sameGroup && item.start_epoch <= previous.end_epoch + 5) {
        previous.end_epoch = Math.max(previous.end_epoch, item.end_epoch);
        previous.duration_seconds = previous.end_epoch - previous.start_epoch;
        previous.segment_count = Math.max(
          Number(previous.segment_count) || 0,
          Number(item.segment_count) || 0,
        );
      } else {
        merged.push({ ...item });
      }
    }
  }
  return merged.sort((left, right) => left.start_epoch - right.start_epoch);
}

export function describePlaybackError(error) {
  const details = [];
  const code = Number(error?.code);
  if (Number.isFinite(code)) details.push(`code ${code}`);
  const category = Number(error?.category);
  if (Number.isFinite(category)) details.push(`category ${category}`);
  if (error?.message) details.push(String(error.message));
  const data = Array.isArray(error?.data)
    ? error.data.filter((item) => ["string", "number"].includes(typeof item)).slice(0, 3)
    : [];
  if (data.length) details.push(data.join(", "));
  return [...new Set(details)].join(" · ") || "unknown media error";
}

export function isUnsupportedPlaybackError(error) {
  const code = Number(error?.code);
  const dataCodes = Array.isArray(error?.data) ? error.data.map(Number).filter(Number.isFinite) : [];
  const description = describePlaybackError(error).toLowerCase();
  return code === 4
    || dataCodes.includes(4)
    || /(?:codec|decode|format|media source).*(?:unsupported|not supported)/.test(description)
    || /(?:unsupported|not supported).*(?:codec|decode|format|media source)/.test(description);
}

/** Only media compatibility failures justify switching playback transports. */
export function isRecordingCompatibilityError(error) {
  const code = Number(error?.code);
  if (Number(error?.category) === 1 || (code >= 1000 && code < 2000)) return false;
  return code === 3 || code === 4
    || code === 4032 // Shaka CONTENT_UNSUPPORTED_BY_BROWSER
    || ([3014, 3016].includes(code) && [3, 4].includes(Number(error?.data?.[0]))) // MediaError
    || (code === 3015 && error?.data?.[0]?.name === "NotSupportedError") // addSourceBuffer
    || /does not support Shaka Player/.test(error?.message || "")
    || (!Number.isFinite(code) && isUnsupportedPlaybackError(error));
}

// The HLS window playlist's clock: rows in time order, each with its media
// offset (the concatenated duration of the rows before it).
export function recordingPlaybackTimeline(rows) {
  let mediaOffset = 0;
  return (rows || [])
    .map((item) => ({
      ...item,
      start_epoch: Number(item.start_epoch),
      end_epoch: Number(item.end_epoch),
    }))
    .filter((item) => Number.isFinite(item.start_epoch) && Number.isFinite(item.end_epoch))
    .sort((left, right) => left.start_epoch - right.start_epoch)
    .map((item) => {
      const duration = Math.max(0.01, Number(item.duration_seconds) || item.end_epoch - item.start_epoch);
      const result = { ...item, media_start: mediaOffset, media_end: mediaOffset + duration };
      mediaOffset += duration;
      return result;
    });
}

export function recordingSegmentAt(rows, epoch) {
  if (!Number.isFinite(epoch)) return null;
  const row = (rows || []).find((item) => Number(item.start_epoch) <= epoch && epoch < Number(item.end_epoch));
  return row ? { start_epoch: Number(row.start_epoch), end_epoch: Number(row.end_epoch) } : null;
}

export function recordingPlayableEpoch(rows, epoch) {
  const valid = (rows || []).filter(row => Number.isFinite(Number(row.start_epoch))
    && Number.isFinite(Number(row.end_epoch)) && Number(row.end_epoch) > Number(row.start_epoch));
  if (!valid.length || !Number.isFinite(epoch)) return null;
  const last = valid.reduce((a, b) => Number(a.end_epoch) > Number(b.end_epoch) ? a : b);
  if (epoch >= Number(last.end_epoch) - 1) return Math.max(Number(last.start_epoch), Number(last.end_epoch) - 1);
  if (valid.some(row => Number(row.start_epoch) <= epoch && epoch < Number(row.end_epoch))) return epoch;
  return valid.reduce((best, row) => {
    const candidate = epoch < Number(row.start_epoch) ? Number(row.start_epoch)
      : Math.max(Number(row.start_epoch), Number(row.end_epoch) - 1);
    return best === null || Math.abs(candidate - epoch) < Math.abs(best - epoch) ? candidate : best;
  }, null);
}

export function recordingSegmentLocalTime(segment, epoch, video = null) {
  if (!segment || !Number.isFinite(epoch)) return null;
  const indexedDuration = Math.max(0, segment.end_epoch - segment.start_epoch);
  const mediaDuration = Number(video?.duration);
  const duration = Number.isFinite(mediaDuration) && mediaDuration > 0 ? mediaDuration : indexedDuration;
  return Math.max(0, Math.min(Math.max(0, duration - 0.01), epoch - segment.start_epoch));
}

// Seeking to the end of the newest footage starts this far before it.
export const RECORDING_EDGE_SNAP_SECONDS = 1;

// A recording that ended this recently is still being written.
export const RECORDING_LIVE_EDGE_RECENT_SECONDS = 120;
// A seek to the end of footage that is still being written starts this far
// before it. Native HLS needs a few whole segments ahead of the start: started
// in the last second of a growing playlist it refetches that segment and waits
// for the next one to be published, which takes a segment length or more.
export const RECORDING_LIVE_EDGE_HOLDBACK_SECONDS = 30;

// The latest time a seek into `latest`, the newest recorded range, may start at.
// Playback continuing from footage it has just played passes `continuing`: it
// must not be moved back, or it would replay what was just shown.
export function recordingEdgeSeekEpoch(latest, nowEpoch, continuing = false) {
  const live = !continuing && Number(nowEpoch) - latest.end_epoch < RECORDING_LIVE_EDGE_RECENT_SECONDS;
  const back = live ? RECORDING_LIVE_EDGE_HOLDBACK_SECONDS : RECORDING_EDGE_SNAP_SECONDS;
  return Math.max(latest.start_epoch, latest.end_epoch - back);
}

// Where playback continues after `segment`, or null when nothing follows yet.
// A range has to reach beyond the edge snap to count. Availability can end a
// fraction of a second after the segment that was just played: a recording is
// listed with its nominal length until its exact length is read. Seeking to
// that sliver snaps back into the played segment and repeats its last second.
export function recordingEpochAfterSegment(segment, ranges) {
  if (!segment) return null;
  const end = Number(segment.end_epoch) + 0.01;
  const next = (ranges || []).find((item) => (
    item.end_epoch - Math.max(end, item.start_epoch) > RECORDING_EDGE_SNAP_SECONDS
  ));
  return next ? Math.max(end, next.start_epoch) : null;
}

export const RECORDING_LIVE_EDGE_STORAGE_KEY = "survng.recordingLiveEdge";

// Opt-in (?liveedge=1, remembered; ?liveedge=0 clears it): at the live edge
// native HLS plays a growing playlist, so new recordings follow on without the
// page loading a new playlist.
export function recordingLiveEdgeFollow(win = typeof window === "undefined" ? null : window) {
  if (!win) return false;
  let requested = null;
  try {
    requested = new URLSearchParams(win.location?.search || "").get("liveedge");
  } catch {
    requested = null;
  }
  try {
    if (requested === "0") {
      win.localStorage?.removeItem(RECORDING_LIVE_EDGE_STORAGE_KEY);
      return false;
    }
    if (requested === "1") {
      win.localStorage?.setItem(RECORDING_LIVE_EDGE_STORAGE_KEY, "1");
      return true;
    }
    return win.localStorage?.getItem(RECORDING_LIVE_EDGE_STORAGE_KEY) === "1";
  } catch {
    // Storage can be unavailable (private mode); the URL still applies.
    return requested === "1";
  }
}

// Rows re-read for a growing playlist replace the loaded ones only when they
// extend them: the player's clock is the rows it was started with, so a
// changed or missing earlier row would move every later media time.
export function extendedPlaybackRows(current, next) {
  if (!Array.isArray(current) || !Array.isArray(next) || next.length <= current.length) return null;
  const unchanged = current.every((row, index) => (
    Number(row.start_epoch) === Number(next[index].start_epoch)
    && Number(row.duration_seconds) === Number(next[index].duration_seconds)
  ));
  return unchanged ? next : null;
}
