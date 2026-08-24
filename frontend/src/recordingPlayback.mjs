import { PREFER_NATIVE_HLS } from "./shared/constants.js";

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

export function seekVideoToTime(video, mediaTime) {
  if (!video || !Number.isFinite(mediaTime)) return;
  if (typeof video.fastSeek === "function") {
    video.fastSeek(mediaTime);
    return;
  }
  video.currentTime = mediaTime;
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
  return continuousClock ? false : drift > 0.2;
}

/** Mobile Safari often pauses while seeking; resume only for intentional autoplay seeks. */
export function shouldResumePlaybackAfterSeek({ pendingSeekMode, autoplay }) {
  return Boolean(autoplay) && (pendingSeekMode === "local" || pendingSeekMode === "window-ready");
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
