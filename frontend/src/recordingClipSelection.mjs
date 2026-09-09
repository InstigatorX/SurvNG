export const CLIP_MINIMUM_DURATION_SECONDS = 1;

function finiteEpoch(value) {
  if (value === null || value === undefined || value === "") return null;
  const epoch = Number(value);
  return Number.isFinite(epoch) ? epoch : null;
}

function validBounds(startEpoch, endEpoch) {
  const start = finiteEpoch(startEpoch);
  const end = finiteEpoch(endEpoch);
  return start !== null && end !== null && end > start ? { start, end } : null;
}

export function clipRangeIsValid(range, startEpoch, endEpoch) {
  const bounds = validBounds(startEpoch, endEpoch);
  const start = finiteEpoch(range?.start);
  const end = finiteEpoch(range?.end);
  return Boolean(bounds && start !== null && end !== null
    && start >= bounds.start && end <= bounds.end
    && end - start >= CLIP_MINIMUM_DURATION_SECONDS);
}

export function canSetClipBoundaryAtPlayhead({ range, kind, playhead, startEpoch, endEpoch }) {
  const bounds = validBounds(startEpoch, endEpoch);
  const target = finiteEpoch(playhead);
  if (!bounds || target === null || !clipRangeIsValid(range, startEpoch, endEpoch)) return false;
  if (kind === "start") {
    return target >= bounds.start && target <= range.end - CLIP_MINIMUM_DURATION_SECONDS && target !== range.start;
  }
  if (kind === "end") {
    return target >= range.start + CLIP_MINIMUM_DURATION_SECONDS && target <= bounds.end && target !== range.end;
  }
  return false;
}

export function setClipBoundaryAtPlayhead(options) {
  if (!canSetClipBoundaryAtPlayhead(options)) return options.range;
  return options.kind === "start"
    ? { ...options.range, start: Number(options.playhead) }
    : { ...options.range, end: Number(options.playhead) };
}

export function clipPreviewReachedEnd(epoch, endEpoch, toleranceSeconds = 0.08) {
  return Number.isFinite(epoch) && Number.isFinite(endEpoch) && epoch >= endEpoch - toleranceSeconds;
}
