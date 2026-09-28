function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function eventEpoch(event) {
  const parsed = Date.parse(String(event?.created_at || ""));
  if (Number.isFinite(parsed)) return parsed / 1000;
  return finiteNumber(event?.start_epoch);
}

export function withoutIsolatedTrackSpikes(samples) {
  if (!Array.isArray(samples) || samples.length < 3) return Array.isArray(samples) ? samples : [];
  const rejected = new Set();
  for (let index = 1; index < samples.length - 1; index += 1) {
    const previous = samples[index - 1];
    const candidate = samples[index];
    const next = samples[index + 1];
    if (![previous, candidate, next].every((sample) => Array.isArray(sample) && sample.length >= 5)) continue;
    const beforeGap = candidate[0] - previous[0];
    const afterGap = next[0] - candidate[0];
    if (beforeGap <= 0 || afterGap <= 0 || beforeGap > 1.5 || afterGap > 1.5) continue;
    const center = (sample) => [(sample[1] + sample[3]) / 2, (sample[2] + sample[4]) / 2];
    const diagonal = (sample) => Math.hypot(sample[3] - sample[1], sample[4] - sample[2]);
    const distance = (left, right) => Math.hypot(left[0] - right[0], left[1] - right[1]);
    const previousCenter = center(previous);
    const candidateCenter = center(candidate);
    const nextCenter = center(next);
    const stableScale = Math.max(1, diagonal(previous), diagonal(next));
    if (
      distance(previousCenter, candidateCenter) > stableScale * 0.40
      && distance(candidateCenter, nextCenter) > stableScale * 0.40
      && distance(previousCenter, nextCenter) < stableScale * 0.20
    ) rejected.add(index);
  }
  return samples.filter((_sample, index) => !rejected.has(index));
}

export function containedFrameTransform(containerSize, sourceSize) {
  const containerWidth = finiteNumber(containerSize?.width);
  const containerHeight = finiteNumber(containerSize?.height);
  const sourceWidth = finiteNumber(sourceSize?.width);
  const sourceHeight = finiteNumber(sourceSize?.height);
  if ([containerWidth, containerHeight, sourceWidth, sourceHeight].some((value) => value === null || value <= 0)) return null;
  const scale = Math.min(containerWidth / sourceWidth, containerHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return {
    x: (containerWidth - width) / 2,
    y: (containerHeight - height) / 2,
    width,
    height,
    scale,
  };
}

function observationEpoch(observation) {
  const direct = finiteNumber(observation?.captured_at_epoch);
  if (direct !== null) return direct;
  const parsed = Date.parse(String(observation?.captured_at || ""));
  return Number.isFinite(parsed) ? parsed / 1000 : null;
}

function sceneReplayTracks(incident) {
  const groups = new Map();
  for (const subject of incident?.scene_objects || []) {
    for (const observation of subject?.observations || []) {
      const key = String(observation?.scene_track_key || "");
      // Stored keys are "tracking:{event}:{run}:{id}". The database column
      // prefixes the event id, so either form has tracking as its own field.
      if (!key.split(":").includes("tracking")) continue;
      const epoch = observationEpoch(observation);
      const box = observation?.box || {};
      const coordinates = [box.x1, box.y1, box.x2, box.y2].map(finiteNumber);
      const width = finiteNumber(observation?.detection_frame_width);
      const height = finiteNumber(observation?.detection_frame_height);
      if (
        epoch === null
        || width === null
        || height === null
        || width <= 0
        || height <= 0
        || coordinates.some((value) => value === null)
        || coordinates[2] <= coordinates[0]
        || coordinates[3] <= coordinates[1]
      ) continue;
      let group = groups.get(key);
      if (!group) {
        group = { label: observation.label || subject?.label || "object", samples: [] };
        groups.set(key, group);
      }
      group.samples.push({ epoch, coordinates, width, height });
    }
  }
  const prepared = [];
  const sizeCounts = new Map();
  for (const group of groups.values()) {
    const localCounts = new Map();
    for (const sample of group.samples) {
      const sizeKey = `${sample.width}x${sample.height}`;
      localCounts.set(sizeKey, (localCounts.get(sizeKey) || 0) + 1);
      sizeCounts.set(sizeKey, (sizeCounts.get(sizeKey) || 0) + 1);
    }
    const sizeKey = [...localCounts.entries()].sort((left, right) => right[1] - left[1])[0][0];
    const [width, height] = sizeKey.split("x").map(Number);
    const history = [];
    for (const sample of group.samples.filter((item) => item.width === width && item.height === height).sort((left, right) => left.epoch - right.epoch)) {
      const point = [sample.epoch, ...sample.coordinates];
      if (history.length && history[history.length - 1][0] === sample.epoch) history[history.length - 1] = point;
      else history.push(point);
    }
    if (history.length) prepared.push({ label: group.label, width, height, history });
  }
  if (!prepared.length) return { tracks: [], width: 0, height: 0 };
  const [width, height] = [...sizeCounts.entries()].sort((left, right) => right[1] - left[1])[0][0].split("x").map(Number);
  const tracks = prepared.filter((track) => track.width === width && track.height === height).map((track, index) => {
    const last = track.history[track.history.length - 1];
    return {
      track_id: index + 1,
      label: track.label,
      state: "confirmed",
      box: { x1: last[1], y1: last[2], x2: last[3], y2: last[4] },
      box_history: track.history,
      trajectory: track.history.map((sample) => [sample[0], (sample[1] + sample[3]) / 2, (sample[2] + sample[4]) / 2]),
    };
  });
  return { tracks, width, height };
}

function earliestHistory(tracks) {
  let earliest = null;
  for (const track of tracks || []) {
    const history = track?.box_history || track?.boxHistory || [];
    for (const sample of history) {
      const epoch = finiteNumber(Array.isArray(sample) ? sample[0] : null);
      if (epoch === null) continue;
      earliest = earliest === null ? epoch : Math.min(earliest, epoch);
    }
  }
  return earliest;
}

export function incidentReplayTracking(event, incident = null) {
  const source = incidentTrackingSource(event, incident);
  const scene = incident?.scene_objects ? incident : event?.scene_objects ? event : null;
  const built = sceneReplayTracks(scene);
  const storedTracks = storedObjectTracks(source);
  const storedStart = earliestHistory(storedTracks);
  const sceneStart = earliestHistory(built.tracks);
  // A reclaimed tracking job saves only its last run. Observations from the
  // earlier runs are still the path, so a later fragment must not hide them.
  const sceneStartsEarlier = storedStart !== null && sceneStart !== null && storedStart - sceneStart > 1;
  if (storedTracks.length && !sceneStartsEarlier) return source;
  const events = event?.events?.length ? event.events : incident?.events || [];
  const withTracks = events.find((candidate) => storedObjectTracks(candidate).length);
  if (withTracks && !sceneStartsEarlier) return withTracks;
  if (!built.tracks.length) return source;
  const base = source?.object_tracking && typeof source.object_tracking === "object" ? source.object_tracking : {};
  const epochs = built.tracks.flatMap((track) => track.box_history.map((sample) => sample[0]));
  const start = Math.min(...epochs);
  const end = Math.max(...epochs);
  const windowStart = finiteNumber(base.window_start_epoch);
  const windowEnd = finiteNumber(base.window_end_epoch);
  return {
    ...(source || event || incident || {}),
    object_tracking: {
      ...base,
      state: base.state || "complete",
      frame_width: built.width,
      frame_height: built.height,
      sample_fps: finiteNumber(base.sample_fps) || 3,
      tracks: built.tracks,
      window_start_epoch: windowStart === null ? start : Math.min(windowStart, start),
      window_end_epoch: windowEnd === null ? end : Math.max(windowEnd, end),
      analyzed_from: new Date(start * 1000).toISOString(),
      analyzed_through: new Date(end * 1000).toISOString(),
    },
  };
}

export function incidentTrackingSource(event, incident = null) {
  if (event?.object_tracking?.tracks?.length || event?.object_tracking?.state) return event;
  const incidentEvents = event?.events?.length ? event.events : incident?.events || [];
  const candidates = incidentEvents.filter(
    (candidate) => candidate?.object_tracking?.tracks?.length || candidate?.object_tracking?.state,
  );
  if (!candidates.length) return null;
  const target = eventEpoch(event);
  if (target === null) return candidates[0];
  return candidates.reduce((best, candidate) => {
    const candidateTime = eventEpoch(candidate);
    const bestTime = eventEpoch(best);
    const candidateDistance = candidateTime === null ? Number.POSITIVE_INFINITY : Math.abs(candidateTime - target);
    const bestDistance = bestTime === null ? Number.POSITIVE_INFINITY : Math.abs(bestTime - target);
    return candidateDistance < bestDistance ? candidate : best;
  });
}

export function storedObjectTracks(event) {
  const tracks = event?.object_tracking?.tracks;
  if (!Array.isArray(tracks)) return [];
  return tracks.flatMap((track) => {
    const box = track?.box || {};
    const coordinates = [box.x1, box.y1, box.x2, box.y2].map(finiteNumber);
    const trackId = finiteNumber(track?.track_id);
    if (!track?.label || trackId === null || coordinates.some((value) => value === null)) return [];
    const trajectory = Array.isArray(track.trajectory)
      ? track.trajectory.flatMap((point) => {
        if (!Array.isArray(point) || point.length < 3) return [];
        const values = point.slice(0, 3).map(finiteNumber);
        return values.some((value) => value === null) ? [] : [values];
      }).sort((left, right) => left[0] - right[0])
      : [];
    const boxHistory = withoutIsolatedTrackSpikes(Array.isArray(track.box_history)
      ? track.box_history.flatMap((sample) => {
        if (!Array.isArray(sample) || sample.length < 5) return [];
        const values = sample.slice(0, 5).map(finiteNumber);
        if (values.some((value) => value === null) || values[3] <= values[1] || values[4] <= values[2]) return [];
        return [values];
      }).sort((left, right) => left[0] - right[0])
      : []);
    const boxTimestamps = new Set(boxHistory.map((sample) => sample[0]));
    const filteredTrajectory = boxHistory.length
      ? trajectory.filter((point) => boxTimestamps.has(point[0]))
      : trajectory;
    const recoveryHistory = Array.isArray(track.reid_recovery_history)
      ? track.reid_recovery_history.flatMap((recovery) => {
        const capturedAt = finiteNumber(recovery?.captured_at);
        const similarity = finiteNumber(recovery?.similarity);
        const recoveryBox = Array.isArray(recovery?.box)
          ? recovery.box.slice(0, 4).map(finiteNumber)
          : [];
        if (capturedAt === null || similarity === null || recoveryBox.length !== 4 || recoveryBox.some((value) => value === null)) return [];
        if (recoveryBox[2] <= recoveryBox[0] || recoveryBox[3] <= recoveryBox[1]) return [];
        return [{
          capturedAt,
          similarity,
          resumedCompletedTrack: Boolean(recovery.resumed_completed_track),
          box: recoveryBox,
        }];
      }).sort((left, right) => left.capturedAt - right.capturedAt)
      : [];
    return [{
      ...track,
      trackId,
      x1: coordinates[0],
      y1: coordinates[1],
      x2: coordinates[2],
      y2: coordinates[3],
      trajectory: filteredTrajectory,
      boxHistory,
      recoveryHistory,
    }];
  }).filter((track) => track.x2 > track.x1 && track.y2 > track.y1);
}

export function trackFrameAt(track, epoch, { holdSeconds = 1, sampleFps = 2, observationsOnly = false } = {}) {
  const samples = track?.boxHistory;
  if (!Array.isArray(samples) || !samples.length || !Number.isFinite(epoch)) return null;
  const first = samples[0];
  const last = samples[samples.length - 1];
  const safeHoldSeconds = Math.max(0, Number(holdSeconds) || 0);
  const expectedInterval = 1 / Math.max(0.1, Number(sampleFps) || 2);
  if (epoch < first[0] || epoch > last[0] + safeHoldSeconds) return null;
  let previous = first;
  let next = first;
  for (const sample of samples) {
    if (sample[0] <= epoch) previous = sample;
    if (sample[0] >= epoch) {
      next = sample;
      break;
    }
    next = sample;
  }
  const span = next[0] - previous[0];
  const progress = span > 0 ? Math.max(0, Math.min(1, (epoch - previous[0]) / span)) : 0;
  const box = previous.slice(1, 5).map((value, index) => value + (next[index + 1] - value) * progress);
  const estimated = epoch > last[0] || (
    span > expectedInterval * 1.75
    && epoch > previous[0] + expectedInterval * 1.25
    && epoch < next[0] - expectedInterval * 0.25
  );
  if (observationsOnly && estimated) return null;
  const path = (track.trajectory || []).filter((point) => point[0] <= epoch).map((point) => point.slice(1, 3));
  const center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
  if (!path.length || path[path.length - 1][0] !== center[0] || path[path.length - 1][1] !== center[1]) path.push(center);
  const recovery = [...(track.recoveryHistory || [])]
    .reverse()
    .find((item) => item.capturedAt <= epoch && epoch - item.capturedAt <= Math.max(0.75, safeHoldSeconds));
  return { box, path, recovery: recovery || null, estimated };
}

export function playbackEpochAt(windowStartEpoch, mediaTime, mediaStartTime) {
  const startEpoch = finiteNumber(windowStartEpoch);
  const currentTime = finiteNumber(mediaTime);
  const originTime = finiteNumber(mediaStartTime);
  if (startEpoch === null || currentTime === null || originTime === null) return null;
  return startEpoch + currentTime - originTime;
}

export function hlsProgramStartEpoch(manifest) {
  if (typeof manifest !== "string") return null;
  const line = manifest.split(/\r?\n/).find((value) => value.startsWith("#EXT-X-PROGRAM-DATE-TIME:"));
  if (!line) return null;
  const parsed = Date.parse(line.slice("#EXT-X-PROGRAM-DATE-TIME:".length).trim());
  return Number.isFinite(parsed) ? parsed / 1000 : null;
}

export function hlsPlaybackOffset(windowStartEpoch, mediaStartEpoch, initialOffset = 0) {
  const windowStart = finiteNumber(windowStartEpoch);
  const mediaStart = finiteNumber(mediaStartEpoch);
  const requestedOffset = finiteNumber(initialOffset);
  if (windowStart === null || mediaStart === null) return Math.max(0, requestedOffset || 0);
  return Math.max(0, windowStart - mediaStart) + Math.max(0, requestedOffset || 0);
}

export function trackingCoverageLabel(tracking) {
  if (!tracking) return "Tracking unavailable";
  if (tracking.state === "active") return "Tracking processing";
  if (tracking.state === "complete" && !tracking.coverage_incomplete) {
    return "Tracking complete";
  }
  const reason = tracking.completion_reason;
  if (["missing_media_while_object_active", "stale_handoff_without_recorded_coverage"].includes(reason)) {
    return "Tracking incomplete — recording coverage unavailable";
  }
  if (reason === "inference_unavailable") return "Tracking incomplete — waiting for inference capacity";
  if (reason === "processing_budget_exhausted") return "Tracking incomplete — processing time limit reached";
  if (reason === "session_stopped") return "Tracking incomplete — session stopped";
  return "Tracking incomplete";
}


// Explicit recorded windows include their lead-in and tail already.
export function recordedIncidentWindow(event, before = 5, after = 5) {
  const events = event?.events?.length ? event.events : [event];
  const windows = events.flatMap(item => {
    const start = finiteNumber(item?.object_tracking?.window_start_epoch);
    const end = finiteNumber(item?.object_tracking?.window_end_epoch);
    return start !== null && end !== null && end > start ? [{ start, end }] : [];
  });
  if (!windows.length) return null;
  const anchors = [...events, ...(event?.motion_observations || [])].map(eventEpoch).filter(value => value !== null);
  return {
    start: Math.min(...windows.map(window => window.start), ...anchors.map(epoch => epoch - before)),
    end: Math.max(...windows.map(window => window.end), ...anchors.map(epoch => epoch + after)),
  };
}

export function trackingCoverageAt(tracking, epoch) {
  if (!tracking || finiteNumber(tracking.window_start_epoch) === null || !Number.isFinite(epoch)) return null;
  const from = Date.parse(tracking.analyzed_from || "") / 1000;
  const through = Date.parse(tracking.analyzed_through || "") / 1000;
  if (!Number.isFinite(from) || !Number.isFinite(through) || epoch < from || epoch > through) {
    return tracking.state === "active" ? "Tracking pending for this footage" : "This footage has not been analyzed";
  }
  return null;
}
