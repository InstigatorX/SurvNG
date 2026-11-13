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

function replayTrackKey(key) {
  // The database prefixes an event id to the original tracking key.
  const parts = String(key || "").split(":");
  const index = parts.indexOf("tracking");
  return index >= 0 ? parts.slice(index).join(":") : null;
}

function sceneReplayTracks(event, incident, sources) {
  const groups = new Map();
  const camera = event?.camera_id || incident?.camera_id;
  const members = event?.scene_clip_window?.episode_id
    ? new Set((event.events || []).map((item) => String(item.id))) : null;
  const allowed = (observation) => (
    (!camera || !observation.camera_id || observation.camera_id === camera)
    && (!members?.size || members.has(String(observation.event_id)))
  );
  function add(key, label, epoch, coordinates, width, height) {
    if (!key || epoch === null || !(width > 0 && height > 0)
      || coordinates.some((value) => value === null)
      || coordinates[2] <= coordinates[0] || coordinates[3] <= coordinates[1]) return;
    let group = groups.get(key);
    if (!group) {
      group = { key, label, samples: [] };
      groups.set(key, group);
    }
    group.samples.push({ epoch, coordinates, width, height });
  }
  const scenes = new Set([event, incident]);
  for (const scene of scenes) {
    for (const subject of scene?.scene_objects || []) {
      for (const observation of subject?.observations || []) {
        if (!allowed(observation)) continue;
        const box = observation.box || {};
        add(replayTrackKey(observation.scene_track_key), observation.label || subject.label || "object",
          observationEpoch(observation), [box.x1, box.y1, box.x2, box.y2].map(finiteNumber),
          finiteNumber(observation.detection_frame_width), finiteNumber(observation.detection_frame_height));
      }
    }
  }
  // Match a saved fragment to a full scene identity only with an overlapping
  // observation. Local numeric IDs alone are reused across events and runs.
  for (const source of sources) {
    const tracking = source.object_tracking;
    const width = finiteNumber(tracking?.frame_width);
    const height = finiteNumber(tracking?.frame_height);
    if (!(width > 0 && height > 0)) continue;
    for (const track of storedObjectTracks(source)) {
      const candidates = [...groups.values()].filter((group) => {
        const parts = group.key.split(":");
        return parts[0] === "tracking" && parts[1] === String(source.representative_event_id || source.id)
          && parts.at(-1) === String(track.trackId) && group.label === track.label;
      });
      const matches = candidates.filter((group) => {
        if (!group.timeIndex) {
          group.timeIndex = new Map();
          for (const point of group.samples) {
            const bucket = Math.round(point.epoch * 1000);
            if (!group.timeIndex.has(bucket)) group.timeIndex.set(bucket, []);
            group.timeIndex.get(bucket).push(point);
          }
        }
        return track.boxHistory.some((sample) => {
          const bucket = Math.round(sample[0] * 1000);
          for (let offset = -2; offset <= 2; offset += 1) {
            if ((group.timeIndex.get(bucket + offset) || []).some((point) => (
              Math.abs(point.epoch - sample[0]) < 0.002
              && point.coordinates.every((value, index) => Math.abs(value / (index % 2 ? point.height : point.width)
                - sample[index + 1] / (index % 2 ? height : width)) < 0.00001)
            ))) return true;
          }
          return false;
        });
      });
      const key = track.replay_key || (matches.length === 1 ? matches[0].key
        : `stored:${source.representative_event_id || source.id}:${track.trackId}`);
      for (const sample of track.boxHistory) add(key, track.label, sample[0], sample.slice(1), width, height);
      const group = groups.get(key);
      if (group) group.metadata = { track, width, height };
    }
  }
  const sizes = new Map();
  for (const group of groups.values()) for (const sample of group.samples) {
    const key = `${sample.width}x${sample.height}`;
    sizes.set(key, (sizes.get(key) || 0) + 1);
  }
  if (!sizes.size) return null;
  const [width, height] = [...sizes].sort((a, b) => b[1] - a[1])[0][0].split("x").map(Number);
  const usedIds = new Set();
  const tracks = [...groups.values()].sort((a, b) => a.key.localeCompare(b.key)).map((group, index) => {
    const history = [];
    for (const sample of group.samples.sort((a, b) => a.epoch - b.epoch)) {
      const point = [sample.epoch, ...sample.coordinates.map((value, index) => value
        * (index % 2 ? height / sample.height : width / sample.width))];
      // Saved history rounds capture times to milliseconds.
      if (history.length && Math.abs(history.at(-1)[0] - point[0]) < 0.002) history[history.length - 1] = point;
      else history.push(point);
    }
    // Stable internal IDs preserve separate event/run identities across refreshes.
    let id = 2166136261;
    for (const char of group.key) id = Math.imul(id ^ char.charCodeAt(0), 16777619) >>> 0;
    while (usedIds.has(id)) id = (id + 1) >>> 0;
    usedIds.add(id);
    const last = history.at(-1);
    const metadata = group.metadata;
    return {
      ...(metadata?.track || {}),
      track_id: id,
      replay_key: group.key,
      display_id: index + 1,
      label: group.label,
      state: "confirmed",
      box: { x1: last[1], y1: last[2], x2: last[3], y2: last[4] },
      box_history: history,
      trajectory: history.map((sample) => [sample[0], (sample[1] + sample[3]) / 2, (sample[2] + sample[4]) / 2]),
      reid_recovery_history: (metadata?.track.recoveryHistory || []).map((recovery) => ({
        captured_at: recovery.capturedAt, similarity: recovery.similarity,
        resumed_completed_track: recovery.resumedCompletedTrack,
        box: recovery.box.map((value, index) => value * (index % 2 ? height / metadata.height : width / metadata.width)),
      })),
    };
  });
  return { tracks, width, height };
}

export function incidentReplayTracking(event, incident = null) {
  const source = incidentTrackingSource(event, incident);
  const camera = event?.camera_id || incident?.camera_id;
  const candidates = [event, ...(event?.events || []), ...(event?.scene_clip_window?.episode_id ? [] : incident?.events || [])];
  const sources = [...new Map(candidates.filter((item) => item?.object_tracking
    && (!camera || !item.camera_id || item.camera_id === camera))
    .map((item) => [String(item.representative_event_id || item.id), item])).values()];
  const built = sceneReplayTracks(event, incident, sources);
  if (!built) return source;
  const base = source?.object_tracking || {};
  let start = Infinity;
  let end = -Infinity;
  for (const track of built.tracks) for (const sample of track.box_history) {
    start = Math.min(start, sample[0]);
    end = Math.max(end, sample[0]);
  }
  const starts = sources.map((item) => finiteNumber(item.object_tracking.window_start_epoch)).filter((value) => value !== null);
  const ends = sources.map((item) => finiteNumber(item.object_tracking.window_end_epoch)).filter((value) => value !== null);
  return {
    ...(event || source || incident || {}),
    object_tracking: {
      ...base,
      state: base.state || "complete",
      frame_width: built.width,
      frame_height: built.height,
      sample_fps: finiteNumber(base.sample_fps) || 3,
      tracks: built.tracks,
      window_start_epoch: Math.min(start, ...starts),
      window_end_epoch: Math.max(end, ...ends),
      analyzed_from: new Date(Math.min(start, ...sources.map((item) => Date.parse(item.object_tracking.analyzed_from) / 1000).filter(Number.isFinite)) * 1000).toISOString(),
      analyzed_through: new Date(Math.max(end, ...sources.map((item) => Date.parse(item.object_tracking.analyzed_through) / 1000).filter(Number.isFinite)) * 1000).toISOString(),
      coverage_incomplete: sources.some((item) => item.object_tracking.coverage_incomplete),
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

// Keep completed paths during replay, with breaks across missing observations.
export function incidentTrackFramesAt(tracks, epoch, options = {}) {
  if (!Number.isFinite(epoch)) return [];
  const maxGap = Math.max(1, 1.75 / Math.max(0.1, Number(options.sampleFps) || 2));
  return tracks.flatMap((track) => {
    const frame = trackFrameAt(track, epoch, options);
    const paths = [];
    let previousEpoch = -Infinity;
    for (const sample of track.boxHistory || []) {
      if (sample[0] > epoch) break;
      if (sample[0] - previousEpoch > maxGap) paths.push([]);
      paths.at(-1).push([(sample[1] + sample[3]) / 2, (sample[2] + sample[4]) / 2]);
      previousEpoch = sample[0];
    }
    if (frame && paths.length) paths.at(-1).push([(frame.box[0] + frame.box[2]) / 2, (frame.box[1] + frame.box[3]) / 2]);
    if (!frame && !paths.length) return [];
    return [{ ...track, ...frame, box: frame?.box || null, paths }];
  });
}

// Snapshot overlays must describe the displayed frame, not the final state of
// each track. Missing history cannot establish that a track was in this image.
export function snapshotObjectTracks(event, capturedEpoch) {
  const sampleFps = event?.object_tracking?.sample_fps || 2;
  return storedObjectTracks(event).flatMap((track) => {
    const frame = trackFrameAt(track, capturedEpoch, {
      holdSeconds: 0,
      sampleFps,
      observationsOnly: true,
    });
    if (!frame) return [];
    const [x1, y1, x2, y2] = frame.box;
    return [{ ...track, x1, y1, x2, y2, path: frame.path }];
  });
}

export function snapshotTrackOverview(event, capturedEpoch) {
  const current = new Map(snapshotObjectTracks(event, capturedEpoch).map((track) => [track.trackId, track]));
  return incidentTrackFramesAt(storedObjectTracks(event), Number.MAX_VALUE).map((track) => {
    const visible = current.get(track.trackId);
    return {
      ...track,
      snapshotBox: visible ? [visible.x1, visible.y1, visible.x2, visible.y2] : null,
    };
  });
}

// Round corners within the adjacent segments' triangle. This avoids spline
// overshoot, preserves endpoints, and leaves observation gaps to the caller.
export function trackPathD(points, radius = 8) {
  const path = points.filter((point, index) => index === 0
    || point[0] !== points[index - 1][0] || point[1] !== points[index - 1][1]);
  if (!path.length) return "";
  const format = (point) => point.map((value) => Number(value.toFixed(3))).join(" ");
  let d = `M ${format(path[0])}`;
  for (let index = 1; index < path.length - 1; index += 1) {
    const previous = path[index - 1];
    const corner = path[index];
    const next = path[index + 1];
    const incoming = Math.hypot(corner[0] - previous[0], corner[1] - previous[1]);
    const outgoing = Math.hypot(next[0] - corner[0], next[1] - corner[1]);
    const distance = Math.min(Math.max(0, radius), incoming / 3, outgoing / 3);
    const before = corner.map((value, axis) => value + (previous[axis] - value) * distance / incoming);
    const after = corner.map((value, axis) => value + (next[axis] - value) * distance / outgoing);
    d += ` L ${format(before)} Q ${format(corner)} ${format(after)}`;
  }
  if (path.length > 1) d += ` L ${format(path.at(-1))}`;
  return d;
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
