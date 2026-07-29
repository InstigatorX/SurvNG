function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function eventEpoch(event) {
  const parsed = Date.parse(String(event?.created_at || ""));
  if (Number.isFinite(parsed)) return parsed / 1000;
  return finiteNumber(event?.start_epoch);
}

export function incidentTrackingSource(event, incident = null) {
  if (event?.object_tracking?.tracks?.length) return event;
  const incidentEvents = event?.events?.length ? event.events : incident?.events || [];
  const candidates = incidentEvents.filter(
    (candidate) => candidate?.object_tracking?.tracks?.length,
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
    const boxHistory = Array.isArray(track.box_history)
      ? track.box_history.flatMap((sample) => {
        if (!Array.isArray(sample) || sample.length < 5) return [];
        const values = sample.slice(0, 5).map(finiteNumber);
        if (values.some((value) => value === null) || values[3] <= values[1] || values[4] <= values[2]) return [];
        return [values];
      }).sort((left, right) => left[0] - right[0])
      : [];
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
      trajectory,
      boxHistory,
      recoveryHistory,
    }];
  }).filter((track) => track.x2 > track.x1 && track.y2 > track.y1);
}

export function trackFrameAt(track, epoch, holdSeconds = 1) {
  const samples = track?.boxHistory;
  if (!Array.isArray(samples) || !samples.length || !Number.isFinite(epoch)) return null;
  const first = samples[0];
  const last = samples[samples.length - 1];
  if (epoch < first[0] || epoch > last[0] + Math.max(0, Number(holdSeconds) || 0)) return null;
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
  const path = (track.trajectory || []).filter((point) => point[0] <= epoch).map((point) => point.slice(1, 3));
  const center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
  if (!path.length || path[path.length - 1][0] !== center[0] || path[path.length - 1][1] !== center[1]) path.push(center);
  const recovery = [...(track.recoveryHistory || [])]
    .reverse()
    .find((item) => item.capturedAt <= epoch && epoch - item.capturedAt <= Math.max(0.75, holdSeconds));
  return { box, path, recovery: recovery || null };
}

export function playbackEpochAt(windowStartEpoch, mediaTime, mediaStartTime) {
  const startEpoch = finiteNumber(windowStartEpoch);
  const currentTime = finiteNumber(mediaTime);
  const originTime = finiteNumber(mediaStartTime);
  if (startEpoch === null || currentTime === null || originTime === null) return null;
  return startEpoch + currentTime - originTime;
}
