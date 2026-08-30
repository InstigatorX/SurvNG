export const AI_DETECTION_SAMPLE_MS = 200;
export const AI_TRACK_PREDICTION_MS = 250;
export const AI_TRACK_FADE_START_MS = 500;
export const AI_TRACK_REMOVE_MS = 1000;

function validBox(box) {
  return box && [box.x1, box.y1, box.x2, box.y2].every((value) => Number.isFinite(Number(value)));
}

function copyBox(box) {
  return { x1: Number(box.x1), y1: Number(box.y1), x2: Number(box.x2), y2: Number(box.y2) };
}

export function debugDetectionIou(left, right) {
  const x1 = Math.max(Number(left.x1), Number(right.x1));
  const y1 = Math.max(Number(left.y1), Number(right.y1));
  const x2 = Math.min(Number(left.x2), Number(right.x2));
  const y2 = Math.min(Number(left.y2), Number(right.y2));
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const leftArea = Math.max(0, left.x2 - left.x1) * Math.max(0, left.y2 - left.y1);
  const rightArea = Math.max(0, right.x2 - right.x1) * Math.max(0, right.y2 - right.y1);
  return intersection / Math.max(1, leftArea + rightArea - intersection);
}

function boundedVelocity(previousBox, nextBox, elapsedMs) {
  const elapsed = Math.max(1, elapsedMs);
  const limit = 2;
  const velocityFor = (key) => Math.max(-limit, Math.min(limit, (nextBox[key] - previousBox[key]) / elapsed));
  return { x1: velocityFor("x1"), y1: velocityFor("y1"), x2: velocityFor("x2"), y2: velocityFor("y2") };
}

function predictedBox(track, now) {
  const age = Math.max(0, Math.min(AI_TRACK_PREDICTION_MS, now - track.seenAt));
  return Object.fromEntries(Object.entries(track.box).map(([key, value]) => [key, value + (track.velocity?.[key] || 0) * age]));
}

function boxCenterDistance(left, right) {
  return Math.hypot(
    (left.x1 + left.x2 - right.x1 - right.x2) / 2,
    (left.y1 + left.y2 - right.y1 - right.y2) / 2,
  );
}

function compatibleNearbyBox(predicted, candidate) {
  const predictedWidth = Math.max(1, predicted.x2 - predicted.x1);
  const predictedHeight = Math.max(1, predicted.y2 - predicted.y1);
  const candidateWidth = Math.max(1, candidate.x2 - candidate.x1);
  const candidateHeight = Math.max(1, candidate.y2 - candidate.y1);
  const areaRatio = (candidateWidth * candidateHeight) / (predictedWidth * predictedHeight);
  const maxDistance = Math.max(18, Math.min(80, Math.hypot(predictedWidth, predictedHeight) * 0.5));
  return areaRatio >= 0.55 && areaRatio <= 1.8 && boxCenterDistance(predicted, candidate) <= maxDistance;
}

function associationScore(track, detection, now) {
  if (track.label !== detection.label) return null;
  const predicted = predictedBox(track, now);
  const overlap = debugDetectionIou(predicted, detection.box);
  if (overlap >= 0.25) return 2 + overlap;
  if (!compatibleNearbyBox(predicted, detection.box)) return null;

  const predictedWidth = Math.max(1, predicted.x2 - predicted.x1);
  const predictedHeight = Math.max(1, predicted.y2 - predicted.y1);
  const distance = boxCenterDistance(predicted, detection.box);
  return 1 - distance / Math.max(18, Math.min(80, Math.hypot(predictedWidth, predictedHeight) * 0.5));
}

export function updateDebugDetectionTracks(existingTracks, detections, now, nextTrackId) {
  const available = existingTracks.filter((track) => now - track.seenAt < AI_TRACK_REMOVE_MS);
  const validDetections = detections.filter((object) => validBox(object?.box));
  const candidates = [];
  available.forEach((track, trackIndex) => {
    validDetections.forEach((detection, detectionIndex) => {
      const score = associationScore(track, detection, now);
      if (score !== null) candidates.push({ detectionIndex, score, trackIndex });
    });
  });
  candidates.sort((left, right) => right.score - left.score);

  const detectionMatches = new Map();
  const matchedTracks = new Set();
  candidates.forEach(({ detectionIndex, trackIndex }) => {
    if (detectionMatches.has(detectionIndex) || matchedTracks.has(trackIndex)) return;
    detectionMatches.set(detectionIndex, available[trackIndex]);
    matchedTracks.add(trackIndex);
  });

  return validDetections.slice(0, 40).map((object, detectionIndex) => {
    const previous = detectionMatches.get(detectionIndex) || null;
    const box = copyBox(object.box);
    const depthMeters = Number(object?.depth_stats?.median_m);
    return {
      id: previous?.id || nextTrackId(), label: object.label, confidence: Number(object.confidence) || 0, box,
      depthMeters: Number.isFinite(depthMeters) && depthMeters > 0 ? depthMeters : null,
      velocity: previous ? boundedVelocity(previous.box, box, now - previous.seenAt) : { x1: 0, y1: 0, x2: 0, y2: 0 },
      renderBox: previous?.renderBox || box, renderedAt: previous?.renderedAt ?? now, seenAt: now, opacity: 1,
    };
  });
}

export function advanceDebugDetectionTracks(tracks, now) {
  return tracks.flatMap((track) => {
    const age = now - track.seenAt;
    if (age >= AI_TRACK_REMOVE_MS) return [];
    const predictionAge = Math.max(0, Math.min(AI_TRACK_PREDICTION_MS, age));
    const target = Object.fromEntries(Object.entries(track.box).map(([key, value]) => [key, value + (track.velocity?.[key] || 0) * predictionAge]));
    const elapsed = Math.max(0, Math.min(100, now - (track.renderedAt ?? now)));
    const correction = 1 - Math.exp(-elapsed / 45);
    const rendered = track.renderBox || track.box;
    const renderBox = Object.fromEntries(Object.keys(target).map((key) => [key, rendered[key] + (target[key] - rendered[key]) * correction]));
    const opacity = age <= AI_TRACK_FADE_START_MS ? 1 : Math.max(0, 1 - (age - AI_TRACK_FADE_START_MS) / (AI_TRACK_REMOVE_MS - AI_TRACK_FADE_START_MS));
    return [{ ...track, renderBox, renderedAt: now, opacity }];
  });
}
