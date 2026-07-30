import assert from "node:assert/strict";
import { containedFrameTransform, incidentTrackingSource, playbackEpochAt, storedObjectTracks, trackFrameAt } from "../src/objectTrackReplay.mjs";

const tracks = storedObjectTracks({ object_tracking: { tracks: [{
  track_id: 7,
  label: "person",
  box: { x1: 20, y1: 10, x2: 50, y2: 80 },
  trajectory: [[11, 35, 45], [10, 25, 45], ["bad", 0, 0]],
  box_history: [[11, 20, 10, 50, 80], [10, 10, 10, 40, 80], [12, 5, 5, 4, 8]],
  reid_recovery_history: [
    { captured_at: 11, similarity: 0.91, resumed_completed_track: true, box: [20, 10, 50, 80] },
    { captured_at: "bad", similarity: 1, box: [0, 0, 1, 1] },
  ],
}] } });

assert.equal(tracks.length, 1);
assert.deepEqual(tracks[0].trajectory, [[10, 25, 45], [11, 35, 45]]);
assert.deepEqual(tracks[0].boxHistory, [[10, 10, 10, 40, 80], [11, 20, 10, 50, 80]]);
assert.deepEqual(tracks[0].recoveryHistory, [{ capturedAt: 11, similarity: 0.91, resumedCompletedTrack: true, box: [20, 10, 50, 80] }]);
assert.equal(trackFrameAt(tracks[0], 9.9), null);
assert.deepEqual(trackFrameAt(tracks[0], 10.5)?.box, [15, 10, 45, 80]);
assert.deepEqual(trackFrameAt(tracks[0], 10.5)?.path, [[25, 45], [30, 45]]);
assert.equal(trackFrameAt(tracks[0], 10.5)?.estimated, false);
assert.deepEqual(trackFrameAt(tracks[0], 11.5, { holdSeconds: 1, sampleFps: 2 })?.box, [20, 10, 50, 80]);
assert.equal(trackFrameAt(tracks[0], 11.5, { holdSeconds: 1, sampleFps: 2 })?.estimated, true);
assert.equal(trackFrameAt(tracks[0], 11.5, { holdSeconds: 1, sampleFps: 2 })?.recovery?.similarity, 0.91);
assert.equal(trackFrameAt(tracks[0], 12.1, { holdSeconds: 1, sampleFps: 2 }), null);

const sparseTrack = storedObjectTracks({ object_tracking: { tracks: [{
  track_id: 8,
  label: "car",
  box: { x1: 40, y1: 10, x2: 70, y2: 80 },
  trajectory: [[10, 25, 45], [14, 55, 45]],
  box_history: [[10, 10, 10, 40, 80], [14, 40, 10, 70, 80]],
}] } })[0];
assert.equal(trackFrameAt(sparseTrack, 10.5, { holdSeconds: 3, sampleFps: 2 })?.estimated, false);
assert.equal(trackFrameAt(sparseTrack, 12, { holdSeconds: 3, sampleFps: 2 })?.estimated, true);
assert.deepEqual(trackFrameAt(sparseTrack, 12, { holdSeconds: 3, sampleFps: 2 })?.box, [25, 10, 55, 80]);

assert.deepEqual(storedObjectTracks({ object_tracking: { tracks: [{ track_id: 1, label: "person", box: {} }] } }), []);

const earlierTracked = {
  id: 10,
  created_at: "2026-07-28T22:36:29+00:00",
  object_tracking: { tracks: [{ track_id: 1 }] },
};
const laterTracked = {
  id: 12,
  created_at: "2026-07-28T22:36:49+00:00",
  object_tracking: { tracks: [{ track_id: 2 }] },
};
const selectedChild = {
  id: 11,
  created_at: "2026-07-28T22:36:31+00:00",
  start_epoch: Date.parse("2026-07-28T22:30:00+00:00") / 1000,
  events: [laterTracked, earlierTracked],
};
assert.equal(incidentTrackingSource(selectedChild)?.id, 10);
assert.equal(incidentTrackingSource({
  ...selectedChild,
  created_at: "2026-07-28T22:36:47+00:00",
})?.id, 12);
assert.equal(incidentTrackingSource(earlierTracked), earlierTracked);
assert.equal(incidentTrackingSource({ id: 13, events: [] }), null);
assert.equal(
  incidentTrackingSource({ id: 14, events: [] }, { events: [earlierTracked] })?.id,
  10,
);

assert.equal(playbackEpochAt(1000, 8, 8), 1000);
assert.equal(playbackEpochAt(1000, 29, 8), 1021);
assert.equal(playbackEpochAt(1000, 0, 0), 1000);
assert.equal(playbackEpochAt(1000, "bad", 0), null);

assert.deepEqual(containedFrameTransform({ width: 1600, height: 900 }, { width: 1920, height: 1080 }), {
  x: 0,
  y: 0,
  width: 1600,
  height: 900,
  scale: 5 / 6,
});
assert.deepEqual(containedFrameTransform({ width: 1600, height: 900 }, { width: 1920, height: 2560 }), {
  x: 462.5,
  y: 0,
  width: 675,
  height: 900,
  scale: 0.3515625,
});
assert.equal(containedFrameTransform({ width: 0, height: 900 }, { width: 1920, height: 1080 }), null);

console.log("object track replay tests passed");
