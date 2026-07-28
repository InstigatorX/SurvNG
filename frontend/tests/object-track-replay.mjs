import assert from "node:assert/strict";
import { storedObjectTracks, trackFrameAt } from "../src/objectTrackReplay.mjs";

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
assert.deepEqual(trackFrameAt(tracks[0], 11.5, 1)?.box, [20, 10, 50, 80]);
assert.equal(trackFrameAt(tracks[0], 11.5, 1)?.recovery?.similarity, 0.91);
assert.equal(trackFrameAt(tracks[0], 12.1, 1), null);

assert.deepEqual(storedObjectTracks({ object_tracking: { tracks: [{ track_id: 1, label: "person", box: {} }] } }), []);

console.log("object track replay tests passed");
