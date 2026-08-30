import assert from "node:assert/strict";
import { AI_TRACK_REMOVE_MS, advanceDebugDetectionTracks, debugDetectionIou, updateDebugDetectionTracks } from "../src/debugDetectionTracks.mjs";

assert.equal(debugDetectionIou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 5, y1: 0, x2: 15, y2: 10 }), 1 / 3);

let nextId = 1;
const allocateId = () => nextId++;
let tracks = updateDebugDetectionTracks([], [{ label: "person", confidence: 0.9, box: { x1: 0, y1: 0, x2: 10, y2: 20 } }], 0, allocateId);
assert.equal(tracks[0].id, 1);
tracks = updateDebugDetectionTracks(tracks, [{ label: "person", confidence: 0.8, box: { x1: 5, y1: 0, x2: 15, y2: 20 } }], 100, allocateId);
assert.equal(tracks[0].id, 1);
assert.equal(tracks[0].velocity.x1, 0.05, "velocity is derived from the two real measurements");

tracks = updateDebugDetectionTracks(tracks, [{ label: "person", confidence: 0.8, box: { x1: 20, y1: 0, x2: 30, y2: 20 } }], 300, allocateId);
assert.equal(tracks.length, 1, "a fast subject remains one local track when its predicted box overlaps");
assert.equal(tracks[0].id, 1);

tracks = updateDebugDetectionTracks(tracks, [{ label: "person", confidence: 0.8, box: { x1: 50, y1: 0, x2: 60, y2: 20 } }], 500, allocateId);
assert.equal(tracks.length, 1, "a single nearby same-label box reuses the local track when IoU is temporarily absent");
assert.equal(tracks[0].id, 1);

const duplicateTracks = [
  { ...tracks[0], id: 20, box: { x1: 40, y1: 0, x2: 50, y2: 20 }, renderBox: { x1: 40, y1: 0, x2: 50, y2: 20 }, velocity: { x1: 0, y1: 0, x2: 0, y2: 0 }, seenAt: 500 },
  { ...tracks[0], id: 21, box: { x1: 49, y1: 0, x2: 59, y2: 20 }, renderBox: { x1: 49, y1: 0, x2: 59, y2: 20 }, velocity: { x1: 0, y1: 0, x2: 0, y2: 0 }, seenAt: 500 },
];
const deduplicated = updateDebugDetectionTracks(duplicateTracks, [{ label: "person", confidence: 0.9, box: { x1: 51, y1: 0, x2: 61, y2: 20 } }], 700, allocateId);
assert.equal(deduplicated.length, 1, "a completed detector frame retires an unmatched ghost box");
assert.equal(deduplicated[0].id, 21, "the closest overlapping track wins when duplicate candidates exist");

const replaced = updateDebugDetectionTracks(deduplicated, [{ label: "person", confidence: 0.9, box: { x1: 300, y1: 0, x2: 310, y2: 20 } }], 900, allocateId);
assert.equal(replaced.length, 1, "a large detector jump replaces the old box instead of displaying both");
assert.notEqual(replaced[0].id, 21);

const twoPeople = updateDebugDetectionTracks([], [
  { label: "person", confidence: 0.9, box: { x1: 0, y1: 0, x2: 20, y2: 40 } },
  { label: "person", confidence: 0.9, box: { x1: 200, y1: 0, x2: 220, y2: 40 } },
], 1000, allocateId);
const twoPeopleMoved = updateDebugDetectionTracks(twoPeople, [
  { label: "person", confidence: 0.9, box: { x1: 205, y1: 0, x2: 225, y2: 40 } },
  { label: "person", confidence: 0.9, box: { x1: 5, y1: 0, x2: 25, y2: 40 } },
], 1200, allocateId);
assert.deepEqual(twoPeopleMoved.map((track) => track.id), [twoPeople[1].id, twoPeople[0].id], "global association does not depend on detector ordering");
assert.deepEqual(updateDebugDetectionTracks(twoPeopleMoved, [], 1400, allocateId), [], "an empty successful detector frame clears stale boxes");

let rendered = advanceDebugDetectionTracks(tracks, 600);
assert.ok(rendered[0].renderBox.x1 > 0 && rendered[0].renderBox.x1 < 200, "rendered box eases toward bounded prediction");
rendered = advanceDebugDetectionTracks(rendered, 1100);
assert.equal(rendered[0].opacity, 0.8, "missing tracks fade after a short hold");
assert.deepEqual(advanceDebugDetectionTracks(rendered, 500 + AI_TRACK_REMOVE_MS), [], "missing tracks are removed conservatively");

console.log("debug detection track smoothing tests passed");
