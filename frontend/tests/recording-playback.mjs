import assert from "node:assert/strict";
import {
  describePlaybackError,
  isUnsupportedPlaybackError,
  playbackMediaTimeForEpoch,
  playbackRowsCoverEpoch,
} from "../src/recordingPlayback.mjs";

const rows = [
  { start_epoch: 100, end_epoch: 110 },
  { start_epoch: 120, end_epoch: 130 },
];

assert.equal(playbackRowsCoverEpoch(rows, 105), true);
assert.equal(playbackRowsCoverEpoch(rows, 115), false);
assert.equal(playbackRowsCoverEpoch(rows, 130), false);
assert.equal(playbackRowsCoverEpoch([], 105), false);
assert.equal(playbackMediaTimeForEpoch([
  { start_epoch: 100, end_epoch: 110, media_start: 0, media_end: 10 },
  { start_epoch: 120, end_epoch: 130, media_start: 10, media_end: 20 },
], 125), 15);
assert.equal(playbackMediaTimeForEpoch(rows, 105), null);
assert.equal(playbackMediaTimeForEpoch([], 105), null);

assert.equal(describePlaybackError({ code: 1001, category: 1, message: "request failed" }), "code 1001 · category 1 · request failed");
assert.equal(describePlaybackError(null), "unknown media error");
assert.equal(isUnsupportedPlaybackError({ code: 4, message: "source unsupported" }), true);
assert.equal(isUnsupportedPlaybackError({ code: 1001, category: 1, message: "network request failed" }), false);

console.log("recording playback tests passed");
