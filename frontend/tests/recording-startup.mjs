import assert from "node:assert/strict";
import { recordingPlayableEpoch } from "../src/recordingPlayback.mjs";

const rows = [{ start_epoch: 100, end_epoch: 110 }, { start_epoch: 120, end_epoch: 130 }];
assert.equal(recordingPlayableEpoch(rows, 150), 129, "start inside finalized footage, not the open tail");
assert.equal(recordingPlayableEpoch(rows, 129.99), 129);
assert.equal(recordingPlayableEpoch(rows, 124.25), 124.25, "preserve explicit seeks within footage");
assert.equal(recordingPlayableEpoch(rows, 118), 120, "snap gaps to actual media");
assert.equal(recordingPlayableEpoch(rows, 99), 100);
assert.equal(recordingPlayableEpoch([{ start_epoch: 100, end_epoch: 100.4 }], 101), 100);
assert.equal(recordingPlayableEpoch([], 101), null);
assert.equal(recordingPlayableEpoch(rows, null), null);
console.log("recording startup and finalized-media positioning tests passed");
