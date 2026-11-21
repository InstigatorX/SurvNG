import assert from "node:assert/strict";
import {
  allCamerasGridColumns,
  recordingGridBestEpoch,
} from "../src/recordingGrid.mjs";

assert.deepEqual([0, 1, 2, 3, 4, 5, 9, 10, 16].map(allCamerasGridColumns), [1, 1, 2, 2, 2, 3, 3, 4, 4], "the grid stays roughly square");

const ranges = [
  { camera_id: "ahead", start_epoch: 980, end_epoch: 1000 },
  { camera_id: "one", start_epoch: 900, end_epoch: 990 },
  { camera_id: "two", start_epoch: 910, end_epoch: 990 },
  { camera_id: "three", start_epoch: 920, end_epoch: 988 },
];
assert.equal(recordingGridBestEpoch(ranges, 999), 986);
assert.equal(recordingGridBestEpoch([], 999), null);
assert.equal(recordingGridBestEpoch(ranges, Number.NaN), null);

console.log("recording grid tests passed");
