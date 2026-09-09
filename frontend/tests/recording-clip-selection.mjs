import assert from "node:assert/strict";
import {
  CLIP_MINIMUM_DURATION_SECONDS,
  canSetClipBoundaryAtPlayhead,
  clipPreviewReachedEnd,
  clipRangeIsValid,
  setClipBoundaryAtPlayhead,
} from "../src/recordingClipSelection.mjs";

const range = { start: 120, end: 180 };
assert.equal(CLIP_MINIMUM_DURATION_SECONDS, 1);
assert.equal(clipRangeIsValid(range, 100, 200), true);
assert.equal(clipRangeIsValid({ start: null, end: 20 }, 0, 30), false);
assert.equal(clipRangeIsValid({ start: 0, end: 20 }, null, 30), false);
assert.equal(clipRangeIsValid({ start: 120, end: 120.5 }, 100, 200), false);
assert.equal(clipRangeIsValid({ start: 99, end: 180 }, 100, 200), false);
assert.equal(clipRangeIsValid({ start: 120, end: 201 }, 100, 200), false);

assert.deepEqual(
  setClipBoundaryAtPlayhead({ range, kind: "start", playhead: 150, startEpoch: 100, endEpoch: 200 }),
  { start: 150, end: 180 },
);
assert.deepEqual(
  setClipBoundaryAtPlayhead({ range, kind: "start", playhead: 179, startEpoch: 100, endEpoch: 200 }),
  { start: 179, end: 180 },
);
assert.deepEqual(
  setClipBoundaryAtPlayhead({ range, kind: "end", playhead: 121, startEpoch: 100, endEpoch: 200 }),
  { start: 120, end: 121 },
);
assert.deepEqual(
  setClipBoundaryAtPlayhead({ range, kind: "end", playhead: 999, startEpoch: 100, endEpoch: 200 }),
  range,
);
assert.equal(canSetClipBoundaryAtPlayhead({ range, kind: "start", playhead: 120, startEpoch: 100, endEpoch: 200 }), false);
assert.equal(canSetClipBoundaryAtPlayhead({ range, kind: "end", playhead: 150, startEpoch: 100, endEpoch: 200 }), true);
assert.deepEqual(
  setClipBoundaryAtPlayhead({ range, kind: "start", playhead: null, startEpoch: 100, endEpoch: 200 }),
  range,
);
assert.equal(canSetClipBoundaryAtPlayhead({ range, kind: "start", playhead: 180, startEpoch: 100, endEpoch: 200 }), false);
assert.equal(canSetClipBoundaryAtPlayhead({ range, kind: "end", playhead: 120, startEpoch: 100, endEpoch: 200 }), false);

assert.equal(clipPreviewReachedEnd(179.91, 180), false);
assert.equal(clipPreviewReachedEnd(179.92, 180), true);
assert.equal(clipPreviewReachedEnd(180.5, 180), true);
assert.equal(clipPreviewReachedEnd(null, 180), false);

console.log("recording clip selection tests passed");
