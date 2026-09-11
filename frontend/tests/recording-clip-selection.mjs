import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
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

const source = readFileSync(new URL("../src/timeline/TimelinePages.jsx", import.meta.url), "utf8");
const endedHandler = source.slice(source.indexOf("  function handleRecordingEnded("), source.indexOf("  function handleRecordingSeeked("));
for (const useSegmentPlayback of [false, true]) {
  for (const previewComplete of [false, true]) {
    const expectedEpoch = useSegmentPlayback ? 100 : 200;
    const video = { currentTime: 10 };
    let continued = false;
    const context = vm.createContext({
      Number, videoRef: { current: video }, pendingSeekEpochRef: { current: null },
      useSegmentPlayback, nativeSegment: { start_epoch: 90 },
      mediaTimeToEpoch: (time) => 190 + time,
      finishClipPreviewAtEnd(element, epoch) {
        assert.equal(element, video);
        assert.equal(epoch, expectedEpoch);
        return clipPreviewReachedEnd(epoch, expectedEpoch + (previewComplete ? 0 : 10));
      },
      continueRecordingPlayback() { continued = true; },
    });
    vm.runInContext(endedHandler, context);
    context.handleRecordingEnded({ currentTarget: video });
    assert.equal(continued, !previewComplete, "continue across segments only while the selected preview is unfinished");
    continued = false;
    context.pendingSeekEpochRef.current = 500;
    context.handleRecordingEnded({ currentTarget: video });
    assert.equal(continued, false, "outgoing clip ending must not replace a pending seek");
  }
}

console.log("recording clip selection tests passed");
