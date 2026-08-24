import assert from "node:assert/strict";
import {
  adjustRecordingExportRange,
  describePlaybackError,
  gridPlaybackNeedsSeek,
  isUnsupportedPlaybackError,
  mergeRecordingAvailability,
  playbackMediaTimeForEpoch,
  playbackRowsCoverEpoch,
  prefersJpegScrubPreview,
  scrubPreviewBucketSeconds,
  scrubPreviewDelayMs,
  seekVideoToTime,
  shouldResumePlaybackAfterSeek,
} from "../src/recordingPlayback.mjs";

const exportRange = { start: 100, end: 200 };
assert.deepEqual(adjustRecordingExportRange({ range: exportRange, kind: "start", key: "ArrowRight", startEpoch: 0, endEpoch: 300 }), { start: 101, end: 200 });
assert.deepEqual(adjustRecordingExportRange({ range: exportRange, kind: "end", key: "ArrowLeft", shiftKey: true, startEpoch: 0, endEpoch: 300 }), { start: 100, end: 140 });
assert.deepEqual(adjustRecordingExportRange({ range: exportRange, kind: "start", key: "End", startEpoch: 0, endEpoch: 300 }), { start: 199, end: 200 });
assert.deepEqual(adjustRecordingExportRange({ range: exportRange, kind: "end", key: "Home", startEpoch: 0, endEpoch: 300 }), { start: 100, end: 101 });
assert.equal(adjustRecordingExportRange({ range: exportRange, kind: "start", key: "Enter", startEpoch: 0, endEpoch: 300 }), null);

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
assert.equal(playbackMediaTimeForEpoch([
  { start_epoch: 100, end_epoch: 110, media_start: 0, media_end: 10 },
], 110.2, 0.25), 9.99);
assert.equal(playbackMediaTimeForEpoch([
  { start_epoch: 100, end_epoch: 110, media_start: 0, media_end: 10 },
], 110.3, 0.25), null);
assert.ok(Math.abs(playbackMediaTimeForEpoch([
  { start_epoch: 100, end_epoch: 110, media_start: 0, media_end: 10 },
  { start_epoch: 110, end_epoch: 120, media_start: 10, media_end: 20 },
], 110.2, 0.4) - 10.2) < 0.0001);
assert.equal(gridPlaybackNeedsSeek({ currentTime: 4, targetTime: 5.2, playing: true, epochDelta: 0.5 }), false);
assert.equal(gridPlaybackNeedsSeek({ currentTime: 1, targetTime: 5.2, playing: true, epochDelta: 0.5 }), false);
assert.equal(gridPlaybackNeedsSeek({ currentTime: 4, targetTime: 5.2, playing: true, epochDelta: 10 }), true);
assert.equal(gridPlaybackNeedsSeek({ currentTime: 5, targetTime: 5.2, playing: false, epochDelta: 0 }), true);

assert.equal(shouldResumePlaybackAfterSeek({ pendingSeekMode: "local", autoplay: true }), true);
assert.equal(shouldResumePlaybackAfterSeek({ pendingSeekMode: "window-ready", autoplay: true }), true);
assert.equal(shouldResumePlaybackAfterSeek({ pendingSeekMode: "local", autoplay: false }), false);
assert.equal(shouldResumePlaybackAfterSeek({ pendingSeekMode: "window", autoplay: true }), false);
assert.equal(shouldResumePlaybackAfterSeek({ pendingSeekMode: null, autoplay: true }), false);

assert.equal(prefersJpegScrubPreview({ preferNativeHls: true, coarsePointer: false }), true);
assert.equal(prefersJpegScrubPreview({ preferNativeHls: false, coarsePointer: true }), true);
assert.equal(prefersJpegScrubPreview({ preferNativeHls: false, coarsePointer: false }), false);
assert.equal(scrubPreviewDelayMs({ coarsePointer: true }), 450);
assert.equal(scrubPreviewDelayMs({ coarsePointer: false }), 250);
assert.equal(scrubPreviewBucketSeconds({ coarsePointer: true }), 10);
assert.equal(scrubPreviewBucketSeconds({ coarsePointer: false }), 5);

let fastSeekTime = null;
let currentTimeValue = 0;
seekVideoToTime({
  fastSeek(value) { fastSeekTime = value; },
  get currentTime() { return currentTimeValue; },
  set currentTime(value) { currentTimeValue = value; },
}, 12.5);
assert.equal(fastSeekTime, 12.5);
seekVideoToTime({ set currentTime(value) { currentTimeValue = value; } }, 3.25);
assert.equal(currentTimeValue, 3.25);

const mergedAvailability = mergeRecordingAvailability(
  [
    { camera_id: "gate", source: "live", start_epoch: 100, end_epoch: 110 },
    { camera_id: "garage", source: "live", start_epoch: 100, end_epoch: 110 },
  ],
  [
    { camera_id: "gate", source: "live", start_epoch: 109, end_epoch: 120 },
    { camera_id: "gate", source: "main", start_epoch: 109, end_epoch: 125 },
  ],
);
assert.equal(mergedAvailability.length, 3);
assert.deepEqual(
  mergedAvailability.find((item) => item.camera_id === "gate" && item.source === "live"),
  { camera_id: "gate", source: "live", start_epoch: 100, end_epoch: 120, duration_seconds: 20, segment_count: 0 },
);

assert.equal(describePlaybackError({ code: 1001, category: 1, message: "request failed" }), "code 1001 · category 1 · request failed");
assert.equal(describePlaybackError(null), "unknown media error");
assert.equal(isUnsupportedPlaybackError({ code: 4, message: "source unsupported" }), true);
assert.equal(isUnsupportedPlaybackError({ code: 1001, category: 1, message: "network request failed" }), false);

console.log("recording playback tests passed");
