import assert from "node:assert/strict";
import {
  recordingSegmentAt,
  recordingSegmentLocalTime,
  recordingEpochAfterSegment,
  recordingEdgeSeekEpoch,
  recordingLiveEdgeFollow,
  extendedPlaybackRows,
  adjustRecordingExportRange,
  describePlaybackError,
  gridPlaybackNeedsSeek,
  isUnsupportedPlaybackError,
  mergeRecordingAvailability,
  recordingCoverageUnion,
  playbackMediaTimeForEpoch,
  playbackRowsCoverEpoch,
  prefersJpegScrubPreview,
  recordingSeekToleranceSeconds,
  scrubPreviewBucketSeconds,
  scrubPreviewDelayMs,
  seekVideoToTime,
  seekWatchdogDelayMs,
  shouldResumePlaybackAfterSeek,
  shouldUseFastSeek,
  videoReachedSeekTarget,
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
assert.equal(gridPlaybackNeedsSeek({ currentTime: 3.4, targetTime: 5.2, playing: true, epochDelta: 0.5 }), false, "normal lag on a running clock is left alone");
assert.equal(gridPlaybackNeedsSeek({ currentTime: 1, targetTime: 5.2, playing: true, epochDelta: 0.5 }), true, "a stalled tile is pulled back to the clock");
assert.equal(gridPlaybackNeedsSeek({ currentTime: 1.5, targetTime: 5.2, playing: true, epochDelta: 2 }), false, "a 4x clock steps 2 s per tick and tolerates more lag");
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
assert.equal(shouldUseFastSeek({ coarsePointer: true }), false);
assert.equal(shouldUseFastSeek({ preferNativeHls: true }), false);
assert.equal(shouldUseFastSeek({ coarsePointer: false, preferNativeHls: false }), true);
assert.equal(recordingSeekToleranceSeconds({ coarsePointer: true }), 0.35);
assert.equal(seekWatchdogDelayMs({ coarsePointer: true }), 3000);
assert.equal(seekWatchdogDelayMs({ local: true, coarsePointer: true }), 400);
assert.equal(videoReachedSeekTarget({ currentTime: 4.92 }, 5, 0.35), true);
assert.equal(videoReachedSeekTarget({ currentTime: 4.2 }, 5, 0.35), false);

let fastSeekTime = null;
let currentTimeValue = 0;
seekVideoToTime({
  fastSeek(value) { fastSeekTime = value; },
  get currentTime() { return currentTimeValue; },
  set currentTime(value) { currentTimeValue = value; },
}, 12.5, { coarsePointer: false, preferNativeHls: false });
assert.equal(fastSeekTime, 12.5);
fastSeekTime = null;
currentTimeValue = 0;
seekVideoToTime({
  fastSeek(value) { fastSeekTime = value; },
  set currentTime(value) { currentTimeValue = value; },
}, 4.5, { coarsePointer: true });
assert.equal(fastSeekTime, null);
assert.equal(currentTimeValue, 4.5);

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

// A mobile source is one real recording segment, including leading/interior gaps.
const mobileRows = [{ start_epoch: 150, end_epoch: 180 }, { start_epoch: 200, end_epoch: 210 }];
const mobileSegment = recordingSegmentAt(mobileRows, 155);
assert.deepEqual(mobileSegment, { start_epoch: 150, end_epoch: 180 });
assert.equal(recordingSegmentLocalTime(mobileSegment, 155, { duration: 30 }), 5);
assert.equal(recordingSegmentAt(mobileRows, 190), null);
assert.equal(recordingEpochAfterSegment(mobileSegment, mobileRows), 200);
assert.equal(recordingEpochAfterSegment(mobileRows[1], mobileRows), null);
assert.equal(recordingEpochAfterSegment(mobileRows[1], [...mobileRows, { start_epoch: 210, end_epoch: 220 }]), 210.01);

// Live edge: a new recording is listed with its nominal 10 s until its exact
// length (9.964 s here) is read, so availability ends a sliver after the
// segment that just finished. That sliver is not footage to continue into:
// seeking to it snaps back and replays the last second, over and over.
{
  const played = { start_epoch: 1000, end_epoch: 1009.964 };
  assert.equal(recordingEpochAfterSegment(played, [{ start_epoch: 400, end_epoch: 1010 }]), null,
    "a nominal end just past the played segment is not a next recording");
  assert.equal(recordingEpochAfterSegment(played, [{ start_epoch: 400, end_epoch: 1020 }]), 1009.974,
    "the next recording continues right after the played one");
  assert.equal(
    recordingEpochAfterSegment(played, [{ start_epoch: 400, end_epoch: 1010 }, { start_epoch: 1300, end_epoch: 1310 }]),
    1300,
    "a sliver at the end of a range is skipped for the next range",
  );
}

// Growing playlist opt-in, remembered like the transport override.
{
  const store = new Map();
  const win = (search) => ({
    location: { search },
    localStorage: {
      getItem: (key) => (store.has(key) ? store.get(key) : null),
      setItem: (key, value) => store.set(key, String(value)),
      removeItem: (key) => store.delete(key),
    },
  });
  assert.equal(recordingLiveEdgeFollow(win("")), true, "on unless turned off");
  assert.equal(recordingLiveEdgeFollow(win("?camera=foyer&liveedge=0")), false);
  assert.equal(recordingLiveEdgeFollow(win("?camera=foyer")), false, "remembered without the parameter");
  assert.equal(recordingLiveEdgeFollow(win("?liveedge=1")), true);
  assert.equal(recordingLiveEdgeFollow(win("")), true, "back on");
  assert.equal(recordingLiveEdgeFollow({ location: { search: "?liveedge=0" }, get localStorage() { throw new Error("blocked"); } }), false);
  assert.equal(recordingLiveEdgeFollow({ location: { search: "" }, get localStorage() { throw new Error("blocked"); } }), true);
  assert.equal(recordingLiveEdgeFollow(null), false);
}

// A growing playlist's rows are only ever extended in place.
{
  const loaded = [{ start_epoch: 100, duration_seconds: 9.964 }, { start_epoch: 110, duration_seconds: 9.964 }];
  const longer = [...loaded.map((row) => ({ ...row })), { start_epoch: 120, duration_seconds: 9.964 }];
  assert.equal(extendedPlaybackRows(loaded, longer), longer);
  assert.equal(extendedPlaybackRows(loaded, loaded.map((row) => ({ ...row }))), null, "nothing new");
  assert.equal(
    extendedPlaybackRows(loaded, [loaded[0], { start_epoch: 110, duration_seconds: 10 }, longer[2]]),
    null,
    "an earlier row changed: media times would move",
  );
  assert.equal(extendedPlaybackRows(null, longer), null);
}

// All cameras: the rail draws the union of every camera's (and source's) ranges.
assert.deepEqual(recordingCoverageUnion([
  { camera_id: "b", source: "main", start_epoch: 100, end_epoch: 160 },
  { camera_id: "a", source: "live", start_epoch: 100, end_epoch: 130 },
  { camera_id: "a", source: "main", start_epoch: 140, end_epoch: 150 },
  { camera_id: "c", source: "live", start_epoch: 160, end_epoch: 170 },
  { camera_id: "a", source: "live", start_epoch: 200, end_epoch: 210 },
  { camera_id: "a", source: "live", start_epoch: "bad", end_epoch: 210 },
]), [{ start_epoch: 100, end_epoch: 170 }, { start_epoch: 200, end_epoch: 210 }], "overlapping and touching ranges merge; real gaps stay");
assert.deepEqual(recordingCoverageUnion(null), []);

// A seek to the end of footage still being written starts a few segments back,
// so native HLS has media ahead of it; older footage keeps the one-second snap.
assert.equal(recordingEdgeSeekEpoch({ start_epoch: 1000, end_epoch: 2000 }, 2010, false, true), 1970);
assert.equal(recordingEdgeSeekEpoch({ start_epoch: 1990, end_epoch: 2000 }, 2010, false, true), 1990);
assert.equal(recordingEdgeSeekEpoch({ start_epoch: 1000, end_epoch: 2000 }, 2200, false, true), 1999);
// Playback continuing past what it just played is never moved back.
assert.equal(recordingEdgeSeekEpoch({ start_epoch: 1000, end_epoch: 2000 }, 2010, true, true), 1999);
// A player that loads a finished playlist (Shaka, per-file MP4, live edge
// turned off) keeps the one-second snap: starting 30 s back buys it nothing.
assert.equal(recordingEdgeSeekEpoch({ start_epoch: 1000, end_epoch: 2000 }, 2010), 1999);
assert.equal(recordingEdgeSeekEpoch({ start_epoch: 1000, end_epoch: 2000 }, 2010, false, false), 1999);
