import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { recordingSegmentAt } from "../src/recordingPlayback.mjs";

const source = readFileSync(new URL("../src/timeline/TimelinePages.jsx", import.meta.url), "utf8");
function functionSource(name, nextName) {
  return source.slice(source.indexOf(`  function ${name}(`), source.indexOf(`  function ${nextName}(`));
}

// Crossing the detail-window boundary consumes already fetched rows, preserving
// the exact segment URL warmed by the second video instead of waiting for HTTP.
for (const autoplay of [true, false]) {
  const calls = [];
  const warmWindow = { start: 110, end: 1010, rows: [{ start_epoch: 110, end_epoch: 120 }] };
  const context = vm.createContext({
    Number, performance, recordingSegmentAt, snapToRecording: (value) => value,
    activeCameraId: "gate", isAllCameras: false, useSegmentPlayback: true,
    timelineView: { startEpoch: 0, endEpoch: 1010 },
    loadedPlaybackWindow: { start: 0, end: 110 }, playbackTimeline: [{ start_epoch: 100, end_epoch: 110 }],
    prefetchedNativeWindow: warmWindow, nativeSegment: { start_epoch: 100, end_epoch: 110 },
    windowAround: (start) => ({ start, end: start + 900 }), playbackRowsCoverEpoch: () => false,
    videoRef: { current: { paused: true, readyState: 4 } }, autoplayRef: { current: true },
    desiredEpochRef: {}, pendingSeekEpochRef: {}, pendingSeekModeRef: {}, playbackRequestRef: { current: 1 },
    setHeroSeeking() {}, setFollowTarget() {}, setPlaybackError() {}, setPlaybackErrorStage() {},
    setPlayhead() {}, setPlaybackWindow() {}, setPlaybackNotice() {}, clearSeekWatchdog() {},
    setHeroPlaying: (value) => calls.push(["playing", value]),
    setPlaybackDetail: (value) => calls.push(["detail", value]),
    setNativeSegment: (value) => calls.push(["segment", value]),
    requestPlaybackWindow: () => { throw new Error("Warm window must not be fetched again"); },
  });
  vm.runInContext(functionSource("playAt", "panTimelineViewport"), context);
  context.playAt(110.01, autoplay);
  assert.equal(calls.find(([name]) => name === "detail")[1].rows, warmWindow.rows);
  assert.equal(calls.find(([name]) => name === "segment")[1].start_epoch, 110);
  assert.equal(context.pendingSeekEpochRef.current, 110.01);
  assert.equal(context.autoplayRef.current, autoplay);
  if (!autoplay) assert.deepEqual(calls[0], ["playing", false]);
}

// A loading incoming video is paused even while the UI still intends to play.
// The Pause button must cancel that intent rather than issue another play().
const paused = [];
const toggle = vm.createContext({
  videoRef: { current: { paused: true, pause: () => paused.push("pause") } },
  useSegmentPlayback: true, heroSeeking: true, autoplayRef: { current: true },
  setHeroPlaying: (value) => paused.push(value), requestRecordingPlay: () => paused.push("play"),
});
vm.runInContext(functionSource("toggleHeroPlayback", "beginFrameSearch"), toggle);
toggle.toggleHeroPlayback();
assert.deepEqual(paused, [false, "pause"]);
assert.equal(toggle.autoplayRef.current, false);

// A watchdog belonging to the outgoing element cannot seek or complete the
// incoming clip, including when the same element has been recycled for a URL.
for (const recycle of [false, true]) {
  const callbacks = [];
  let url = "first.mp4";
  const video = { getAttribute: () => url, currentTime: 0 };
  const context = vm.createContext({
    Number, clearSeekWatchdog() {}, seekWatchdogRef: {}, pendingSeekEpochRef: { current: 110 },
    videoRef: { current: video }, recordingSeekToleranceSeconds: () => .1, seekWatchdogDelayMs: () => 1000,
    window: { setTimeout: (callback) => callbacks.push(callback) },
    videoReachedSeekTarget: () => false,
    completePendingNativeSeek: () => { throw new Error("Stale seek completed"); },
  });
  vm.runInContext(functionSource("scheduleNativeSeekWatchdog", "handleNativeSegmentMetadata"), context);
  context.scheduleNativeSeekWatchdog(video, 5);
  if (recycle) url = "second.mp4";
  else context.videoRef.current = {};
  callbacks[0]();
  assert.equal(video.currentTime, 0);
  assert.equal(callbacks.length, 1);
}

console.log("native Timeline handoff, pause intent, and stale seek tests passed");
