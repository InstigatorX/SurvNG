import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { recordingPlaybackTransport, seekVideoToTime, isRecordingCompatibilityError, describePlaybackError, recordingSegmentAt, playbackRowsCoverEpoch, videoReachedSeekTarget, recordingSeekToleranceSeconds } from "../src/recordingPlayback.mjs";

for (const error of [{ code: 3 }, { code: 4 }, { code: 4032, category: 4 }, { code: 3014, data: [3] }, { code: 3015, data: [{ name: "NotSupportedError" }] }, { code: 3016, data: [3] }, { code: 3016, data: [4] }, new Error("This browser does not support Shaka Player")]) {
  assert.equal(isRecordingCompatibilityError(error), true, JSON.stringify(error));
}
for (const error of [{ code: 2 }, { code: 4, category: 1 }, { code: 1001, data: [4] }, { code: 1002 }, { code: 3014, data: [2] }, { code: 3015, data: [{ name: "InvalidStateError" }] }, { code: 3016, data: [2, 4] }, { code: 6001, data: [4] }, { code: 7000 }, new Error("Network error")]) {
  assert.equal(isRecordingCompatibilityError(error), false, JSON.stringify(error));
}
const source = readFileSync(new URL("../src/timeline/TimelinePages.jsx", import.meta.url), "utf8");
const handler = source.slice(source.indexOf("  function handleRecordingError("), source.indexOf("  function retryRecordingPlayback("));
function fixture({ playing = true, pending = null, rows = true, transcoded = false, original = false } = {}) {
  const calls = [];
  const context = vm.createContext({
    Number, Math, console: { warn() {} }, isRecordingCompatibilityError, describePlaybackError, recordingSegmentAt,
    useTranscodedPlayback: transcoded, useSegmentPlayback: transcoded || original, nativeScope: "gate:main:100:900", activeCameraId: "gate", source: "main",
    originalFallbackRef: { current: original ? "gate:main:100:900" : null },
    transcodeFallbackRef: { current: transcoded ? "gate:main:100:900" : null },
    codecFallbackRef: { current: false }, availableSources: ["main", "live"],
    desiredEpochRef: { current: 105 }, pendingSeekEpochRef: { current: pending }, pendingSeekModeRef: {},
    autoplayRef: { current: playing }, playbackRetryRef: { current: { attempts: 0, timer: 9 } },
    playbackTimeline: rows ? [{ start_epoch: 100, end_epoch: 110 }] : [], hasPlaybackMedia: true,
    clearSeekWatchdog: () => calls.push(["clearSeek"]),
    setHeroSeeking: (x) => calls.push(["seeking", x]), setNativeSegment: (x) => calls.push(["segment", x]),
    setOriginalScope: (x) => calls.push(["originalScope", x]), setTranscodeScope: (x) => calls.push(["transcodeScope", x]), setPlaybackError() {}, setPlaybackErrorStage() {}, setPlaybackNotice() {},
    requestPlaybackWindow: (x) => calls.push(["window", x]), windowAround: (x) => ({ start: x - 10, end: x + 10 }),
    setSource: (x) => calls.push(["source", x]), setManifestRetryToken: () => calls.push(["hlsRetry"]),
    setNativeSegmentRetryToken: () => calls.push(["mp4Retry"]),
    window: { clearTimeout: (x) => calls.push(["clearTimer", x]), setTimeout: (callback) => { calls.push(["retry"]); callback(); } },
  });
  vm.runInContext(handler, context);
  return { context, calls };
}
for (const playing of [true, false]) {
  for (const pending of [null, 107.25]) {
    const { context, calls } = fixture({ playing, pending });
    context.handleRecordingError({ code: 4032 });
    assert.equal(context.autoplayRef.current, playing);
    assert.equal(context.pendingSeekEpochRef.current, pending ?? 105);
    assert.equal(context.pendingSeekModeRef.current, "native-ready");
    assert.equal(calls.find(([name]) => name === "segment")[1].start_epoch, 100);
    assert.equal(calls.find(([name]) => name === "originalScope")[1], "gate:main:100:900");
    assert.ok(!calls.some(([name]) => name === "source" || name === "window"));
    assert.equal(context.playbackRetryRef.current.attempts, 0);
    const count = calls.length;
    context.handleRecordingError({ code: 4 });
    assert.equal(calls.length, count, "duplicate outgoing errors must not start another handoff");
  }
}
{
  const { context, calls } = fixture({ rows: false, pending: 125 });
  context.handleRecordingError({ code: 3 });
  assert.equal(context.pendingSeekModeRef.current, "window");
  assert.equal(calls.find(([name]) => name === "window")[1].start, 115);
}
for (const error of [{ code: 1001, category: 1 }, { code: 4, category: 1 }]) {
  const { context, calls } = fixture();
  context.playbackRetryRef.current.timer = null;
  context.handleRecordingError(error);
  assert.ok(calls.some(([name]) => name === "hlsRetry"));
  assert.ok(!calls.some(([name]) => ["originalScope", "transcodeScope", "source", "mp4Retry"].includes(name)));
}
// Only an original-file decode failure may start the encoder. Its network
// failures retry the original-file player, and retained intent is unchanged.
for (const playing of [true, false]) {
  const { context, calls } = fixture({ original: true, playing });
  context.handleRecordingError({ code: 3 });
  assert.ok(calls.some(([name]) => name === "transcodeScope"));
  assert.ok(!calls.some(([name]) => name === "originalScope" || name === "source"));
  assert.equal(context.autoplayRef.current, playing);
  assert.equal(context.pendingSeekEpochRef.current, 105);
  const count = calls.length;
  context.handleRecordingError({ code: 3 });
  assert.equal(calls.length, count);
}
{
  const { context, calls } = fixture({ original: true });
  context.playbackRetryRef.current.timer = null;
  context.handleRecordingError({ code: 2 });
  assert.ok(calls.some(([name]) => name === "mp4Retry"));
  assert.ok(!calls.some(([name]) => name === "transcodeScope"));
}

// Browsers may report a missing MP4 as format error4. Check HTTP availability
// before codec escalation; ignore a response for a replaced source or camera.
const nativeErrorHandler = source.slice(source.indexOf("  async function handleNativeRecordingError("), source.indexOf("  function handleRecordingError("));
for (const result of [200, 404, 503, "offline"]) {
  for (const stale of [false, true]) {
    const calls = [];
    let src = "original.mp4";
    const video = { error: { code: 4 }, getAttribute: () => src };
    const context = vm.createContext({
      AbortController, videoRef: { current: video }, nativeScope: "gate", transport: "original",
      playbackTransportRef: { current: { scope: "gate", mode: "original" } },
      isRecordingCompatibilityError, handleRecordingError: (error) => calls.push(error),
      window: { setTimeout: () => 1, clearTimeout() {} },
      fetch: async (_url, options) => {
        assert.equal(options.headers.Range, "bytes=0-0");
        if (stale) src = "new.mp4";
        if (result === "offline") throw new Error("Network unavailable");
        return { ok: result === 200, status: result, body: { cancel: async () => {} } };
      },
    });
    vm.runInContext(nativeErrorHandler, context);
    await context.handleNativeRecordingError({ currentTarget: video });
    assert.equal(calls.length, stale ? 0 : 1);
    if (!stale) assert.equal(isRecordingCompatibilityError(calls[0]), result === 200);
  }
}

// HLS is the default; fast native playback copies original MP4, and only a
// format/decode failure opts into original files first. Network errors never encode.
for (const nativeHls of [true, false]) {
  for (const rate of [.5, 1, 2, 4]) {
    assert.equal(recordingPlaybackTransport({ nativeHls, rate }), nativeHls && rate > 2 ? "original" : "hls");
    assert.equal(recordingPlaybackTransport({ nativeHls, rate, preferOriginal: true }), "original");
    assert.equal(recordingPlaybackTransport({ nativeHls, rate, incompatible: true, preferOriginal: true }), "transcode");
  }
}

// Changing speed changes transport at the retained wall-clock position without
// changing play/pause intent. This also covers an in-flight seek and cold index.
const switchHandler = source.slice(source.indexOf("  function switchRecordingTransport("), source.indexOf("  useEffect(switchRecordingTransport,"));
for (const [transport, requestedTransport] of [["hls", "original"], ["original", "hls"], ["original", "transcode"], ["hls", "transcode"]]) {
  for (const playing of [false, true]) {
    for (const pending of [null, 107.25]) {
      for (const rows of [[], [{ start_epoch: 100, end_epoch: 110 }]]) {
        const calls = [];
        const context = vm.createContext({
          Number, transport, requestedTransport, nativeScope: "gate", playbackTransport: { scope: "gate", mode: transport },
          autoplayRef: { current: playing }, desiredEpochRef: { current: 105 }, pendingSeekEpochRef: { current: pending },
          pendingSeekModeRef: {}, playbackTimeline: rows, recordingSegmentAt, clearSeekWatchdog() {},
          playbackRetryRef: { current: { attempts: 2, timer: 5 } },
          window: { clearTimeout() {} }, setHeroSeeking() {}, setNativeSegment() {},
          setPlaybackTransport: (value) => calls.push(["transport", value]),
          requestPlaybackWindow: (value) => calls.push(["window", value]), windowAround: (epoch) => ({ start: epoch - 10 }),
        });
        vm.runInContext(switchHandler, context);
        context.switchRecordingTransport();
        assert.equal(context.pendingSeekEpochRef.current, pending ?? 105);
        assert.equal(context.autoplayRef.current, playing);
        assert.equal(calls[0][1].mode, requestedTransport);
        assert.equal(context.playbackRetryRef.current.attempts, 0);
        assert.equal(context.pendingSeekModeRef.current, !rows.length ? "window" : requestedTransport === "hls" ? "window-ready" : "native-ready");
        assert.equal(calls.some(([name]) => name === "window"), !rows.length);
      }
    }
  }
}
// A newly mounted HLS element for the old window cannot consume an in-flight
// seek to another window when returning from fast original playback.
const readyHandler = source.slice(source.indexOf("  function handleRecordingReady("), source.indexOf("  function handleRecordingTimeUpdate("));
{
  const video = { playbackRate: 1 };
  const context = vm.createContext({
    Number, videoRef: { current: video }, useSegmentPlayback: false, transport: "hls", requestedTransport: "hls",
    originalFallbackRef: {}, transcodeFallbackRef: {}, nativeScope: "gate", playbackRate: 2, normalizedTimelinePlaybackRate: (rate) => rate,
    playbackRetryRef: { current: { attempts: 0 } }, pendingSeekEpochRef: { current: 2005 },
    desiredEpochRef: { current: 2005 }, snapToRecording: (value) => value,
    loadedPlaybackWindow: { start: 1000, end: 1900 },
    playbackTimeline: [{ start_epoch: 1000, end_epoch: 1010 }], playbackRowsCoverEpoch: () => false,
    epochToPlaybackMediaTime: () => { throw new Error("Old window must not clamp pending seek"); },
  });
  vm.runInContext(readyHandler, context);
  context.handleRecordingReady(null, video);
  assert.equal(context.pendingSeekEpochRef.current, 2005);
  assert.equal(context.desiredEpochRef.current, 2005);
}

// Availability merges small gaps between files. A seek inside such a gap in
// the loaded window must snap to media and release subsequent playhead updates.
{
  const video = { playbackRate: 1, currentTime: 0, paused: true, fastSeek() { throw new Error("Native HLS must use an exact seek"); } };
  const playheads = [];
  const plays = [];
  const context = vm.createContext({
    Number, Math, performance, videoReachedSeekTarget, recordingSeekToleranceSeconds, nativeHls: true, videoRef: { current: video }, useSegmentPlayback: false,
    transport: "hls", requestedTransport: "hls", originalFallbackRef: {}, transcodeFallbackRef: {}, nativeScope: "gate",
    playbackRate: 1, normalizedTimelinePlaybackRate: (rate) => rate,
    playbackRetryRef: { current: { attempts: 0 } }, pendingSeekEpochRef: { current: 1005.1 },
    pendingSeekModeRef: { current: "window" }, desiredEpochRef: { current: 1005.1 },
    loadedPlaybackWindow: { start: 1000, end: 1900 },
    playbackTimeline: [
      { start_epoch: 1000, end_epoch: 1005, media_start: 0, media_end: 5 },
      { start_epoch: 1005.2, end_epoch: 1010, media_start: 5, media_end: 9.8 },
    ],
    playbackRowsCoverEpoch, snapToRecording: (value) => value,
    autoplayRef: { current: true }, ignorePauseUntilRef: {}, ignorePauseAfterSeekMs: () => 900,
    shouldResumePlaybackAfterSeek: ({ autoplay }) => autoplay,
    requestRecordingPlay: (element) => plays.push(element),
    seekVideoToTime,
    setPlayhead: (epoch) => playheads.push(epoch), setPlaybackNotice() {}, setPlaybackError() {},
    setPlaybackErrorStage() {}, setHeroSeeking() {}, clearSeekWatchdog() {}, scheduleSeekWatchdog() {},
  });
  const mappings = source.slice(source.indexOf("  function mediaTimeToEpoch("), source.indexOf("  function windowAround("));
  const completion = source.slice(source.indexOf("  function completePendingRecordingSeek("), source.indexOf("  function completePendingNativeSeek("));
  const timeUpdate = source.slice(source.indexOf("  function handleRecordingTimeUpdate("), source.indexOf("  function handleRecordingSeeked("));
  vm.runInContext(mappings + readyHandler + completion + timeUpdate, context);
  context.handleRecordingReady(null, video);
  assert.equal(context.pendingSeekModeRef.current, "window-ready", "a gap in the loaded window must not leave readiness waiting forever");
  assert.equal(video.currentTime, 5, "seek to the next playable segment boundary");
  video.seeking = true;
  context.completePendingRecordingSeek(video);
  assert.equal(context.pendingSeekModeRef.current, "window-ready", "watchdog must not declare a still-seeking native video ready");
  video.seeking = false;
  video.currentTime = 0;
  context.completePendingRecordingSeek(video);
  assert.equal(context.pendingSeekModeRef.current, "window-ready", "a late seeked event at the old position must not acknowledge the new seek");
  assert.equal(plays.length, 0, "do not resume the wrong footage");
  video.currentTime = 5;
  context.completePendingRecordingSeek(video);
  assert.equal(context.pendingSeekEpochRef.current, null);
  assert.equal(context.pendingSeekModeRef.current, null);
  assert.equal(context.desiredEpochRef.current, 1005.2);
  assert.equal(plays.length, 1, "preserve the scrub's playback intent");
  video.currentTime = 6;
  context.handleRecordingTimeUpdate({ currentTarget: video });
  assert.equal(playheads.at(-1), 1006.2, "playhead updates must resume after the gap seek");
}

// Subsequent scrubs within the current native HLS playlist must also bypass
// Safari fastSeek, not just the initial metadata/ready seek.
{
  const video = { currentTime: 0, paused: false, fastSeek() { throw new Error("Repeated native HLS scrub used fastSeek"); } };
  const context = vm.createContext({
    Number, nativeHls: true, useSegmentPlayback: false, isAllCameras: false, activeCameraId: "gate",
    timelineView: { startEpoch: 1000, endEpoch: 1900 }, loadedPlaybackWindow: { start: 1000, end: 1900 },
    playbackTimeline: [{ start_epoch: 1000, end_epoch: 1010 }], playbackRowsCoverEpoch,
    snapToRecording: (time) => time, windowAround: () => ({ start: 1000, end: 1900 }),
    videoRef: { current: video }, autoplayRef: {}, pendingSeekEpochRef: {}, pendingSeekModeRef: {}, desiredEpochRef: {},
    playbackRequestRef: { current: 0 }, epochToPlaybackMediaTime: (epoch) => epoch - 1000, seekVideoToTime,
    setHeroSeeking() {}, setFollowTarget() {}, setPlaybackError() {}, setPlaybackErrorStage() {}, setPlayhead() {},
    setPlaybackWindow() {}, setPlaybackNotice() {}, scheduleSeekWatchdog() {},
  });
  const playAt = source.slice(source.indexOf("  function playAt("), source.indexOf("  function panTimelineViewport("));
  vm.runInContext(playAt, context);
  for (const target of [1002, 1008, 1004]) {
    context.playAt(target, true);
    assert.equal(video.currentTime, target - 1000);
    assert.equal(context.pendingSeekModeRef.current, "local");
    assert.equal(context.pendingSeekEpochRef.current, target);
  }
}

// A fresh camera/day starts with its requested default rather than carrying a
// previous camera's codec fallback or transport, including initial ?speed=4.
for (const requestedTransport of ["hls", "original"]) {
  const calls = [];
  const context = vm.createContext({
    transport: requestedTransport, requestedTransport, nativeScope: "new-camera",
    playbackTransport: { scope: "old-camera", mode: "transcode" },
    setPlaybackTransport: (value) => calls.push(value),
  });
  vm.runInContext(switchHandler, context);
  context.switchRecordingTransport();
  assert.equal(calls[0].mode, requestedTransport);
  assert.equal(calls[0].scope, "new-camera");
}

// Original fast-play URLs cannot accidentally invoke the mobile encoder.
const urls = readFileSync(new URL("../src/shared/mediaUrls.js", import.meta.url), "utf8");
const segmentUrlFunction = urls.slice(urls.indexOf("export function recordingSegmentUrl("), urls.indexOf("export function recordingMobileWindowUrl(")) .replaceAll("export ", "");
const urlContext = vm.createContext({ URLSearchParams, appUrl: (value) => `/survng${value}` });
vm.runInContext(segmentUrlFunction, urlContext);
for (const transcode of [false, true]) {
  const url = new URL(urlContext.recordingSegmentUrl("gate", 107.25, "main", transcode), "http://nvr.test");
  assert.equal(url.pathname, "/survng/api/cameras/gate/recordings/segment.mp4");
  assert.equal(url.searchParams.get("mobile"), String(transcode));
  assert.equal(url.searchParams.get("epoch"), "107.250");
}
assert.match(urlContext.recordingMobileSegmentUrl("gate", 107.25, "main"), /mobile=true/);

// Timers from an outgoing HLS source cannot seek a replacement video or finish
// its pending MP4 seek, including a second watchdog tick already queued.
const watchdog = source.slice(source.indexOf("  function scheduleSeekWatchdog("), source.indexOf("  function completePendingRecordingSeek("));
for (const secondTick of [false, true]) {
  for (const recycle of [false, true]) {
    const timers = [];
    let src = "day.m3u8?reload=0";
    const video = { currentTime: 0, getAttribute: () => src };
    const context = vm.createContext({
      Number, videoRef: { current: video }, seekWatchdogRef: {}, clearSeekWatchdog() {},
      recordingSeekToleranceSeconds: () => .35, seekWatchdogDelayMs: () => 3000,
      pendingSeekEpochRef: { current: 125 }, pendingSeekModeRef: { current: "window-ready" },
      videoReachedSeekTarget: () => false, prefersJpegScrubPreview: () => true,
      completePendingRecordingSeek: () => { throw new Error("Stale HLS timer completed seek"); },
      window: { setTimeout: (fn) => { timers.push(fn); return timers.length; } },
    });
    vm.runInContext(watchdog, context);
    context.scheduleSeekWatchdog(video, 25);
    if (secondTick) timers.shift()();
    const timeBefore = video.currentTime;
    if (recycle) src = "day.m3u8?reload=1";
    else context.videoRef.current = { currentTime: 0 };
    timers.shift()();
    assert.equal(video.currentTime, timeBefore);
    assert.equal(timers.length, 0);
  }
}

console.log("HLS default, codec fallback, retained playback intent, and network retry tests passed");
