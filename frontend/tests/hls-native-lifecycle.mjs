import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { nativePlaylistStart, nativePlaylistStartStep } from "../src/recordingPlayback.mjs";

const source = readFileSync(new URL("../src/shared/RecordingHlsVideo.jsx", import.meta.url), "utf8");
const constants = source.slice(source.indexOf("const NATIVE_START_EVENTS"), source.indexOf("// Let Safari own the playlist"))
  + source.slice(source.indexOf("const NATIVE_START_WATCHDOG_MS"), source.indexOf("export const RecordingHlsVideo"));
const effectStart = source.indexOf("  useEffect(() => {\n    if (!nativeHls) return undefined;");
const effect = source.slice(effectStart, source.indexOf("\n\n  if (!nativeHls) {"));
const listeners = new Map();
const loads = [];
const ready = [];
const errors = [];
const requests = [];
const starting = [];
const timers = [];
const corrections = [];
const needsPlay = [];
const video = {
  src: null, readyState: 0, error: null, seeking: false, currentTime: 0, paused: true, muted: false,
  rejectPlay: false,
  play() {
    if (this.rejectPlay) return Promise.reject(new Error("NotAllowedError"));
    this.paused = false;
    return Promise.resolve();
  },
  seekable: { length: 0, start() { return 0; }, end() { return 0; } },
  addEventListener(name, handler) { listeners.set(name, handler); },
  removeEventListener(name, handler) { if (listeners.get(name) === handler) listeners.delete(name); },
  dispatch(name) { listeners.get(name)?.({ type: name }); },
  getAttribute() { return this.src; },
  setAttribute(_name, value) { this.src = value; },
  removeAttribute() { this.src = null; },
  pause() {},
  load() { loads.push(this.src); this.readyState = 0; this.error = null; this.seeking = false; this.currentTime = 0; },
};
let cleanup;
const context = vm.createContext({
  nativeHls: true, videoRef: { current: video }, src: "a.m3u8", AbortController,
  callbacks: { current: {} }, console,
  // Controllable timers: the startup watchdog is fired explicitly below.
  window: { setTimeout: (fn, ms) => { timers.push({ fn, ms, cleared: false }); return timers.length; }, clearTimeout: (id) => { if (timers[id - 1]) timers[id - 1].cleared = true; } },
  nativePlaylistStart, nativePlaylistStartStep,
  seekVideoToTime(element, time) { corrections.push(time); element.currentTime = time; },
  setStartingSrc: (value) => starting.push(value),
  useEffect(run) { cleanup = run(); },
  fetch(url, options) { return new Promise((resolve) => requests.push({ url, options, resolve })); },
});
vm.runInContext(constants, context);
function render(src, startTime = null, options = {}) {
  context.src = src;
  context.callbacks.current = {
    src,
    startTime,
    muted: Boolean(options.muted),
    autoPlay: options.autoPlay !== false,
    onStartNeedsPlay: (element) => needsPlay.push(element.src),
    onReady: (_player, element) => ready.push([src, element, element.currentTime]),
    onError: (error) => errors.push(error),
  };
}
function mount() { vm.runInContext(`(function(src) { ${effect} })(src)`, context); }
function metadata() { video.readyState = 1; listeners.get("loadedmetadata")(); }

render("a.m3u8");
mount();
metadata();
assert.deepEqual(ready, [["a.m3u8", video, 0]]);
video.error = { code: 4 };
const oldFailure = listeners.get("error")();
const oldReady = listeners.get("loadedmetadata");
render("b.m3u8");
oldReady();
assert.equal(ready.length, 1, "outgoing metadata must not seek the newly requested window");
video.seeking = true;
cleanup();
assert.equal(requests[0].options.signal.aborted, true);
mount();
assert.equal(video.seeking, false, "changing windows releases the pending media seek");
assert.deepEqual(loads, ["a.m3u8", null, "b.m3u8"], "release old media before loading the next playlist on the same element");
listeners.get("loadedmetadata")();
assert.equal(ready.length, 1, "queued old metadata while the new resource is empty must not report ready");
requests[0].resolve({ ok: true, text: async () => "#EXTM3U" });
await oldFailure;
assert.equal(errors.length, 0, "an outgoing format probe must not trigger the new window's codec fallback");
metadata();
assert.deepEqual(ready.at(-1), ["b.m3u8", video, 0]);

for (const src of ["a.m3u8", "b.m3u8", "a.m3u8"]) {
  render(src);
  cleanup();
  mount();
  metadata();
  assert.deepEqual(ready.at(-1), [src, video, 0]);
}
// Development effect replay must reload the same source after cleanup too.
cleanup();
mount();
metadata();
assert.equal(video.src, "a.m3u8");
assert.equal(ready.length, 6);
cleanup();
render("");
mount();
assert.equal(video.src, null, "an empty camera transition must not request the document as media");
metadata();
assert.equal(ready.length, 6);
cleanup();
assert.equal(listeners.size, 0);
assert.equal(video.src, null);

// Native startup at an offset follows the playlist's EXT-X-START. It never
// writes currentTime while Safari loads, stays muted until the position
// settles, and reports ready only for the settled element.
ready.length = 0;
render("window.m3u8", 42.5);
mount();
assert.equal(video.muted, true, "the lead-in plays muted");
metadata();
assert.equal(ready.length, 0, "metadata alone cannot confirm the start");
assert.equal(video.currentTime, 0, "startup does not write currentTime");
assert.equal(video.paused, false, "autoplay starts so the native player presents frames");
video.readyState = 4;
video.currentTime = 42.5;
video.dispatch("canplay");
assert.deepEqual(ready.at(-1), ["window.m3u8", video, 42.5], "ready reports the settled position");
assert.deepEqual(corrections, []);
assert.equal(video.muted, false, "the caller's mute setting returns once settled");
assert.deepEqual(starting.slice(-2), ["window.m3u8", null], "the video is hidden only while this load's start settles");
assert.equal(listeners.has("timeupdate"), false, "start watchers release after ready");
cleanup();

// A start that plays from the wrong place is corrected exactly once.
ready.length = 0;
video.readyState = 0;
video.paused = true;
render("wrong.m3u8", 12, { muted: true });
mount();
metadata();
video.readyState = 4;
video.currentTime = 0.2;
video.dispatch("seeked");
video.dispatch("timeupdate");
video.currentTime = 0.4;
video.dispatch("timeupdate");
assert.deepEqual(corrections, [12], "one correction after playback advanced from the wrong position");
assert.equal(ready.length, 0, "the correction settles before ready");
video.dispatch("seeked");
assert.deepEqual(ready.at(-1), ["wrong.m3u8", video, 12]);
assert.equal(video.muted, true, "a muted caller stays muted");
cleanup();

// After a correction, a start Safari paused resumes (Timeline's rule).
ready.length = 0;
video.readyState = 0;
video.paused = true;
video.currentTime = 0;
render("resume.m3u8", 8);
mount();
metadata();
video.readyState = 4;
video.currentTime = 0.2;
video.dispatch("timeupdate");
video.currentTime = 0.4;
video.dispatch("timeupdate");
assert.deepEqual(corrections.at(-1), 8);
video.paused = true;
video.dispatch("seeked");
assert.deepEqual(ready.at(-1), ["resume.m3u8", video, 8]);
assert.equal(video.paused, false, "a corrected, on-target start that Safari paused resumes");
cleanup();

// A start that never settles is not left hidden forever.
const watchdog = () => timers.filter((timer) => timer.ms === 4000 && !timer.cleared).at(-1);
ready.length = 0;
video.readyState = 0;
video.paused = true;
video.currentTime = 0;
render("stuck-playing.m3u8", 20);
mount();
metadata();
video.readyState = 4;
video.seeking = true;
watchdog().fn();
assert.deepEqual(ready.at(-1), ["stuck-playing.m3u8", video, 0], "a playing start stuck seeking is revealed where it is");
assert.equal(starting.at(-1), null);
cleanup();
video.seeking = false;

ready.length = 0;
needsPlay.length = 0;
video.readyState = 0;
video.paused = true;
render("stuck-paused.m3u8", 20, { autoPlay: false });
mount();
metadata();
watchdog().fn();
assert.deepEqual(needsPlay, ["stuck-paused.m3u8"], "a paused start that never settles asks for play");
assert.equal(ready.length, 0);
cleanup();

// Autoplay refused (for example Low Power Mode): the caller is asked to play.
needsPlay.length = 0;
video.readyState = 0;
video.paused = true;
video.rejectPlay = true;
render("refused.m3u8", 20);
mount();
metadata();
await Promise.resolve();
await Promise.resolve();
assert.deepEqual(needsPlay, ["refused.m3u8"], "a rejected play() is reported, not swallowed");
video.rejectPlay = false;
cleanup();

// A paused wrong position cannot be judged; the caller is asked to play.
needsPlay.length = 0;
ready.length = 0;
video.readyState = 0;
video.paused = true;
video.currentTime = 0;
render("paused.m3u8", 30, { autoPlay: false });
mount();
metadata();
assert.equal(video.paused, true, "no autoplay was requested");
video.readyState = 3;
video.dispatch("canplay");
assert.deepEqual(needsPlay, ["paused.m3u8"]);
assert.equal(ready.length, 0);
cleanup();

// Without an offset, ready still follows metadata (Timeline owns its start).
ready.length = 0;
video.readyState = 0;
render("plain.m3u8");
mount();
metadata();
assert.deepEqual(ready.at(-1), ["plain.m3u8", video, 0]);
cleanup();

console.log("native HLS source lifecycle and stale callback tests passed");
