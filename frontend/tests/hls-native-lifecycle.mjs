import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../src/shared/RecordingHlsVideo.jsx", import.meta.url), "utf8");
const helper = source.slice(
  source.indexOf("function seekableContains("),
  source.indexOf("// Let Safari own the playlist"),
);
const effect = source.slice(source.indexOf("  useEffect(() => {"), source.indexOf("\n\n  return nativeHls"));
const listeners = new Map();
const loads = [];
const ready = [];
const errors = [];
const requests = [];
const video = {
  src: null, readyState: 0, error: null, seeking: false, currentTime: 0,
  seekable: { length: 0, start() { return 0; }, end() { return 0; } },
  addEventListener(name, handler) { listeners.set(name, handler); },
  removeEventListener(name, handler) { if (listeners.get(name) === handler) listeners.delete(name); },
  getAttribute() { return this.src; },
  setAttribute(_name, value) { this.src = value; },
  removeAttribute() { this.src = null; },
  pause() {},
  load() { loads.push(this.src); this.readyState = 0; this.error = null; this.seeking = false; this.currentTime = 0; },
};
let cleanup;
const context = vm.createContext({
  nativeHls: true, videoRef: { current: video }, src: "a.m3u8", AbortController,
  callbacks: { current: {} }, window: { setTimeout, clearTimeout },
  useEffect(run) { cleanup = run(); },
  fetch(url, options) { return new Promise((resolve) => requests.push({ url, options, resolve })); },
});
vm.runInContext(helper, context);
function render(src, startTime = null) {
  context.src = src;
  context.callbacks.current = {
    src,
    startTime,
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

// Native Safari must wait for a seekable range before applying startTime, then
// report ready at the requested media offset instead of t=0.
ready.length = 0;
loads.length = 0;
render("window.m3u8", 42.5);
mount();
video.seekable = { length: 0, start() { return 0; }, end() { return 0; } };
metadata();
assert.equal(ready.length, 0, "do not finish ready before the target is seekable");
assert.equal(video.currentTime, 0);
assert.equal(listeners.has("progress"), true);
video.seekable = {
  length: 1,
  start() { return 0; },
  end() { return 90; },
};
video.readyState = 2;
listeners.get("progress")();
assert.equal(video.currentTime, 42.5);
assert.deepEqual(ready.at(-1), ["window.m3u8", video, 42.5]);
assert.equal(listeners.has("progress"), false, "start watchers release after ready");
cleanup();

console.log("native HLS source lifecycle and stale callback tests passed");
