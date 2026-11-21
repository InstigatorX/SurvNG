import assert from "node:assert/strict";
import { nativePlaylistStart, nativePlaylistStartStep } from "../src/recordingPlayback.mjs";

// Native HLS startup is owned by EXT-X-START. The shared step decides, per
// media event, whether to keep waiting, ask for play, correct once, or settle.
const video = (state) => ({ readyState: 4, seeking: false, paused: false, currentTime: 0, ...state });

{
  const start = nativePlaylistStart(5);
  assert.deepEqual(nativePlaylistStartStep(start, video({ readyState: 1 }), "loadedmetadata"), { action: "wait" }, "metadata alone cannot confirm the start");
  assert.deepEqual(nativePlaylistStartStep(start, video({ seeking: true }), "timeupdate"), { action: "wait" }, "an in-flight seek is left alone");
  assert.deepEqual(nativePlaylistStartStep(start, video({ currentTime: 5.05 }), "timeupdate"), { action: "settle", atTarget: true });
}
{
  const start = nativePlaylistStart(5);
  assert.deepEqual(nativePlaylistStartStep(start, video({ paused: true, readyState: 3, currentTime: 0 }), "canplay"), { action: "needs-play" }, "a paused wrong position needs play before it can be judged");
  assert.deepEqual(nativePlaylistStartStep(start, video({ paused: true, readyState: 2, currentTime: 0 }), "loadeddata"), { action: "wait" });
}
{
  // Wrong position: wait for playback to advance 0.1 s, correct once, then settle.
  const start = nativePlaylistStart(5);
  assert.equal(nativePlaylistStartStep(start, video({ currentTime: 0.2 }), "seeked").action, "wait", "a seeked event at the wrong place is not proof of playback");
  assert.equal(nativePlaylistStartStep(start, video({ currentTime: 0.2 }), "timeupdate").action, "wait");
  assert.equal(nativePlaylistStartStep(start, video({ currentTime: 0.25 }), "timeupdate").action, "wait", "less than 0.1 s of progress");
  assert.equal(nativePlaylistStartStep(start, video({ currentTime: 0.35 }), "timeupdate").action, "correct");
  assert.equal(start.corrected, true);
  assert.equal(nativePlaylistStartStep(start, video({ currentTime: 0.4 }), "timeupdate").action, "wait", "the correction gets its own settle window");
  assert.deepEqual(nativePlaylistStartStep(start, video({ currentTime: 0.6 }), "timeupdate"), { action: "settle", atTarget: false }, "never a second correction");
}
{
  const start = nativePlaylistStart(5);
  nativePlaylistStartStep(start, video({ currentTime: 2 }), "timeupdate");
  assert.equal(nativePlaylistStartStep(start, video({ currentTime: 1 }), "timeupdate").action, "wait", "time moving backwards restarts the progress check");
  assert.equal(start.firstWrongTime, 1);
}

console.log("Native playlist start tests passed");
