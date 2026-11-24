import assert from "node:assert/strict";
import {
  FRAME_STEP_EDGE_RECHECK_MS,
  FRAME_STEP_RETRY_MS,
  acceptRepeatedStep,
  createFrameStepper,
  createHoldRepeater,
  formatFrameTime,
  frameIndexAt,
  frameListing,
  frameStepKey,
  frameStepWidth,
} from "../src/frameStepping.mjs";

// Frame lists: about 19.84 fps, not a 1/20 grid; segments are not aligned.
const FPS = 19.84;
function segment(start, end, { previous = null, next = null, offset = 0 } = {}) {
  const frames = [];
  for (let time = start + offset; time < end - 1e-9; time += 1 / FPS) frames.push(Number(time.toFixed(6)));
  return { source: "main", start, end, frames, previous, next };
}
// Three segments: [1000, 1009.98), [1009.98, 1020.05) (continuing), a gap,
// then [1030, 1040).
const SEGMENTS = [
  segment(1000, 1009.98, { next: 1009.98 }),
  segment(1009.98, 1020.05, { previous: 1009.979, offset: 0.012 }),
  segment(1030, 1040),
];

function harness({ segments = SEGMENTS, ...options } = {}) {
  let clock = 0;
  const timers = [];
  const listings = [];
  const images = [];
  const released = [];
  const changes = [];
  const stepper = createFrameStepper({
    loadFrames: (epoch, signal) => new Promise((resolve, reject) => {
      listings.push({ epoch, resolve, reject, signal });
      signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
    }),
    loadImage: (epoch, signal) => new Promise((resolve, reject) => {
      images.push({ epoch, resolve, reject, signal });
      signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
    }),
    release: (entry) => released.push(entry.epoch),
    onChange: (snapshot) => changes.push(snapshot),
    now: () => clock,
    schedule: (callback, ms) => { const timer = { at: clock + ms, callback }; timers.push(timer); return timer; },
    cancel: (timer) => { const index = timers.indexOf(timer); if (index >= 0) timers.splice(index, 1); },
    ...options,
  });
  const flush = () => new Promise((done) => setTimeout(done, 0));
  const find = (epoch) => segments.find((item) => item.start <= epoch && epoch < item.end) || null;
  return {
    stepper, listings, images, released, changes,
    get state() { return stepper.snapshot(); },
    advance(ms) {
      clock += ms;
      for (const timer of timers.filter((item) => item.at <= clock)) {
        timers.splice(timers.indexOf(timer), 1);
        timer.callback();
      }
    },
    // Answer the oldest open listing request from the fixture segments.
    async answerListing() {
      await flush();
      const request = listings.find((item) => !item.done && !item.signal.aborted);
      assert.ok(request, "a listing request is open");
      request.done = true;
      request.resolve(find(request.epoch));
      await flush();
      return request.epoch;
    },
    async answerImage({ fail = false } = {}) {
      await flush();
      const request = images.find((item) => !item.done && !item.signal.aborted);
      assert.ok(request, "an image request is open");
      request.done = true;
      if (fail) request.reject(new Error("500"));
      else request.resolve({ image: `img:${request.epoch.toFixed(6)}`, actual: request.epoch });
      await flush();
      return request.epoch;
    },
    flush,
    openImages: () => images.filter((item) => !item.done && !item.signal.aborted).map((item) => item.epoch),
    openListings: () => listings.filter((item) => !item.done && !item.signal.aborted).map((item) => item.epoch),
  };
}

const [first, second] = SEGMENTS;

// Helpers.
assert.equal(frameStepWidth({ playerWidth: 390, pixelRatio: 3 }), 1280, "an iPhone gets a 1280 px frame");
assert.equal(frameStepWidth({ playerWidth: 1100, pixelRatio: 2 }), 1920);
assert.equal(frameStepWidth({ playerWidth: 800, pixelRatio: 1 }), 960);
assert.equal(frameStepWidth({ playerWidth: 300, pixelRatio: 1 }), 640);
assert.equal(frameIndexAt([10, 10.05, 10.1], 9), 0, "before the first frame: the first");
assert.equal(frameIndexAt([10, 10.05, 10.1], 10.07), 1, "between frames: the earlier");
assert.equal(frameIndexAt([10, 10.05, 10.1], 10.0999996), 2, "a time printed to the microsecond finds its frame");
assert.equal(frameIndexAt([10, 10.05, 10.1], 99), 2);
assert.equal(formatFrameTime(1000.35, "UTC"), "12:16:40.350 AM", "to the millisecond");
assert.equal(formatFrameTime(1000.0496, "UTC"), "12:16:40.050 AM", "rounded, not truncated");
assert.equal(frameListing({ start: 1, end: 2, frames: [] }), null, "no frames: nothing to step through");
assert.equal(frameListing({ start: 1, end: 2, frames: ["x"] }), null);
assert.deepEqual(frameListing({ start: 1, end: 2, frames: [1.5], next: null }), { start: 1, end: 2, frames: [1.5], previous: null, next: null, source: null });
const key = (fields) => ({ key: ".", target: null, ...fields });
assert.equal(frameStepKey(key({ key: "," })), -1);
assert.equal(frameStepKey(key({ key: "." })), 1);
assert.equal(frameStepKey(key({ key: ".", ctrlKey: true })), 0, "modified keys are not steps");
assert.equal(frameStepKey(key({ key: ".", defaultPrevented: true })), 0);
assert.equal(frameStepKey(key({ key: ".", target: { closest: (selector) => (selector.includes("input") ? {} : null) } })), 0, "typing in a field is not a step");
assert.equal(frameStepKey(key({ key: "k" })), 0);
assert.equal(acceptRepeatedStep(null, 1000), true);
assert.equal(acceptRepeatedStep(1000, 1030), false, "key repeat faster than the hold rate is thinned");
assert.equal(acceptRepeatedStep(1000, 1090), true);

// The first step: the listing of where the player is, then the next frame.
{
  const h = harness();
  const at = first.frames[40] + 0.01; // between frames 40 and 41
  h.stepper.step(1, at);
  assert.equal(h.state.active, true);
  assert.equal(h.state.known, false, "the frame is not known until the listing arrives");
  assert.equal(await h.answerListing(), at);
  assert.equal(h.state.epoch, first.frames[41], "one frame forward from the frame on screen");
  assert.equal(h.state.loading, true);
  assert.deepEqual(h.openImages(), [first.frames[41]]);
  await h.answerImage();
  assert.equal(h.state.loading, false);
  assert.equal(h.state.image, `img:${first.frames[41].toFixed(6)}`);
  assert.equal(h.state.shownEpoch, first.frames[41]);
  // The frame after it is fetched ahead.
  assert.deepEqual(h.openImages(), [first.frames[42]], "the next frame in the direction of travel is warmed");
  await h.answerImage();
  assert.deepEqual(h.openImages(), [first.frames[51]], "then one about half a second on, so the next GOP is decoded early");
  h.stepper.step(1);
  assert.equal(h.state.epoch, first.frames[42]);
  assert.equal(h.state.loading, false, "a warmed frame shows at once");
  // Back twice: frames already loaded need no request.
  h.stepper.step(-1);
  h.stepper.step(-1);
  assert.equal(h.state.epoch, first.frames[40]);
  await h.flush();
  assert.deepEqual(h.openImages(), [first.frames[40]], "a frame fetched ahead gives way to the one wanted");
  await h.answerImage();
  h.stepper.step(1);
  assert.equal(h.state.loading, false);
  assert.equal(h.state.epoch, first.frames[41]);
}

// Holding: steps outrun the images; only the latest wanted frame is loaded
// next, never the frames passed on the way.
{
  const h = harness();
  h.stepper.step(1, first.frames[10]);
  await h.answerListing();
  assert.deepEqual(h.openImages(), [first.frames[11]]);
  for (let i = 0; i < 6; i += 1) h.stepper.step(1);
  assert.equal(h.state.epoch, first.frames[17]);
  assert.equal(h.state.loading, true);
  assert.deepEqual(h.openImages(), [first.frames[11]], "one request at a time, no queue");
  await h.answerImage(); // frame 11 arrives late
  assert.equal(h.state.shownEpoch, null, "a frame that is no longer wanted is not shown as the wanted one");
  assert.deepEqual(h.openImages(), [first.frames[17]], "the next request is for the latest wanted frame");
  await h.answerImage();
  assert.equal(h.state.shownEpoch, first.frames[17]);
  const passed = first.frames.slice(12, 17);
  assert.ok(!h.images.some((request) => passed.includes(request.epoch)), "frames 12 to 16 were never requested");
}

// Across a segment boundary and back, with frame times that are not a grid.
{
  const h = harness();
  const last = first.frames.length - 1;
  h.stepper.step(-1, first.frames[last]);
  await h.answerListing();
  h.stepper.step(1);
  assert.equal(h.state.epoch, first.frames[last]);
  // Near the end the next segment's listing is fetched ahead.
  await h.answerImage();
  assert.deepEqual(h.openListings(), [first.next], "the segment ahead is listed before it is reached");
  await h.answerListing();
  h.stepper.step(1);
  assert.equal(h.state.epoch, second.frames[0], "the frame after a segment's last is the next segment's first");
  assert.equal(h.state.notice, null);
  h.stepper.step(-1);
  assert.equal(h.state.epoch, first.frames[last], "and back again");
}

// A step into a segment not listed yet waits for it; more presses meanwhile
// are not queued.
{
  const h = harness();
  const last = first.frames.length - 1;
  h.stepper.step(1, first.frames[last]);
  await h.answerListing(); // own segment; the step then needs the next one
  assert.deepEqual(h.openListings(), [first.next]);
  assert.equal(h.state.epoch, first.frames[last], "stays on the last frame while the next segment loads");
  h.stepper.step(1);
  h.stepper.step(1);
  await h.answerListing();
  assert.equal(h.state.epoch, second.frames[0], "three presses while waiting moved one frame, not three");
}

// Into a gap: stops at the last frame and says so; no error.
{
  const h = harness();
  const last = second.frames.length - 1;
  h.stepper.step(1, second.frames[last - 1]);
  await h.answerListing();
  assert.equal(h.state.epoch, second.frames[last]);
  h.stepper.step(1);
  assert.equal(h.state.epoch, second.frames[last], "no frame past the end of the recording");
  assert.equal(h.state.notice, "end");
  assert.deepEqual(h.openListings(), [], "nothing is asked: the server said no recording follows");
  h.stepper.step(-1);
  assert.equal(h.state.notice, null, "a step back clears the notice");
  // At the very start of a recording.
  const g = harness();
  g.stepper.step(-1, first.frames[0]);
  await g.answerListing();
  assert.equal(g.state.epoch, first.frames[0]);
  assert.equal(g.state.notice, "start");
}

// The newest segment: after a while its "next" is asked about again.
{
  const segments = [segment(1000, 1001), segment(1001, 1002, { previous: 1000.999 })];
  const h = harness({ segments });
  h.stepper.step(-1, 1000.99);
  await h.answerListing();
  h.stepper.step(1); // back on the last frame
  h.stepper.step(1);
  assert.equal(h.state.notice, "end");
  h.advance(FRAME_STEP_EDGE_RECHECK_MS + 1);
  segments[0] = { ...segments[0], next: 1001 };
  h.stepper.step(1);
  await h.flush();
  assert.equal(h.openListings().length, 1, "the segment is listed again");
  await h.answerListing();
  assert.deepEqual(h.openListings(), [1001]);
  await h.answerListing();
  assert.equal(h.state.epoch, segments[1].frames[0], "stepping continues into the newly indexed segment");
}

// A failed frame is retried, then reported; it is not an error of the player.
{
  const h = harness();
  h.stepper.step(1, first.frames[5]);
  await h.answerListing();
  for (const wait of FRAME_STEP_RETRY_MS) {
    await h.answerImage({ fail: true });
    assert.deepEqual(h.openImages(), [], `waits ${wait} ms before asking again`);
    h.advance(wait);
  }
  await h.answerImage({ fail: true });
  assert.equal(h.state.notice, "error");
  assert.equal(h.state.active, true, "still stepping: the next press tries again");
  h.stepper.step(1);
  assert.equal(h.state.notice, null);
  await h.flush();
  assert.deepEqual(h.openImages(), [first.frames[7]]);
}

// Leaving stops everything and takes the frame down; images stay cached.
{
  const h = harness();
  h.stepper.step(1, first.frames[5]);
  await h.answerListing();
  await h.answerImage();
  h.stepper.step(1);
  assert.equal(h.openImages().length, 1);
  h.stepper.leave();
  assert.deepEqual(h.state, { active: false });
  assert.deepEqual(h.openImages(), [], "an open request is aborted");
  assert.equal(h.stepper.epoch(), null);
  // Stepping again starts from the player's position.
  h.stepper.step(-1, first.frames[7]);
  assert.equal(h.state.epoch, first.frames[6]);
  assert.equal(h.state.loading, false, "a frame loaded before leaving is still there");
  h.stepper.close();
  assert.ok(h.released.includes(first.frames[6]), "closing releases the loaded frames");
}

// Bounded memory: images and listings.
{
  const h = harness({ imageLimit: 4, segmentLimit: 2 });
  h.stepper.step(1, first.frames[0]);
  await h.answerListing();
  for (let i = 0; i < 10; i += 1) {
    await h.answerImage();
    h.stepper.step(1);
  }
  assert.ok(h.stepper.stats().images <= 4, `at most 4 frames kept (${h.stepper.stats().images})`);
  assert.ok(h.released.length >= 6, "frames dropped from memory are released");
}

// Hold to repeat.
{
  let clock = 0;
  const timers = [];
  const steps = [];
  const hold = createHoldRepeater({
    onStep: (direction) => steps.push([clock, direction]),
    schedule: (callback, ms) => { const timer = { at: clock + ms, callback }; timers.push(timer); return timer; },
    cancel: (timer) => { const index = timers.indexOf(timer); if (index >= 0) timers.splice(index, 1); },
  });
  const run = (ms) => {
    const end = clock + ms;
    for (;;) {
      timers.sort((a, b) => a.at - b.at);
      const timer = timers[0];
      if (!timer || timer.at > end) break;
      timers.shift();
      clock = timer.at;
      timer.callback();
    }
    clock = end;
  };
  hold.press(1);
  assert.deepEqual(steps, [[0, 1]], "one step at once");
  run(399);
  assert.equal(steps.length, 1, "nothing more before the hold delay");
  run(1);
  assert.deepEqual(steps.at(-1), [400, 1]);
  run(900);
  assert.equal(steps.length, 12, "then one every 90 ms: 11 in the next 0.9 s");
  hold.release();
  run(1000);
  assert.equal(steps.length, 12, "nothing after release");
  assert.equal(hold.holding(), false);
  hold.press(-1);
  hold.press(1);
  run(400);
  assert.deepEqual(steps.slice(-3).map(([, direction]) => direction), [-1, 1, 1], "a new press replaces the old hold");
  hold.release();
}

console.log("frame stepping: listing, steps, hold, latest-wins images, boundaries, gaps, retries, memory");
