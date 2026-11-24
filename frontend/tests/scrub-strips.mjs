import assert from "node:assert/strict";
import {
  SCRUB_STRIP_GAP_SECONDS,
  SCRUB_STRIP_NEAR_SECONDS,
  SCRUB_STRIP_RETRY_MS,
  SCRUB_STRIP_TIMEOUT_MS,
  createScrubStripStore,
  scrubStripLayout,
  scrubStripTile,
  scrubStripTileWidth,
} from "../src/scrubStrips.mjs";

const headers = (values) => ({ get: (name) => values[name] ?? null });
const layout = scrubStripLayout(headers({
  "X-SurvNG-Strip-Start": "103.000",
  "X-SurvNG-Strip-End": "113.000",
  "X-SurvNG-Strip-Step": "1",
  "X-SurvNG-Strip-Count": "10",
  "X-SurvNG-Strip-Columns": "6",
  "X-SurvNG-Strip-Rows": "2",
  "X-SurvNG-Strip-Source": "live",
}));
assert.deepEqual(layout, { start: 103, end: 113, step: 1, count: 10, columns: 6, rows: 2, source: "live" });
// A response without a usable layout (an older server, a proxy that strips
// headers) is not drawn from: a wrong tile is worse than none.
assert.equal(scrubStripLayout(headers({})), null);
assert.equal(scrubStripLayout(headers({
  "X-SurvNG-Strip-Start": "103", "X-SurvNG-Strip-End": "113", "X-SurvNG-Strip-Step": "1",
  "X-SurvNG-Strip-Count": "13", "X-SurvNG-Strip-Columns": "6", "X-SurvNG-Strip-Rows": "2",
})), null);

// Tile i is the frame i seconds into the segment: 1920x480 holds 6x2 tiles.
assert.deepEqual(scrubStripTile(layout, 103.0, 1920, 480), { index: 0, epoch: 103, sx: 0, sy: 0, sw: 320, sh: 240 });
assert.deepEqual(scrubStripTile(layout, 108.9, 1920, 480), { index: 5, epoch: 108, sx: 1600, sy: 0, sw: 320, sh: 240 });
assert.deepEqual(scrubStripTile(layout, 109.2, 1920, 480), { index: 6, epoch: 109, sx: 0, sy: 240, sw: 320, sh: 240 });
// The segment runs a little past its last whole second; that tail, and
// anything outside, shows the nearest real tile and never a blank one.
assert.equal(scrubStripTile(layout, 112.99, 1920, 480).index, 9);
assert.equal(scrubStripTile({ ...layout, count: 9 }, 112.5, 1920, 480).index, 8);
assert.equal(scrubStripTile(layout, 90, 1920, 480).index, 0);
assert.equal(scrubStripTile(layout, 108, 0, 0), null);
// A long segment thinned to one tile every 5 s.
assert.equal(scrubStripTile({ ...layout, end: 403, step: 5, count: 60, columns: 6, rows: 10 }, 139, 1920, 2400).index, 7);

assert.equal(scrubStripTileWidth({ coarsePointer: true, playerWidth: 2000, pixelRatio: 3 }), 320);
assert.equal(scrubStripTileWidth({ playerWidth: 1200, pixelRatio: 1 }), 480);
assert.equal(scrubStripTileWidth({ playerWidth: 700, pixelRatio: 1 }), 320);

// A store whose loads are resolved by hand, on a clock the test moves.
// Timers run when the clock passes them.
function harness(options = {}) {
  const pending = [];
  const released = [];
  const timers = [];
  let changes = 0;
  let clock = 1_000_000;
  const store = createScrubStripStore({
    load: (epoch, signal) => new Promise((resolve, reject) => pending.push({ epoch, resolve, reject, signal })),
    onChange: () => { changes += 1; },
    release: (strip) => released.push(strip.start),
    now: () => clock,
    schedule: (callback, ms) => {
      const timer = { callback, at: clock + ms };
      timers.push(timer);
      return timer;
    },
    cancel: (timer) => {
      const index = timers.indexOf(timer);
      if (index >= 0) timers.splice(index, 1);
    },
    ...options,
  });
  const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
  return {
    store, pending, released, settle, timers,
    changes: () => changes,
    advance: (ms) => {
      clock += ms;
      for (const timer of timers.filter((item) => item.at <= clock)) {
        timers.splice(timers.indexOf(timer), 1);
        timer.callback();
      }
    },
  };
}
const segment = (start) => ({ start, end: start + 10, step: 1, count: 10, columns: 6, rows: 2 });

// Two requests at a time and no queue: when one finishes, the next is for
// where the playhead is then. The places it passed in between are never
// asked for.
{
  const t = harness();
  assert.deepEqual(t.store.frameAt(105), { strip: null, gap: false, loading: true });
  t.store.want(105);
  t.store.want(106);   // the same segment: not asked for twice
  t.store.want(125);
  t.store.want(145);
  t.store.want(165);
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch), [105, 125]);
  t.pending[0].resolve(segment(100));
  await t.settle();
  assert.equal(t.changes(), 1);
  assert.equal(t.store.frameAt(105).strip.start, 100);
  // The playhead has moved on while that strip was made. Until its own
  // segment arrives, the nearest loaded one stands in, marked as loading.
  assert.deepEqual(t.store.frameAt(165), { strip: t.store.frameAt(105).strip, gap: false, loading: true });
  assert.equal(scrubStripTile(t.store.frameAt(165).strip, 165, 1920, 480).index, 9);
  assert.equal(scrubStripTile(t.store.frameAt(95).strip, 95, 1920, 480).index, 0);
  // Not from further away than that: it would be a different scene.
  assert.equal(t.store.frameAt(110 + SCRUB_STRIP_NEAR_SECONDS + 1).strip, null);
  // 145 was passed while both requests were busy; 165 is where it is now.
  assert.deepEqual(t.pending.map((item) => item.epoch), [105, 125, 165]);
  t.pending[1].resolve(segment(120));
  await t.settle();
  // Its own segment is still loading, so the one after it cannot be named.
  assert.equal(t.pending.length, 3);
  t.pending[2].resolve(segment(160));
  await t.settle();
  // At a loaded segment: the one the drag is heading into is fetched, and
  // once that shows where it ends, the one after it.
  assert.deepEqual(t.pending.map((item) => item.epoch).slice(3), [170.5]);
  t.pending[3].resolve(segment(170));
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch).slice(4), [180.5]);
  t.pending[4].resolve(segment(180));
  await t.settle();
  // Two segments ahead, not a run of them.
  assert.equal(t.pending.length, 5);
  // Moving within what is loaded asks only for what keeps it two ahead.
  t.store.want(168);
  t.store.want(169);
  await t.settle();
  assert.equal(t.pending.length, 5);
  t.store.want(171);
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch).slice(5), [190.5]);
  // Turning back fetches the segment behind.
  t.store.want(161);
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch).slice(6), [159.5]);
}

// The playhead's own segment and the next load together once the first shows
// which way the drag is going and where the segment ends.
{
  const t = harness();
  t.store.want(105);
  t.store.want(107);
  await t.settle();
  t.pending[0].resolve(segment(100));
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch), [105, 110.5]);
  t.store.want(112);
  await t.settle();
  // 112 is in the segment already being fetched ahead: no second request.
  assert.equal(t.pending.length, 2);
}

// A time with no recording is remembered for a while, then asked about again.
// Only that second is remembered: the answer names a time, and footage may
// begin right next to it.
{
  const t = harness();
  t.store.want(208);
  await t.settle();
  t.pending[0].resolve(null);
  await t.settle();
  assert.deepEqual(t.store.frameAt(208.4), { strip: null, gap: true, loading: false });
  t.store.want(208.6);
  await t.settle();
  assert.equal(t.pending.length, 1);
  // The recording ends at 207: one second away is still asked about.
  assert.equal(t.store.frameAt(206.5).gap, false);
  t.store.want(206.5);
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch), [208, 206.5]);
  t.pending[1].resolve({ ...segment(200), end: 207, count: 7 });
  await t.settle();
  assert.equal(t.store.frameAt(206.5).strip.start, 200);
}
{
  const t = harness();
  t.store.want(208);
  await t.settle();
  t.pending[0].resolve(null);
  await t.settle();
  t.advance(SCRUB_STRIP_GAP_SECONDS * 1000 + 1);
  assert.equal(t.store.frameAt(208.4).gap, false);
  t.store.want(208.4);
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch), [208, 208.4]);
}

// A failed request (the generator is busy) is not repeated at once, and is
// repeated by itself after the wait: the playhead may be resting.
{
  const t = harness();
  t.store.want(305);
  await t.settle();
  t.pending[0].reject(new Error("429"));
  await t.settle();
  t.store.want(306);
  await t.settle();
  assert.equal(t.pending.length, 1);
  assert.equal(t.store.frameAt(305).loading, true);
  t.advance(SCRUB_STRIP_RETRY_MS + 1);
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch), [305, 306]);
}

// A request that never answers gives its slot back.
{
  const t = harness();
  t.store.want(355);
  await t.settle();
  assert.equal(t.pending.length, 1);
  t.advance(SCRUB_STRIP_TIMEOUT_MS + 1);
  assert.equal(t.pending[0].signal.aborted, true);
  assert.equal(t.changes(), 1);
  t.advance(SCRUB_STRIP_RETRY_MS + 1);
  await t.settle();
  assert.equal(t.pending.length, 2);
  // Its late answer is released, not kept.
  t.pending[0].resolve(segment(350));
  await t.settle();
  assert.deepEqual(t.released, [350]);
  assert.equal(t.store.size, 0);
}

// Memory is bounded: the strip used longest ago goes first, and its image is
// released. Looking at a strip counts as using it.
{
  const t = harness({ limit: 2 });
  const loadAt = async (epoch) => {
    t.store.idle();
    t.store.want(epoch);
    await t.settle();
    t.pending.at(-1).resolve(segment(epoch - 5));
    await t.settle();
    t.advance(10);
  };
  await loadAt(405);
  await loadAt(505);
  t.store.frameAt(405);
  t.advance(10);
  await loadAt(605);
  assert.equal(t.store.size, 2);
  assert.deepEqual(t.released, [500]);
  assert.equal(t.store.frameAt(405).strip.start, 400);
  // The same segment arriving again replaces the copy held.
  t.store.idle();
  t.store.want(408);
  await t.settle();
  assert.equal(t.pending.length, 3);
  await loadAt(605);
  assert.equal(t.pending.length, 3);
}

// Strips that overlap stay side by side. The sub and main streams are cut at
// different times; if one replaced the other, fetching ahead would evict the
// strip under a resting playhead and ask for it again, without end.
{
  const t = harness();
  const live = { start: 100, end: 110, step: 1, count: 10, columns: 6, rows: 2, source: "live" };
  const main = { start: 105, end: 115, step: 1, count: 10, columns: 6, rows: 2, source: "main" };
  t.store.want(102);
  t.store.want(103);
  await t.settle();
  t.pending[0].resolve(live);
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch), [102, 110.5]);
  t.pending[1].resolve(main);
  await t.settle();
  assert.equal(t.store.size, 2);
  assert.deepEqual(t.released, []);
  assert.equal(t.store.frameAt(103).strip, live);
  // Resting at 103: one more segment ahead, then nothing.
  assert.deepEqual(t.pending.map((item) => item.epoch), [102, 110.5, 115.5]);
  t.pending[2].resolve(segment(115));
  await t.settle();
  assert.equal(t.pending.length, 3);
}

// One short segment does not make a later one look like two.
{
  const t = harness();
  t.store.want(501);
  await t.settle();
  t.pending[0].resolve({ ...segment(500), end: 503.5, count: 4 });
  await t.settle();
  t.store.idle();
  t.store.want(601);
  t.store.want(607);
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch), [501, 601]);
}

// The end of a scrub forgets which way it was going.
{
  const t = harness();
  t.store.want(705);
  t.store.want(706);
  await t.settle();
  t.store.idle();
  t.pending[0].resolve(segment(700));
  await t.settle();
  t.store.want(706);
  await t.settle();
  // A new scrub that has not moved yet fetches nothing ahead.
  assert.equal(t.pending.length, 1);
}

// Closing releases everything and ignores a request that finishes afterwards.
{
  const t = harness();
  t.store.want(705);
  await t.settle();
  t.pending[0].resolve(segment(700));
  await t.settle();
  t.store.idle();
  t.store.want(805);
  await t.settle();
  t.store.close();
  t.pending[1].resolve(segment(800));
  await t.settle();
  assert.deepEqual(t.released, [700, 800]);
  assert.equal(t.store.size, 0);
  const changesAtClose = t.changes();
  t.store.want(905);
  await t.settle();
  assert.equal(t.pending.length, 2);
  assert.equal(t.changes(), changesAtClose);
}

console.log("scrub strip tests passed");
