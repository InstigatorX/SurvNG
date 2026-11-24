import assert from "node:assert/strict";
import {
  SCRUB_STRIP_GAP_SECONDS,
  SCRUB_STRIP_RETRY_MS,
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
function harness(options = {}) {
  const pending = [];
  const released = [];
  let changes = 0;
  let clock = 1_000_000;
  const store = createScrubStripStore({
    load: (epoch) => new Promise((resolve, reject) => pending.push({ epoch, resolve, reject })),
    onChange: () => { changes += 1; },
    release: (strip) => released.push(strip.start),
    now: () => clock,
    ...options,
  });
  const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
  return {
    store, pending, released, settle,
    changes: () => changes,
    advance: (ms) => { clock += ms; },
  };
}
const segment = (start) => ({ start, end: start + 10, step: 1, count: 10, columns: 6, rows: 2 });

// One request at a time, and always for where the playhead is now: the
// places it passed while the first strip was being made are never asked for.
{
  const t = harness();
  assert.deepEqual(t.store.frameAt(105), { strip: null, gap: false, loading: true });
  t.store.want(105);
  t.store.want(125);
  t.store.want(165);
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch), [105]);
  t.pending[0].resolve(segment(100));
  await t.settle();
  assert.equal(t.changes(), 1);
  assert.equal(t.store.frameAt(105).strip.start, 100);
  assert.deepEqual(t.pending.map((item) => item.epoch), [105, 165]);
  t.pending[1].resolve(segment(160));
  await t.settle();
  // Idle at a loaded segment: the one the drag is heading into is fetched.
  assert.deepEqual(t.pending.map((item) => item.epoch), [105, 165, 170.5]);
  t.pending[2].resolve(segment(170));
  await t.settle();
  // One segment ahead, not a run of them.
  assert.equal(t.pending.length, 3);
  // Moving within what is loaded asks for nothing, even at the far edge of
  // the segment already fetched ahead.
  t.store.want(168);
  t.store.want(169);
  await t.settle();
  assert.equal(t.pending.length, 3);
  // Turning back fetches the segment behind.
  t.store.want(161);
  await t.settle();
  assert.deepEqual(t.pending.map((item) => item.epoch).slice(3), [159.5]);
}

// A time with no recording is remembered for a while, then asked about again.
{
  const t = harness();
  t.store.want(205);
  await t.settle();
  t.pending[0].resolve(null);
  await t.settle();
  assert.deepEqual(t.store.frameAt(205), { strip: null, gap: true, loading: false });
  t.store.want(206);
  await t.settle();
  assert.equal(t.pending.length, 1);
  t.advance(SCRUB_STRIP_GAP_SECONDS * 1000 + 1);
  assert.equal(t.store.frameAt(205).loading, true);
  t.store.want(207);
  await t.settle();
  assert.equal(t.pending.length, 2);
}

// A failed request (the generator is busy) is not repeated at once.
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
  t.store.want(307);
  await t.settle();
  assert.equal(t.pending.length, 2);
}

// Memory is bounded: the strip used longest ago goes first, and its image is
// released. A strip that overlaps a new one is replaced by it.
{
  const t = harness({ limit: 2 });
  for (const [index, epoch] of [405, 505, 605].entries()) {
    t.store.idle();
    t.store.want(epoch);
    await t.settle();
    t.pending[index].resolve(segment(epoch - 5));
    await t.settle();
    t.advance(10);
  }
  assert.equal(t.store.size, 2);
  assert.deepEqual(t.released, [400]);
  assert.equal(t.store.frameAt(405).loading, true);
  t.store.idle();
  t.store.want(612);
  await t.settle();
  t.pending[3].resolve(segment(603));
  await t.settle();
  assert.deepEqual(t.released, [400, 600]);
  assert.equal(t.store.frameAt(606).strip.start, 603);
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
