import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { timelinePanViewport, timelineViewport } from "../src/timelineWorkspace.mjs";

const source = readFileSync(new URL("../src/timeline/TimelinePages.jsx", import.meta.url), "utf8");
function functionsBetween(first, next) {
  const start = source.indexOf(`  function ${first}(`);
  const end = source.indexOf(`  function ${next}(`, start);
  assert.ok(start >= 0 && end > start, `actual ${first} handlers must be present`);
  return source.slice(start, end);
}
function gestureHarness() {
  const seeks = [], pans = [], previews = [], drafts = [];
  const captures = new Set();
  const target = {
    getBoundingClientRect: () => ({ left: 10, width: 360 }),
    setPointerCapture: id => captures.add(id),
    hasPointerCapture: id => captures.has(id),
    releasePointerCapture: id => captures.delete(id),
  };
  const context = vm.createContext({
    Math, Number, duration: 3600, startEpoch: 21600, offset: 1200,
    dragRef: { current: null }, draftRef: { current: 1200 },
    previewHideTimerRef: { current: null }, previewTimerRef: { current: null },
    previewManifestUrl: "", prefersJpegScrubPreview: () => true,
    window: { clearTimeout() {} }, setLocalPreviewEnabled() {},
    setDraft: value => drafts.push(value), setScrubbing() {}, hidePreviewAfterDelay() {},
    schedulePreview: value => previews.push(value),
    onSeek: epoch => seeks.push(epoch), onPanViewport: delta => pans.push(delta),
  });
  vm.runInContext(functionsBetween("updateDraft", "exportEpochAtPointer"), context);
  const event = (clientX, pointerType = "touch", pointerId = 1) => ({
    currentTarget: target, clientX, pointerType, pointerId, preventDefault() {},
  });
  return { context, seeks, pans, previews, drafts, captures, event };
}

// Pointer capture allows a swipe to continue beyond the visible track. The
// resulting pan must exceed the original hour without moving playback.
{
  const { context, seeks, pans, previews, captures, event } = gestureHarness();
  context.startDrag(event(310));
  context.moveDrag(event(250));
  context.moveDrag(event(70));
  context.moveDrag(event(-140));
  context.finishDrag(event(-140));
  assert.ok(pans.reduce((sum, delta) => sum + delta, 0) > 3600);
  assert.deepEqual(seeks, []);
  assert.deepEqual(previews, [], "viewport browsing should not load scrub previews");
  assert.equal(context.dragRef.current, null);
  assert.equal(captures.size, 0);
}

// A short tap, including a little finger jitter, retains precise seek behavior.
for (const jitter of [0, 3, -3]) {
  const { context, seeks, pans, event } = gestureHarness();
  context.startDrag(event(190));
  context.moveDrag(event(190 + jitter));
  context.finishDrag(event(190 + jitter));
  assert.deepEqual(pans, []);
  assert.deepEqual(seeks, [21600 + (180 + jitter) * 10]);
}

for (const mode of ["touch", "mouse"]) {
  const { context, seeks, captures, drafts, event } = gestureHarness();
  context.startDrag(event(100, mode));
  context.moveDrag(event(200, mode));
  context.finishDrag(event(200, mode), true);
  assert.deepEqual(seeks, [], `cancelled ${mode} gesture must not commit a seek`);
  assert.equal(drafts.at(-1), 1200);
  assert.equal(captures.size, 0);
}

// Touch drags on the playhead fine scrub without moving the viewport.
{
  const { context, seeks, pans, previews, event } = gestureHarness();
  context.startDrag(event(130)); // current playhead at 1200 seconds / 3600
  context.moveDrag(event(175));
  context.finishDrag(event(175));
  assert.deepEqual(pans, [], "grabbing the playhead must scrub, not pan");
  assert.deepEqual(seeks, [21600 + 1245], "fine scrubbing moves one second per pixel");
  assert.ok(previews.length > 0);
}

{
  const { context, seeks, pans, previews, event } = gestureHarness();
  // Mouse drags still scrub the selected hour, including preview updates.
  context.startDrag(event(100, "mouse"));
  context.moveDrag(event(200, "mouse"));
  context.finishDrag(event(240, "mouse"));
  assert.deepEqual(pans, []);
  assert.deepEqual(seeks, [21600 + 2300]);
  assert.ok(previews.length > 0);
}

// An unrelated pointer cannot steal the captured gesture or commit its seek.
{
  const { context, seeks, pans, event } = gestureHarness();
  context.startDrag(event(190));
  context.moveDrag(event(290, "touch", 2));
  context.finishDrag(event(290, "touch", 2));
  assert.deepEqual(pans, []);
  assert.deepEqual(seeks, []);
  assert.equal(context.dragRef.current.pointerId, 1);
  context.finishDrag(event(190), true);
}

function panHarness({ anchor = 43200, dayEnd = 86400, view = { startEpoch: 41400, endEpoch: 45000 } } = {}) {
  const updates = [], following = [];
  const context = vm.createContext({
    Number, dayStart: 0, dayEnd, incidentRangeHours: 1, timelineView: view,
    timelineViewport, timelinePanViewport,
    setFollowPlayhead: value => following.push(value),
    setTimelineViewportAnchor: update => updates.push(update),
  });
  vm.runInContext(functionsBetween("panTimelineViewport", "returnToPlayhead"), context);
  function flush() {
    for (const update of updates.splice(0)) {
      assert.equal(typeof update, "function", "batched gestures must compose functional state updates");
      anchor = update(anchor);
    }
    return anchor;
  }
  return { context, flush, following };
}

// Simulate six pointer moves before React commits another render. They must
// accumulate instead of each reusing the old render's viewport center.
{
  const { context, flush, following } = panHarness();
  for (let i = 0; i < 6; i++) context.panTimelineViewport(900);
  assert.equal(flush(), 48600);
  assert.ok(following.every(value => value === false));
}

// Clamp each queued update at the day boundary, including a gesture reversing
// direction before React renders. This also covers 23- and 25-hour local days.
for (const dayEnd of [23, 24, 25].map(hours => hours * 3600)) {
  const { context, flush } = panHarness({ anchor: null, dayEnd, view: { startEpoch: 0, endEpoch: 3600 } });
  context.panTimelineViewport(-7200);
  context.panTimelineViewport(900);
  assert.equal(flush(), 2700);
  context.panTimelineViewport(dayEnd * 2);
  context.panTimelineViewport(-900);
  assert.equal(flush(), dayEnd - 2700);
}
console.log("Timeline mobile gesture tests passed");
