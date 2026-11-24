// The React layer of timeline scrub frames, in a real browser: the
// useScrubStrips hook, ScrubFrameOverlay, and the scrub reporting of the rail
// (useVerticalScrub, EdgeRail) and the classic scrubber (RecordingTimeline).
// The pure strip store is covered by tests/scrub-strips.mjs.
//   node tests/scrub-frame-react.mjs
// The page is tests/fixtures/scrub-frame-react.jsx; preview strips are served
// by a route in this file that the test answers one request at a time.
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const temporary = mkdtempSync(join(tmpdir(), "scrub-frame-react-"));
let browser, server;

// Segments are 10 s, aligned to 10; [2000, 2100) has no recording.
const GAP = { start: 2000, end: 2100 };
function stripHeaders(epoch, source) {
  const start = Math.floor(epoch / 10) * 10;
  return {
    "X-SurvNG-Strip-Start": String(start), "X-SurvNG-Strip-End": String(start + 10),
    "X-SurvNG-Strip-Step": "1", "X-SurvNG-Strip-Count": "10",
    "X-SurvNG-Strip-Columns": "4", "X-SurvNG-Strip-Rows": "3", "X-SurvNG-Strip-Source": source,
  };
}

let stripPng = null;
async function open(scenario, query = "", contextOptions = {}) {
  const context = await browser.newContext({ viewport: { width: 900, height: 700 }, ...contextOptions });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  // Strip requests wait here until the test answers them.
  const requests = [];
  await page.route("**/api/cameras/*/recordings/preview-strip.jpg*", (route) => {
    const url = new URL(route.request().url());
    requests.push({
      camera: url.pathname.split("/")[3], epoch: Number(url.searchParams.get("epoch")),
      source: url.searchParams.get("source"), route, answered: false,
    });
  });
  await page.route("**/api/**", (route) => (route.request().url().includes("preview-strip.jpg") ? route.fallback() : route.fulfill({ status: 404 })));
  await page.goto(`${server.resolvedUrls.local[0]}?scenario=${scenario}${query}`);
  await page.waitForSelector("body[data-ready=true]");
  if (!stripPng) stripPng = Buffer.from(await page.evaluate(() => window.h.stripPng()), "base64");
  const answer = async (request) => {
    request.answered = true;
    if (request.epoch >= GAP.start && request.epoch < GAP.end) {
      await request.route.fulfill({ status: 404, headers: { "X-SurvNG-Gap-Start": String(GAP.start), "X-SurvNG-Gap-End": String(GAP.end) } });
    } else {
      await request.route.fulfill({ status: 200, contentType: "image/png", headers: stripHeaders(request.epoch, request.source), body: stripPng });
    }
  };
  // Answer the open request for `epoch` (or the oldest one) and wait for the
  // page to have received it.
  page.answer = async (epoch = null) => {
    const match = () => requests.find((item) => !item.answered && (epoch === null || Math.floor(item.epoch / 10) === Math.floor(epoch / 10)));
    // A request reaches the route a moment after the page made it.
    for (let tries = 0; !match() && tries < 40; tries += 1) await page.waitForTimeout(25);
    const request = match();
    assert.ok(request, `a strip request for ${epoch} is open (open: ${JSON.stringify(page.open())})`);
    const before = await page.evaluate(() => window.log.length);
    await answer(request);
    await page.waitForFunction((count) => window.log.length > count, before, { timeout: 3000 }).catch(() => {});
    return request;
  };
  page.answerAll = async () => { for (const request of requests.filter((item) => !item.answered)) await answer(request); };
  page.open = () => requests.filter((item) => !item.answered).map((item) => item.epoch);
  // The open requests once `count` of them have reached the route (or 1 s).
  page.openSoon = async (count = 1) => {
    for (let tries = 0; page.open().length < count && tries < 40; tries += 1) await page.waitForTimeout(25);
    return page.open();
  };
  page.requests = requests;
  page.errors = errors;
  page.log = () => page.evaluate(() => window.log.slice());
  page.clearLog = () => page.evaluate(() => { window.log.length = 0; });
  page.overlay = () => page.evaluate(() => window.h.overlay());
  page.waitLog = (predicate, message, timeout = 3000) => page.waitForFunction(
    (source) => window.log.some(new Function("entry", `return ${source}`)), predicate, { timeout },
  ).catch(async () => { throw new Error(`${message}: ${JSON.stringify(await page.log())}`); });
  return page;
}
const settle = (page, ms = 150) => page.waitForTimeout(ms);
const lastFrame = (log) => log.filter((entry) => entry.frame).at(-1);
const ends = (log) => log.filter((entry) => entry.frame === false || entry.end);

async function hookTests() {
  // Frames arrive as strips load; the overlay draws the right tile.
  {
    const page = await open("hook");
    await page.evaluate(() => window.h.scrub(1005));
    let log = await page.log();
    assert.deepEqual(log.at(-1), { frame: true, epoch: 1005, start: null, source: null, index: null, gap: false, loading: true, held: false },
      "a scrub publishes a loading frame at once");
    assert.deepEqual(await page.openSoon(), [1005], "the playhead's own segment is requested");
    assert.equal(page.requests[0].camera, "cam1");
    assert.equal(page.requests[0].source, "main");
    let overlay = await page.overlay();
    assert.equal(overlay.up, true);
    assert.equal(overlay.drawn, "false", "nothing drawn until a strip has loaded");
    assert.equal(overlay.visibility, "hidden", "the player's own picture shows until a frame is drawn");
    await page.answer(1005);
    log = await page.log();
    assert.deepEqual(lastFrame(log), { frame: true, epoch: 1005, start: 1000, source: "main", index: 5, gap: false, loading: false, held: false },
      "the loaded strip is delivered with the tile for the time");
    overlay = await page.overlay();
    assert.equal(overlay.drawn, "true");
    assert.equal(overlay.tile, 5, "the overlay draws tile 5 of the strip");
    assert.deepEqual(overlay.canvas, [40, 30], "the canvas is one tile in size");
    assert.equal(overlay.loading, false);

    // Moving within a loaded segment needs no request.
    await page.evaluate(() => window.h.scrub(1008));
    assert.equal(lastFrame(await page.log()).index, 8);
    assert.equal((await page.overlay()).tile, 8);
    assert.ok(!page.open().some((epoch) => Math.floor(epoch / 10) === 100), "no request for a segment already loaded");

    // Into the gap: "No recording".
    await page.evaluate(() => window.h.scrub(2050));
    await page.answer(2050);
    log = await page.log();
    assert.equal(lastFrame(log).gap, true);
    overlay = await page.overlay();
    assert.equal(overlay.gap, true);
    assert.match(overlay.text, /No recording/);
    assert.equal(overlay.status, "No recording");

    // An uncommitted end clears at once.
    await page.evaluate(() => window.h.scrub(1005));
    await page.clearLog();
    await page.evaluate(() => window.h.scrub(null));
    assert.deepEqual(await page.log(), [{ frame: false, committed: false }], "an uncommitted end clears at once");
    assert.equal((await page.overlay()).up, false, "the overlay is taken down at once");

    // An uncommitted end while the time is still loading does not hold it.
    await page.answerAll(); // free the request slots held by fetches ahead
    await settle(page);
    await page.evaluate(() => window.h.scrub(1045));
    await page.clearLog();
    await page.evaluate(() => window.h.scrub(null));
    await page.answer(1045);
    await settle(page);
    assert.deepEqual(await page.log(), [{ frame: false, committed: false }], "an uncommitted end stops delivering frames at once");

    // A second end is ignored.
    await page.evaluate(() => { window.h.scrub(1006); window.h.scrub(null, { committed: true }); });
    await page.clearLog();
    await page.evaluate(() => window.h.scrub(null));
    assert.deepEqual(await page.log(), [], "a double end is ignored");
    assert.deepEqual(page.errors, []);
    await page.context().close();
  }

  // After a committed end the released time keeps loading; its frame is
  // delivered held, then loading stops.
  {
    const page = await open("hook");
    await page.evaluate(() => window.h.scrub(1005));
    await page.answer(1005);
    await page.evaluate(() => window.h.scrub(1015)); // moving forward
    assert.deepEqual(await page.openSoon(), [1015]);
    await page.clearLog();
    await page.evaluate(() => window.h.scrub(null, { committed: true }));
    assert.deepEqual(await page.log(), [{ frame: false, committed: true }]);
    // The overlay keeps the last frame up meanwhile.
    assert.equal((await page.overlay()).tile, 9, "the stand-in frame stays up after release");
    // An uncommitted end while held is ignored too: the hold is not cut short.
    await page.evaluate(() => window.h.scrub(null));
    assert.deepEqual(await page.log(), [{ frame: false, committed: true }], "an end during a hold is ignored");
    await page.answer(1015);
    const log = await page.log();
    assert.deepEqual(log.at(-1), { frame: true, epoch: 1015, start: 1010, source: "main", index: 5, gap: false, loading: false, held: true },
      "the released time's frame is delivered, marked held");
    assert.equal((await page.overlay()).tile, 5, "the held frame replaces the stand-in in the overlay");
    // Loading has stopped: nothing is fetched ahead and no more frames come.
    await settle(page, 300);
    assert.deepEqual(page.open(), [], "nothing is fetched after the held frame arrived");
    assert.equal((await page.log()).length, log.length, "no frame after the held one");
    assert.deepEqual(page.errors, []);
    await page.context().close();
  }

  // ... and stops after ~4 s regardless.
  {
    const page = await open("hook");
    await page.evaluate(() => window.h.scrub(1005));
    await page.answer(1005);
    await page.evaluate(() => window.h.scrub(1035));
    await page.evaluate(() => window.h.scrub(null, { committed: true }));
    await page.clearLog();
    await settle(page, 4300);
    await page.answer(1035).catch(() => {});
    await settle(page);
    assert.deepEqual(await page.log(), [], "a released time answered after 4 s is no longer delivered");
    assert.deepEqual(page.errors, []);
    await page.context().close();
  }

  // A new scrub cancels a hold.
  {
    const page = await open("hook");
    await page.evaluate(() => window.h.scrub(1005));
    await page.answer(1005);
    await page.evaluate(() => window.h.scrub(1035));
    await page.evaluate(() => window.h.scrub(null, { committed: true }));
    await settle(page, 500);
    await page.evaluate(() => window.h.scrub(1205));
    assert.equal(lastFrame(await page.log()).held, false, "frames of a new scrub are not held");
    // Past the hold's 4 s the new scrub is still served.
    await settle(page, 3900);
    await page.clearLog();
    await page.answer(1205);
    const frame = lastFrame(await page.log());
    assert.ok(frame, "the new scrub's frame is delivered after the old hold would have ended");
    assert.equal(frame.epoch, 1205);
    assert.equal(frame.held, false);
    assert.equal(frame.index, 5);
    assert.deepEqual(page.errors, []);
    await page.context().close();
  }

  // Camera and source changes, unmount mid-scrub.
  {
    const page = await open("hook");
    await page.evaluate(() => window.h.scrub(1005));
    await page.answer(1005);
    await page.clearLog();
    await page.evaluate(() => window.h.setCamera("cam2"));
    let taken = ends(await page.log());
    assert.ok(taken.length >= 1 && taken.every((entry) => entry.committed === false), "a camera change mid-scrub takes the frame down");
    assert.equal((await page.overlay()).up, false);
    // The new camera's strips are asked for from the new camera.
    await page.evaluate(() => window.h.scrub(1105));
    assert.equal((await page.answer(1105)).camera, "cam2", "the new camera's strips are requested");

    // A held frame survives a change of stream alone.
    await page.evaluate(() => window.h.scrub(1135));
    await page.evaluate(() => window.h.scrub(null, { committed: true }));
    await page.evaluate(() => window.h.setSeeking(true));
    await page.clearLog();
    await page.evaluate(() => window.h.setSource("live"));
    assert.deepEqual(await page.log(), [], "a source change alone does not take down a held frame");
    await settle(page, 400);
    assert.equal((await page.overlay()).up, true, "the held frame is still up after a source change");
    assert.equal((await page.overlay()).tile, 9);

    // A source change mid-scrub (not held) does end it.
    await page.evaluate(() => window.h.scrub(1105));
    await page.clearLog();
    await page.evaluate(() => window.h.setSource("main"));
    assert.deepEqual(await page.log(), [{ frame: false, committed: false }], "a scrub in progress loses its frames on a source change");

    // A camera change takes down a held frame too.
    await page.evaluate(() => window.h.scrub(1165));
    await page.evaluate(() => window.h.scrub(null, { committed: true }));
    await settle(page, 400);
    assert.equal((await page.overlay()).up, true, "the frame is held while the player seeks");
    await page.clearLog();
    await page.evaluate(() => window.h.setCamera("cam1"));
    assert.deepEqual(await page.log(), [{ frame: false, committed: false }], "a camera change takes down a held frame");
    assert.equal((await page.overlay()).up, false);
    await page.evaluate(() => window.h.setSeeking(false));

    // Unmount mid-scrub.
    await page.evaluate(() => window.h.scrub(1105));
    await page.clearLog();
    await page.evaluate(() => window.h.unmount());
    assert.deepEqual(await page.log(), [{ frame: false, committed: false }], "unmounting mid-scrub ends it");
    assert.equal((await page.overlay()).up, false);
    assert.deepEqual(page.errors, []);
    await page.context().close();
  }

  // All cameras: no store, no requests.
  {
    const page = await open("hook", "&camera=");
    await page.evaluate(() => { window.h.scrub(1005); window.h.scrub(1025); window.h.scrub(null, { committed: true }); });
    await settle(page, 300);
    assert.equal(page.requests.length, 0, "all-cameras mode requests nothing");
    assert.deepEqual(await page.log(), [], "all-cameras mode delivers no frames");
    assert.equal((await page.overlay()).up, false);
    assert.deepEqual(page.errors, []);
    await page.context().close();
  }
  console.log("useScrubStrips: delivery, hold, cancel, camera/source change, unmount, all cameras, double end");
}

async function overlayTests() {
  const page = await open("overlay");
  const show = (spec, options) => page.evaluate(([s, o]) => window.h.show(s, o), [spec, options ?? {}]);
  const renders = () => page.evaluate(() => window.h.renders);
  let overlay = await page.overlay();
  assert.equal(overlay.statusPresent, true, "the status region exists before any frame is up");
  assert.equal(overlay.status, "");
  assert.equal(overlay.up, false);

  // A loading frame with nothing to draw yet hides the canvas.
  await show({ strip: null, index: 3, loading: true });
  overlay = await page.overlay();
  assert.equal(overlay.drawn, "false");
  assert.equal(overlay.visibility, "hidden");
  await show({ strip: "a", index: 7 });
  overlay = await page.overlay();
  assert.equal(overlay.tile, 7, "draws the right tile");
  assert.deepEqual(overlay.canvas, [40, 30]);
  assert.equal(overlay.drawn, "true");
  assert.equal(overlay.loading, false);
  assert.equal(overlay.opacity, "1");
  // Keeps the last picture while the next loads, dimmed.
  await show({ strip: null, index: 8, loading: true });
  overlay = await page.overlay();
  assert.equal(overlay.drawn, "true", "the last picture stays up while the next loads");
  assert.equal(overlay.visibility, "visible");
  assert.equal(overlay.tile, 7);
  assert.equal(overlay.loading, true);
  assert.equal(overlay.opacity, "0.82", "a loading frame is dimmed");
  // A stand-in strip while loading is drawn, dimmed.
  await show({ strip: "a", index: 2, loading: true });
  overlay = await page.overlay();
  assert.equal(overlay.tile, 2);
  assert.equal(overlay.loading, true);

  // Identical frames do not re-render.
  await show({ strip: "a", index: 4 });
  let before = await renders();
  for (let i = 0; i < 6; i += 1) await show({ strip: "a", index: 4 });
  // React may render once more before it can bail out of an unchanged state
  // (its eager bailout needs an idle fiber); what matters is that repeated
  // identical frames, as a resting scrub reports them, do not each render.
  assert.ok(await renders() - before <= 1, `identical frames do not re-render (${await renders() - before} renders for 6 frames)`);
  await show({ strip: "a", index: 5 });
  assert.ok(await renders() > before, "a different tile does render");
  before = await renders();
  await show({ strip: "b", index: 5 });
  assert.ok(await renders() > before, "the same tile of another strip renders");

  // A gap.
  await show({ strip: null, index: 0, gap: true });
  overlay = await page.overlay();
  assert.equal(overlay.gap, true);
  assert.match(overlay.text, /No recording/);
  assert.equal(overlay.status, "No recording", "the status region gets its text for a gap");

  // An uncommitted end clears at once.
  await show(null);
  overlay = await page.overlay();
  assert.equal(overlay.up, false, "an uncommitted end clears at once");
  assert.equal(overlay.status, "");
  assert.equal(overlay.statusPresent, true, "the status region stays");

  // A held frame does not put a picture up by itself...
  await show({ strip: "a", index: 6, held: true });
  assert.equal((await page.overlay()).up, false, "a held frame with nothing up shows nothing");
  // ... but replaces one that is up.
  await show({ strip: "a", index: 1, loading: true });
  await show(null, { committed: true });
  await show({ strip: "a", index: 6, held: true });
  overlay = await page.overlay();
  assert.equal(overlay.tile, 6, "a held frame replaces the picture while one is up");
  assert.equal(overlay.loading, false);

  // Released, not seeking: stays up for ~300 ms, then clears.
  const timing = await page.evaluate(async () => {
    const { h } = window;
    await h.show({ strip: "a", index: 3 });
    h.setSeeking(false);
    await h.show(null, { committed: true });
    await h.sleep(200);
    const at200 = h.overlay().up;
    await h.sleep(400);
    return { at200, at600: h.overlay().up };
  });
  assert.equal(timing.at200, true, "a committed release stays up for at least ~300 ms");
  assert.equal(timing.at600, false, "and clears once the player is not seeking");

  // Released while seeking: stays until seeking goes false.
  const seeking = await page.evaluate(async () => {
    const { h } = window;
    await h.show({ strip: "a", index: 3 });
    h.setSeeking(true);
    await h.show(null, { committed: true });
    await h.sleep(1500);
    const during = h.overlay().up;
    h.setSeeking(false);
    await h.sleep(50);
    return { during, after: h.overlay().up };
  });
  assert.equal(seeking.during, true, "a committed release stays up while the player seeks");
  assert.equal(seeking.after, false, "and clears when seeking ends");

  // A seek that never ends: cleared after ~4 s regardless.
  const stuck = await page.evaluate(async () => {
    const { h } = window;
    await h.show({ strip: "a", index: 3 });
    h.setSeeking(true);
    await h.show(null, { committed: true });
    await h.sleep(3800);
    const at3800 = h.overlay().up;
    await h.sleep(800);
    return { at3800, at4600: h.overlay().up };
  });
  assert.equal(stuck.at3800, true);
  assert.equal(stuck.at4600, false, "a committed release clears after ~4 s regardless");

  // A new frame during a release cancels the pending clear.
  const resumed = await page.evaluate(async () => {
    const { h } = window;
    await h.show({ strip: "a", index: 3 });
    h.setSeeking(false);
    await h.show(null, { committed: true });
    await h.sleep(100);
    await h.show({ strip: "a", index: 4 });
    await h.sleep(600);
    return h.overlay();
  });
  assert.equal(resumed.up, true, "a new scrub after a release is not cleared by the old release");
  assert.equal(resumed.tile, 4);
  assert.deepEqual(page.errors, []);
  await page.context().close();
  console.log("ScrubFrameOverlay: tiles, loading, gap, renders, held, release timing, status region");
}

async function railTests() {
  const page = await open("rail");
  const { RAIL_TOP } = await page.evaluate(() => window.h.constants);
  const scroller = page.locator(".rail-scroller");
  await settle(page, 300);
  // Programmatic and playback scrolls are not scrubs.
  const top0 = await scroller.evaluate((element) => element.scrollTop);
  assert.ok(top0 > 0, "the rail scrolled to the playhead");
  await page.evaluate((value) => window.h.setPlayhead(value), RAIL_TOP - 9000);
  await scroller.evaluate((element) => { element.scrollTop += 37; });
  await settle(page, 400);
  assert.notEqual(await scroller.evaluate((element) => element.scrollTop), top0, "playback moved the rail");
  assert.deepEqual(await page.log(), [], "playback and programmatic scrolls do not scrub");

  // A user scroll reports positions, then ends committed and seeks.
  await scroller.hover();
  await page.mouse.wheel(0, 300);
  await page.waitLog("entry.seek !== undefined", "the settled scroll seeks");
  const log = await page.log();
  const epochs = log.filter((entry) => Number.isFinite(entry.epoch));
  assert.ok(epochs.length > 0, "a wheel scroll reports scrub positions");
  assert.ok(epochs.at(-1).epoch < RAIL_TOP - 9000, "scrolling down scrubs earlier");
  const endIndex = log.findIndex((entry) => entry.end);
  const seekIndex = log.findIndex((entry) => entry.seek !== undefined);
  assert.deepEqual(log[endIndex], { end: true, committed: true }, "the scrub ends committed");
  assert.ok(endIndex < seekIndex, "the end is reported before the seek");
  assert.equal(log[seekIndex].seek, epochs.at(-1).epoch, "it seeks to the last scrubbed time");
  assert.equal(log.filter((entry) => entry.end).length, 1);
  // The rail following the player there is not a scrub.
  await settle(page, 600);
  assert.equal((await page.log()).length, log.length, "the follow-up playback scroll does not scrub");

  // Unmount mid-scrub ends it.
  await page.clearLog();
  await page.evaluate(() => { window.h.unmountOnScrub = true; });
  await page.mouse.wheel(0, 200);
  await page.waitForFunction(() => !document.querySelector(".rail-scroller"));
  await settle(page, 300);
  const unmounted = await page.log();
  assert.ok(unmounted.some((entry) => Number.isFinite(entry.epoch)));
  assert.deepEqual(ends(unmounted), [{ end: true, committed: false }], "unmounting mid-scrub ends it, uncommitted");
  assert.equal(unmounted.some((entry) => entry.seek !== undefined), false, "and does not seek");
  assert.deepEqual(page.errors, []);
  await page.context().close();
  console.log("useVerticalScrub: user scrolls scrub and commit, playback scrolls do not, unmount ends");
}

async function edgeTests() {
  {
    const page = await open("edge");
    const track = await page.locator(".edge-rail-track").boundingBox();
    assert.ok(track.height > 400, "the edge rail track is laid out");
    const x = track.x + track.width / 2;
    await page.mouse.move(x, track.y + track.height * 0.25);
    await page.mouse.down();
    await page.mouse.move(x, track.y + track.height * 0.5, { steps: 4 });
    await page.waitForSelector(".edge-rail-loupe");
    await settle(page, 300);
    assert.equal(await page.locator(".edge-rail-loupe img").count(), 0, "with onScrub and a camera the loupe shows no image");
    assert.equal(page.requests.length, 0);
    await page.mouse.up();
    const log = await page.log();
    const epochs = log.filter((entry) => Number.isFinite(entry.epoch));
    assert.ok(epochs.length >= 4, "pointer moves report scrub positions");
    assert.ok(epochs.at(-1).epoch < epochs[0].epoch, "moving down scrubs earlier");
    const endIndex = log.findIndex((entry) => entry.end);
    assert.deepEqual(log[endIndex], { end: true, committed: true }, "pointer up ends committed");
    assert.deepEqual(log.slice(endIndex + 1), [{ seek: epochs.at(-1).epoch }], "then seeks to the last position");
    assert.equal(await page.locator(".edge-rail-loupe").count(), 0);

    // Unmount mid-gesture.
    await page.clearLog();
    await page.mouse.move(x, track.y + track.height * 0.3);
    await page.mouse.down();
    await page.mouse.move(x, track.y + track.height * 0.4, { steps: 2 });
    await page.evaluate(() => window.h.unmount());
    await page.mouse.up();
    const unmounted = await page.log();
    assert.ok(unmounted.some((entry) => Number.isFinite(entry.epoch)));
    assert.deepEqual(ends(unmounted), [{ end: true, committed: false }], "unmounting mid-gesture ends the scrub");
    assert.deepEqual(page.errors, []);
    await page.context().close();
  }
  // Without onScrub the loupe keeps its own picture (shows the check above
  // looks at the right element).
  {
    const page = await open("edge", "&scrub=false");
    const track = await page.locator(".edge-rail-track").boundingBox();
    await page.mouse.move(track.x + track.width / 2, track.y + track.height * 0.25);
    await page.mouse.down();
    await page.mouse.move(track.x + track.width / 2, track.y + track.height * 0.4, { steps: 2 });
    await page.waitForSelector(".edge-rail-loupe img", { state: "attached", timeout: 2000 });
    await page.mouse.up();
    await page.context().close();
  }
  console.log("EdgeRail: reports moves, ends committed on up, ends on unmount, no loupe image");
}

async function classicTests() {
  {
    const page = await open("classic");
    const { CLASSIC_START } = await page.evaluate(() => window.h.constants);
    const box = await page.getByRole("slider", { name: "Recording timeline" }).boundingBox();
    assert.ok(box.width > 500, "the classic track is laid out");
    const y = box.y + box.height / 2;
    const at = (fraction) => box.x + box.width * fraction;
    // A seek drag reports positions and ends committed.
    await page.mouse.move(at(0.1), y);
    await page.mouse.down();
    await page.mouse.move(at(0.5), y, { steps: 4 });
    await page.mouse.up();
    let log = await page.log();
    const epochs = log.filter((entry) => Number.isFinite(entry.epoch)).map((entry) => entry.epoch);
    assert.ok(epochs.length >= 4, "a seek drag reports scrub positions");
    assert.ok(Math.abs(epochs[0] - (CLASSIC_START + 360)) < 30, `starts at the pointer (${epochs[0] - CLASSIC_START})`);
    assert.ok(Math.abs(epochs.at(-1) - (CLASSIC_START + 1800)) < 30, `follows the pointer (${epochs.at(-1) - CLASSIC_START})`);
    const endIndex = log.findIndex((entry) => entry.end);
    assert.deepEqual(log[endIndex], { end: true, committed: true }, "a seek drag ends committed");
    assert.equal(log.filter((entry) => entry.end).length, 1);
    const seek = log.find((entry) => entry.seek !== undefined);
    assert.ok(seek && log.indexOf(seek) > endIndex, "and then seeks");
    assert.ok(Math.abs(seek.seek - epochs.at(-1)) < 30);

    // A cancelled drag ends uncommitted and does not seek.
    await page.clearLog();
    await page.mouse.move(at(0.2), y);
    await page.mouse.down();
    await page.mouse.move(at(0.3), y, { steps: 2 });
    await page.getByRole("slider", { name: "Recording timeline" }).evaluate((input) => {
      input.dispatchEvent(new PointerEvent("pointercancel", { bubbles: true, pointerId: 1, pointerType: "mouse" }));
    });
    await page.mouse.up();
    log = await page.log();
    assert.ok(log.some((entry) => Number.isFinite(entry.epoch)));
    assert.deepEqual(ends(log), [{ end: true, committed: false }], "a cancelled drag ends uncommitted");
    assert.equal(log.some((entry) => entry.seek !== undefined), false, "a cancelled drag does not seek");

    // Unmount mid-drag.
    await page.clearLog();
    await page.mouse.move(at(0.2), y);
    await page.mouse.down();
    await page.mouse.move(at(0.25), y, { steps: 2 });
    await page.evaluate(() => window.h.unmount());
    await page.mouse.up();
    log = await page.log();
    assert.ok(log.some((entry) => Number.isFinite(entry.epoch)));
    assert.deepEqual(ends(log), [{ end: true, committed: false }], "unmounting mid-drag ends the scrub");
    assert.equal(page.requests.length, 0, "the classic scrubber requests no strips itself");
    assert.deepEqual(page.errors, []);
    await page.context().close();
  }
  // A touch swipe pans and ends uncommitted (only if it had scrubbed; a pan
  // never does).
  {
    const page = await open("classic", "", { hasTouch: true, isMobile: true });
    const box = await page.getByRole("slider", { name: "Recording timeline" }).boundingBox();
    const y = box.y + box.height / 2;
    const client = await page.context().newCDPSession(page);
    const touch = (type, x) => client.send("Input.dispatchTouchEvent", { type, touchPoints: type === "touchEnd" ? [] : [{ x, y, id: 1 }] });
    await touch("touchStart", box.x + box.width * 0.8);
    for (const fraction of [0.75, 0.65, 0.55, 0.45]) await touch("touchMove", box.x + box.width * fraction);
    await touch("touchEnd");
    await settle(page);
    const log = await page.log();
    assert.ok(log.some((entry) => entry.pan !== undefined), `the swipe pans (${JSON.stringify(log)})`);
    assert.equal(log.some((entry) => Number.isFinite(entry.epoch)), false, "a pan does not scrub");
    assert.equal(log.some((entry) => entry.seek !== undefined), false, "a pan does not seek");
    assert.deepEqual(ends(log), [], "a pan that never scrubbed reports no end");

    // A touch on the playhead fine-scrubs, then releases committed.
    await page.clearLog();
    const playheadX = box.x + box.width * (1200 / 3600);
    await touch("touchStart", playheadX);
    for (const dx of [5, 10, 20, 30]) await touch("touchMove", playheadX + dx);
    await touch("touchEnd");
    await settle(page);
    const fine = await page.log();
    assert.ok(fine.some((entry) => Number.isFinite(entry.epoch)), `a playhead touch scrubs (${JSON.stringify(fine)})`);
    assert.deepEqual(ends(fine), [{ end: true, committed: true }], "a touch scrub ends committed on release");
    assert.deepEqual(page.errors, []);
    await page.context().close();
  }
  console.log("RecordingTimeline: seek drag commits, cancel/unmount end uncommitted, pans do not scrub");
}

try {
  server = await createServer({
    root: join(dirname(fileURLToPath(import.meta.url)), ".."), configFile: false, logLevel: "error",
    cacheDir: join(temporary, "vite-cache"), server: { host: "127.0.0.1", port: 0 },
    plugins: [{ name: "scrub-frame-react-fixture", configureServer(vite) {
      vite.middlewares.use((req, res, next) => {
        if (req.url.split("?")[0] !== "/") return next();
        res.setHeader("Content-Type", "text/html");
        res.end('<!doctype html><html data-theme="dark"><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><div id="root"></div><script type="module" src="/tests/fixtures/scrub-frame-react.jsx"></script></body></html>');
      });
    } }],
  });
  await server.listen();
  browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
  const only = process.env.SCRUB_FRAME_ONLY || "";
  for (const [name, run] of [["hook", hookTests], ["overlay", overlayTests], ["rail", railTests], ["edge", edgeTests], ["classic", classicTests]]) {
    if (!only || only.split(",").includes(name)) await run();
  }
  console.log("Scrub-frame React checks passed");
} finally {
  await browser?.close();
  await server?.close();
  rmSync(temporary, { recursive: true, force: true });
}
