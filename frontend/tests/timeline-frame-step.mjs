// Frame stepping on the full Timeline page (RecordingsPage), in Chromium:
// the step controls and keys, the exact-frame overlay and time readout,
// segment boundaries, a gap, holding a control, resuming play from the
// stepped frame, and what ends stepping (a scrub, another camera, play).
// The page is tests/fixtures/timeline-interactions.jsx against mocked APIs;
// frames.json and frame.jpg are answered here, with frame times at
// 19.84 fps that are not a 1/20 grid.
//   node tests/timeline-frame-step.mjs
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createReadStream, mkdirSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { chromium } from "playwright";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const temporary = mkdtempSync(join(tmpdir(), "timeline-frame-step-"));
const media = join(temporary, "media");
const date = "2026-09-09";
const dayStart = Date.parse(`${date}T00:00:00Z`) / 1000;
const noon = dayStart + 12 * 3600;
const cameras = ["gate", "yard"].map((id) => ({ id, name: id[0].toUpperCase() + id.slice(1), enabled: true, recording: true, sub_recording: true, online: true }));
// Frame stepping sees no recording in [noon + 20, noon + 40).
const GAP = { start: noon + 20, end: noon + 40 };
const FPS = 19.84;
const frameLog = [];
let frameDelayMs = 0;
let server;
let browser;

function segmentFrames(start) {
  const frames = [];
  for (let k = 0; start + 0.013 + k / FPS < start + 10; k += 1) frames.push(Number((start + 0.013 + k / FPS).toFixed(6)));
  return frames;
}
const inGap = (epoch) => epoch >= GAP.start && epoch < GAP.end;
function frameAt(epoch) {
  const start = Math.floor(epoch / 10) * 10;
  const frames = segmentFrames(start);
  let index = 0;
  for (let i = 0; i < frames.length; i += 1) if (frames[i] <= epoch + 0.0005) index = i;
  return frames[index];
}
function json(res, payload) { res.setHeader("Content-Type", "application/json"); res.setHeader("Cache-Control", "no-store"); res.end(JSON.stringify(payload)); }
function serveFile(req, res, file) {
  const size = statSync(file).size;
  let start = 0; let end = size - 1;
  res.setHeader("Content-Type", ({ ".ts": "video/mp2t", ".jpg": "image/jpeg" })[extname(file)] || "application/octet-stream");
  res.setHeader("Accept-Ranges", "bytes");
  if (req.headers.range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
    start = match[1] ? Number(match[1]) : Math.max(0, size - Number(match[2]));
    if (match[1] && match[2]) end = Math.min(end, Number(match[2]));
    res.statusCode = 206; res.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
  }
  res.setHeader("Content-Length", end - start + 1);
  if (req.method === "HEAD") res.end();
  else createReadStream(file, { start, end }).on("error", () => res.destroy()).pipe(res);
}
function rows(start, end) {
  return Array.from({ length: Math.ceil((end - start) / 10) }, (_, i) => ({
    start_epoch: start + i * 10, end_epoch: Math.min(end, start + (i + 1) * 10),
    duration_seconds: 10, media_start: i * 10, media_end: (i + 1) * 10,
    path: `/fixture/${start + i * 10}.mp4`, name: `${start + i * 10}.mp4`, size_bytes: 10000,
  }));
}

const settle = (page, ms = 150) => page.waitForTimeout(ms);
const framesOf = (epoch) => segmentFrames(Math.floor(epoch / 10) * 10);
const label = (epoch) => {
  const date = new Date(Math.round(epoch * 1000));
  const hours = date.getUTCHours() % 12 || 12;
  return `${hours}:${String(date.getUTCMinutes()).padStart(2, "0")}:${String(date.getUTCSeconds()).padStart(2, "0")}.${String(date.getUTCMilliseconds()).padStart(3, "0")}`;
};

try {
  mkdirSync(media);
  execFileSync(process.env.FFMPEG_PATH || "ffmpeg", ["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=12:d=30", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "12", "-sc_threshold", "0", "-f", "hls", "-hls_time", "10", "-hls_playlist_type", "vod", "-hls_segment_filename", join(media, "%d.ts"), join(media, "index.m3u8")]);
  execFileSync(process.env.FFMPEG_PATH || "ffmpeg", ["-hide_banner", "-loglevel", "error", "-i", join(media, "0.ts"), "-frames:v", "1", join(media, "preview.jpg")]);
  server = await createServer({ root, configFile: false, logLevel: "error", cacheDir: join(temporary, "vite"), server: { host: "127.0.0.1", port: 0 }, plugins: [{ name: "frame-step-fixture", configureServer(vite) {
    vite.middlewares.use((req, res, next) => {
      const url = new URL(req.url, "http://localhost");
      const path = url.pathname;
      if (path === "/" || path === "/timeline") {
        res.setHeader("Content-Type", "text/html"); res.end('<!doctype html><html data-theme="dark"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frame step fixture</title></head><body><div id="root"></div><script type="module" src="/tests/fixtures/timeline-interactions.jsx"></script></body></html>'); return;
      }
      if (path === "/api/events/stream") { res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" }); res.write(": fixture\n\n"); req.on("close", () => res.end()); return; }
      if (path === "/api/cameras") { json(res, cameras); return; }
      if (path === "/api/config") { json(res, { cameras, detector: { tracking: { camera_transition_routes: [] } } }); return; }
      if (path === "/api/semantic-search/status") { json(res, { enabled: false, ready: false }); return; }
      if (path === "/api/exports") { json(res, { exports: [] }); return; }
      if (path === "/api/system/status") { json(res, {}); return; }
      const recording = /^\/api\/cameras\/([^/]+)\/recordings\/(.+)$/.exec(path);
      if (recording) {
        const [, camera, kind] = recording;
        const start = Number(url.searchParams.get("start_epoch"));
        const end = Number(url.searchParams.get("end_epoch"));
        const epoch = Number(url.searchParams.get("epoch"));
        if (kind === "day") { json(res, { available_sources: ["main", "live"], availability: [{ start_epoch: start + 6 * 3600, end_epoch: start + 22 * 3600 }], incidents: [] }); return; }
        if (kind === "window") { json(res, { start_epoch: start, end_epoch: end, recordings: rows(start, end) }); return; }
        if (kind === "updates") { json(res, { availability: [], incidents: [] }); return; }
        if (kind === "day.m3u8") {
          const lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:10", "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-PLAYLIST-TYPE:VOD"];
          for (let i = 0; i < Math.ceil((end - start) / 10); i++) { if (i && i % 3 === 0) lines.push("#EXT-X-DISCONTINUITY"); lines.push("#EXTINF:10.000,", `/frame-step-media/${i % 3}.ts?window=${start}&part=${i}`); }
          lines.push("#EXT-X-ENDLIST"); res.setHeader("Content-Type", "application/vnd.apple.mpegurl"); res.end(lines.join("\n") + "\n"); return;
        }
        if (kind === "preview.jpg") { serveFile(req, res, join(media, "preview.jpg")); return; }
        if (kind === "frames.json") {
          frameLog.push({ kind: "list", camera, epoch, source: url.searchParams.get("source") });
          if (inGap(epoch)) { res.writeHead(404); res.end(); return; }
          const segment = Math.floor(epoch / 10) * 10;
          json(res, {
            source: url.searchParams.get("source"), start: segment, end: segment + 10, frames: segmentFrames(segment),
            previous: inGap(segment - 0.001) ? null : segment - 0.001,
            next: inGap(segment + 10) ? null : segment + 10,
          });
          return;
        }
        if (kind === "frame.jpg") {
          const actual = frameAt(epoch);
          const entry = { kind: "frame", camera, epoch, actual, width: Number(url.searchParams.get("width")), at: Date.now() };
          frameLog.push(entry);
          const level = Math.round((actual * FPS) % 200) + 30;
          const body = `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="48"><rect width="64" height="48" fill="rgb(${level},${level},${level})"/></svg>`;
          setTimeout(() => {
            res.writeHead(200, { "Content-Type": "image/svg+xml", "X-SurvNG-Actual-Timestamp": actual.toFixed(6), "Cache-Control": "private, max-age=3600" });
            res.end(body);
          }, frameDelayMs);
          return;
        }
      }
      if (/^\/frame-step-media\/[012]\.ts$/.test(path)) { serveFile(req, res, join(media, path.split("/").at(-1))); return; }
      if (path.startsWith("/api/")) { res.writeHead(404); res.end(`Unmocked fixture API: ${path}`); return; }
      next();
    });
  } }] });
  await server.listen();
  browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
  await run();
  console.log("Timeline frame stepping checks passed");
} finally {
  await browser?.close();
  await server?.close();
  rmSync(temporary, { recursive: true, force: true });
}

async function open(at, { viewport = { width: 1440, height: 1000 }, ...options } = {}) {
  const page = await browser.newPage({ viewport, ...options });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.errors = errors;
  await page.goto(`${server.resolvedUrls.local[0]}timeline?camera=gate&date=${date}&at=${at}&source=main`);
  await page.waitForFunction(() => {
    const state = window.timelineFixture?.snapshot();
    return state?.readyState >= 2 && !state.seeking && Number.isFinite(state.playhead);
  });
  const hud = page.locator(".recording-hero-controls");
  if (!(await page.evaluate(() => window.timelineFixture.snapshot().paused))) {
    await hud.getByRole("button", { name: "Pause", exact: true }).click();
  }
  await page.waitForFunction(() => window.timelineFixture.snapshot().paused);
  page.hud = hud;
  page.read = () => page.evaluate(() => window.timelineFixture.snapshot());
  page.overlay = () => page.evaluate(() => {
    const element = document.querySelector(".recording-frame-step");
    return element ? {
      epoch: Number(element.dataset.epoch) || null, drawn: element.dataset.drawn, loading: element.classList.contains("loading"),
      notice: element.querySelector(".recording-frame-step-notice")?.textContent || "",
      label: document.querySelector(".timeline-hud-day .recordings-v2-date-toggle")?.textContent || "",
    } : null;
  });
  // Wait until the overlay shows the frame `epoch`, loaded.
  page.shows = (epoch, message = "") => page.waitForFunction((wanted) => {
    const element = document.querySelector(".recording-frame-step");
    return element && !element.classList.contains("loading") && Math.abs(Number(element.dataset.epoch) - wanted) < 1e-5;
  }, epoch, { timeout: 5000 }).catch(async () => { throw new Error(`${message}: overlay ${JSON.stringify(await page.overlay())}`); });
  return page;
}

// Wait for a seek to finish, then pause the player if it is playing.
// A play() still pending reads as paused, so wait a while for it to start.
async function pauseWhenSettled(page) {
  await page.waitForFunction(() => {
    const state = window.timelineFixture.snapshot();
    return state.readyState >= 2 && !state.seeking;
  });
  const playing = await page.waitForFunction(() => !window.timelineFixture.snapshot().paused, null, { timeout: 1500 })
    .then(() => true, () => false);
  if (playing) await page.hud.getByRole("button", { name: "Pause", exact: true }).click();
  await page.waitForFunction(() => window.timelineFixture.snapshot().paused);
}

// Step forward one frame at a time until `epoch` is shown.
async function stepTo(page, epoch, message) {
  for (let tries = 0; tries < 12; tries += 1) {
    await page.keyboard.press(".");
    await page.waitForFunction(() => {
      const element = document.querySelector(".recording-frame-step");
      return element && !element.classList.contains("loading") && element.dataset.epoch;
    });
    const shown = (await page.overlay()).epoch;
    if (Math.abs(shown - epoch) < 1e-5) return;
    assert.ok(shown < epoch, `${message}: stepped past ${epoch} to ${shown}`);
  }
  throw new Error(`${message}: not reached`);
}

async function run() {
  // Keys and controls step one real frame at a time; the readout shows it
  // to the millisecond.
  {
    const page = await open(noon + 5);
    const start = (await page.read()).playhead;
    assert.equal(await page.overlay(), null, "nothing over the player before stepping");
    const frames = framesOf(start);
    const onScreen = frames.filter((time) => time <= start + 0.0005).at(-1);
    const index = frames.indexOf(onScreen);
    await page.keyboard.press(".");
    await page.shows(frames[index + 1], "'.' steps one frame forward");
    if (process.env.FRAME_STEP_SCREENSHOTS) await page.screenshot({ path: join(process.env.FRAME_STEP_SCREENSHOTS, "desktop.png") });
    let overlay = await page.overlay();
    assert.equal(overlay.drawn, "true");
    assert.ok(overlay.label.includes(label(frames[index + 1])), `the readout shows the frame's time to the millisecond (${overlay.label})`);
    await page.keyboard.press(".");
    await page.shows(frames[index + 2], "a second step");
    await page.keyboard.press(",");
    await page.keyboard.press(",");
    await page.keyboard.press(",");
    await page.shows(frames[index - 1], "',' steps back past where stepping began");
    await page.hud.getByRole("button", { name: "Next frame", exact: true }).click();
    await page.shows(frames[index], "the Next frame control");
    await page.hud.getByRole("button", { name: "Previous frame", exact: true }).click();
    await page.shows(frames[index - 1], "the Previous frame control");
    // Keyboard activation of a control steps once.
    await page.hud.getByRole("button", { name: "Next frame", exact: true }).focus();
    await page.keyboard.press("Enter");
    await page.shows(frames[index], "Enter on the control");
    const widths = new Set(frameLog.filter((entry) => entry.kind === "frame").map((entry) => entry.width));
    const heroWidth = await page.evaluate(() => document.querySelector(".recording-hero-media").clientWidth);
    assert.equal(widths.size, 1, "one frame width throughout");
    const [width] = widths;
    assert.ok(width >= heroWidth * 0.9 && width < Math.max(640, heroWidth * 1.9), `frames about as wide as the player (${width} for ${heroWidth} px)`);
    // Typing in a field is not stepping.
    const listsBefore = frameLog.length;
    await page.evaluate(() => {
      const input = document.createElement("input");
      input.id = "typing";
      document.body.append(input);
    });
    await page.focus("#typing");
    await page.keyboard.type(".,.");
    await settle(page);
    assert.equal(frameLog.length, listsBefore, "'.' typed into a field does not step");
    await page.evaluate(() => document.getElementById("typing").remove());

    // Once stepping pauses the player is seeked to the stepped frame...
    const stepped = frames[index];
    await page.waitForFunction((wanted) => Math.abs(window.timelineFixture.snapshot().playhead - wanted) < 0.1, stepped, { timeout: 5000 });
    assert.ok(await page.overlay(), "... and the exact frame stays up over it");
    // ... and play resumes from it.
    await page.hud.getByRole("button", { name: "Play", exact: true }).click();
    await page.waitForFunction(() => !window.timelineFixture.snapshot().paused && !document.querySelector(".recording-frame-step"), null, { timeout: 5000 });
    const resumed = await page.read();
    assert.ok(resumed.playhead >= stepped - 0.1 && resumed.playhead < stepped + 1.5, `play resumes at the stepped frame (${resumed.playhead} vs ${stepped})`);
    // Stepping is not offered while playing.
    assert.equal(await page.hud.getByRole("button", { name: "Next frame", exact: true }).isDisabled(), true);
    const before = frameLog.length;
    await page.keyboard.press(".");
    await settle(page);
    assert.equal(frameLog.length, before, "'.' does nothing while playing");
    assert.deepEqual(page.errors, []);
    await page.close();
  }

  // Across a segment boundary, and back.
  {
    const page = await open(noon + 9.8);
    const last = framesOf(noon).at(-1);
    const next = framesOf(noon + 10)[0];
    await stepTo(page, last, "the segment's last frame");
    await page.keyboard.press(".");
    await page.shows(next, "the next segment's first frame");
    await page.keyboard.press(",");
    await page.shows(last, "and back");
    assert.equal(await page.locator(".recordings-v2-error").count(), 0);
    assert.deepEqual(page.errors, []);
    await page.close();
  }

  // Into a time with no recording: stops at the last frame and says so.
  {
    const page = await open(noon + 19.8);
    const last = framesOf(noon + 10).at(-1);
    await stepTo(page, last, "the last recorded frame");
    await page.keyboard.press(".");
    await page.waitForFunction(() => /No recording after this frame/.test(document.querySelector(".recording-frame-step")?.textContent || ""));
    const overlay = await page.overlay();
    assert.ok(Math.abs(overlay.epoch - last) < 1e-5, "the last frame stays up");
    assert.equal(await page.locator('[role="status"]', { hasText: "No recording after this frame" }).count(), 1, "the notice is announced");
    assert.equal(await page.locator(".recordings-v2-error").count(), 0, "a gap is not an error");
    await page.keyboard.press(",");
    await page.waitForFunction(() => !document.querySelector(".recording-frame-step-notice"));
    assert.deepEqual(page.errors, []);
    await page.close();
  }

  // Holding a control: steps repeat; slow frames are skipped, not queued.
  {
    const page = await open(noon + 2);
    frameDelayMs = 250;
    const first = frameLog.length;
    const button = page.hud.getByRole("button", { name: "Next frame", exact: true });
    const box = await button.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await settle(page, 1500);
    await page.mouse.up();
    const wanted = await page.evaluate(() => document.querySelector(".timeline-hud-day .recordings-v2-date-toggle").textContent);
    await settle(page, 800);
    const requested = frameLog.slice(first).filter((entry) => entry.kind === "frame");
    const start = framesOf(noon + 2).filter((time) => time <= noon + 2.0005).at(-1);
    const overlay = await page.overlay();
    const moved = framesOf(noon + 2).indexOf(overlay.epoch) - framesOf(noon + 2).indexOf(start);
    assert.ok(moved >= 10 && moved <= 16, `held for 1.5 s: about 12 steps (${moved})`);
    assert.ok(requested.length < moved, `frames passed while held are not all fetched (${requested.length} requests for ${moved} steps)`);
    assert.ok(wanted.includes(label(overlay.epoch)), "the last frame wanted is the one shown");
    assert.equal(overlay.loading, false);
    frameDelayMs = 0;
    assert.deepEqual(page.errors, []);
    await page.close();
  }

  // Play pressed before stepping has settled: playback is not paused again
  // by the seek that would have followed.
  {
    const page = await open(noon + 5);
    await page.keyboard.press(".");
    await page.waitForSelector(".recording-frame-step[data-drawn=true]");
    await page.waitForTimeout(800); // settled: the player has been seeked
    // Playback that takes a second to start, as a native HLS reload can.
    // A pause() meanwhile cancels it, as a pause during a pending play does.
    await page.evaluate(() => {
      const { play, pause } = HTMLMediaElement.prototype;
      const pending = new WeakMap();
      HTMLMediaElement.prototype.play = function slowPlay() {
        const token = {};
        pending.set(this, token);
        return new Promise((done) => setTimeout(done, 1000)).then(() => {
          if (pending.get(this) !== token) throw new DOMException("The play() request was interrupted by a call to pause().", "AbortError");
          return play.call(this);
        });
      };
      HTMLMediaElement.prototype.pause = function cancellingPause() {
        pending.delete(this);
        return pause.call(this);
      };
    });
    // A step and Play within the settle delay.
    await page.keyboard.press(".");
    await page.hud.getByRole("button", { name: "Play", exact: true }).click();
    await page.waitForTimeout(2000);
    const state = await page.read();
    assert.equal(state.paused, false, "still playing after the settle delay has passed");
    assert.equal(await page.overlay(), null);
    assert.deepEqual(page.errors, []);
    await page.close();
  }

  // A step right after a seek into another 15-minute window starts from the
  // time asked for, not from the new window's start.
  {
    const page = await open(noon + 5);
    const before = frameLog.length;
    await page.hud.getByRole("button", { name: "Back 10 seconds", exact: true }).click();
    await page.keyboard.press(".");
    await page.waitForSelector(".recording-frame-step[data-drawn=true]", { timeout: 5000 });
    const first = frameLog.slice(before).find((entry) => entry.kind === "list");
    assert.ok(Math.abs(first.epoch - (noon - 5)) < 1, `steps from the time sought (${first.epoch - noon} s from noon)`);
    assert.ok(Math.abs((await page.overlay()).epoch - (noon - 5)) < 1);
    assert.deepEqual(page.errors, []);
    await page.close();
  }

  // A scrub, a seek, another camera: the stepped frame comes down.
  {
    const page = await open(noon + 5);
    await page.keyboard.press(".");
    await page.waitForSelector(".recording-frame-step[data-drawn=true]");
    const scroller = page.locator(".event-rail-scroller");
    const box = await scroller.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -40);
    await page.waitForFunction(() => !document.querySelector(".recording-frame-step"), null, { timeout: 3000 });
    // The rail commits its scrub as a seek that plays; pause once it has.
    await pauseWhenSettled(page);

    await page.keyboard.press(".");
    await page.waitForSelector(".recording-frame-step[data-drawn=true]");
    await page.hud.getByRole("button", { name: "Back 10 seconds", exact: true }).click();
    await page.waitForFunction(() => !document.querySelector(".recording-frame-step"), null, { timeout: 3000 });
    await pauseWhenSettled(page);

    await page.keyboard.press(".");
    await page.waitForSelector(".recording-frame-step[data-drawn=true]");
    const listsBefore = frameLog.filter((entry) => entry.camera === "yard").length;
    await page.locator(".timeline-camera-picker button").first().click();
    await page.getByRole("option", { name: /Yard/ }).click();
    await page.waitForFunction(() => window.timelineFixture.snapshot().context?.camera_id === "yard");
    assert.equal(await page.overlay(), null, "another camera takes the stepped frame down");
    // Switching camera starts playback.
    await pauseWhenSettled(page);
    await page.keyboard.press(".");
    await page.waitForSelector(".recording-frame-step[data-drawn=true]");
    assert.ok(frameLog.filter((entry) => entry.camera === "yard").length > listsBefore, "the new camera's frames are asked for");
    assert.deepEqual(page.errors, []);
    await page.close();
  }

  // The same in the Catch up and Cameras layouts; not in All cameras.
  for (const layout of ["Catch up", "Cameras"]) {
    const page = await open(noon + 5);
    await page.getByRole("group", { name: "Timeline view" }).getByRole("button", { name: layout, exact: true }).click();
    await page.waitForFunction(() => window.timelineFixture.snapshot().paused);
    const frames = framesOf((await page.read()).playhead);
    await page.hud.getByRole("button", { name: "Next frame", exact: true }).click();
    await page.waitForSelector(".recording-frame-step[data-drawn=true]");
    const overlay = await page.overlay();
    assert.ok(frames.includes(overlay.epoch), `${layout}: steps to a real frame`);
    assert.deepEqual(page.errors, []);
    await page.close();
  }
  {
    const page = await open(noon + 5);
    await page.locator(".timeline-camera-picker button").first().click();
    await page.getByRole("option", { name: /All cameras/ }).click();
    await page.waitForSelector(".all-cameras-controls");
    assert.equal(await page.getByRole("button", { name: "Next frame", exact: true }).count(), 0, "no stepping in the all-cameras grid");
    const before = frameLog.length;
    await page.keyboard.press(".");
    await settle(page);
    assert.equal(frameLog.length, before, "nor by key");
    await page.close();
  }

  // A phone: the controls are full-size tap targets and a tap steps once.
  {
    const page = await open(noon + 5, { viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, deviceScaleFactor: 3 });
    const button = page.hud.getByRole("button", { name: "Next frame", exact: true });
    await button.scrollIntoViewIfNeeded();
    const box = await button.boundingBox();
    assert.ok(box.width >= 44 && box.height >= 44, `a 44 px tap target (${box.width}x${box.height})`);
    const before = frameLog.filter((entry) => entry.kind === "frame").length;
    await button.tap();
    await page.waitForSelector(".recording-frame-step[data-drawn=true]");
    const frames = frameLog.filter((entry) => entry.kind === "frame").slice(before);
    assert.equal(frames[0].width, 1280, "a phone gets a 1280 px frame for a 390 pt player at 3x");
    if (process.env.FRAME_STEP_SCREENSHOTS) await page.screenshot({ path: join(process.env.FRAME_STEP_SCREENSHOTS, "phone.png") });
    assert.deepEqual(page.errors, []);
    await page.close();
  }
}
