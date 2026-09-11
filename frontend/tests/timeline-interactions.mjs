import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { chromium, webkit } from "playwright";

// TIMELINE_INTERACTIONS_SERVE=1 keeps the actual page available to Safari/iAB.
// TIMELINE_INTERACTIONS_BROWSER=webkit selects Playwright's native HLS engine.
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const temporary = mkdtempSync(join(tmpdir(), "timeline-interactions-"));
const media = join(temporary, "media");
const requests = [];
const date = "2026-09-09";
const dayStart = Date.parse(`${date}T00:00:00Z`) / 1000;
const cameras = ["gate", "yard", "driveway"].map((id) => ({ id, name: id[0].toUpperCase() + id.slice(1), enabled: true, recording: true, sub_recording: true, online: true }));
let server;
let browser;
const streams = new Set();
function json(res, payload) { res.setHeader("Content-Type", "application/json"); res.setHeader("Cache-Control", "no-store"); res.end(JSON.stringify(payload)); }
function serveFile(req, res, file) {
  const size = statSync(file).size;
  let start = 0; let end = size - 1;
  res.setHeader("Content-Type", ({ ".ts": "video/mp2t", ".mp4": "video/mp4", ".jpg": "image/jpeg" })[extname(file)] || "application/octet-stream");
  res.setHeader("Accept-Ranges", "bytes");
  if (req.headers.range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
    if (!match || (!match[1] && !match[2])) { res.writeHead(416); res.end(); return; }
    start = match[1] ? Number(match[1]) : Math.max(0, size - Number(match[2]));
    if (match[1] && match[2]) end = Math.min(end, Number(match[2]));
    if (start > end || start >= size) { res.writeHead(416, { "Content-Range": `bytes */${size}` }); res.end(); return; }
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
try {
  mkdirSync(media);
  execFileSync(process.env.FFMPEG_PATH || "ffmpeg", ["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=12:d=30", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "120", "-sc_threshold", "0", "-f", "hls", "-hls_time", "10", "-hls_playlist_type", "vod", "-hls_segment_filename", join(media, "%d.ts"), join(media, "index.m3u8")]);
  execFileSync(process.env.FFMPEG_PATH || "ffmpeg", ["-hide_banner", "-loglevel", "error", "-i", join(media, "0.ts"), "-c", "copy", "-movflags", "+faststart", join(media, "segment.mp4")]);
  execFileSync(process.env.FFMPEG_PATH || "ffmpeg", ["-hide_banner", "-loglevel", "error", "-i", join(media, "0.ts"), "-frames:v", "1", join(media, "preview.jpg")]);
  server = await createServer({ root, configFile: false, logLevel: "error", cacheDir: join(temporary, "vite"), server: { host: "127.0.0.1", port: Number(process.env.TIMELINE_INTERACTIONS_PORT) || 0 }, plugins: [{ name: "full-timeline-fixture", configureServer(vite) {
    vite.middlewares.use((req, res, next) => {
      const url = new URL(req.url, "http://localhost");
      const path = url.pathname;
      if (path === "/" || path === "/timeline") {
        res.setHeader("Content-Type", "text/html"); res.end('<!doctype html><html data-theme="dark"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full Timeline interaction fixture</title></head><body><div id="root"></div><script type="module" src="/tests/fixtures/timeline-interactions.jsx"></script></body></html>'); return;
      }
      if (path === "/timeline-interactions-requests.json") { json(res, requests); return; }
      if (path.startsWith("/api/") || path.startsWith("/timeline-interactions-media/")) requests.push({ id: requests.length + 1, path, query: url.search, method: req.method, range: req.headers.range || "", at: Date.now() });
      if (path === "/api/events/stream") { res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" }); res.write(': fixture connected\n\n'); streams.add(res); req.on("close", () => streams.delete(res)); return; }
      if (path === "/api/cameras") { json(res, cameras); return; }
      if (path === "/api/config") { json(res, { cameras, detector: { tracking: { camera_transition_routes: [{ from_camera: "gate", to_camera: "yard", bidirectional: true }] } } }); return; }
      if (path === "/api/semantic-search/status") { json(res, { enabled: false, ready: false }); return; }
      if (path === "/api/exports") { json(res, { exports: [] }); return; }
      if (path === "/api/system/status") { json(res, {}); return; }
      const recording = /^\/api\/cameras\/([^/]+)\/recordings\/(.+)$/.exec(path);
      if (recording) {
        const kind = recording[2];
        const start = Number(url.searchParams.get("start_epoch"));
        const end = Number(url.searchParams.get("end_epoch"));
        if (kind === "day") { json(res, { available_sources: ["main", "live"], availability: [{ start_epoch: start + 6 * 3600, end_epoch: start + 22 * 3600 }], incidents: [] }); return; }
        if (kind === "window") { json(res, { start_epoch: start, end_epoch: end, recordings: rows(start, end) }); return; }
        if (kind === "updates") { json(res, { availability: [], incidents: [] }); return; }
        if (kind === "day.m3u8") {
          const lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:10", "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-PLAYLIST-TYPE:VOD"];
          for (let i = 0; i < Math.ceil((end - start) / 10); i++) { if (i && i % 3 === 0) lines.push("#EXT-X-DISCONTINUITY"); lines.push("#EXTINF:10.000,", `/timeline-interactions-media/${i % 3}.ts?window=${start}&part=${i}`); }
          lines.push("#EXT-X-ENDLIST"); res.setHeader("Content-Type", "application/vnd.apple.mpegurl"); res.end(lines.join("\n") + "\n"); return;
        }
        if (kind === "segment.mp4") { serveFile(req, res, join(media, "segment.mp4")); return; }
        if (kind === "preview.jpg") { serveFile(req, res, join(media, "preview.jpg")); return; }
      }
      if (/^\/timeline-interactions-media\/[012]\.ts$/.test(path)) { serveFile(req, res, join(media, path.split("/").at(-1))); return; }
      if (path.startsWith("/api/")) { res.writeHead(404); res.end(`Unmocked fixture API: ${path}`); return; }
      next();
    });
  } }] });
  await server.listen();
  const url = `${server.resolvedUrls.local[0]}timeline?camera=gate&date=${date}&at=${dayStart + 12 * 3600 + 5}&source=main`;
  console.log(`Full RecordingsPage fixture: ${url}\nDiagnostics: ${server.resolvedUrls.local[0]}timeline-interactions-requests.json`);
  if (process.env.TIMELINE_INTERACTIONS_SERVE === "1") {
    await new Promise(done => { process.once("SIGINT", done); process.once("SIGTERM", done); });
  } else {
    const engine = process.env.TIMELINE_INTERACTIONS_BROWSER || "chromium";
    const macChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
    browser = engine === "webkit" ? await webkit.launch() : await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || (existsSync(macChrome) ? macChrome : undefined), headless: true });
    const page = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(() => window.timelineFixture?.snapshot().readyState >= 2);
    if (process.env.TIMELINE_INTERACTIONS_SCREENSHOT) await page.screenshot({ path: process.env.TIMELINE_INTERACTIONS_SCREENSHOT });
    await runInteractions(page, url, engine);
    assert.deepEqual(errors, [], "full mobile page must not raise JavaScript errors");
    const desktop = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    await desktop.goto(url);
    await desktop.waitForFunction(() => window.timelineFixture?.snapshot().readyState >= 2);
    await desktop.locator('.recordings-toolbar-day-controls .recordings-v2-date-toggle').click();
    await desktop.getByRole("dialog", { name: "Choose recording day" }).getByRole("button", { name: "8", exact: true }).click();
    await desktop.waitForFunction(() => new URL(location.href).searchParams.get("date") === "2026-09-08");
    await desktop.getByRole("group", { name: "Visible time scale" }).getByRole("button", { name: "2h", exact: true }).click();
    await desktop.waitForFunction(() => window.timelineFixture.snapshot().rangeSeconds === 7200);
    console.log("Full Timeline interaction checks passed");
  }
} finally {
  await browser?.close();
  for (const response of streams) response.end();
  await server?.close();
  rmSync(temporary, { recursive: true, force: true });
}

async function runInteractions(page, url, engine) {
  const read = () => page.evaluate(() => window.timelineFixture.snapshot());
  const playlistCount = () => requests.filter(request => request.path.endsWith("/day.m3u8")).length;
  const waitReady = () => page.waitForFunction(() => {
    const state = window.timelineFixture?.snapshot();
    return state?.readyState >= 2 && !state.seeking && Number.isFinite(state.playhead)
      && !state.notices.some(message => /Loading recording/.test(message));
  });
  const calendar = page.locator('.recordings-commandbar input[type="date"][aria-label="Recording day"]');
  assert.ok(await calendar.isVisible(), "mobile recording day needs an operable native control");
  const commandBar = await page.locator(".recordings-commandbar").boundingBox();
  for (const control of [calendar, page.getByLabel("Timeline camera", { exact: true })]) {
    const bounds = await control.boundingBox();
    assert.ok(bounds.y >= commandBar.y && bounds.y + bounds.height <= commandBar.y + commandBar.height,
      "mobile date and camera controls must stay inside the visible command bar");
  }
  const controls = await page.locator(".recordings-v2-controls").boundingBox();
  const timeline = await page.locator(".recordings-v2-timeline").boundingBox();
  assert.ok(timeline.y + timeline.height <= controls.y + controls.height + 1,
    "the range navigation must not push the timeline outside its reserved height");
  await calendar.fill("2026-09-08");
  await page.waitForFunction(() => new URL(location.href).searchParams.get("date") === "2026-09-08");
  await waitReady();
  const camera = page.getByLabel("Timeline camera", { exact: true });
  assert.equal(await camera.locator("option").count(), 3, "every fixture camera must remain selectable");
  await camera.selectOption("yard");
  await page.waitForFunction(() => window.timelineFixture.snapshot().context?.camera_id === "yard");
  await waitReady();
  // Reset to a known midday anchor so directional pan assertions never depend
  // on the production policy for choosing a first clip on a different day.
  await page.goto(url);
  await waitReady();
  const scale = page.getByLabel("Visible time span", { exact: true });
  assert.ok(await scale.isVisible());
  const beforeScale = await read();
  for (const hours of [4, 24, 1]) {
    await scale.selectOption(String(hours));
    await page.waitForFunction(seconds => window.timelineFixture.snapshot().rangeSeconds === seconds, hours * 3600);
  }
  assert.equal((await read()).videoId, beforeScale.videoId, "changing the visible span must retain the hero video");
  const beforePan = await read();
  const beforePanManifests = playlistCount();
  for (let i = 0; i < 3; i++) {
    const previous = JSON.stringify((await read()).ticks);
    await page.getByRole("button", { name: "Earlier time range", exact: true }).click();
    await page.waitForFunction(ticks => JSON.stringify(window.timelineFixture.snapshot().ticks) !== ticks, previous);
  }
  for (let i = 0; i < 3; i++) await page.getByRole("button", { name: "Later time range", exact: true }).click();
  await page.waitForFunction(ticks => JSON.stringify(window.timelineFixture.snapshot().ticks) === ticks, JSON.stringify(beforePan.ticks));
  const slider = page.getByRole("slider", { name: "Recording timeline", exact: true });
  const box = await slider.boundingBox();
  assert.ok(box && box.width > 100);
  if (engine === "chromium") {
    // Real touch input exercises pointer capture and swipe/tap discrimination.
    const cdp = await page.context().newCDPSession(page);
    const x = box.x + box.width * 0.72;
    const y = box.y + box.height * 0.5;
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x, y }] });
    for (let i = 1; i <= 8; i++) {
      await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: x - box.width * 0.4 * i / 8, y }] });
      await page.waitForTimeout(20);
    }
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await page.waitForFunction(ticks => JSON.stringify(window.timelineFixture.snapshot().ticks) !== ticks, JSON.stringify(beforePan.ticks));
    assert.ok(Math.abs((await read()).playhead - beforePan.playhead) < 1, "swiping the viewport must not seek the hero");
    await cdp.detach();
  }
  const beforeWheel = JSON.stringify((await read()).ticks);
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  if (engine === "webkit") {
    await slider.dispatchEvent("wheel", { deltaX: 180, deltaY: 0 });
  } else {
    await page.mouse.wheel(180, 0);
  }
  await page.waitForFunction(ticks => JSON.stringify(window.timelineFixture.snapshot().ticks) !== ticks, beforeWheel);
  assert.equal(playlistCount(), beforePanManifests, "viewport panning must not request a new hero playlist");
  assert.equal((await read()).videoId, beforePan.videoId);

  await page.goto(url);
  await waitReady();
  async function tapAtFraction(fraction) {
    const state = await read();
    const start = state.playhead - state.rangeOffset;
    const currentBox = await slider.boundingBox();
    const x = currentBox.x + currentBox.width * fraction;
    const y = currentBox.y + currentBox.height / 2;
    const target = start + state.rangeSeconds * fraction;
    await page.touchscreen.tap(x, y);
    const tolerance = state.rangeSeconds / currentBox.width + 1;
    await page.waitForFunction(({ target, tolerance }) => {
      const s = window.timelineFixture.snapshot();
      const mediaTarget = ((target % 900) + 900) % 900;
      return s.readyState >= 2 && !s.seeking && Math.abs(s.playhead - target) < tolerance
        && Math.abs(s.currentTime - mediaTarget) < tolerance
        && !s.notices.some(message => /Loading recording/.test(message));
    }, { target, tolerance });
    return await read();
  }
  const initial = await read();
  const initialManifests = playlistCount();
  for (const fraction of [0.53, 0.56, 0.54]) {
    const after = await tapAtFraction(fraction);
    assert.equal(after.videoId, initial.videoId, "same-window seeks must keep the same hero video");
    assert.equal(after.src, initial.src, "same-window seeks must retain their source");
  }
  assert.equal(playlistCount(), initialManifests, "same-window seeks must reuse the loaded playlist");
  if ((await read()).paused) await page.locator(".recording-hero-controls").getByRole("button", { name: "Play", exact: true }).click();
  let releaseWindow;
  let windowRequested;
  const blockedWindow = new Promise(resolve => { releaseWindow = resolve; });
  const windowStarted = new Promise(resolve => { windowRequested = resolve; });
  const delayWindow = async route => { windowRequested(); await blockedWindow; await route.continue(); };
  await page.route("**/recordings/window?**", delayWindow);
  const crossSeek = tapAtFraction(0.86);
  await windowStarted;
  assert.equal((await read()).paused, true, "a pending seek stops outgoing footage immediately");
  const requestedPlayhead = (await read()).playhead;
  await page.evaluate(() => window.timelineFixture.video().dispatchEvent(new Event("ended")));
  assert.equal((await read()).playhead, requestedPlayhead, "an outgoing ended event cannot redirect the new seek");
  releaseWindow();
  const cross = await crossSeek;
  await page.unroute("**/recordings/window?**", delayWindow);
  assert.ok(playlistCount() > initialManifests, "cross-window seek must request another playlist");
  const selectedPlaylists = requests.filter(request => request.path.endsWith("/day.m3u8"));
  assert.ok(new Set(selectedPlaylists.map(request => new URLSearchParams(request.query).get("start_epoch"))).size > 1);
  const crossCount = playlistCount();
  const crossState = await read();
  const nextTarget = cross.playhead + 60;
  const currentStart = crossState.playhead - crossState.rangeOffset;
  const repeated = await tapAtFraction((nextTarget - currentStart) / crossState.rangeSeconds);
  assert.equal(repeated.src, cross.src);
  assert.equal(repeated.videoId, cross.videoId);
  assert.equal(playlistCount(), crossCount, "later seek in new window must not reload it");
  const windowCount = requests.filter(request => request.path.endsWith("/window")).length;
  const revisit = await read();
  await tapAtFraction((initial.playhead + 120 - (revisit.playhead - revisit.rangeOffset)) / revisit.rangeSeconds);
  assert.equal(requests.filter(request => request.path.endsWith("/window")).length, windowCount,
    "revisiting a recent archive window reuses its metadata");
  if ((await read()).paused) await page.locator(".recording-hero-controls").getByRole("button", { name: "Play", exact: true }).click();
  const playing = await read();
  await page.waitForFunction(previous => {
    const s = window.timelineFixture.snapshot();
    return !s.paused && !s.seeking && s.playhead > previous + 1;
  }, playing.playhead);
  await page.locator(".recording-hero-controls").getByRole("button", { name: "Pause", exact: true }).click();
  assert.equal((await read()).error, null);

  const beforeCamera = await read();
  const beforeCameraRequests = requests.length;
  let releaseCameraWindow;
  let cameraWindowRequested;
  const cameraWindowGate = new Promise(resolve => { releaseCameraWindow = resolve; });
  const cameraWindowStarted = new Promise(resolve => { cameraWindowRequested = resolve; });
  let cameraWindowFinished;
  const cameraWindowDone = new Promise(resolve => { cameraWindowFinished = resolve; });
  const delayCameraWindow = async route => { cameraWindowRequested(); await cameraWindowGate; await route.continue(); cameraWindowFinished(); };
  await page.route("**/yard/recordings/window?**", delayCameraWindow);
  await page.getByRole("button", { name: "Show Yard recording at the current time", exact: true }).click();
  await cameraWindowStarted;
  await page.waitForTimeout(150);
  const prematurePlaylists = requests.slice(beforeCameraRequests).filter(request => request.path.endsWith("/yard/recordings/day.m3u8"));
  releaseCameraWindow();
  await cameraWindowDone;
  assert.equal(prematurePlaylists.length, 0, "camera switch must wait for its own metadata before requesting a playlist");
  await page.unroute("**/yard/recordings/window?**", delayCameraWindow);
  await page.waitForFunction(() => {
    const s = window.timelineFixture.snapshot();
    return s.context?.camera_id === "yard" && s.readyState >= 2 && !s.paused && !s.seeking;
  }).catch(async error => { console.error("Camera switch state:", await read()); throw error; });
  assert.equal((await read()).videoId, beforeCamera.videoId, "camera changes retain Safari's authorized video element");
  assert.ok(Math.abs((await read()).playhead - beforeCamera.playhead) < 5, "camera switch retains the selected time");

  // Drag the visible playhead instead of panning the surrounding time range.
  if (engine === "chromium") {
    await page.locator(".recording-hero-controls").getByRole("button", { name: "Pause", exact: true }).click();
    const fineBefore = await read();
    const fineBox = await slider.boundingBox();
    const fineX = fineBox.x + fineBox.width * fineBefore.rangeOffset / fineBefore.rangeSeconds;
    const fineY = fineBox.y + fineBox.height / 2;
    const fineInput = await page.context().newCDPSession(page);
    await fineInput.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x: fineX, y: fineY }] });
    await fineInput.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: fineX + 40, y: fineY }] });
    await fineInput.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await fineInput.detach();
    await page.waitForFunction(target => {
      const s = window.timelineFixture.snapshot();
      return !s.seeking && !s.paused && Math.abs(s.playhead - target) < 3;
    }, fineBefore.playhead + 40);
    assert.deepEqual((await read()).ticks, fineBefore.ticks, "fine scrub must retain the browsed time range");
  }
}
