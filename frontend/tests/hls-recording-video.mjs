import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, webkit } from "playwright";
import { createServer } from "vite";

// HLS_RECORDING_FIXTURES points to exported production playlists + manifest.json.
// Generate those with the repository's Python environment:
//   python frontend/tests/fixtures/generate-recording-hls.py --output /tmp/survng-hls
//   HLS_RECORDING_FIXTURES=/tmp/survng-hls node frontend/tests/hls-recording-video.mjs
// Without that export, FFmpeg creates a portable H.264 playback smoke fixture.
// HLS_RECORDING_SERVE=1 leaves the visible fixture running for manual browser QA.
// HLS_RECORDING_BROWSER=webkit selects installed Playwright WebKit for native HLS.
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const temporary = mkdtempSync(join(tmpdir(), "hls-recording-video-"));
const normalFixtures = ["h264", "gap", "unknown", "hevc", "mixed"];
const requests = [];
let browser;
let server;
let releaseManual;

function portableFixture() {
  const directory = join(temporary, "media", "h264");
  mkdirSync(directory, { recursive: true });
  execFileSync(process.env.FFMPEG_PATH || "ffmpeg", [
    "-hide_banner", "-loglevel", "error",
    ...["red", "lime", "blue"].flatMap((color) => ["-f", "lavfi", "-i", `color=c=${color}:s=320x180:r=12:d=10`]),
    "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
    "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[video]", "-map", "[video]", "-map", "3:a",
    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-bf", "2", "-g", "120", "-sc_threshold", "0", "-c:a", "aac",
    "-f", "hls", "-hls_time", "10", "-hls_playlist_type", "vod", "-hls_segment_type", "fmp4",
    "-hls_fmp4_init_filename", "init.mp4", "-hls_segment_filename", join(directory, "%d.m4s"), join(directory, "index.m3u8"),
  ]);
  return { directory: join(temporary, "media"), description: "Portable FFmpeg H.264 smoke fixture (not a production remux export).",
    manifest: { h264: { playlist: "h264/index.m3u8", duration: 30, boundaries: [10, 20], colors: ["red", "lime", "blue"] } } };
}

function serveFile(req, res, file) {
  const size = statSync(file).size;
  const contentType = ({ ".m3u8": "application/vnd.apple.mpegurl", ".m4s": "video/mp4", ".mp4": "video/mp4", ".ts": "video/mp2t", ".aac": "audio/aac" })[extname(file)] || "application/octet-stream";
  res.setHeader("Content-Type", contentType);
  res.setHeader("Accept-Ranges", "bytes");
  res.setHeader("Cache-Control", "no-store");
  let start = 0;
  let end = size - 1;
  if (req.headers.range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
    if (!match || (!match[1] && !match[2])) {
      res.writeHead(416, { "Content-Range": `bytes */${size}` }); res.end(); return;
    }
    if (!match[1]) start = Math.max(0, size - Number(match[2]));
    else { start = Number(match[1]); if (match[2]) end = Math.min(end, Number(match[2])); }
    if (start > end || start >= size) {
      res.writeHead(416, { "Content-Range": `bytes */${size}` }); res.end(); return;
    }
    res.statusCode = 206;
    res.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
  }
  res.setHeader("Content-Length", end - start + 1);
  if (req.method === "HEAD") { res.end(); return; }
  createReadStream(file, { start, end }).on("error", () => res.destroy()).pipe(res);
}

try {
  const productionDirectory = process.env.HLS_RECORDING_FIXTURES || "";
  const data = productionDirectory
    ? { directory: resolve(productionDirectory), manifest: JSON.parse(readFileSync(join(productionDirectory, "manifest.json"), "utf8")), description: "Production recording HLS export: independent 10-second recordings, remuxed without video transcoding." }
    : portableFixture();
  const manifest = Object.fromEntries(normalFixtures.filter((name) => data.manifest[name]).map((name) => [name, data.manifest[name]]));
  const fixture = { description: data.description, playlists: Object.fromEntries(Object.entries(manifest).map(([name, value]) => [name, value.playlist])), manifest };
  assert.ok(fixture.playlists.h264, "fixture manifest must include h264");
  server = await createServer({
    root, configFile: false, logLevel: "error", cacheDir: join(temporary, "vite-cache"),
    server: { host: "127.0.0.1", port: Number(process.env.HLS_RECORDING_PORT) || 0 },
    plugins: [{ name: "recording-hls-fixture", configureServer(vite) {
      vite.middlewares.use((req, res, next) => {
        const url = new URL(req.url, "http://localhost");
        if (url.pathname === "/") {
          res.setHeader("Content-Type", "text/html");
          res.end('<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recording HLS browser checks</title></head><body><div id="root"></div><script type="module" src="/tests/fixtures/hls-recording-video.jsx"></script></body></html>'); return;
        }
        if (url.pathname === "/fixture-config.json" || url.pathname === "/fixture-requests.json") {
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(url.pathname === "/fixture-config.json" ? fixture : requests)); return;
        }
        if (!url.pathname.startsWith("/media/")) { next(); return; }
        requests.push({ path: url.pathname, query: url.search, range: req.headers.range || "", method: req.method });
        if (url.searchParams.has("offline")) { res.writeHead(503, { "Content-Type": "text/plain" }); res.end("Simulated recording storage outage"); return; }
        let file;
        try { file = resolve(data.directory, decodeURIComponent(url.pathname.slice("/media/".length))); }
        catch { res.writeHead(400); res.end(); return; }
        if (!file.startsWith(data.directory + sep) || !existsSync(file) || !statSync(file).isFile()) {
          res.writeHead(404); res.end("Fixture not found"); return;
        }
        serveFile(req, res, file);
      });
    } }],
  });
  await server.listen();
  const url = server.resolvedUrls.local[0];
  console.log(`${fixture.description}\nHLS browser fixture: ${url}`);
  // Exercise the actual HTTP server's byte-range behavior before any media test.
  const playlist = await fetch(`${url}media/${fixture.playlists.h264}`);
  assert.match(playlist.headers.get("content-type"), /mpegurl/);
  const text = await playlist.text();
  const init = /#EXT-X-MAP:URI="([^"]+)"/.exec(text)?.[1];
  assert.ok(init, "fMP4 fixture should advertise its init segment");
  const initUrl = new URL(init, `${url}media/${fixture.playlists.h264}`);
  const range = await fetch(initUrl, { headers: { Range: "bytes=0-15" } });
  assert.equal(range.status, 206); assert.equal((await range.arrayBuffer()).byteLength, 16);
  assert.match(range.headers.get("content-range"), /^bytes 0-15\//);
  const suffix = await fetch(initUrl, { headers: { Range: "bytes=-8" } });
  assert.equal(suffix.status, 206); assert.equal((await suffix.arrayBuffer()).byteLength, 8);
  const invalid = await fetch(initUrl, { headers: { Range: "bytes=999999999-" } });
  assert.equal(invalid.status, 416);

  if (process.env.HLS_RECORDING_SERVE === "1") {
    console.log("Serve-only mode: open the URL above. Ctrl-C stops the server.");
    await new Promise((resolveDone) => {
      releaseManual = resolveDone;
      process.once("SIGINT", resolveDone); process.once("SIGTERM", resolveDone);
    });
  } else {
    const engine = process.env.HLS_RECORDING_BROWSER || "chromium";
    assert.ok(["chromium", "webkit"].includes(engine), "HLS_RECORDING_BROWSER must be chromium or webkit");
    const macChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
    const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || (existsSync(macChrome) ? macChrome : undefined);
    browser = engine === "webkit"
      ? await webkit.launch({ headless: true })
      : await chromium.launch({ executablePath, headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
    const page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(() => window.hlsFixture?.snapshot().ready > 0 && window.hlsFixture.snapshot().readyState >= 2);
    const snapshot = () => page.evaluate(() => window.hlsFixture.snapshot());
    async function waitColor(channel) {
      await page.waitForFunction((expected) => {
        const video = window.hlsFixture.video();
        if (video.readyState < 2 || video.seeking) return false;
        const canvas = document.createElement("canvas"); canvas.width = canvas.height = 1;
        const ctx = canvas.getContext("2d"); ctx.drawImage(video, 0, 0, 1, 1);
        const rgb = [...ctx.getImageData(0, 0, 1, 1).data].slice(0, 3);
        return rgb[expected] > 100 && rgb.every((value, index) => index === expected || value < 70);
      }, channel);
    }
    const initial = await snapshot();
    assert.ok(initial.paused && Math.abs(initial.duration - 30) < 0.5);
    await waitColor(0);
    // Native WebKit HLS above 2× may enter I-frame trick-play mode. Timeline's
    // high-speed transport fallback is covered separately; exercise HLS at 2×.
    const continuousRate = engine === "webkit" || initial.transport === "Native HLS" ? 2 : 4;
    await page.getByRole("button", { name: `Playback speed ${continuousRate}×`, exact: true }).click();
    await page.getByRole("button", { name: "Play", exact: true }).click();
    await page.waitForFunction(() => window.hlsFixture.snapshot().time > 20.5);
    await page.getByRole("button", { name: "Pause", exact: true }).click();
    await waitColor(2);
    const crossed = await snapshot();
    assert.equal(crossed.id, initial.id, "one DOM video must span all recording segments");
    assert.equal(crossed.metadata, initial.metadata, "crossing clip boundaries must not reload media metadata");
    const frames = await page.evaluate(() => window.hlsFixture.frames);
    for (const channel of [0, 1, 2]) assert.ok(frames.some((frame) => frame.rgb[channel] > 100 && frame.rgb.every((value, i) => i === channel || value < 70)), `playback must decode color ${channel} across boundaries`);
    assert.ok(frames.every((frame) => Math.max(...frame.rgb) > 70), "decoded pixel samples must not be black");
    const playingFrames = frames.filter((frame) => !frame.paused && frame.playRun > 0);
    assert.ok(playingFrames.every((frame, index) => index === 0 || frame.time >= playingFrames[index - 1].time - 0.1), "media time must not wrap to earlier recordings at a segment boundary");
    assert.equal(await page.evaluate(() => window.hlsFixture.events.filter((event) => event.name === "error").length), 0, "normal HLS playback must not emit playback errors");
    assert.equal(await page.evaluate(() => window.hlsFixture.events.filter((event) => event.name === "ended").length), 0, "individual recordings must not end the HLS timeline");
    await page.evaluate(() => window.hlsFixture.seek(window.hlsFixture.video().duration - 0.15, true));
    await page.waitForFunction(() => window.hlsFixture.snapshot().ended && !window.hlsFixture.snapshot().playIntent);
    assert.equal(await page.evaluate(() => window.hlsFixture.events.filter((event) => event.name === "ended").length), 1, "only the full timeline end clears playback intent");
    for (const [time, channel] of [[9.75, 0], [10.25, 1], [20.25, 2]]) {
      await page.getByRole("button", { name: `Seek ${time}s`, exact: true }).click();
      await waitColor(channel);
      const sought = await snapshot();
      assert.ok(sought.paused && Math.abs(sought.time - time) < 0.1, `paused seek to ${time} must remain paused at the correct frame`);
    }
    await page.evaluate(() => { window.hlsFixture.video().playbackRate = 1; window.hlsFixture.seek(9.75, true); });
    await page.waitForFunction(() => window.hlsFixture.snapshot().time > 10.4);
    await waitColor(1);
    await page.getByRole("button", { name: "Seek 20.25s", exact: true }).click();
    await waitColor(2);
    assert.equal((await snapshot()).paused, false, "playing seek must preserve play intent");
    await page.getByRole("button", { name: "Pause", exact: true }).click();

    const beforeSwitch = await snapshot();
    await page.getByRole("button", { name: "Switch playlist URL", exact: true }).click();
    await page.waitForFunction((ready) => window.hlsFixture.snapshot().ready > ready, beforeSwitch.ready);
    await waitColor(1);
    const switched = await snapshot();
    assert.ok(switched.metadata > beforeSwitch.metadata && switched.paused && Math.abs(switched.time - 10.25) < 0.1, "changed playlist must deliver fresh metadata and preserve paused target");
    if (initial.transport === "Native HLS") assert.notEqual(switched.id, beforeSwitch.id, "native playlist replacement must create a fresh video element");
    else assert.equal(switched.id, beforeSwitch.id, "Shaka retains its attached video across playlist replacement");

    await page.getByRole("button", { name: "Simulate network failure", exact: true }).click();
    await page.waitForFunction(() => window.hlsFixture.error?.category === 1, null, { timeout: 30000 });
    assert.equal((await snapshot()).source.includes("offline=1"), true);
    await page.getByRole("button", { name: "Retry playlist", exact: true }).click();
    await page.waitForFunction((ready) => window.hlsFixture.snapshot().ready > ready && !window.hlsFixture.error, switched.ready);
    await waitColor(1);
    assert.ok((await snapshot()).paused, "network recovery retains pause intent");

    // Revisit the exact same two playlist URLs repeatedly, like scrubbing back
    // and forth between timeline windows. Native video identity changes only
    // at the window boundary; seeks and recording segments keep that identity.
    const errorsBeforeWindows = await page.evaluate(() => window.hlsFixture.events.filter((event) => event.name === "error").length);
    for (const [windowName, playing, target, channel] of [["B", false, 20.25, 2], ["A", true, 10.25, 1], ["B", true, 20.25, 2], ["A", false, 10.25, 1]]) {
      const before = await snapshot();
      await page.getByRole("button", { name: `Window ${windowName} · ${playing ? "playing" : "paused"}`, exact: true }).click();
      await page.waitForFunction(({ ready, target, playing }) => {
        const value = window.hlsFixture.snapshot();
        return value.ready > ready && value.readyState >= 2 && !value.seeking && value.paused === !playing
          && (playing ? value.time > target + 0.2 && value.time < target + 2 : Math.abs(value.time - target) < 0.1)
          && value.activeFrames > 0;
      }, { ready: before.ready, target, playing });
      await waitColor(channel);
      const loaded = await snapshot();
      assert.notEqual(loaded.source, before.source, "alternating windows must load distinct playlist URLs");
      assert.ok(loaded.metadata > before.metadata, "every revisited window must deliver fresh metadata");
      if (initial.transport === "Native HLS") assert.notEqual(loaded.id, before.id, "native HLS must replace the video when revisiting a window");
      else assert.equal(loaded.id, before.id, "Shaka must keep its attached video while changing windows");
      assert.equal(loaded.lastActiveFrame.id, loaded.id, "pixel samples must observe the new active element");
      assert.equal(loaded.lastActiveFrame.source, loaded.source, "decoded-frame evidence belongs to the loaded window");

      await page.evaluate(() => window.hlsFixture.seek(9.75, false));
      await page.waitForFunction(() => {
        const value = window.hlsFixture.snapshot();
        return value.paused && !value.seeking && Math.abs(value.time - 9.75) < 0.1;
      });
      await waitColor(0);
      await page.evaluate(() => window.hlsFixture.seek(10.25, false));
      await page.waitForFunction(() => {
        const value = window.hlsFixture.snapshot();
        return value.paused && !value.seeking && Math.abs(value.time - 10.25) < 0.1;
      });
      await waitColor(1);

      await page.evaluate(() => window.hlsFixture.seek(9.75, true));
      await page.waitForFunction(() => {
        const value = window.hlsFixture.snapshot();
        return !value.paused && !value.seeking && value.time > 10.4 && value.time < 12;
      });
      await waitColor(1);
      const progressed = await snapshot();
      assert.equal(progressed.id, loaded.id, "paused seeks and playing segment crossings must retain the window's video");
      assert.equal(progressed.metadata, loaded.metadata, "seeking within a playlist must not reload metadata");
      assert.equal(await page.evaluate(() => window.hlsFixture.events.filter((event) => event.name === "error").length), errorsBeforeWindows, "repeated window seeks must not produce playback errors");
      await page.getByRole("button", { name: "Pause", exact: true }).click();
    }

    // Gap/discontinuity exports exercise production timestamp continuity while
    // codec-changing fixtures remain explicit because HEVC availability varies.
    const extras = ["gap", "unknown", ...(process.env.HLS_RECORDING_HEVC === "1" ? ["hevc", "mixed"] : [])].filter((name) => fixture.playlists[name]);
    for (const name of extras) {
      const before = await snapshot();
      await page.getByRole("combobox", { name: "Recording fixture" }).selectOption(name);
      await page.waitForFunction((ready) => window.hlsFixture.snapshot().ready > ready, before.ready);
      const expected = fixture.manifest[name];
      assert.ok(Math.abs((await snapshot()).duration - expected.duration) < 0.5, `${name} duration must exclude recording-time gaps`);
      for (const [index, color] of expected.colors.entries()) {
        await page.evaluate((time) => window.hlsFixture.seek(time, false), index * 10 + 0.25);
        await waitColor({ red: 0, lime: 1, green: 1, blue: 2 }[color]);
      }
    }
    assert.deepEqual(pageErrors, [], "no uncaught browser errors");
    assert.ok(await page.evaluate(() => window.hlsFixture.events.every((event) => event.ownsRef)), "callbacks must use the forwarded DOM video");
    console.log(`HLS recording browser tests passed: ${initial.transport} at ${continuousRate}×; continuous 10-second clips, repeated window replacement and paused/playing seeks, fresh metadata, network retry${extras.length ? `, ${extras.join(", ")}` : ""}.`);
    if (process.env.HLS_RECORDING_HEVC !== "1") console.log("HEVC/mixed automated coverage not requested; use HLS_RECORDING_HEVC=1 on a compatible browser, or the visible codec selector.");
  }
} finally {
  if (releaseManual) { process.removeListener("SIGINT", releaseManual); process.removeListener("SIGTERM", releaseManual); }
  await browser?.close();
  await server?.close();
  rmSync(temporary, { recursive: true, force: true });
}
