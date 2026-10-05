import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const temporary = mkdtempSync(join(tmpdir(), "incident-recording-player-"));
let browser, server;
try {
  const clip = join(temporary, "segment.mp4");
  execFileSync(process.env.FFMPEG_PATH || "ffmpeg", ["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
    "color=c=blue:s=160x90:r=10:d=2", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", clip]);
  const bytes = readFileSync(clip);
  server = await createServer({
    root: join(dirname(fileURLToPath(import.meta.url)), ".."), configFile: false, logLevel: "error",
    cacheDir: join(temporary, "vite-cache"), server: { host: "127.0.0.1", port: 0 },
    plugins: [{ name: "incident-recording-fixture", configureServer(vite) {
      vite.middlewares.use((req, res, next) => {
        if (req.url.split("?")[0] !== "/") return next();
        res.setHeader("Content-Type", "text/html");
        res.end('<div id="root"></div><script type="module" src="/tests/fixtures/incident-recording-player.jsx"></script>');
      });
    } }],
  });
  await server.listen();
  browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
  // Exercise the original-MP4 transport selected on iOS, using actual decoded
  // media and natural ended events rather than synthetic episode completion.
  const page = await browser.newPage({ userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1" });
  const errors = [], requests = [];
  await page.setViewportSize({ width: 390, height: 844 });
  let rows = [900, 902, 904], failOriginal = false, failTranscode = false, missingSub = false;
  page.on("pageerror", error => errors.push(error.message));
  await page.route("**/api/**", async route => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith("/recordings/window")) {
      const start = Number(url.searchParams.get("start_epoch"));
      return route.fulfill({ json: { camera_id: "fixture", source: url.searchParams.get("source"), start_epoch: start, end_epoch: start + 900,
        recordings: (missingSub && url.searchParams.get("source") === "live" ? [] : rows).filter(epoch => epoch >= start && epoch < start + 900)
          .map(epoch => ({ start_epoch: epoch, end_epoch: epoch + 2, duration_seconds: 2 })) } });
    }
    if (!url.pathname.endsWith("/segment.mp4")) return route.fulfill({ status: 404 });
    requests.push({ source: url.searchParams.get("source"), epoch: Number(url.searchParams.get("epoch")), transcode: url.searchParams.get("mobile") === "true" });
    if (url.searchParams.get("mobile") === "true" ? failTranscode : failOriginal)
      return route.fulfill({ contentType: "video/mp4", body: "unsupported media fixture" });
    const range = /^bytes=(\d+)-(\d*)$/.exec(route.request().headers().range || "");
    const headers = { "Accept-Ranges": "bytes" };
    if (range) {
      const start = Number(range[1]), end = Math.min(bytes.length - 1, range[2] ? Number(range[2]) : bytes.length - 1);
      headers["Content-Range"] = `bytes ${start}-${end}/${bytes.length}`;
      return route.fulfill({ status: 206, contentType: "video/mp4", headers, body: bytes.subarray(start, end + 1) });
    }
    return route.fulfill({ contentType: "video/mp4", headers, body: bytes });
  });
  const open = async (query = "") => { requests.length = 0; await page.goto(server.resolvedUrls.local[0] + query); };
  const waitPlaying = epoch => page.waitForFunction(value => [...document.querySelectorAll("video")].some(video =>
    video.src && Number(new URL(video.src).searchParams.get("epoch")) === value && !video.paused && video.readyState >= 2), epoch);
  const waitComplete = () => page.waitForFunction(() => document.querySelector("#ended").textContent === "1");
  const seek = epoch => page.getByRole("slider").evaluate((input, value) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }, String(epoch));

  await open();
  for (const epoch of rows) {
    await waitPlaying(epoch);
    await page.waitForFunction(value => {
      const video = document.querySelector("video.active");
      return video?.src && Number(new URL(video.src).searchParams.get("epoch")) === value;
    }, epoch);
    const display = await page.evaluate(() => {
      const active = document.querySelector("video.active");
      const bounds = active.getBoundingClientRect();
      const topVideo = document.elementsFromPoint(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2)
        .find(element => element.tagName === "VIDEO");
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 1;
      const context = canvas.getContext("2d");
      context.drawImage(active, 0, 0, 1, 1);
      return { activeOnTop: topVideo === active,
        standbyHidden: [...document.querySelectorAll("video.standby")].every(video => getComputedStyle(video).visibility === "hidden"),
        blue: context.getImageData(0, 0, 1, 1).data[2] };
    });
    assert.equal(display.standbyHidden, true, "incident CSS must hide the inactive native video");
    assert.equal(display.activeOnTop, true, "a blank standby element must not cover a playing segment");
    assert.ok(display.blue > 150, "each displayed segment must have a decoded blue frame");
  }
  await waitComplete();
  assert.ok(requests.every(row => row.source === "live" && !row.transcode), "mobile starts with native Sub");
  assert.deepEqual([...new Set(requests.map(row => row.epoch))], rows, "every segment must play before episode completion");

  await open("?autoplay=false");
  await page.waitForFunction(() => [...document.querySelectorAll("video")].some(v => v.readyState >= 2));
  assert.equal(await page.locator("video").first().evaluate(v => v.paused), true);
  await seek(903);
  await waitPlaying(902);
  assert.equal(await page.locator("#ended").textContent(), "0", "cross-segment seek must not finish the episode");
  await seek(900.5);
  await waitPlaying(900);

  await open("?autoplay=false");
  await page.waitForFunction(() => document.querySelector("video.active")?.readyState >= 2);
  await seek(903);
  await waitPlaying(902);
  await page.getByRole("button", { name: "Pause recording", exact: true }).click();
  await page.getByRole("button", { name: "Main", exact: true }).click();
  await page.waitForFunction(() => {
    const video = document.querySelector("video.active");
    return video?.readyState >= 2 && new URL(video.src).searchParams.get("source") === "main";
  });
  assert.equal(await page.locator("video.active").evaluate(v => v.paused), true, "switch preserves pause");
  assert.ok(requests.some(row => row.source === "main" && row.epoch === 902 && !row.transcode), "switch preserves position");
  assert.equal(await page.getByRole("button", { name: "Main", exact: true }).getAttribute("aria-pressed"), "true");

  await page.setViewportSize({ width: 1280, height: 900 });
  await open("?autoplay=false");
  await page.waitForFunction(() => document.querySelector("video.active")?.readyState >= 2);
  assert.equal(requests[0].source, "main", "desktop defaults to Main");
  await page.getByRole("button", { name: "Sub", exact: true }).click();
  await page.waitForFunction(() => {
    const video = document.querySelector("video.active");
    return video?.readyState >= 2 && new URL(video.src).searchParams.get("source") === "live";
  });
  await page.setViewportSize({ width: 390, height: 844 });
  missingSub = true;
  await open();
  await waitPlaying(900);
  assert.equal(requests[0].source, "main", "missing Sub falls back to native Main");
  assert.equal(requests[0].transcode, false);
  missingSub = false;

  // Missing wall-clock intervals are skipped, not mistaken for episode end.
  rows = [900, 904];
  await open();
  await waitComplete();
  assert.deepEqual([...new Set(requests.map(row => row.epoch))], rows);

  rows = [1798, 1800, 1802];
  await open("?start=1798&end=1804");
  await waitComplete();
  assert.deepEqual([...new Set(requests.map(row => row.epoch))], rows, "handoff must cross the 15-minute window boundary");
  await open("?start=1798&end=1804&autoplay=false");
  await page.waitForFunction(() => [...document.querySelectorAll("video")].some(v => v.readyState >= 2));
  await seek(1802.5);
  await waitPlaying(1802);
  assert.deepEqual([...new Set(requests.map(row => row.epoch))], [1798, 1802], "cross-window seek must load only the requested segment");

  rows = [900, 902, 904]; failOriginal = true;
  await open();
  await waitPlaying(900);
  assert.ok(requests.some(row => row.transcode), "unsupported original must retry as compatible MP4");
  assert.deepEqual([...new Set(requests.map(row => `${row.source}:${row.transcode}`))],
    ["live:false", "main:false", "main:true"], "Sub failure tries native Main before transcoding Main");
  await waitComplete();
  assert.deepEqual([...new Set(requests.filter(row => row.transcode).map(row => row.epoch))], rows);

  failTranscode = true;
  await open();
  await page.getByText("No recording window found", { exact: true }).waitFor();
  assert.equal(requests.filter(row => row.transcode).length, 1, "transcode failure must not loop");

  failOriginal = false; failTranscode = false;
  await open("?autoplay=false");
  await page.waitForFunction(() => [...document.querySelectorAll("video")].some(v => v.readyState >= 2));
  await page.locator("video").first().evaluate(video => {
    Object.defineProperty(video, "error", { value: { code: 2, message: "network failure" } });
    video.dispatchEvent(new Event("error"));
  });
  await page.getByText("No recording window found", { exact: true }).waitFor();
  assert.equal(requests.some(row => row.transcode), false, "network failures must not trigger expensive transcoding");
  assert.deepEqual(errors, []);
  console.log("Incident native autoplay, segment advance, seeking, gaps, window rollover and compatibility fallback passed");
} finally {
  await browser?.close();
  await server?.close();
  rmSync(temporary, { recursive: true, force: true });
}
