import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

// Requires FFmpeg with H.264/AAC and a Chromium browser. Override either binary
// with FFMPEG_PATH or PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH when needed.
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const temporary = mkdtempSync(join(tmpdir(), "native-recording-video-"));
const held = new Map();
const requests = [];
let browser;
let server;
let releaseManual;

function hold(name) {
  let release;
  const promise = new Promise((resolve) => { release = resolve; });
  held.set(name, promise);
  return () => { held.delete(name); release(); };
}

try {
  for (const [name, color] of [["red", "red"], ["green", "lime"], ["blue", "blue"]]) {
    execFileSync(process.env.FFMPEG_PATH || "ffmpeg", ["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", `color=c=${color}:s=320x180:r=24:d=2`, "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", "-shortest", join(temporary, `${name}.mp4`)]);
  }
  server = await createServer({
    root, configFile: false, logLevel: "error", cacheDir: join(temporary, "vite-cache"),
    server: { host: "127.0.0.1", port: Number(process.env.NATIVE_RECORDING_PORT) || 0 },
    plugins: [{ name: "native-recording-video-fixture", configureServer(vite) {
      vite.middlewares.use((req, res, next) => {
        // Serve the same fixture to an existing browser when process launch is
        // sandboxed: NATIVE_RECORDING_SERVE=1 node tests/native-recording-video.mjs.
        if (process.env.NATIVE_RECORDING_SERVE === "1") {
          if (req.url === "/test/hold") {
            releaseManual = hold("blue-delayed.mp4"); res.end("held"); return;
          }
          if (req.url === "/test/release") {
            releaseManual?.(); res.end("released"); return;
          }
          if (req.url?.startsWith("/clips/")) {
            const name = req.url.split("/").at(-1);
            const send = () => {
              if (name.startsWith("broken")) { res.statusCode = 404; res.end(); return; }
              const color = name.split("-")[0].replace(".mp4", "");
              if (!["red", "green", "blue"].includes(color)) { res.statusCode = 404; res.end(); return; }
              res.setHeader("Content-Type", "video/mp4");
              const bytes = readFileSync(join(temporary, `${color}.mp4`));
              const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || "");
              res.setHeader("Accept-Ranges", "bytes");
              if (range) {
                const start = Number(range[1]);
                const end = Math.min(bytes.length - 1, range[2] ? Number(range[2]) : bytes.length - 1);
                res.statusCode = 206;
                res.setHeader("Content-Range", `bytes ${start}-${end}/${bytes.length}`);
                res.setHeader("Content-Length", end - start + 1);
                res.end(bytes.subarray(start, end + 1));
              } else {
                res.setHeader("Content-Length", bytes.length);
                res.end(bytes);
              }
            };
            if (held.has(name)) held.get(name).then(send);
            else send();
            return;
          }
        }
        if (req.url === "/") {
          res.setHeader("Content-Type", "text/html");
          res.end('<html><body style="margin:0"><div id="root"></div><script type="module" src="/tests/fixtures/native-recording-video.jsx"></script></body></html>');
          return;
        }
        next();
      });
    } }],
  });
  await server.listen();
  if (process.env.NATIVE_RECORDING_SERVE === "1") {
    console.log(`Native recording fixture: ${server.resolvedUrls.local[0]}`);
    await new Promise(() => {});
  }
  const macChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || (existsSync(macChrome) ? macChrome : undefined);
  browser = await chromium.launch({ executablePath, headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
  const page = await browser.newPage({ viewport: { width: 640, height: 480 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/clips/*", async (route) => {
    const name = new URL(route.request().url()).pathname.split("/").at(-1);
    requests.push(name);
    if (held.has(name)) await held.get(name);
    if (name.startsWith("broken")) {
      await route.fulfill({ status: 404, body: "Missing recording" }).catch(() => {});
      return;
    }
    const color = name.split("-")[0].replace(".mp4", "");
    await route.fulfill({ status: 200, contentType: "video/mp4", body: readFileSync(join(temporary, `${color}.mp4`)) }).catch(() => {});
  });
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(() => Boolean(window.harness));
  await page.evaluate(() => {
    window.standbyPlayback = [];
    document.addEventListener("play", (event) => {
      if (event.isTrusted && event.target instanceof HTMLVideoElement && event.target !== window.harness.video()) {
        window.standbyPlayback.push(event.target.getAttribute("src"));
      }
    }, true);
  });
  const configure = (patch) => page.evaluate((value) => window.harness.configure(value), patch);
  const snapshot = () => page.evaluate(() => window.harness.snapshot());
  const waitVisible = (src) => page.waitForFunction((expected) => window.harness.snapshot().some((v) => v.src === expected && v.visible && v.activeRef && v.ready >= 2 && !v.seeking), src);
  async function assertColor(channel, label) {
    // Sample the composited screenshot, not drawImage(video): a decoded but
    // hidden element alone cannot prove that the user sees a retained frame.
    const screenshot = await page.locator("#stage").screenshot();
    const rgb = await page.evaluate(async (data) => {
      const image = new Image();
      image.src = `data:image/png;base64,${data}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width; canvas.height = image.height;
      const ctx = canvas.getContext("2d"); ctx.drawImage(image, 0, 0);
      return [...ctx.getImageData(image.width / 2, image.height / 2, 1, 1).data].slice(0, 3);
    }, screenshot.toString("base64"));
    assert.ok(rgb[channel] > 180 && rgb.filter((_, index) => index !== channel).every((value) => value < 70), `${label}: expected channel ${channel}, saw ${rgb}`);
  }

  await configure({ src: "/clips/red.mp4", nextSrc: "/clips/green.mp4", muted: false });
  await waitVisible("/clips/red.mp4");
  await page.waitForFunction(() => window.harness.snapshot().some((v) => v.src === "/clips/green.mp4" && v.ready >= 2));
  const warm = (await snapshot()).find((v) => v.src === "/clips/green.mp4");
  assert.ok(requests.includes("green.mp4"), "next clip requested before outgoing clip ends");
  assert.ok(warm.paused && warm.muted && !warm.visible && !warm.activeRef, "preloading must neither play nor expose standby audio");
  await assertColor(0, "first decoded frame");

  await configure({ advance: true, playing: true });
  await page.evaluate(async () => { const video = window.harness.video(); video.currentTime = video.duration - 0.2; await video.play(); });
  await waitVisible("/clips/green.mp4");
  assert.equal((await snapshot()).find((v) => v.activeRef).id, warm.id, "handoff must promote the same preloaded DOM element");
  await assertColor(1, "automatic clip boundary");
  assert.equal(await page.evaluate(() => window.harness.events.filter((e) => e.name === "ended" && e.src === "/clips/red.mp4").length), 1);
  await page.evaluate(() => window.harness.video().pause());
  await configure({ advance: false, playing: false, playbackRate: 1.5 });

  const releaseBlue = hold("blue-delayed.mp4");
  await configure({ src: "/clips/blue-delayed.mp4", nextSrc: "/clips/red-next.mp4", seek: 0.75 });
  await page.waitForFunction(() => window.harness.video()?.getAttribute("src") === "/clips/blue-delayed.mp4");
  await assertColor(1, "outgoing frame while incoming request is blocked");
  assert.ok((await snapshot()).find((v) => v.visible).ready >= 2, "outgoing retains its decoded frame");
  assert.ok(!requests.includes("red-next.mp4"), "lookahead cannot recycle the displayed outgoing element");
  releaseBlue();
  await waitVisible("/clips/blue-delayed.mp4");
  const sought = (await snapshot()).find((v) => v.activeRef);
  assert.ok(sought.paused && Math.abs(sought.time - 0.75) < 0.05, "paused source change preserves the requested seek");
  assert.equal(await page.evaluate(() => window.harness.video().playbackRate), 1.5);
  await assertColor(2, "new clip appears after paused seek completes");

  const releaseRed = hold("red-superseded.mp4");
  await configure({ src: "/clips/red-superseded.mp4", nextSrc: "", seek: 0.4 });
  await page.waitForFunction(() => window.harness.video()?.getAttribute("src") === "/clips/red-superseded.mp4");
  await assertColor(2, "frame survives first pending source");
  await configure({ src: "/clips/green-latest.mp4", seek: 0.6 });
  await waitVisible("/clips/green-latest.mp4");
  releaseRed();
  await assertColor(1, "latest request wins after superseding delayed source");
  assert.ok(Math.abs((await snapshot()).find((v) => v.activeRef).time - 0.6) < 0.05);
  const stale = await page.evaluate(() => {
    const before = window.harness.events.length;
    const standby = [...document.querySelectorAll("video")].find((v) => v !== window.harness.video());
    for (const type of ["loadedmetadata", "seeked", "timeupdate", "play", "pause", "error", "ended", "loadeddata", "canplay"]) standby.dispatchEvent(new Event(type));
    return window.harness.events.length - before;
  });
  assert.equal(stale, 0, "standby/stale events must not reach active callbacks");
  await assertColor(1, "stale events cannot change the displayed clip");

  await configure({ src: "/clips/broken.mp4", nextSrc: "/clips/red-next.mp4" });
  await page.waitForFunction(() => window.harness.events.some((e) => e.name === "error" && e.src === "/clips/broken.mp4"));
  await assertColor(1, "failed incoming clip retains outgoing frame");
  assert.ok((await snapshot()).every((v) => v.paused), "no standby or failed clip starts playing");
  assert.deepEqual(await page.evaluate(() => window.standbyPlayback), [], "standby must never play, even transiently");
  assert.ok(await page.evaluate(() => window.harness.events.every((event) => event.ownsRef)), "every callback exposes the current active DOM video");
  assert.deepEqual(errors, [], "browser must have no uncaught errors");
  console.log("native recording video browser tests passed (preload, real-node handoff, retained pixels, paused seek, superseded sources, stale events, error retention, standby audio)");
} finally {
  await browser?.close();
  await server?.close();
  rmSync(temporary, { recursive: true, force: true });
}
