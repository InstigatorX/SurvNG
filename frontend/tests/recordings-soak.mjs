import { chromium } from "playwright";

const baseUrl = process.env.SURVNG_URL || "http://127.0.0.1:8088";
const camera = process.env.CAMERA || "front-door";
const source = process.env.SOURCE || "main";
const soakSeconds = Math.max(10, Number(process.env.SOAK_SECONDS || 90));
const scrubCount = Math.max(4, Number(process.env.SCRUBS || 12));
const today = new Intl.DateTimeFormat("en-CA", {
  timeZone: process.env.TZ_NAME || "America/New_York",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
}).format(new Date());
const recordingDate = process.env.DATE || today;
const url = `${baseUrl}/recordings?camera=${encodeURIComponent(camera)}&date=${recordingDate}&source=${source}`;

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const failures = [];
page.on("pageerror", (error) => failures.push(`page: ${error.message}`));
page.on("requestfailed", (request) => {
  if (/recordings|\.m4s|\.mp4|\.m3u8/.test(request.url())) {
    const errorText = request.failure()?.errorText || "failed";
    if (errorText !== "net::ERR_ABORTED") failures.push(`request: ${errorText} ${request.url()}`);
  }
});

try {
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30_000 });
  const video = page.locator(".recordings-v2-player video");
  await video.waitFor({ state: "visible", timeout: 30_000 });
  await page.waitForFunction(() => {
    const element = document.querySelector(".recordings-v2-player video");
    return element && element.readyState >= 2 && element.videoWidth > 0;
  }, null, { timeout: 45_000 });

  await page.evaluate(() => {
    const element = document.querySelector(".recordings-v2-player video");
    window.__recordingSoak = { waiting: 0, stalled: 0, errors: 0, playing: 0 };
    element.addEventListener("waiting", () => { window.__recordingSoak.waiting += 1; });
    element.addEventListener("stalled", () => { window.__recordingSoak.stalled += 1; });
    element.addEventListener("error", () => { window.__recordingSoak.errors += 1; });
    element.addEventListener("playing", () => { window.__recordingSoak.playing += 1; });
  });

  const track = page.locator(".recordings-v2-track");
  const trackBox = await track.boundingBox();
  const ranges = await page.locator(".recordings-v2-track > span").evaluateAll((elements) => (
    elements.map((element) => ({
      left: Number.parseFloat(element.style.left),
      width: Number.parseFloat(element.style.width),
    })).filter((range) => Number.isFinite(range.left) && Number.isFinite(range.width) && range.width > 0)
  ));
  if (!trackBox || !ranges.length) throw new Error("recording timeline has no availability ranges");
  const firstRange = ranges[0];
  const lastRange = ranges[ranges.length - 1];
  const earlyPercent = firstRange.left + Math.min(firstRange.width * 0.15, 0.5);
  const latePercent = lastRange.left + Math.max(lastRange.width * 0.85, lastRange.width - 0.5);
  if (latePercent - earlyPercent < 2) throw new Error("recording day is too short for cross-window scrubbing");

  for (const [index, percent] of [earlyPercent, latePercent, earlyPercent].entries()) {
    const manifestRequest = page.waitForRequest(
      (request) => request.url().includes("/recordings/day.m3u8?"),
      { timeout: 20_000 },
    );
    await page.mouse.click(trackBox.x + (trackBox.width * percent) / 100, trackBox.y + trackBox.height / 2);
    await manifestRequest;
    await page.waitForFunction(() => {
      const element = document.querySelector(".recordings-v2-player video");
      return element && !element.seeking && element.readyState >= 2 && element.videoWidth > 0;
    }, null, { timeout: 30_000 });
    const before = await video.evaluate(async (element) => {
      await element.play();
      return element.currentTime;
    });
    await page.waitForTimeout(1500);
    const after = await video.evaluate((element) => element.currentTime);
    if (after < before + 0.5) failures.push(`window scrub ${index + 1}: playback did not advance (${before.toFixed(2)} -> ${after.toFixed(2)})`);
  }

  const duration = await video.evaluate((element) => element.duration);
  if (!Number.isFinite(duration) || duration < 30) throw new Error(`invalid media duration ${duration}`);
  const stride = Math.max(11, Math.floor(duration / (scrubCount + 2)));
  for (let index = 1; index <= scrubCount; index += 1) {
    const target = Math.min(duration - 4, index * stride + 9.25);
    await video.evaluate(async (element, nextTime) => {
      element.currentTime = nextTime;
      await element.play();
    }, target);
    await page.waitForFunction((expected) => {
      const element = document.querySelector(".recordings-v2-player video");
      return element && !element.seeking && element.readyState >= 2 && element.currentTime >= expected;
    }, target, { timeout: 20_000 });
    const before = await video.evaluate((element) => element.currentTime);
    await page.waitForTimeout(2200);
    const after = await video.evaluate((element) => element.currentTime);
    if (after < before + 0.8) failures.push(`scrub ${index}: playback did not advance (${before.toFixed(2)} -> ${after.toFixed(2)})`);
  }

  const continuousStart = Math.max(0, Math.min(duration - soakSeconds - 2, duration * 0.55));
  await video.evaluate(async (element, start) => {
    element.currentTime = start;
    await element.play();
  }, continuousStart);
  const startedAt = await video.evaluate((element) => element.currentTime);
  await page.waitForTimeout(soakSeconds * 1000);
  const endedAt = await video.evaluate((element) => element.currentTime);
  const stats = await page.evaluate(() => window.__recordingSoak);
  if (endedAt < startedAt + soakSeconds * 0.75) {
    failures.push(`continuous playback advanced only ${(endedAt - startedAt).toFixed(1)}s of ${soakSeconds}s`);
  }
  if (stats.errors) failures.push(`video element errors: ${stats.errors}`);

  console.log(JSON.stringify({ camera, source, recordingDate, duration, scrubCount, soakSeconds, startedAt, endedAt, stats, failures }, null, 2));
  if (failures.length) process.exitCode = 1;
} finally {
  await browser.close();
}
