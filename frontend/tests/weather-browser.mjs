// Uses the isolated synthetic server: node tools/camera-workspace-preview.mjs.
import assert from "node:assert/strict";
import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  const cameraRequests = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => { if (request.url().includes("/api/cameras/")) cameraRequests.push(request.url()); });
  let enabled = true;
  let unavailable = false;
  let imageryUnavailable = false;
  const now = Math.floor(Date.now() / 1000);
  let releaseConditions;
  const initialConditions = new Promise((resolve) => { releaseConditions = resolve; });
  await page.route("**/api/config", async (route) => {
    const response = await route.fetch();
    const config = await response.json();
    await route.fulfill({ json: { ...config, weather: { enabled, name: "Home weather", latitude: 40.7, longitude: -74, radar_zoom: 6, units: "imperial", animate: true } } });
  });
  await page.route("**/api/weather/conditions", async (route) => {
    await initialConditions;
    if (unavailable) return route.fulfill({ status: 503, json: {} });
    await route.fulfill({ json: { updated_at: now, stale: false, data: { time: now, temperature_2m: 72, apparent_temperature: 70, relative_humidity_2m: 55, wind_speed_10m: 8, wind_gusts_10m: 14, weather_code: 2, hourly: [{ time: now + 3600, temperature_2m: 74, precipitation_probability: 25 }] } } });
  });
  await page.route("**/api/weather/radar", (route) => route.fulfill({ json: { updated_at: now, stale: false, data: { frames: [0, 1, 2].map((index) => ({ time: now - (2 - index) * 600, path: `/v2/radar/${now - (2 - index) * 600}` })) } } }));
  await page.route("https://tile.openstreetmap.org/**", (route) => route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="#cbd8cf"/><path d="M0 100L256 150M80 0L160 256" stroke="#f7f4df" stroke-width="8"/></svg>' }));
  await page.route("https://tilecache.rainviewer.com/**", (route) => imageryUnavailable ? route.fulfill({ status: 503, body: "Unavailable" }) : route.fulfill({ contentType: "image/svg+xml", body: `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">${route.request().url().includes("/coverage/") ? "" : '<ellipse cx="190" cy="90" rx="30" ry="80" fill="#49a454"/>'}</svg>` }));
  const url = process.env.LIVE_PREVIEW_URL || "http://127.0.0.1:5182/";
  await page.goto(url);
  const tile = page.locator(".weather-tile");
  await tile.getByText("Loading conditions…").waitFor();
  releaseConditions();
  await tile.getByText("Partly cloudy", { exact: true }).waitFor();
  assert.match(await tile.textContent(), /72°F/);
  assert.equal(await tile.locator(".camera-tile-poster, video").count(), 0);
  await tile.getByRole("button", { name: "Pause radar" }).click();
  await tile.getByRole("slider", { name: "Radar frame" }).fill("0");
  await tile.getByRole("button", { name: "Expand weather" }).click();
  const dialog = page.getByRole("dialog", { name: "Home weather weather and radar" });
  await dialog.waitFor();
  await dialog.getByText("25% precip.").waitFor();
  await dialog.getByRole("combobox", { name: "Radar zoom" }).selectOption("7");
  await page.keyboard.press("Escape");
  await dialog.waitFor({ state: "detached" });

  await page.getByRole("button", { name: "Custom", exact: true }).click();
  const resize = tile.getByRole("button", { name: /^Resize Home weather/ });
  await resize.focus();
  await page.keyboard.press("Enter");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Enter");
  const saved = await page.evaluate(() => localStorage.getItem("survng.liveCustomLayout.v1"));
  assert.ok(JSON.parse(saved).sizes["weather:local"]);
  await page.reload();
  await tile.waitFor();
  assert.equal(await page.evaluate(() => localStorage.getItem("survng.liveCustomLayout.v1")), saved);
  await tile.getByRole("button", { name: "Make primary weather view" }).click();
  await page.locator(".weather-tile.focus-primary").waitFor();
  await page.waitForTimeout(300);
  if (process.env.WEATHER_SCREENSHOT) await page.screenshot({ path: process.env.WEATHER_SCREENSHOT });

  unavailable = true;
  const resume = async () => page.evaluate(() => {
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    document.dispatchEvent(new Event("visibilitychange"));
    Object.defineProperty(document, "hidden", { configurable: true, value: false });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await resume();
  await tile.getByText(/Stale · Conditions/).waitFor();
  assert.match(await tile.textContent(), /72°F/);
  unavailable = false;
  await resume();
  await tile.getByText(/Stale · Conditions/).waitFor({ state: "detached" });

  await page.setViewportSize({ width: 390, height: 844 });
  await tile.getByRole("button", { name: "Make primary weather view" }).click();
  await page.locator(".weather-tile.mobile-primary").waitFor();
  if (process.env.WEATHER_SCREENSHOT) await page.screenshot({ path: process.env.WEATHER_SCREENSHOT.replace(".png", "-mobile.png") });
  await tile.getByRole("button", { name: "Expand weather" }).click();
  await dialog.waitFor();
  const bounds = await dialog.boundingBox();
  assert.ok(bounds.width <= 390 && bounds.height <= 844);
  await dialog.getByRole("button", { name: "Close weather" }).click();
  assert.ok(!cameraRequests.some((request) => request.includes("weather")));
  unavailable = true;
  imageryUnavailable = true;
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.reload();
  await tile.getByText("Conditions unavailable · retrying").waitFor();
  await tile.getByText("Radar imagery unavailable", { exact: true }).waitFor();
  await tile.getByText("Radar coverage unknown.", { exact: true }).waitFor();
  await tile.getByRole("button", { name: "Play radar", exact: true }).waitFor();
  enabled = false;
  await page.reload();
  await page.locator(".camera-tile").first().waitFor();
  assert.equal(await tile.count(), 0);
  assert.deepEqual(errors, []);
  console.log("weather browser tests passed: loading, radar controls, expansion, custom layout persistence, focus, stale/recovery, mobile, imagery failure, reduced motion, disabled");
} finally {
  await browser.close();
}
