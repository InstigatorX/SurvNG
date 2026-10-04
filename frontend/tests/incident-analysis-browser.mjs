import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";

const root = process.env.SURVNG_FRONTEND_BUILD ? new URL(`file://${process.env.SURVNG_FRONTEND_BUILD.replace(/\/$/, "")}/`) : new URL("../../survng/static/", import.meta.url);
const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, "http://localhost").pathname;
    const file = path.startsWith("/static/") ? path.slice(8) : "index.html";
    res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : file.endsWith(".woff2") ? "font/woff2" : "text/html");
    res.end(await readFile(new URL(file, root)));
  } catch { res.writeHead(404); res.end(); }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const event = { id: 41, camera_id: "gate", created_at: "2026-10-03T12:00:00Z", labels: ["person"], objects: [{ label: "person", confidence: .8 }], snapshot_path: "cover.webp" };
  const incident = { ...event, id: "scene-1", incident_id: "scene-1", revision: 1, summary: "Person at Gate.", representative_event_id: 41, event_ids: [41], events: [event], has_objects: true, start_epoch: 1791028800, last_epoch: 1791028810, start_at: event.created_at, end_at: "2026-10-03T12:00:10Z", episodes: [{ id: "ep-1", camera_id: "gate", start_at: event.created_at, end_at: "2026-10-03T12:00:10Z", event_ids: [41] }] };
  let status = "deferred", posts = 0, statusError = false;
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/analysis")) {
      if (route.request().method() === "POST") { posts++; status = "queued"; }
      return route.fulfill({ status: statusError ? 503 : 200, json: { mode: "on_demand", enabled: true, status, episodes: [{ episode_id: "ep-1", status }], remaining: 1 } });
    }
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/config")) return route.fulfill({ json: {} });
    if (path.endsWith("/cameras")) return route.fulfill({ json: [{ id: "gate", name: "Gate" }] });
    if (path.endsWith("/faces/people")) return route.fulfill({ json: [] });
    if (path.endsWith("/incidents/search")) return route.fulfill({ json: { items: [incident], total: 1 } });
    if (path.includes("/incidents/notification/")) return route.fulfill({ json: { incident, camera_name: "Gate", notification: { state: "complete" } } });
    if (path.endsWith("/incidents/detail") || path.endsWith("/incidents/scene-1")) return route.fulfill({ json: incident });
    if (path.endsWith("/events/41")) return route.fulfill({ json: event });
    if (path.endsWith(".jpg")) return route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="800"><rect width="1280" height="800" fill="#30483d"/></svg>' });
    if (path.endsWith("/event-clip/settings")) return route.fulfill({ json: { before_seconds: 5, after_seconds: 5 } });
    return route.fulfill({ status: 404, json: {} });
  });
  const origin = `http://127.0.0.1:${server.address().port}`;
  await page.goto(`${origin}/incidents?incident_id=scene-1`);
  const inspector = page.locator("#incident-inspector");
  const analyze = inspector.getByRole("button", { name: "Analyze extra details" });
  await analyze.waitFor();
  await inspector.getByRole("progressbar", { name: "Extra analysis progress" }).waitFor();
  assert.equal(await inspector.locator(".incident-inspector-extra-analysis").evaluate((section) => section.previousElementSibling?.querySelector("h3")?.textContent), "Faces");
  assert.equal(posts, 0, "desktop selected feed preview never admits optional work");
  await analyze.click();
  await page.getByText("Extra details queued", { exact: true }).waitFor();
  assert.equal(posts, 1);

  await page.goto(`${origin}/incidents/incident-scene-1`);
  await page.getByText("Extra details queued", { exact: true }).waitFor();
  const play = page.getByRole("button", { name: "Play incident", exact: true });
  assert.equal(await play.isEnabled(), true, "analysis never disables playback");
  await play.click();
  await page.locator(".incident-recording-player").waitFor();
  statusError = true;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.getByRole("button", { name: "Retry extra details" }).waitFor();
  assert.equal(await page.locator(".incident-recording-player").isVisible(), true, "analysis errors do not replace video");
  statusError = false; status = "partial";
  await page.getByRole("button", { name: "Retry extra details" }).click();
  await page.getByText("Extra details partly available", { exact: true }).waitFor();
  const terminalPosts = posts;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.getByText("Extra details partly available", { exact: true }).waitFor();
  assert.equal(posts, terminalPosts, "partial work does not automatically restart");

  status = "deferred";
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${origin}/incidents?incident_id=scene-1`);
  await page.locator(".event-overlay").getByText("Extra details queued", { exact: true }).waitFor();
  assert.equal(posts, terminalPosts + 1, "mobile open overlay admits work");
  assert.deepEqual(errors, []);
  console.log("Incident analysis desktop opt-in, detail playback independence, errors, partial results and mobile overlay checks passed");
} finally { await browser.close(); await new Promise((resolve) => server.close(resolve)); }
