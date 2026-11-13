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
  const page = await browser.newPage({ viewport: { width: 390, height: 1000 } });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const epoch = Date.parse("2026-11-12T23:50:49Z") / 1000;
  const makeTrack = (id, start, end) => ({
    track_id: id, label: "person", box: { x1: 500, y1: 100, x2: 600, y2: 500 },
    box_history: [[start, 100 * id, 100, 100 * id + 100, 500], [end, 100 * id + 20, 120, 100 * id + 120, 520]],
  });
  let event = { id: 145498, camera_id: "downstairs", created_at: new Date(epoch * 1000).toISOString(),
    snapshot_captured_at: new Date((epoch + .5) * 1000).toISOString(), snapshot_path: "cover.jpg", labels: ["person"], objects: [],
    object_tracking: { state: "complete", frame_width: 1280, frame_height: 960, sample_fps: 2,
      tracks: [makeTrack(1, epoch, epoch + 1), makeTrack(2, epoch + 10, epoch + 11), makeTrack(3, epoch - 10, epoch - 9)] },
  };
  let fixture = null;
  if (process.env.SURVNG_TRACK_OVERVIEW_FIXTURE) {
    fixture = JSON.parse(await readFile(process.env.SURVNG_TRACK_OVERVIEW_FIXTURE, "utf8"));
    event = fixture.event;
  }
  const incident = { ...event, id: "scene-1", incident_id: "scene-1", revision: 1, representative_event_id: event.id,
    event_ids: [event.id], events: [event], has_objects: true, start_at: event.created_at, end_at: event.created_at,
    ...(fixture?.scene || {}),
  };
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/config")) return route.fulfill({ json: {} });
    if (path.endsWith("/cameras")) return route.fulfill({ json: [{ id: "downstairs", name: "Downstairs" }] });
    if (path.endsWith("/faces/people")) return route.fulfill({ json: [] });
    if (path.endsWith("/incidents/search")) return route.fulfill({ json: { items: [incident], total: 1 } });
    if (path.includes("/incidents/notification/")) return route.fulfill({ json: { incident, notification: { state: "complete" } } });
    if (path.endsWith("/incidents/detail") || path.endsWith("/incidents/scene-1")) return route.fulfill({ json: incident });
    if (path.endsWith(`/events/${event.id}`)) return route.fulfill({ json: event });
    if (path.endsWith(".jpg")) return route.fulfill(fixture?.image_path
      ? { contentType: "image/jpeg", body: await readFile(fixture.image_path) }
      : { contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="960"><rect width="1280" height="960" fill="#30483d"/></svg>' });
    if (path.endsWith("/event-clip/settings")) return route.fulfill({ json: { before_seconds: 5, after_seconds: 5 } });
    return route.fulfill({ status: 404, json: {} });
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/incidents?incident_id=scene-1`);
  await page.getByRole("button", { name: "Show stored object tracks", exact: true }).click();
  const overlay = page.locator(".event-overlay .object-track-layer");
  await overlay.waitFor();
  await overlay.locator(".object-track-box").first().waitFor();
  assert.equal(await overlay.locator("svg > g").count(), fixture ? 5 : 3, "snapshot renders every retained track, including past and future observations");
  assert.equal(await overlay.locator(".object-track-box").count(), 1, "only the track at capture time receives a box");
  assert.ok(await overlay.locator(".object-track-trail").count() >= 2);
  assert.equal(await overlay.getByText("Full incident paths · boxes at snapshot time").isVisible(), true);
  if (process.env.SURVNG_TRACK_OVERVIEW_SCREENSHOT) await page.locator(".event-detail-media").screenshot({ path: process.env.SURVNG_TRACK_OVERVIEW_SCREENSHOT, animations: "disabled" });
  assert.deepEqual(errors, []);
  console.log("snapshot track overview browser tests passed");
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
