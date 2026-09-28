import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";
const root = new URL("../../survng/static/", import.meta.url);
const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, "http://localhost").pathname;
    if (path.startsWith("/static/")) { res.writeHead(404); res.end(); return; }
    const file = path.startsWith("/survng/static/") ? path.slice(15) : "index.html";
    let body = await readFile(new URL(file, root));
    if (file === "index.html") body = Buffer.from(body.toString().replaceAll('="/static/', '="/survng/static/').replace("<head>", '<head><script>window.__SURVNG_BASE_PATH__="/survng"</script>'));
    res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : file.endsWith(".woff2") ? "font/woff2" : "text/html");
    res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const objects = [
    { label: "car", confidence: .948, incident_eligible: false, incident_ineligible_reasons: ["outside_incident_zone"], box: { x1: 700, y1: 200, x2: 1200, y2: 650 }, detection_frame_width: 1280, detection_frame_height: 800 },
    { label: "person", confidence: .9, incident_eligible: true, box: { x1: 400, y1: 250, x2: 550, y2: 700 }, detection_frame_width: 1280, detection_frame_height: 800 },
  ];
  const event = { id: 79090, camera_id: "gate", created_at: "2026-09-24T20:25:45Z", kind: "motion", objects, snapshot_path: "snapshot.webp" };
  const incident = { ...event, id: "gate-79090", representative_event_id: 79090, event_ids: [79090], events: [event], labels: ["person"], duration_seconds: 15 };
  let deferDetails = false;
  let pendingDetail;
  const summary = { ...incident, objects: [], events: [{ ...event, objects: [] }] };
  await page.route("**/api/**", route => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/cameras")) return route.fulfill({ json: [{ id: "gate", name: "Gate" }] });
    if (path.endsWith("/config")) return route.fulfill({ json: {} });
    if (path.endsWith("/faces/people")) return route.fulfill({ json: [] });
    if (path.endsWith("/incidents/search")) return route.fulfill({ json: { items: [summary], total: 1 } });
    if (path.endsWith("/incidents/detail")) {
      if (deferDetails) { pendingDetail = route; return; }
      return route.fulfill({ json: incident });
    }
    if (path.includes("/incidents/by-event/")) return route.fulfill({ json: incident });
    if (path.endsWith("/events/79090")) return route.fulfill({ json: event });
    if (path.endsWith("/snapshot.jpg") || path.endsWith("/thumbnail.jpg") || path.endsWith("/preview.jpg")) return route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="800"><rect width="1280" height="800" fill="#30483d"/></svg>' });
    return route.fulfill({ json: {} });
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/survng/incidents`);
  const detections = page.locator(".incident-summary-objects .inspector-detection");
  const excludedBox = page.locator(".object-box.excluded").first();
  await page.getByText("outside incident zone", { exact: true }).waitFor();
  assert.equal(await detections.count(), 2);
  assert.equal(await page.getByRole("button", { name: /show excluded/i }).count(), 0);
  await excludedBox.waitFor();
  assert.match(await excludedBox.textContent(), /car/);
  assert.doesNotMatch(await page.locator(".incident-inspector").textContent(), /excluded/i);
  assert.doesNotMatch(await excludedBox.textContent(), /excluded/i);
  assert.equal(await excludedBox.evaluate(el => getComputedStyle(el).borderTopStyle), "dashed");
  // Entries keep their original indexes for object selection/search.
  await excludedBox.click();
  assert.equal(await excludedBox.getAttribute("aria-pressed"), "true");

  // A visibility refresh uses the same path as polling and SSE resynchronization.
  // Hold the response to verify full details never regress to the compact summary.
  deferDetails = true;
  const refresh = async () => {
    pendingDetail = null;
    const refreshed = page.waitForResponse(response => response.url().includes("/incidents/search"));
    await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
    await refreshed;
    for (let attempt = 0; !pendingDetail && attempt < 100; attempt += 1) {
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    assert.ok(pendingDetail, "refresh requests fresh details");
    assert.equal(await detections.count(), 2, "details remain visible during refresh");
    assert.equal(await excludedBox.getAttribute("aria-pressed"), "true");
  };
  await refresh();
  const failed = page.waitForResponse(response => response.url().includes("/incidents/detail"));
  await pendingDetail.fulfill({ status: 503, json: {} });
  await failed;
  assert.equal(await detections.count(), 2, "failed refresh retains details");
  await refresh();
  await pendingDetail.fulfill({ json: {
    ...incident,
    objects: [objects[0], { ...objects[1], label: "bicycle" }],
    events: [{ ...event, objects: [objects[0], { ...objects[1], label: "bicycle" }] }],
  } });
  await detections.filter({ hasText: "bicycle" }).waitFor();
  assert.equal(await detections.count(), 2);
  assert.deepEqual(errors, []);
  console.log("incident annotations and background refresh browser test passed");
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
