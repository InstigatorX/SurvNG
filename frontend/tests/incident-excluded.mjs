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
  await page.route("**/api/**", route => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/cameras")) return route.fulfill({ json: [{ id: "gate", name: "Gate" }] });
    if (path.endsWith("/config")) return route.fulfill({ json: {} });
    if (path.endsWith("/faces/people")) return route.fulfill({ json: [] });
    if (path.endsWith("/incidents/search")) return route.fulfill({ json: { items: [incident], total: 1 } });
    if (path.endsWith("/incidents/detail") || path.includes("/incidents/by-event/")) return route.fulfill({ json: incident });
    if (path.endsWith("/events/79090")) return route.fulfill({ json: event });
    if (path.endsWith("/snapshot.jpg") || path.endsWith("/thumbnail.jpg") || path.endsWith("/preview.jpg")) return route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="800"><rect width="1280" height="800" fill="#30483d"/></svg>' });
    return route.fulfill({ json: {} });
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/survng/incidents?event_ids=79090`);
  const toggle = page.getByRole("button", { name: "Show excluded (1)", exact: true });
  await toggle.waitFor();
  const detections = page.locator(".incident-summary-objects .inspector-detection");
  assert.equal(await toggle.getAttribute("aria-pressed"), "false");
  assert.equal(await detections.count(), 1);
  assert.equal(await page.locator(".object-box.excluded").count(), 0);
  await toggle.click();
  assert.equal(await toggle.getAttribute("aria-pressed"), "true");
  assert.equal(await detections.count(), 2);
  await page.getByText("Excluded · outside incident zone", { exact: true }).waitFor();
  await page.locator(".object-box.excluded").first().waitFor();
  assert.match(await page.locator(".object-box.excluded").first().textContent(), /car.*Excluded/);
  // Excluded entries keep their original indexes for object selection/search.
  await page.locator(".object-box.excluded").first().click();
  assert.equal(await page.locator(".object-box.excluded").first().getAttribute("aria-pressed"), "true");
  await toggle.click();
  assert.equal(await detections.count(), 1);
  assert.equal(await page.locator(".object-box.excluded").count(), 0);
  assert.deepEqual(errors, []);
  console.log("incident excluded-detection toggle browser test passed");
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
