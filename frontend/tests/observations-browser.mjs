import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";

const root = process.env.SURVNG_FRONTEND_BUILD ? new URL(`file://${process.env.SURVNG_FRONTEND_BUILD.replace(/\/$/, "")}/`) : new URL("../../survng/static/", import.meta.url);
const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, "http://localhost").pathname;
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
  const page = await browser.newPage({ viewport: { width: 1300, height: 1000 } });
  page.setDefaultTimeout(10000);
  const errors = [], requests = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const observation = { id: "raw-1", camera_id: "room", captured_at: "2026-09-28T12:00:00Z", label: "person", confidence: .279, certainty: "possible", snapshot_url: "/api/observations/raw-1/snapshot", snapshot_available: true, detection_frame_width: 896, detection_frame_height: 512, box: { x1: 89.6, y1: 51.2, x2: 448, y2: 256 }, image: { width: 896, height: 512, source: "live", captured_at: "2026-09-28T12:00:00Z", analyzed_frame: true } };
  let pending = { id: "window-1", camera_id: "room", captured_at: observation.captured_at, status: "pending", summary: "Possible person detections", reason: "awaiting_activity_evidence", observations: [observation], coverage: { state: "sampled" } };
  const established = { ...pending, id: "window-2", status: "established", summary: "Person crossed the driveway", incident_id: "incident-2" };
  let failList = false, failDetail = false;
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url()), path = url.pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/cameras")) return route.fulfill({ json: [{ id: "room", name: "Room" }] });
    if (path.endsWith("/config")) return route.fulfill({ json: {} });
    if (path.endsWith("/observations")) {
      requests.push(url);
      if (failList) return route.fulfill({ status: 503, json: {} });
      const items = [pending, established].filter((item) => !url.searchParams.get("status") || item.status === url.searchParams.get("status"));
      return route.fulfill({ json: { items, total: items.length } });
    }
    if (path.endsWith("/observations/window-1")) return route.fulfill({ status: failDetail ? 503 : 200, json: failDetail ? {} : pending });
    if (path.endsWith("/snapshot")) return route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="896" height="512"><rect width="896" height="512" fill="#30483d"/></svg>' });
    return route.fulfill({ json: {} });
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/survng/observations`);
  const view = page.locator(".observations-page");
  await view.getByRole("heading", { name: "Observations", exact: true }).waitFor();
  await view.getByRole("button", { name: /Awaiting confirmation Room/ }).waitFor();
  assert.equal(requests[0].searchParams.get("limit"), "25");
  const interval = Date.parse(requests[0].searchParams.get("end_at")) - Date.parse(requests[0].searchParams.get("start_at"));
  assert.ok(interval >= 23 * 3600000 && interval <= 25 * 3600000, "default request is one timezone-aware day, never entire history");
  assert.match(await view.getByRole("link", { name: "Open established incident" }).getAttribute("href"), /incident_id=incident-2/);
  await view.getByRole("button", { name: /Awaiting confirmation Room/ }).click();
  const detail = view.getByRole("region", { name: "Observation details" });
  await detail.getByRole("heading", { name: "Awaiting confirmation" }).waitFor();
  assert.match(page.url(), /observation_id=window-1/);
  assert.equal(await detail.getByRole("button", { name: /Possible person/ }).isVisible(), false);
  await detail.getByText("Retained object observations (1)", { exact: true }).click();
  await detail.getByRole("button", { name: /Possible person/ }).click();
  await detail.locator(".incident-observation-image img").waitFor();
  assert.match(await detail.locator(".incident-observation-image img").getAttribute("src"), /\/observations\/raw-1\/snapshot$/);
  assert.match(await detail.locator("figcaption").textContent(), /896 × 512.*Analyzed frame/);
  assert.equal(await detail.locator(".incident-observation-box").evaluate((element) => element.style.left), "10%");
  failList = true; failDetail = true;
  await view.getByRole("button", { name: "Refresh observations" }).click();
  await view.getByText(/Observations could not be loaded/).waitFor();
  await detail.getByText(/Supporting observations could not be loaded/).waitFor();
  assert.equal(await detail.locator(".incident-observation-image img").count(), 1, "temporary refresh failure preserves already reviewed evidence");
  failList = false; failDetail = false;
  pending = { ...pending, status: "incomplete", reason: "recording_unavailable" };
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await detail.getByRole("heading", { name: "Analysis incomplete" }).waitFor();
  assert.match(await detail.textContent(), /does not establish that nothing happened/);
  await view.getByLabel("Status", { exact: true }).selectOption("not_established");
  await view.getByText(/No observations match this day/).waitFor();
  assert.match(page.url(), /status=not_established/);
  assert.equal(await detail.count(), 0);
  await view.getByLabel("Status", { exact: true }).selectOption("all");
  await view.getByLabel("Camera", { exact: true }).selectOption("room");
  await view.getByRole("button", { name: /Analysis incomplete Room/ }).waitFor();
  assert.equal(requests.at(-1).searchParams.get("camera_id"), "room");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`http://127.0.0.1:${server.address().port}/survng/observations?observation_id=window-1`);
  await detail.getByRole("heading", { name: "Analysis incomplete" }).waitFor();
  assert.ok(await view.evaluate((element) => element.scrollWidth <= window.innerWidth), "mobile review fits viewport");
  assert.deepEqual(errors, []);
  console.log("observations review: bounded queries, uncertainty, source provenance, errors, reconnect, and mobile links passed");
} finally { await browser.close(); await new Promise((resolve) => server.close(resolve)); }
