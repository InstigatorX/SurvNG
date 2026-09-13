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
  for (const width of [390, 1440]) {
    const page = await browser.newPage({ viewport: { width, height: 844 }, isMobile: width < 500, hasTouch: width < 500, ...(width < 500 ? { userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1" } : {}) });
    page.setDefaultTimeout(10000);
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    let status = 200, imageFailure = false;
    const detail = { camera_name: "Front Door", notification: { state: "updated", revision: 2, summary: "Alex detected at Front Door.", people: ["Alex"], zones: ["Porch"], representative_event_id: 42, started_at: "2026-09-13T01:00:00Z" }, incident: { id: "incident-front-door-41", camera_id: "front-door", representative_event_id: 42, created_at: "2026-09-13T01:00:10Z", start_epoch: 1789261200, last_epoch: 1789261210, events: [{ id: 42, created_at: "2026-09-13T01:00:10Z", labels: ["person"] }, { id: 41, created_at: "2026-09-13T01:00:00Z", labels: [] }] } };
    await page.route("**/api/**", (route) => {
      const path = new URL(route.request().url()).pathname;
      if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
      if (path.includes("/incidents/notification/")) return route.fulfill({ status, json: detail });
      if (path.endsWith("/thumbnail.jpg")) return route.fulfill({ status: imageFailure ? 404 : 200, contentType: "image/svg+xml", body: imageFailure ? "" : '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="800"><rect width="1280" height="800" fill="#30483d"/><text x="440" y="400" fill="white" font-size="48">Incident image</text></svg>' });
      if (path.endsWith("/event-clip/settings")) return route.fulfill({ json: { before_seconds: 5, after_seconds: 5 } });
      return route.fulfill({ status: 404, json: {} });
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/survng/incidents/incident-front-door-41`);
    await page.getByRole("heading", { name: "Alex detected at Front Door." }).waitFor();
    assert.equal(await page.locator(".assistant-panel").count(), 0);
    assert.equal(await page.getByRole("link", { name: "Live view", exact: true }).getAttribute("href"), "/survng/?camera=front-door");
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    const box = await page.getByRole("button", { name: "Play incident", exact: true }).boundingBox();
    assert.ok(box.height >= 44 && box.y + box.height < 844);
    await page.screenshot({ path: `/tmp/survng-incident-detail-${width}.png`, fullPage: true });
    await page.getByRole("button", { name: "First detection", exact: true }).click();
    assert.match(await page.locator(".incident-detail-hero img").getAttribute("src"), /events\/41\//);
    await page.getByRole("button", { name: "Play incident", exact: true }).click();
    await page.getByRole("button", { name: "Close playback" }).click();
    status = 503;
    await page.evaluate(() => window.dispatchEvent(new Event("online")));
    await page.getByRole("alert").waitFor();
    assert.ok(await page.getByRole("heading", { name: "Alex detected at Front Door." }).isVisible());
    status = 200;
    detail.notification.state = "complete";
    detail.notification.completed_at = "2026-09-13T01:01:00Z";
    await page.getByRole("button", { name: "Retry", exact: true }).click();
    await page.getByText("Completed", { exact: true }).waitFor();
    assert.equal(await page.getByRole("link", { name: "Live view", exact: true }).count(), 0);
    await page.getByRole("button", { name: "Final image", exact: true }).waitFor();
    imageFailure = true;
    detail.notification.revision++;
    await page.evaluate(() => window.dispatchEvent(new Event("online")));
    await page.locator(".incident-detail-hero").getByText("Image unavailable").waitFor();
    status = 404;
    await page.evaluate(() => window.dispatchEvent(new Event("online")));
    await page.getByRole("heading", { name: "Incident unavailable" }).waitFor();
    assert.deepEqual(errors, []);
    await page.close();
  }
  console.log("Incident detail mobile/desktop, playback controls, reconnect, completion, missing image and expired link checks passed");
} finally { await browser.close(); await new Promise((resolve) => server.close(resolve)); }
