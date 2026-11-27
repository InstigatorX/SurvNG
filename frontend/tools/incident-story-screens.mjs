// Render the incident detail storyboard states from the built frontend with
// fixture APIs, for comparing against the approved mockups.
// node tools/incident-story-screens.mjs <outdir> [width]
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { chromium } from "playwright";
import { CAMERAS, queueFor, sceneSvg, storyIncident } from "../tests/fixtures/incident-story.mjs";

const out = process.argv[2] || "/tmp/incident-story";
const width = Number(process.argv[3] || 1440);
await mkdir(out, { recursive: true });
const root = new URL("../../survng/static/", import.meta.url);
const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, "http://localhost").pathname;
    const file = path.startsWith("/survng/static/") ? path.slice(15) : "index.html";
    let body = await readFile(new URL(file, root));
    if (file === "index.html") body = Buffer.from(body.toString().replaceAll('="/static/', '="/survng/static/').replace("<head>", '<head><script>window.__SURVNG_BASE_PATH__="/survng";localStorage.setItem("survng.theme","dark")</script>'));
    res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : file.endsWith(".woff2") ? "font/woff2" : "text/html");
    res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
const incident = storyIncident();
const observationCamera = new Map(incident.scene_objects.flatMap((object) => object.observations.map((item) => [item.id, item.camera_id])));
try {
  const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: "dark" });
  page.on("pageerror", (error) => console.error("pageerror", error.message));
  await page.route("**/api/**", (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/api/cameras")) return route.fulfill({ json: CAMERAS });
    if (path.includes("/incidents/notification/")) return route.fulfill({ json: { incident, notification: { state: "complete", revision: 7 }, camera_name: "Front Driveway" } });
    if (path.endsWith("/incidents/review-queue")) return route.fulfill({ json: queueFor(incident.incident_id) });
    if (path.includes("/incidents/observations/")) return route.fulfill({ contentType: "image/svg+xml", body: sceneSvg(observationCamera.get(decodeURIComponent(path.split("/")[4]))) });
    if (path.endsWith(".jpg")) return route.fulfill({ contentType: "image/svg+xml", body: sceneSvg("front-driveway") });
    if (path.endsWith("/analysis")) return route.fulfill({ json: { status: "complete", enabled: true, mode: "eager", episodes: [] } });
    return route.fulfill({ json: {} });
  });
  const base = `http://127.0.0.1:${server.address().port}/survng/incidents/${incident.incident_id}`;
  await page.goto(base);
  await page.getByRole("heading", { level: 1, name: /Car arrives/ }).waitFor();
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${out}/state-1-open.png`, fullPage: true });
  await page.locator(".story-card", { hasText: "Person" }).click();
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${out}/state-2-pick.png`, fullPage: true });
  const doorBar = page.locator(".story-lane").nth(1).locator(".story-bar").first();
  const box = await doorBar.boundingBox();
  await page.mouse.click(box.x + box.width * 0.25, box.y + box.height / 2);
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${out}/state-3-follow.png`, fullPage: true });
  await page.locator(".story-card", { hasText: "Dog" }).click();
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${out}/state-4-correct.png`, fullPage: true });
  await page.goto(`${base}?review=1`);
  await page.getByRole("heading", { level: 1, name: /Car arrives/ }).waitFor();
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${out}/state-5-review.png`, fullPage: true });
  console.log(`screens written to ${out}`);
} finally { await browser.close(); server.close(); }
