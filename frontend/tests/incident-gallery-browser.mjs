import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";
import { INCIDENT_GALLERY_CHUNK } from "../src/incidentNavigation.mjs";

// The incidents page opens on a gallery that loads as it is scrolled, with
// filters in a left pane and the incident's place and time on each picture.
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
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720"><rect width="1280" height="720" fill="#30483d"/></svg>';
const CAMERAS = [{ id: "front-door", name: "Front Door" }, { id: "driveway", name: "Front Driveway" }];
const LABELS = [["person"], ["car"], ["person", "car"]];
const TOTAL = 150;
// Fixed instants keep the stamp assertion independent of when the test runs.
const START = Date.parse("2026-03-08T23:31:00Z");
const incidents = Array.from({ length: TOTAL }, (_, index) => {
  const labels = LABELS[index % LABELS.length];
  const camera = CAMERAS[index % CAMERAS.length].id;
  const at = new Date(START - index * 240000).toISOString();
  return { id: `incident-${index}`, incident_id: `incident-${index}`, revision: 1, camera_id: camera, camera_ids: [camera], created_at: at, start_at: at, end_at: at,
    duration_seconds: 20, representative_event_id: 1000 + index, event_ids: [1000 + index], labels, has_objects: true, zones: ["Entry"], snapshot_path: "cover.webp",
    objects: labels.map((label) => ({ label, confidence: index % 2 ? .9 : .6, box: { x1: 300, y1: 200, x2: 600, y2: 600 }, detection_frame_width: 1280, detection_frame_height: 720 })) };
});

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const searches = [];
  let failMore = false;
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/api/cameras")) return route.fulfill({ json: CAMERAS });
    if (path.endsWith("/face-names")) return route.fulfill({ json: { people: [] } });
    if (path.endsWith("/incidents/search")) {
      const limit = Number(url.searchParams.get("limit")), offset = Number(url.searchParams.get("offset"));
      const floor = Number(url.searchParams.get("min_confidence") || 0), label = url.searchParams.get("object_label");
      searches.push({ limit, offset, floor, label, type: url.searchParams.get("event_type") });
      if (failMore && offset > 0) return route.fulfill({ status: 500, json: {} });
      const matched = incidents.filter((item) => (!label || item.labels.includes(label)) && item.objects.some((object) => object.confidence >= floor));
      return route.fulfill({ json: { items: matched.slice(offset, offset + limit), total: matched.length, facets: { camera_ids: CAMERAS.map((camera) => camera.id), labels: ["car", "person"], zones: ["Entry"] } } });
    }
    if (path.endsWith("/incidents/detail")) return route.fulfill({ json: incidents.find((item) => item.id === url.searchParams.get("incident_id")) || incidents[0] });
    if (path.endsWith(".jpg") || path.endsWith(".webp") || path.endsWith("/snapshot")) return route.fulfill({ contentType: "image/svg+xml", body: svg });
    return route.fulfill({ json: {} });
  });
  const tiles = page.locator("#incident-results .live-activity-item");
  const list = page.locator("#incident-results");
  const status = page.locator(".incident-gallery-status");
  const toEnd = () => list.evaluate((element) => { element.scrollTop = element.scrollHeight; });

  // Gallery first: nothing is opened, so the address stays a plain list link.
  await page.goto(`${origin}/survng/incidents`);
  await tiles.first().waitFor();
  assert.equal(await page.locator(".gallery-expanded").count(), 1);
  assert.equal(await tiles.count(), INCIDENT_GALLERY_CHUNK);
  assert.deepEqual(searches.map(({ limit, offset }) => [limit, offset]), [[INCIDENT_GALLERY_CHUNK, 0]]);
  assert.equal(new URL(page.url()).search, "");
  assert.equal(await page.locator(".incident-pager").count(), 0, "the gallery has no pages");
  assert.match(await status.innerText(), new RegExp(`Showing ${INCIDENT_GALLERY_CHUNK} of ${TOTAL}`));

  // Place and time sit on the picture, object icons at the bottom left.
  const first = tiles.first();
  assert.equal(await first.locator(".live-activity-copy").count(), 0);
  assert.equal(await first.locator(".incident-hud-where b").innerText(), "Front Door");
  assert.match(await first.locator(".incident-hud-where time").innerText(), /^[A-Z][a-z]{2} \d{1,2}, \d{1,2}:\d{2}(am|pm)$/);
  assert.equal(await first.locator(".incident-hud-objects .pill").count(), 1);
  const hud = await first.locator(".incident-hud-where").boundingBox();
  const icons = await first.locator(".incident-hud-objects").boundingBox();
  const thumb = await first.locator(".live-activity-thumb").boundingBox();
  assert.ok(hud.x - thumb.x < 12 && hud.y - thumb.y < 12, "where and when at the upper left");
  assert.ok(icons.x - thumb.x < 12 && thumb.y + thumb.height - (icons.y + icons.height) < 12, "objects at the lower left");

  // Filters sit in a pane left of the gallery.
  const pane = page.getByRole("complementary", { name: "Incident filters" });
  const paneBox = await pane.boundingBox();
  assert.ok(paneBox.x + paneBox.width <= (await list.boundingBox()).x + 1);
  assert.equal(await page.locator(".incidents-desktop-toolbar").count(), 0);

  // Per row: pictures grow to fill the row at every setting. Smaller pictures bring
  // the end of the list within reach, which asks for more; that request fails here.
  failMore = true;
  const perRow = page.getByLabel("Incidents per row");
  for (const columns of [3, 6, 4]) {
    await perRow.fill(String(columns));
    const boxes = await Promise.all(Array.from({ length: columns + 1 }, (_, index) => tiles.nth(index).boundingBox()));
    assert.ok(boxes.slice(0, columns).every((box) => Math.abs(box.y - boxes[0].y) < 1), `${columns} per row share a row`);
    assert.ok(boxes[columns].y > boxes[0].y, `tile ${columns + 1} starts the next row`);
    const listBox = await list.boundingBox();
    assert.ok(listBox.width - (boxes[columns - 1].x + boxes[columns - 1].width - listBox.x) < 28, "the row fills the width");
    const picture = await tiles.first().locator(".live-activity-thumb").boundingBox();
    assert.ok(Math.abs(picture.width - boxes[0].width) < 4, "the picture fills its tile");
  }
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem("survng.incidentGalleryColumns.v1"))), 4);

  // A failed load keeps what is shown and offers a retry.
  await toEnd();
  await status.getByText("Could not load more incidents.").waitFor();
  assert.equal(await tiles.count(), INCIDENT_GALLERY_CHUNK);
  assert.deepEqual(searches.at(-1), { limit: INCIDENT_GALLERY_CHUNK, offset: INCIDENT_GALLERY_CHUNK, floor: 0, label: null, type: "object" },
    "scrolling near the end asks for the incidents after the ones shown");
  failMore = false;
  await status.getByRole("button", { name: "Retry" }).click();
  await page.waitForFunction((count) => document.querySelectorAll("#incident-results .live-activity-item").length === count, 2 * INCIDENT_GALLERY_CHUNK);
  await toEnd();
  await page.waitForFunction((count) => document.querySelectorAll("#incident-results .live-activity-item").length === count, TOTAL);
  assert.equal(await tiles.first().locator(".incident-hud-where b").innerText(), "Front Door", "earlier tiles stay");
  assert.match(await status.innerText(), new RegExp(`All ${TOTAL} incidents`));
  const settled = searches.length;
  await toEnd();
  await page.waitForTimeout(400);
  assert.equal(searches.length, settled, "nothing more to ask for at the end");

  // Confidence threshold and object choice go to the search and reset the list.
  await page.getByRole("slider", { name: "Confidence threshold" }).fill("80");
  await page.waitForFunction(() => document.querySelectorAll("#incident-results .live-activity-item").length === 60);
  assert.equal(searches.at(-1).floor, 0.8);
  assert.equal(searches.at(-1).offset, 0);
  assert.match(await pane.locator(".incident-filter-confidence-readout").innerText(), /75 results\s+80%/);
  await page.getByRole("radio", { name: "Person" }).check();
  await page.waitForFunction(() => document.querySelectorAll("#incident-results .live-activity-item").length === 50);
  assert.deepEqual([searches.at(-1).label, searches.at(-1).floor], ["person", 0.8]);
  const clear = pane.getByRole("button", { name: "Clear filters (2)" });
  await clear.click();
  await page.waitForFunction((count) => document.querySelectorAll("#incident-results .live-activity-item").length === count, INCIDENT_GALLERY_CHUNK);
  assert.equal(await page.getByRole("slider", { name: "Confidence threshold" }).inputValue(), "0");
  // Motion-only incidents carry no detection confidence.
  await pane.getByRole("button", { name: "Motion" }).click();
  await page.waitForFunction(() => document.querySelector(".incident-filter-confidence input").disabled);
  assert.equal(searches.at(-1).type, "motion");
  assert.equal(searches.at(-1).floor, 0);
  await pane.getByRole("button", { name: "Objects" }).click();
  await tiles.first().waitFor();

  // A tile opens the viewer on that incident; Escape returns to the gallery with it marked.
  await toEnd();
  await page.waitForFunction((count) => document.querySelectorAll("#incident-results .live-activity-item").length === count, 2 * INCIDENT_GALLERY_CHUNK);
  await tiles.nth(70).locator(".live-activity-select").click();
  await page.waitForFunction(() => !document.querySelector(".gallery-expanded"));
  await page.waitForFunction(() => new URLSearchParams(location.search).get("incident_id") === "incident-70");
  assert.equal(await pane.count(), 0, "the viewer keeps its width unless filters are asked for");
  await page.locator(".incident-pager").waitFor();
  await page.locator("#incident-results .live-activity-item.selected").waitFor();
  await page.getByRole("button", { name: /^Filters/ }).click();
  await pane.waitFor();
  await page.getByRole("button", { name: /^Filters/ }).click();
  await page.keyboard.press("Escape");
  await page.locator(".gallery-expanded").waitFor();
  const chosen = page.locator("#incident-results .live-activity-item.selected");
  await chosen.waitFor();
  // The gallery reloads far enough to hold the open incident.
  await page.waitForFunction((count) => document.querySelectorAll("#incident-results .live-activity-item").length >= count, 2 * INCIDENT_GALLERY_CHUNK);
  await page.waitForFunction(() => {
    const box = document.querySelector("#incident-results .live-activity-item.selected")?.getBoundingClientRect();
    const frame = document.querySelector("#incident-results").getBoundingClientRect();
    return box && box.top >= frame.top && box.bottom <= frame.bottom;
  });

  // A link to one incident still opens its viewer.
  await page.goto(`${origin}/survng/incidents?incident_id=incident-3`);
  await page.locator(".incident-pager").waitFor();
  assert.equal(await page.locator(".gallery-expanded").count(), 0);
  assert.deepEqual(errors, []);
  await page.close();
  console.log("Incident gallery: default view, scroll loading, per-row sizing, picture overlay, filter pane, confidence and viewer round trip passed");
} finally { await browser.close(); await new Promise((resolve) => server.close(resolve)); }
