import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";
import { INCIDENT_GALLERY_CHUNK } from "../src/incidentNavigation.mjs";
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

// Review marks against a list that really changes: a marked incident leaves the
// Unreviewed list on the server at once, while the gallery still shows it.
const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720"><rect width="1280" height="720" fill="#30483d"/></svg>';
const START = Date.parse("2026-03-08T23:31:00Z");
function world(total) {
  const incidents = Array.from({ length: total }, (_, index) => {
    const at = new Date(START - index * 60000).toISOString();
    return { id: `incident-${index}`, incident_id: `incident-${index}`, revision: 1, camera_id: "front-door", camera_ids: ["front-door"], created_at: at, start_at: at, end_at: at,
      duration_seconds: 20, representative_event_id: 1000 + index, event_ids: [1000 + index], review_verdict: null, labels: ["person"], has_objects: true, zones: [], snapshot_path: "available",
      objects: [{ label: "person", confidence: .9 }] };
  });
  const state = { incidents, delay: 0, searches: [] };
  state.route = async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/api/cameras")) return route.fulfill({ json: [{ id: "front-door", name: "Front Door" }] });
    if (path.endsWith("/incidents/reviews")) {
      const body = route.request().postDataJSON();
      for (const item of incidents) if (body.incident_ids.includes(item.id)) item.review_verdict = body.verdict;
      return route.fulfill({ json: {} });
    }
    if (path.endsWith("/review")) {
      const id = decodeURIComponent(path.split("/").at(-2));
      incidents.find((item) => item.id === id).review_verdict = route.request().postDataJSON().verdict;
      return route.fulfill({ json: {} });
    }
    if (path.endsWith("/incidents/search")) {
      const limit = Number(url.searchParams.get("limit")), offset = Number(url.searchParams.get("offset")), review = url.searchParams.get("review");
      state.searches.push({ limit, offset, review });
      const matched = incidents.filter((item) => !review || (item.review_verdict === "reviewed") === (review === "reviewed"));
      const body = { items: matched.slice(offset, offset + limit).map((item) => ({ ...item })), total: matched.length, facets: { camera_ids: ["front-door"], labels: ["person"], zones: [] } };
      if (state.delay) await new Promise((resolve) => setTimeout(resolve, state.delay));
      return route.fulfill({ json: body });
    }
    if (path.endsWith(".jpg") || path.endsWith(".webp") || path.endsWith("/snapshot")) return route.fulfill({ contentType: "image/svg+xml", body: svg });
    return route.fulfill({ json: {} });
  };
  return state;
}
const shownIds = (page) => page.locator("#incident-results .live-activity-item .live-activity-select").evaluateAll((links) => links.map((link) => link.closest(".live-activity-item").dataset.incidentId || link.getAttribute("href") || ""));

try {
  // Desktop.
  const desk = world(200);
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/api/**", desk.route);
  await page.goto(`${origin}/survng/incidents`);
  const tiles = page.locator("#incident-results .live-activity-item");
  const list = page.locator("#incident-results");
  const status = page.locator(".incident-gallery-status");
  await tiles.first().waitFor();
  // Mark three in the first screen, then read on to the end: nothing further down is skipped.
  for (const index of [0, 1, 2]) await tiles.nth(index).getByRole("checkbox").click();
  await page.waitForFunction(() => document.querySelectorAll("#incident-results input[type=checkbox]:checked").length === 3);
  await page.mouse.move(5, 5);
  for (let waited = 0; waited < 20000 && !(await status.textContent()).includes("for this day"); waited += 150) {
    await list.evaluate((element) => { element.scrollTop = element.scrollHeight; });
    await page.waitForTimeout(150);
  }
  assert.match(await status.textContent(), /for this day/);
  const count = await tiles.count();
  assert.ok(count === 200 || count === 197, `every unreviewed incident is shown (${count})`);
  assert.ok(desk.searches.some((search) => search.offset > 0 && search.offset % 60 !== 0), "later chunks start where the server's list now starts");
  // Bulk marking sends only what still needs it.
  await list.evaluate((element) => { element.scrollTop = 0; });

  // A filter switched there and back while the first answer is still on its way does not leave the list dimmed.
  desk.delay = 500;
  const reviewFilter = page.getByRole("group", { name: "Review filter" });
  await reviewFilter.getByRole("button", { name: "Reviewed", exact: true }).click();
  await reviewFilter.getByRole("button", { name: "Reviewed", exact: true }).click();
  await page.waitForTimeout(2500);
  assert.equal(await page.locator("#incident-results.updating").count(), 0, "the list is not left marked as updating");
  assert.equal(await list.getAttribute("aria-busy"), "false");
  desk.delay = 0;
  assert.deepEqual(errors, []);
  await page.close();

  // Phone: after a filter change, scrolling still loads more and the list starts at its top.
  const pocket = world(300);
  const phone = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  phone.setDefaultTimeout(10000);
  const phoneErrors = [];
  phone.on("pageerror", (error) => phoneErrors.push(error.message));
  await phone.route("**/api/**", pocket.route);
  await phone.goto(`${origin}/survng/incidents`);
  const phoneTiles = phone.locator("#incident-results .live-activity-item");
  await phoneTiles.first().waitFor();
  const toEnd = () => phone.evaluate(() => { const element = document.querySelector("#incident-results"); element.scrollTop = element.scrollHeight; window.scrollTo(0, document.documentElement.scrollHeight); });
  await toEnd();
  await phone.waitForFunction(() => document.querySelectorAll("#incident-results .live-activity-item").length >= 120);
  await toEnd();
  const phoneReview = phone.getByRole("group", { name: "Review filter" }).getByRole("button", { name: "Reviewed", exact: true });
  if (!(await phoneReview.isVisible())) await phone.getByRole("button", { name: "Filters", exact: true }).click();
  await phoneReview.click();
  const done = phone.getByRole("button", { name: "Done", exact: true });
  if (await done.isVisible()) await done.click();
  await phone.waitForFunction(() => document.querySelectorAll("#incident-results .live-activity-item").length === 60);
  assert.equal(await phone.evaluate(() => document.querySelector("#incident-results").scrollTop), 0, "a new list starts at its top");
  await toEnd();
  await phone.waitForFunction(() => document.querySelectorAll("#incident-results .live-activity-item").length >= 120, null, { timeout: 5000 });
  // The last control clears the bottom navigation.
  for (let waited = 0; waited < 20000 && !(await phone.locator(".incident-earlier-day").count()); waited += 150) { await toEnd(); await phone.waitForTimeout(150); }
  await toEnd();
  await phone.waitForTimeout(300);
  const clear = await phone.evaluate(() => {
    const button = document.querySelector(".incident-earlier-day").getBoundingClientRect();
    const frame = document.querySelector(".mobile-incidents").getBoundingClientRect();
    const nav = document.querySelector(".mobile-workspace-nav")?.getBoundingClientRect();
    return button.bottom <= frame.bottom + 1 && (!nav || button.bottom <= nav.top + 1);
  });
  assert.ok(clear, "the Earlier day button is whole, above the bottom navigation");
  assert.deepEqual(phoneErrors, []);
  await phone.close();
  console.log("Incident gallery review marks: offsets after marking, filter there-and-back, phone scroll loading and reset passed");
} finally { await browser.close(); await new Promise((resolve) => server.close(resolve)); }
