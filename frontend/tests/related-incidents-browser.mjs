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
  const event = { id: 41, camera_id: "gate", created_at: "2026-10-03T12:00:00Z", labels: ["person"], objects: [{ label: "cat", snapshot_visible: false }, { label: "person", confidence: .8 }, { label: "dog", confidence: .7 }], snapshot_path: "cover.webp" };
  const incident = { ...event, id: "scene-1", incident_id: "scene-1", revision: 1, summary: "Person at Gate.", representative_event_id: 41, event_ids: [41], events: [event], has_objects: true, start_epoch: 1791028800, last_epoch: 1791028810, start_at: event.created_at, end_at: "2026-10-03T12:00:10Z", episodes: [{ id: "ep-1", camera_id: "gate", start_at: event.created_at, end_at: "2026-10-03T12:00:10Z", event_ids: [41] }] };
  let missing = false;
  const visualRequests = [];
  let visualOutcome = "results";
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/semantic-search/visual")) {
      visualRequests.push(route.request().postDataJSON());
      return route.fulfill({ status: visualOutcome === "error" ? 503 : 200, json: visualOutcome === "error"
        ? { detail: "Visual search unavailable" }
        : { results: visualOutcome === "empty" ? [] : [{ event: { ...event, id: 43 }, score: .8, match_strength: "visual_similarity" }] } });
    }
    if (path.endsWith("/appearance-matches")) return route.fulfill({ json: { matches: [] } });
    if (path.endsWith("/related-incidents")) return route.fulfill({ json: { matches: [42, 43, 44].map((id) => ({ event_id: id, camera_id: "gate", created_at: event.created_at, visually_similar: true })) } });
    if (path.includes("/incidents/by-event/")) return route.fulfill({ status: path.endsWith("/42") ? 200 : 404, json: { ...incident, id: "scene-2", representative_event_id: 42, events: [{ ...event, id: 42 }] } });
    if (path.endsWith("/events/43")) return route.fulfill({ json: { ...event, id: 43 } });
    if (path.endsWith("/events/44")) return route.fulfill({ status: missing ? 200 : 404, json: { ...event, id: 44 } });
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
  const related = page.locator(".incident-related").filter({ has: page.getByRole("heading", { name: "Related incidents", exact: true }) });
  const tiles = related.locator(".incident-related-grid button");
  await tiles.first().waitFor();
  for (const [index, id] of [[0, 42], [1, 43]]) {
    await tiles.nth(index).click();
    await page.waitForFunction((id) => document.querySelector(`.incident-related-grid button[aria-pressed="true"] img`)?.getAttribute("src").includes(`/events/${id}/`), id);
    assert.match(page.url(), /incident_id=scene-1/);
    assert.ok(await page.locator(`.incident-desktop-focus img[src*="/events/${id}/"]`).count() > 0);
  }
  await tiles.nth(2).click();
  await page.getByRole("alert").filter({ hasText: "no longer available" }).waitFor();
  assert.equal(await tiles.nth(1).getAttribute("aria-pressed"), "true");
  missing = true;
  await tiles.nth(2).click();
  await page.waitForFunction(() => document.querySelectorAll('.incident-related-grid button')[2]?.getAttribute('aria-pressed') === 'true');
  await related.getByRole("button", { name: "Selected incident", exact: true }).click();
  assert.equal(await related.locator('[aria-pressed="true"]').count(), 0);
  // Explicit controls must work without snapshot annotation boxes, and use
  // snapshot-visible object indexes rather than the raw detection array.
  const actions = page.getByRole("group", { name: "Find similar objects" });
  assert.equal(await actions.getByRole("button").count(), 2);
  assert.equal(visualRequests.length, 0, "opening details must not start a search");
  const similar = page.locator(".incident-visual-similar");
  await actions.getByRole("button", { name: "Find similar: person (object 1)", exact: true }).click();
  await similar.locator(".incident-visual-similar-card").waitFor();
  assert.equal(visualRequests.at(-1).event_id, 41);
  assert.equal(visualRequests.at(-1).object_index, 0);
  await similar.getByRole("button", { name: "Clear", exact: true }).click();
  await similar.waitFor({ state: "hidden" });
  visualOutcome = "error";
  await actions.getByRole("button", { name: "Find similar: dog (object 2)", exact: true }).click();
  await similar.getByText("Visual search unavailable", { exact: true }).waitFor();
  assert.equal(visualRequests.at(-1).event_id, 41);
  assert.equal(visualRequests.at(-1).object_index, 1);
  await similar.getByRole("button", { name: "Clear", exact: true }).click();
  visualOutcome = "empty";
  await actions.getByRole("button", { name: "Find similar: dog (object 2)", exact: true }).click();
  await similar.getByText("No similar incidents in the indexes yet.", { exact: true }).waitFor();
  assert.deepEqual(errors, []);
  console.log("Related selection and explicit Find similar: results, object indexes, clear, error and empty states passed");
} finally { await browser.close(); await new Promise((resolve) => server.close(resolve)); }
