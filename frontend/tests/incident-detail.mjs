import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";
import { CAMERAS, queueFor, sceneSvg, storyIncident } from "./fixtures/incident-story.mjs";

const root = process.env.SURVNG_FRONTEND_BUILD ? new URL(`file://${process.env.SURVNG_FRONTEND_BUILD.replace(/\/$/, "")}/`) : new URL("../../survng/static/", import.meta.url);
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
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="800"><rect width="1280" height="800" fill="#30483d"/></svg>';

// Legacy notification-only incident: no scene objects or episodes, people from faces.
async function legacyChecks(width) {
  const page = await browser.newPage({ viewport: { width, height: 844 }, isMobile: width < 500, hasTouch: width < 500, ...(width < 500 ? { userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1" } : {}) });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let status = 200, imageFailure = false;
  const requests = [];
  const detail = { camera_name: "Front Door", notification: { state: "active", revision: 2, summary: "Alex detected at Front Door.", people: ["Alex"], zones: ["Porch"], representative_event_id: 42, started_at: "2026-09-13T01:00:00Z" }, incident: { id: "incident-front-door-41", camera_id: "front-door", representative_event_id: 42, created_at: "2026-09-13T01:00:10Z", start_epoch: 1789261200, last_epoch: 1789261210, events: [{ id: 42, camera_id: "front-door", created_at: "2026-09-13T01:00:10Z", labels: ["person"] }, { id: 41, camera_id: "front-door", created_at: "2026-09-13T01:00:00Z", labels: [] }] } };
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    requests.push(path);
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/cameras")) return route.fulfill({ json: [{ id: "front-door", name: "Front Door" }] });
    if (path.endsWith("/config")) return route.fulfill({ json: {} });
    if (path.endsWith("/faces/people")) return route.fulfill({ json: [] });
    if (path.endsWith("/incidents/search")) return route.fulfill({ json: { items: [detail.incident], total: 1, facets: { camera_ids: ["front-door"], labels: ["person"], zones: ["Porch"] } } });
    if (path.endsWith("/incidents/review-queue")) return route.fulfill({ json: { items: [{ incident_id: "incident-front-door-41", verdict: null }], total: 1, remaining: 1 } });
    if (path.includes("/incidents/notification/")) return route.fulfill({ status, json: detail });
    if (path.endsWith("/thumbnail.jpg") || path.endsWith("/preview.jpg")) return route.fulfill({ status: imageFailure ? 404 : 200, contentType: "image/svg+xml", body: imageFailure ? "" : svg });
    if (path.endsWith("/event-clip/settings")) return route.fulfill({ json: { before_seconds: 5, after_seconds: 5 } });
    return route.fulfill({ status: 404, json: {} });
  });
  if (width < 500) {
    // The incident list stays as it was: on phones a card opens this page.
    await page.goto(`${origin}/survng/incidents`);
    const filters = page.getByRole("button", { name: "Filters", exact: true });
    await filters.waitFor();
    await filters.click();
    await page.getByLabel("Incident camera", { exact: true }).selectOption("front-door");
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.getByRole("button", { name: "Reset filters" }).click();
    await page.getByRole("button", { name: "Done", exact: true }).click();
    await page.locator(".incident-card-open").first().click();
    await page.waitForURL("**/survng/incidents/incident-front-door-41");
  } else {
    await page.goto(`${origin}/survng/incidents/incident-front-door-41`);
  }
  const heading = page.getByRole("heading", { level: 1, name: "Alex detected at Front Door." });
  await heading.waitFor();
  assert.equal(await page.locator(".assistant-panel").count(), 0);
  assert.ok(await page.locator(".app-shell .topbar").count(), "the detail page sits inside the workspace shell");
  assert.equal(await page.getByRole("link", { name: "Live view" }).getAttribute("href"), "/survng/?camera=front-door");
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  assert.match(await page.locator(".story-aside").innerText(), /Motion only/);
  assert.ok(await page.getByRole("button", { name: "Previous" }).isDisabled());
  await page.getByText("Evidence images (4)").click();
  const frames = page.locator(".incident-detail-frame");
  assert.equal(await frames.count(), 4);
  await frames.nth(1).click();
  const viewer = page.getByRole("dialog", { name: "Evidence image viewer" });
  await viewer.waitFor();
  assert.match(await viewer.locator("img").getAttribute("src"), /events\/41\//);
  await viewer.getByRole("button", { name: "Next", exact: true }).click();
  assert.match(await viewer.locator("img").getAttribute("src"), /recordings\/preview/);
  await page.keyboard.press("Escape");
  assert.equal(await viewer.count(), 0);
  const play = page.getByRole("button", { name: "Play incident" });
  const box = await play.boundingBox();
  assert.ok(box.height >= 40);
  await play.click();
  await page.locator(".incident-recording-player").waitFor();
  assert.ok(requests.some((path) => path.includes("/recordings/window") || path.includes("/day.m3u8")));
  assert.equal(requests.some((path) => path.endsWith("/clip.mp4")), false);
  await page.getByRole("button", { name: "Close playback" }).click();
  status = 503;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.getByRole("alert").waitFor();
  assert.ok(await heading.isVisible());
  status = 200;
  detail.notification.state = "complete";
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await page.getByRole("link", { name: "Live view" }).waitFor({ state: "detached" });
  imageFailure = true;
  detail.notification.revision++;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.locator(".story-frame").getByText("Image unavailable").waitFor();
  status = 404;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.getByRole("heading", { name: "Incident unavailable" }).waitFor();
  if (width > 500) {
    await page.goto(`${origin}/survng/incidents/incident-%E0`);
    await page.getByRole("heading", { name: "Incident unavailable" }).waitFor();
  }
  assert.deepEqual(errors, []);
  await page.close();
}

// The storyboard incident through the five approved states.
async function storyChecks(width) {
  const page = await browser.newPage({ viewport: { width, height: 900 } });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let incident = storyIncident();
  const queue = queueFor(incident.incident_id);
  const verdicts = [], corrections = [], requests = [];
  const cameraOf = new Map(incident.scene_objects.flatMap((object) => object.observations.map((item) => [item.id, item.camera_id])));
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    requests.push(`${route.request().method()} ${path}`);
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/api/cameras")) return route.fulfill({ json: CAMERAS });
    if (path.includes("/incidents/notification/")) {
      const id = decodeURIComponent(path.split("/").pop());
      return route.fulfill({ json: { incident: id === incident.incident_id ? incident : { ...storyIncident(), id, incident_id: id }, notification: { state: "complete", revision: incident.revision }, camera_name: "Front Driveway" } });
    }
    if (path.endsWith("/incidents/review-queue")) return route.fulfill({ json: queue });
    if (path.endsWith("/review")) {
      const id = decodeURIComponent(path.match(/incidents\/([^/]+)\/review$/)[1]);
      const body = route.request().postDataJSON(); verdicts.push([id, body.verdict]);
      const item = queue.items.find((candidate) => candidate.incident_id === id);
      if (item) item.verdict = body.verdict;
      return route.fulfill({ json: { incident_id: item?.incident_id, verdict: body.verdict } });
    }
    if (path.endsWith("/corrections")) {
      const body = route.request().postDataJSON(); corrections.push(body);
      if (body.operation === "associate") {
        const [keep, ...others] = body.object_ids;
        const folded = incident.scene_objects.filter((object) => others.includes(object.id));
        incident = { ...incident, revision: incident.revision + 1, scene_objects: incident.scene_objects.filter((object) => !others.includes(object.id)).map((object) => object.id === keep
          ? { ...object, observations: [...object.observations, ...folded.flatMap((item) => item.observations)], visible_intervals: [...object.visible_intervals, ...folded.flatMap((item) => item.visible_intervals)] } : object) };
        return route.fulfill({ json: incident });
      }
      incident = { ...incident, revision: incident.revision + 1, scene_objects: incident.scene_objects.map((object) => object.id === body.object_id ? { ...object, label: body.label, certainty: "observed" } : object) };
      return route.fulfill({ json: incident });
    }
    if (path.includes("/recordings/window")) {
      const start = Number(url.searchParams.get("start_epoch")), end = Number(url.searchParams.get("end_epoch"));
      return route.fulfill({ json: { camera_id: path.match(/cameras\/([^/]+)\//)[1], source: "main", start_epoch: start, end_epoch: end, recordings: [{ start_epoch: start, end_epoch: end, duration_seconds: end - start, name: "segment" }] } });
    }
    if (path.includes("/incidents/observations/")) return route.fulfill({ contentType: "image/svg+xml", body: sceneSvg(cameraOf.get(decodeURIComponent(path.split("/")[4]))) });
    if (path.endsWith(".jpg")) return route.fulfill({ contentType: "image/svg+xml", body: svg });
    return route.fulfill({ json: {} });
  });
  const base = `${origin}/survng/incidents/${incident.incident_id}`;
  await page.goto(base);
  // 1 · Open: best frame, full strength, nothing selected.
  await page.getByRole("heading", { level: 1, name: "Car arrives, person walks to Front Door" }).waitFor();
  assert.match(await page.locator(".story-meta").innerText(), /1 min 42 s · 3 cameras · 3 observed objects/);
  assert.match(await page.locator(".story-frame-chips").innerText(), /Front Driveway[\s\S]*Best frame/);
  assert.equal(await page.locator(".story-box").count(), 2);
  assert.equal(await page.locator(".story-bar.dimmed, .story-card.dimmed, .story-box.dimmed").count(), 0);
  assert.deepEqual(await page.locator(".story-lane-name").allInnerTexts(), ["Front Driveway", "Front Door", "Side Yard"]);
  assert.equal(await page.locator(".story-bar.vehicle.stationary").count(), 1);
  assert.equal(await page.locator(".story-bar.uncertain").count(), 1);
  assert.deepEqual(await page.locator(".story-card-title").allInnerTexts(), ["Person", "Car", "Dog uncertain"]);
  await page.getByRole("heading", { name: "Why this is an incident" }).waitFor();
  assert.match(await page.locator(".story-chips").innerText(), /Established[\s\S]*Person in Entry[\s\S]*Side Yard analysis pending/);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  // Shortcuts are review-mode only.
  await page.keyboard.press("e");
  await page.waitForTimeout(300);
  assert.equal(verdicts.length, 0);
  // The evidence frame is contained in its box at every width (no cropped side).
  const frameBox = await page.locator(".story-frame").boundingBox();
  const imageBox = await page.locator(".story-frame-image").boundingBox();
  assert.ok(imageBox.x >= frameBox.x - 1 && imageBox.x + imageBox.width <= frameBox.x + frameBox.width + 1, `image ${JSON.stringify(imageBox)} inside frame ${JSON.stringify(frameBox)}`);
  assert.ok(imageBox.y >= frameBox.y - 1 && imageBox.y + imageBox.height <= frameBox.y + frameBox.height + 1);
  // 2 · Pick an object.
  await page.locator(".story-card", { hasText: "Person" }).click();
  await page.getByRole("heading", { name: "Person · path across cameras" }).waitFor();
  assert.equal(await page.locator(".story-card.dimmed").count(), 2);
  assert.equal(await page.locator(".story-bar.person:not(.dimmed)").count(), 2);
  assert.equal(await page.locator(".story-box.vehicle.dimmed").count(), 1);
  assert.match(await page.locator(".story-frame-chips").innerText(), /Following Person/);
  assert.equal(await page.locator(".story-path li.active").innerText().then((text) => text.includes("Front Driveway")), true);
  // 3 · Follow: a bar on another camera moves the frame there and says so.
  const doorBar = page.locator(".story-lane").nth(1).locator(".story-bar").first();
  const bar = await doorBar.boundingBox();
  await page.mouse.click(bar.x + bar.width * 0.25, bar.y + bar.height / 2);
  await page.getByRole("status").filter({ hasText: "Switched from Front Driveway" }).waitFor();
  assert.match(await page.locator(".story-frame-chips").innerText(), /Front Door/);
  assert.match(await page.locator(".story-path li.active").innerText(), /Front Door/);
  await page.getByRole("button", { name: "Follow Person" }).click();
  await page.locator(".story-frame .incident-recording-player").waitFor();
  // Follow starts from the moment pinned on the storyline (Front Door), not the first camera.
  await page.waitForFunction(() => document.querySelector(".story-frame .incident-recording-player")?.dataset.cameraId === "front-door");
  assert.ok(requests.some((item) => item.includes("/cameras/front-door/recordings/window")));
  // A bar click during playback re-targets the player to that camera.
  const drivewayBar = await page.locator(".story-lane").nth(0).locator(".story-bar.person").boundingBox();
  await page.mouse.click(drivewayBar.x + drivewayBar.width * 0.5, drivewayBar.y + drivewayBar.height / 2);
  await page.waitForFunction(() => document.querySelector(".story-frame .incident-recording-player")?.dataset.cameraId === "front-driveway");
  assert.equal(await page.locator(".story-frame-chip.strong").innerText(), "Front Driveway");
  await page.getByRole("button", { name: "Close playback" }).click();
  // 4 · Correct the uncertain object.
  await page.locator(".story-card", { hasText: "Dog" }).click();
  await page.getByRole("heading", { name: "Correct this object" }).waitFor();
  assert.match(await page.locator(".story-frame-chips").innerText(), /Side Yard[\s\S]*Only supporting image/);
  assert.equal(await page.locator(".story-box.uncertain").count(), 1);
  await page.getByRole("button", { name: "Change label" }).click();
  await page.getByLabel("New label").fill("cat");
  await page.getByRole("button", { name: "Save label" }).click();
  await page.getByText("Label corrected.").waitFor();
  assert.deepEqual(corrections.at(-1), { expected_revision: 7, operation: "label", object_id: "object-dog", label: "cat" });
  await page.locator(".story-card", { hasText: "Person" }).click();
  await page.getByRole("button", { name: "Correct this object" }).click();
  await page.getByRole("button", { name: "Same as another object" }).click();
  await page.getByLabel("Same object as").selectOption("object-car");
  await page.getByRole("button", { name: "Associate", exact: true }).click();
  // The person was folded into the car: the car stays selected, nothing is orphaned-dimmed.
  await page.waitForFunction(() => document.querySelectorAll(".story-card").length === 2);
  assert.deepEqual(corrections.at(-1).object_ids, ["object-car", "object-person"]);
  assert.equal(await page.locator(".story-card[aria-pressed=true]").innerText().then((text) => text.startsWith("Car")), true);
  assert.equal(await page.locator(".story-card.dimmed").count(), 1);
  await page.locator(".story-card", { hasText: "Car" }).click();
  await page.getByRole("heading", { name: "Why this is an incident" }).waitFor();
  assert.equal(await page.locator(".story-card.dimmed, .story-bar.dimmed").count(), 0);
  // 5 · Review mode: one bar, verdicts by key, then the next undecided incident.
  await page.getByRole("button", { name: "Review mode" }).click();
  assert.match(page.url(), /review=1/);
  const reviewBar = page.getByRole("region", { name: "Review" });
  await reviewBar.waitFor();
  assert.match(await reviewBar.innerText(), /21 left/);
  assert.match(await page.locator(".story-crumb").innerText(), /Review mode · 5 of 25/);
  for (const name of ["Reviewed", "Keep for later", "Nothing here"]) assert.ok(await reviewBar.getByRole("button", { name: new RegExp(name) }).isVisible());
  // Typing in a field never triggers review shortcuts, even in review mode.
  await page.locator(".story-card", { hasText: "Cat" }).click();
  await page.getByRole("button", { name: "Correct this object" }).click();
  await page.getByRole("button", { name: "Change label" }).click();
  await page.getByLabel("New label").pressSequentially("exfjk");
  await page.waitForTimeout(300);
  assert.equal(verdicts.length, 0);
  assert.match(page.url(), /incident-story-1\?review=1/);
  await page.evaluate(() => document.activeElement.blur());
  const atUrl = (pattern) => page.waitForFunction((source) => new RegExp(source).test(location.href), pattern.source).catch((error) => { throw new Error(`${pattern} not reached; at ${page.url()}; ${error.message}`); });
  await page.keyboard.press("e");
  await atUrl(/incident-q-5\?review=1/);
  assert.deepEqual(verdicts, [["incident-story-1", "reviewed"]]);
  await page.waitForFunction(() => document.activeElement?.matches(".story-heading h1"));
  assert.match(await page.title(), /SurvNG$/);
  await page.getByRole("region", { name: "Review" }).waitFor();
  await page.keyboard.press("j");
  await atUrl(/incident-q-6\?review=1/);
  assert.equal(verdicts.length, 1);
  await page.keyboard.press("k");
  await atUrl(/incident-q-5\?review=1/);
  await page.goBack();
  await atUrl(/incident-q-6/);
  await page.getByRole("button", { name: "Exit review" }).click();
  assert.equal(await page.getByRole("region", { name: "Review" }).count(), 0);
  await page.keyboard.press("e");
  await page.waitForTimeout(300);
  assert.equal(verdicts.length, 1);
  assert.deepEqual(errors, []);
  await page.close();
}

// Optional provider title: shown when ready; the built-in title otherwise.
async function titleChecks() {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let reply = { status: 200, json: { incident_id: "incident-story-1", status: "ready", title: "Car pulls in and a person walks to the front door", revision: 7, current: true } };
  const titleRequests = [];
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/api/cameras")) return route.fulfill({ json: CAMERAS });
    if (path.includes("/incidents/notification/")) return route.fulfill({ json: { incident: storyIncident(), notification: { state: "complete" }, camera_name: "Front Driveway" } });
    if (path.endsWith("/incident-story-1/title")) { titleRequests.push(path); return route.fulfill(reply); }
    return route.fulfill({ json: {} });
  });
  const url = `${origin}/survng/incidents/incident-story-1`;
  await page.goto(url);
  await page.getByRole("heading", { level: 1, name: "Car pulls in and a person walks to the front door" }).waitFor();
  assert.match(await page.locator(".story-heading h1").getAttribute("title"), /Built-in: Car arrives, person walks to Front Door/);
  reply = { status: 200, json: { incident_id: "incident-story-1", status: "failed", title: null, reason: "provider_failed", message: "AI title unavailable" } };
  await page.goto(url);
  await page.getByRole("heading", { level: 1, name: "Car arrives, person walks to Front Door" }).waitFor();
  await page.getByText("· AI title unavailable").waitFor();
  reply = { status: 200, json: { incident_id: "incident-story-1", status: "disabled", title: null } };
  await page.goto(url);
  await page.getByRole("heading", { level: 1, name: "Car arrives, person walks to Front Door" }).waitFor();
  assert.equal(await page.getByText("AI title unavailable").count(), 0);
  reply = { status: 500, json: {} };
  await page.goto(url);
  await page.getByRole("heading", { level: 1, name: "Car arrives, person walks to Front Door" }).waitFor();
  // The title request follows the incident load; give the last one time to arrive.
  for (let attempt = 0; attempt < 50 && titleRequests.length < 4; attempt += 1) await page.waitForTimeout(100);
  assert.ok(titleRequests.length >= 4);
  assert.deepEqual(errors, []);
  await page.close();
}

// Viewers cannot record verdicts, so a review link opens the plain page.
async function viewerChecks() {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: true, user: { username: "guest", role: "viewer" } } });
    if (path.endsWith("/api/cameras")) return route.fulfill({ json: CAMERAS });
    if (path.includes("/incidents/notification/")) return route.fulfill({ json: { incident: storyIncident(), notification: { state: "complete" }, camera_name: "Front Driveway" } });
    if (path.endsWith("/incidents/review-queue")) return route.fulfill({ json: queueFor("incident-story-1") });
    return route.fulfill({ json: {} });
  });
  await page.goto(`${origin}/survng/incidents/incident-story-1?review=1`);
  await page.getByRole("heading", { level: 1, name: "Car arrives, person walks to Front Door" }).waitFor();
  assert.equal(await page.getByRole("region", { name: "Review" }).count(), 0);
  assert.equal(await page.getByRole("button", { name: /Review mode|Exit review/ }).count(), 0);
  await page.locator(".story-card", { hasText: "Dog" }).click();
  await page.getByText("Only an administrator can correct objects.").waitFor();
  assert.deepEqual(errors, []);
  await page.close();
}

try {
  await viewerChecks();
  await titleChecks();
  for (const width of [360, 390, 1440]) await legacyChecks(width);
  for (const width of [390, 1440]) await storyChecks(width);
  console.log("Incident detail: storyboard states, review keys, corrections, mobile, playback, reconnect, missing image and expired link checks passed");
} finally { await browser.close(); await new Promise((resolve) => server.close(resolve)); }
