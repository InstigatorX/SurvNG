import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";
import { BASE, CAMERAS, queueFor, sceneSvg, storyIncident } from "./fixtures/incident-story.mjs";

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
    // On phones the list is the gallery, and a picture opens this page.
    await page.goto(`${origin}/survng/incidents`);
    const filters = page.getByRole("button", { name: "Filters", exact: true });
    await filters.waitFor();
    await filters.click();
    await page.getByRole("checkbox", { name: "Front Door", exact: true }).check();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.getByRole("button", { name: "Reset filters" }).click();
    assert.equal(await page.getByRole("checkbox", { name: "Front Door", exact: true }).isChecked(), false);
    await page.getByRole("button", { name: "Done", exact: true }).click();
    await page.locator("#incident-results .live-activity-select").first().click();
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
  assert.match(await page.locator(".story-meta").innerText(), /1 min 42 s · 3 cameras/);
  assert.equal(await page.getByRole("heading", { name: "Sightings (3)" }).count(), 1);
  assert.equal(await page.getByText("Not recognized", { exact: false }).count(), 0);
  assert.match(await page.locator(".story-frame-chips").innerText(), /Front Driveway[\s\S]*Best frame/);
  assert.equal(await page.locator(".story-box").count(), 2);
  assert.equal(await page.locator(".story-bar.dimmed, .story-card.dimmed, .story-box.dimmed").count(), 0);
  assert.deepEqual(await page.locator(".story-lane-name").allInnerTexts(), ["Front Driveway", "Front Door", "Side Yard"]);
  assert.equal(await page.locator(".story-bar.vehicle.stationary").count(), 1);
  assert.equal(await page.locator(".story-bar.uncertain").count(), 1);
  assert.deepEqual(await page.locator(".story-card-title").allInnerTexts(), ["Person", "Car", "Dog uncertain"]);
  await page.getByRole("heading", { name: "Why this is an incident" }).waitFor();
  assert.match(await page.locator(".story-chips").innerText(), /^Met alert criteria · Person in Entry\s+Side Yard analysis pending$/);
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
  await doorBar.scrollIntoViewIfNeeded();
  const bar = await doorBar.boundingBox();
  await page.mouse.click(bar.x + bar.width * 0.25, bar.y + bar.height / 2);
  await page.getByRole("status").filter({ hasText: "Switched from Front Driveway" }).waitFor();
  assert.match(await page.locator(".story-frame-chips").innerText(), /Front Door/);
  // The clicked moment lies between analysed frames: the person stays marked, as an estimate.
  await page.locator(".story-box.person.estimated").waitFor();
  assert.match(await page.locator(".story-box.person.estimated").innerText(), /Person · estimated/);
  assert.match(await page.locator(".story-frame-chips").innerText(), /Following Person/);
  assert.match(await page.locator(".story-path li.active").innerText(), /Front Door/);
  await page.getByRole("button", { name: "Follow Person" }).click();
  await page.locator(".story-frame .incident-recording-player").waitFor();
  // Follow starts from the moment pinned on the storyline (Front Door), not the first camera.
  await page.waitForFunction(() => document.querySelector(".story-frame .incident-recording-player")?.dataset.cameraId === "front-door");
  assert.ok(requests.some((item) => item.includes("/cameras/front-door/recordings/window")));
  // A bar click during playback re-targets the player to that camera.
  const driveway = page.locator(".story-lane").nth(0).locator(".story-bar.person");
  await driveway.scrollIntoViewIfNeeded();
  const drivewayBar = await driveway.boundingBox();
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
  assert.match(await page.locator(".story-crumb").innerText(), /5 of 25/);
  assert.ok(await page.getByRole("button", { name: "Exit review" }).isVisible());
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
  let ongoing = false;
  let reviseTo = null;
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/api/cameras")) return route.fulfill({ json: CAMERAS });
    if (path.includes("/incidents/notification/")) return route.fulfill({ json: { incident: { ...storyIncident(), ...(reviseTo ? { revision: reviseTo } : {}) }, notification: { state: ongoing ? "active" : "complete" }, camera_name: "Front Driveway" } });
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
  // A completed incident that reopens goes back to the built-in title.
  reply = { status: 200, json: { incident_id: "incident-story-1", status: "ready", title: "Car pulls in and a person walks to the front door", revision: 7, current: true } };
  await page.goto(url);
  await page.getByRole("heading", { level: 1, name: "Car pulls in and a person walks to the front door" }).waitFor();
  ongoing = true;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.getByRole("heading", { level: 1, name: "Car arrives, person walks to Front Door" }).waitFor();
  // When it settles again with new evidence, the old wording does not come back.
  ongoing = false;
  reviseTo = 8;
  reply = { status: 500, json: {} };
  const settledRequests = titleRequests.length;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  // Wait until the settled page has asked again (and been refused).
  for (let attempt = 0; attempt < 50 && titleRequests.length <= settledRequests; attempt += 1) await page.waitForTimeout(100);
  assert.ok(titleRequests.length > settledRequests, "the settled incident asked for a new wording");
  await page.waitForTimeout(200);
  assert.equal(await page.getByText("Car pulls in and a person walks to the front door").count(), 0);
  reviseTo = null;
  ongoing = true;
  // A growing incident keeps the built-in title and does not ask for an AI one.
  await page.waitForTimeout(500);
  const before = titleRequests.length;
  reply = { status: 200, json: { incident_id: "incident-story-1", status: "ready", title: "Should not show", revision: 7, current: true } };
  await page.goto(url);
  await page.getByRole("heading", { level: 1, name: "Car arrives, person walks to Front Door" }).waitFor();
  await page.waitForTimeout(500);
  assert.equal(titleRequests.length, before);
  ongoing = false;
  reply = { status: 500, json: {} };
  await page.goto(url);
  await page.getByRole("heading", { level: 1, name: "Car arrives, person walks to Front Door" }).waitFor();
  // The title request follows the incident load; give the last one time to arrive.
  for (let attempt = 0; attempt < 50 && titleRequests.length < before + 1; attempt += 1) await page.waitForTimeout(100);
  assert.ok(titleRequests.length >= before + 1);
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

// Follow skips a camera with no recording; Auto theme on a dark OS keeps dark ink;
// an AI wording of an older revision does not replace the built-in title.
async function followThemeChecks() {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: "dark" });
  await page.addInitScript(() => localStorage.setItem("survng.theme", "auto"));
  let drivewayMode = "empty";
  const windowStarts = [];
  // 15-minute windows: the story's Front Door interval is moved to cross one.
  const BOUNDARY = Math.ceil((BASE + 60) / 900) * 900;
  const crossing = storyIncident();
  const door = crossing.scene_objects.find((object) => object.id === "object-person").visible_intervals[1];
  door.start_at = new Date((BOUNDARY - 30) * 1000).toISOString();
  door.end_at = new Date((BOUNDARY + 40) * 1000).toISOString();
  crossing.end_at = door.end_at; crossing.last_epoch = BOUNDARY + 40;
  crossing.episodes[1].end_at = door.end_at;
  const doorSightings = crossing.scene_objects.find((object) => object.id === "object-person").observations.filter((item) => item.camera_id === "front-door");
  doorSightings.forEach((item, index) => { item.captured_at = new Date((BOUNDARY - 25 + index * 25) * 1000).toISOString(); });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/api/**", (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/api/cameras")) return route.fulfill({ json: CAMERAS });
    if (path.includes("/incidents/notification/")) return route.fulfill({ json: { incident: drivewayMode === "boundary" ? crossing : storyIncident(), notification: { state: "complete" }, camera_name: "Front Driveway" } });
    if (path.endsWith("/title")) return route.fulfill({ json: { incident_id: "incident-story-1", status: "ready", title: "Old wording", revision: 6, current: false } });
    if (path.includes("/recordings/window")) {
      const start = Number(url.searchParams.get("start_epoch")), end = Number(url.searchParams.get("end_epoch"));
      // The usual failure: main errors while the (unrecorded) sub stream answers empty.
      if (path.includes("/front-driveway/") && drivewayMode === "error" && url.searchParams.get("source") !== "live") return route.fulfill({ status: 500, json: {} });
      // A clip crossing a window boundary, recorded only in the later window.
      if (drivewayMode === "boundary" && path.includes("/front-door/")) {
        windowStarts.push(start);
        const rows = start >= BOUNDARY ? [{ start_epoch: start, end_epoch: end, duration_seconds: end - start, name: "segment" }] : [];
        return route.fulfill({ json: { camera_id: "front-door", source: "main", start_epoch: start, end_epoch: end, recordings: rows } });
      }
      const recordings = path.includes("/front-driveway/") ? [] : [{ start_epoch: start, end_epoch: end, duration_seconds: end - start, name: "segment" }];
      return route.fulfill({ json: { camera_id: "x", source: "main", start_epoch: start, end_epoch: end, recordings } });
    }
    return route.fulfill({ json: {} });
  });
  await page.goto(`${origin}/survng/incidents/incident-story-1`);
  await page.getByRole("heading", { level: 1, name: "Car arrives, person walks to Front Door" }).waitFor();
  await page.waitForTimeout(300);
  assert.equal(await page.getByText("Old wording").count(), 0);
  assert.equal(await page.locator(".story-uncertain-tag").evaluate((el) => getComputedStyle(el).color), "rgb(196, 181, 253)");
  await page.locator(".story-card", { hasText: "Person" }).click();
  await page.getByRole("button", { name: "Follow Person" }).click();
  await page.waitForFunction(() => document.querySelector(".story-frame .incident-recording-player")?.dataset.cameraId === "front-door");
  assert.match(await page.locator(".story-frame [role=status]").innerText(), /No recording on Front Driveway; switched to Front Door/);
  // A failed request is not an empty recording: Follow stays on that camera.
  drivewayMode = "error";
  await page.reload();
  await page.locator(".story-card", { hasText: "Person" }).click();
  await page.getByRole("button", { name: "Follow Person" }).click();
  await page.locator(".story-frame .incident-recording-player").getByText("No recording window found").waitFor();
  await page.waitForTimeout(500);
  assert.equal(await page.locator(".story-frame .incident-recording-player").getAttribute("data-camera-id"), "front-driveway");
  // Following into a clip that crosses a window boundary finds the later recording.
  drivewayMode = "boundary";
  await page.reload();
  await page.locator(".story-card", { hasText: "Person" }).click();
  const boundaryBar = page.locator(".story-lane").nth(1).locator(".story-bar").first();
  await boundaryBar.scrollIntoViewIfNeeded();
  const doorBar = await boundaryBar.boundingBox();
  await page.mouse.click(doorBar.x + 2, doorBar.y + doorBar.height / 2);
  await page.getByRole("button", { name: "Follow Person" }).click();
  await page.waitForFunction(() => document.querySelector(".story-frame .incident-recording-player")?.dataset.cameraId === "front-door");
  for (let attempt = 0; attempt < 50 && !windowStarts.includes(BOUNDARY); attempt += 1) await page.waitForTimeout(100);
  assert.ok(windowStarts.includes(BOUNDARY - 900) && windowStarts.includes(BOUNDARY), JSON.stringify(windowStarts));
  // (The fixture's segment has no real media, so playback itself is not checked here.)
  await page.waitForTimeout(500);
  assert.equal(await page.locator(".story-frame .incident-recording-player").getAttribute("data-camera-id"), "front-door");
  assert.doesNotMatch(await page.locator(".story-frame [role=status]").innerText(), /No recording on Front Door/);
  assert.deepEqual(errors, []);
  await page.close();
}

async function stillEvidenceChecks() {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, timezoneId: "America/New_York", hasTouch: true });
  page.setDefaultTimeout(5000);
  const incident = storyIncident();
  const person = incident.scene_objects.find((object) => object.id === "object-person");
  const sighting = person.observations.find((observation) => observation.id === "obs-person-door-47");
  sighting.snapshot_available = false;
  delete incident.snapshot_observation_id;
  delete incident.snapshot_url;
  incident.snapshot_captured_at = new Date((BASE + 30.5) * 1000).toISOString();
  const requests = [];
  await page.route("**/api/**", (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/cameras")) return route.fulfill({ json: CAMERAS });
    if (path.includes("/incidents/notification/")) return route.fulfill({ json: { incident, notification: { state: "complete" } } });
    if (path.endsWith("/preview.jpg")) {
      requests.push([path, Number(url.searchParams.get("epoch")), url.searchParams.get("source")]);
      if (url.searchParams.get("source") === "main") return route.fulfill({ status: 500 });
      return route.fulfill({
        contentType: "image/svg+xml", body: svg,
        headers: { "X-SurvNG-Actual-Timestamp": String(BASE + 48) },
      });
    }
    if (path.endsWith("/thumbnail.jpg") || path.endsWith("/snapshot")) return route.fulfill({ contentType: "image/svg+xml", body: svg });
    return route.fulfill({ json: {} });
  });
  await page.goto(`${origin}/survng/incidents/${incident.incident_id}`);
  const frame = page.locator(".story-frame");
  await frame.getByText("Best frame", { exact: false }).waitFor();
  assert.match(await frame.locator(".story-frame-chips time").getAttribute("datetime", { timeout: 1000 }), /15:24:50\.500Z/);
  const doorBar = page.locator(".story-lane").nth(1).locator(".story-bar").first();
  await doorBar.press("Enter");
  await frame.getByText("Nearby recording frame", { exact: false }).waitFor();
  assert.deepEqual(requests.map(([, epoch, source]) => [epoch, source]), [[BASE + 47, "main"], [BASE + 47, "live"]]);
  assert.match(requests[0][0], /\/front-door\//);
  // No analysed frame matches this picture, so nothing is drawn as observed;
  // the followed person is marked with an estimate only.
  assert.equal(await frame.locator(".story-box:not(.estimated)").count(), 0);
  assert.equal(await frame.locator(".story-box.person.estimated").count(), 1);
  assert.match(await frame.locator(".story-frame-chips time").getAttribute("datetime"), /15:25:08\.000Z/);
  assert.match(await page.getByRole("link", { name: "View in Timeline" }).getAttribute("href"), /source=live/);
  await page.close();
}

async function reviewLayoutChecks() {
  const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
  const incident = storyIncident();
  incident.continuity_uncertain = true;
  let disconnected = false;
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/cameras")) return route.fulfill({ json: CAMERAS });
    if (path.includes("/incidents/notification/")) return disconnected
      ? route.fulfill({ status: 503, json: {} })
      : route.fulfill({ json: { incident, notification: { state: "complete" } } });
    if (path.endsWith("/incidents/review-queue")) return route.fulfill({ json: queueFor(incident.incident_id) });
    if (path.endsWith("/snapshot")) return route.fulfill({ contentType: "image/svg+xml", body: svg });
    return route.fulfill({ json: {} });
  });
  await page.goto(`${origin}/survng/incidents/${incident.incident_id}?review=1`);
  await page.getByRole("button", { name: "Reviewed", exact: true }).waitFor();
  const storyline = page.getByRole("region", { name: "Storyline" });
  async function checkDesktopColumns() {
    const line = await storyline.boundingBox();
    const layout = await page.locator(".story-layout").boundingBox();
    const pane = await page.locator(".story-aside").boundingBox();
    assert.ok(Math.abs(line.x - layout.x) <= 1 && Math.abs(line.width - layout.width) <= 1, "storyline spans the player and the right pane");
    assert.ok(pane.width <= 320, `right pane should be at most 320 px, got ${pane.width}`);
  }
  const initial = await storyline.boundingBox();
  assert.ok(initial.y + initial.height <= 768, "the player and storyline fit in desktop Review mode");
  await checkDesktopColumns();
  await page.setViewportSize({ width: 1000, height: 768 });
  disconnected = true;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.getByRole("button", { name: "Retry" }).waitFor();
  await page.waitForFunction(() => document.querySelector(".story-storyline").getBoundingClientRect().bottom <= innerHeight);
  disconnected = false;
  await page.getByRole("button", { name: "Retry" }).click();
  await page.getByRole("button", { name: "Retry" }).waitFor({ state: "detached" });
  await page.getByRole("button", { name: "Exit review" }).click();
  await page.setViewportSize({ width: 2560, height: 1440 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await checkDesktopColumns();
  async function checkTallViewport() {
    const link = await page.getByRole("link", { name: "Open in the incident list" }).boundingBox();
    const spaceBelow = 1440 - (link.y + link.height);
    assert.ok(spaceBelow >= 0 && spaceBelow <= 48, `collapsed evidence controls should leave at most 48 px of bottom space, got ${spaceBelow}`);
    const before = await page.locator(".story-frame").boundingBox();
    await page.getByText(/Evidence images \(/).click();
    assert.equal((await page.locator(".story-frame").boundingBox()).height, before.height, "expanding evidence should keep the player height");
    await page.getByText(/Evidence images \(/).click();
  }
  await checkTallViewport();
  incident.speech = { segments: 1 };
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.getByText("Speech", { exact: true }).waitFor();
  await checkTallViewport();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForFunction(() => {
    const card = document.querySelector(".story-card").getBoundingClientRect();
    const bottomNav = document.querySelector(".mobile-workspace-nav").getBoundingClientRect();
    return card.bottom <= bottomNav.top;
  });
  await page.locator(".story-card", { hasText: "Person" }).click();
  const bounds = await page.evaluate(() => {
    const frame = document.querySelector(".story-frame").getBoundingClientRect();
    const context = document.querySelector(".story-context").getBoundingClientRect();
    return { frameBottom: frame.bottom, contextTop: context.top, contextBottom: context.bottom };
  });
  assert.ok(bounds.contextTop >= bounds.frameBottom && bounds.contextTop - bounds.frameBottom <= 16, "selected sighting sits directly beneath the player on phones");
  assert.ok(bounds.contextBottom <= 844, "selected sighting actions fit in the phone viewport");
  assert.ok(await page.getByText("Some sightings may show the same subject.", { exact: true }).isVisible());
  await page.getByRole("button", { name: "Clear selection" }).press("Enter");
  assert.equal(await page.evaluate(() => document.activeElement.textContent), "Sightings (3)");
  await page.evaluate(() => {
    const frame = document.querySelector(".story-frame");
    scrollTo(0, scrollY + frame.getBoundingClientRect().top + 100);
  });
  await page.locator(".story-bar.uncertain").press("Enter");
  await page.waitForFunction(() => {
    const frame = document.querySelector(".story-frame").getBoundingClientRect();
    const topbar = document.querySelector(".topbar").getBoundingClientRect();
    const actions = document.querySelector(".story-context .story-actions")?.getBoundingClientRect();
    return frame.top >= topbar.bottom && actions?.bottom < innerHeight;
  });
  await page.close();
}

try {
  await reviewLayoutChecks();
  await stillEvidenceChecks();
  await followThemeChecks();
  await viewerChecks();
  await titleChecks();
  for (const width of [360, 390, 1440]) await legacyChecks(width);
  for (const width of [390, 1440]) await storyChecks(width);
  console.log("Incident detail: storyboard states, review keys, corrections, mobile, playback, reconnect, missing image and expired link checks passed");
} finally { await browser.close(); await new Promise((resolve) => server.close(resolve)); }
