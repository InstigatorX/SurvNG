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
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, hasTouch: true });
  page.setDefaultTimeout(10000);
  const errors = [];
  const corrections = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const observed = (id, eventId, label, available = true) => ({ id, event_id: eventId, camera_id: "gate", captured_at: "2026-09-24T20:25:45Z", label, confidence: .75, incident_eligible: false, snapshot_available: available, detection_frame_width: 1280, detection_frame_height: 800, box: { x1: 128, y1: 160, x2: 640, y2: 560 } });
  const first = { id: 80913, camera_id: "gate", created_at: "2026-09-24T20:25:45Z", objects: [{ label: "person", confidence: .75, incident_eligible: false, box: { x1: 128, y1: 160, x2: 640, y2: 560 }, detection_frame_width: 1280, detection_frame_height: 800 }], snapshot_path: "first.webp" };
  const second = { ...first, id: 80914, camera_id: "driveway", created_at: "2026-09-24T20:25:55Z", objects: [{ ...first.objects[0], label: "car" }], snapshot_path: "second.webp" };
  let incident = { ...first, id: "scene-1", incident_id: "scene-1", revision: 4, representative_event_id: 80914, event_ids: [80913, 80914], events: [second, first], labels: ["person", "car"], has_objects: true, duration_seconds: 10, start_at: first.created_at, end_at: second.created_at, summary: "Person and car observed at Gate.", coverage: { state: "incomplete", gaps: [{ start_at: first.created_at, end_at: second.created_at, reason: "missing_video" }] }, episodes: [{ id: "episode-1", camera_id: "gate", start_at: first.created_at, end_at: second.created_at, coverage: { state: "sampled" } }, { id: "episode-2", camera_id: "driveway", start_at: second.created_at, end_at: "2026-09-24T20:26:07Z", event_ids: [80914], coverage: { state: "historical" } }], scene_objects: [
    { id: "object-1", label: "person", certainty: "supported", continuity_uncertain: true, first_seen_at: first.created_at, last_seen_at: second.created_at, source_event_id: 80913, object_index: 0, observations: [observed("obs-1", 80913, "person"), observed("obs-2", 80914, "person", false)] },
    { id: "object-2", label: "car", certainty: "uncertain", first_seen_at: second.created_at, source_event_id: 80914, object_index: 0, observations: [observed("obs-3", 80914, "car")] },
  ] };
  incident.continuity_uncertain = true;
  incident.establishment = { status: "established", reason: "supported_movement", summary: "A person moved across the gate view.", supporting_observation_ids: ["obs-1"], policy_version: 1 };
  incident.scene_objects[0].observations[0].image = { width: 1280, height: 800, source: "live_substream", captured_at: first.created_at, analyzed_frame: true };
  incident.scene_objects[0].observations[0].review_image = { url: "/api/review-image.jpg", width: 2560, height: 1600, source: "main", captured_at: second.created_at, analyzed_frame: false };
  incident.snapshot_url = "/api/incidents/observations/obs-1/snapshot";
  incident.snapshot_observation_id = "obs-1";
  incident.activity = [
    { kind: "last_seen", object_id: "object-1", label: "person", camera_id: "gate", captured_at: second.created_at, observation_id: "obs-2" },
    { kind: "appeared", object_id: "object-1", label: "person", camera_id: "gate", captured_at: first.created_at, observation_id: "obs-1" },
  ];
  incident.alert_decisions = [{ event_id: 80913, eligible: false, objects: [{ label: "person", eligible: false, reasons: ["outside_incident_zone"], zones: [] }] }];
  let conflict = false;
  const clipRequests = [];
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path.endsWith("/clip.mp4")) {
      clipRequests.push(path);
      return route.fulfill({ status: 404, body: "" });
    }
    if (path.includes("/recordings/window")) {
      const start = Number(url.searchParams.get("start_epoch"));
      const end = Number(url.searchParams.get("end_epoch"));
      const cameraId = decodeURIComponent(path.split("/cameras/")[1].split("/")[0]);
      return route.fulfill({ json: { camera_id: cameraId, source: url.searchParams.get("source") || "main", start_epoch: start, end_epoch: end, recordings: [{ start_epoch: start, end_epoch: end, duration_seconds: Math.max(0.01, end - start), name: "segment" }] } });
    }
    if (path.endsWith("/event-clip/settings")) return route.fulfill({ json: { before_seconds: 5, after_seconds: 5 } });
    if (path.includes("/incidents/notification/")) return route.fulfill({ json: { incident, notification: { state: "active", revision: incident.revision }, camera_name: "Gate" } });
    if (path.endsWith("/auth/session")) return route.fulfill({ json: { enabled: false } });
    if (path.endsWith("/cameras")) return route.fulfill({ json: [{ id: "gate", name: "Gate" }] });
    if (path.endsWith("/config")) return route.fulfill({ json: { cameras: [{ id: "gate", zones: [{ name: "Road", enabled: true, behavior: "incident", color: "#22c55e", points: [{ x: 0.1, y: 0.2 }, { x: 0.8, y: 0.2 }, { x: 0.5, y: 0.9 }] }] }] } });
    if (path.endsWith("/faces/people")) return route.fulfill({ json: [] });
    if (path.endsWith("/incidents/search")) return route.fulfill({ json: { items: [incident], total: 1 } });
    if (path.endsWith("/incidents/detail")) return route.fulfill({ json: url.searchParams.get("incident_id") === "scene-2" ? { ...incident, id: "scene-2", incident_id: "scene-2", revision: 7 } : incident });
    if (path.endsWith("/corrections")) {
      const body = route.request().postDataJSON(); corrections.push(body);
      if (conflict) return route.fulfill({ status: 409, json: {} });
      incident = { ...incident, revision: incident.revision + 1, scene_objects: incident.scene_objects.map((object) => body.operation === "label" && object.id === body.object_id ? { ...object, label: body.label } : object) };
      return route.fulfill({ json: incident });
    }
    if (path.includes("/events/80913") && !path.endsWith(".jpg")) return route.fulfill({ json: first });
    if (path.includes("/events/80914") && !path.endsWith(".jpg")) return route.fulfill({ json: second });
    if (path.endsWith("/snapshot") || path.endsWith(".jpg")) return route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="800"><rect width="1280" height="800" fill="#30483d"/></svg>' });
    return route.fulfill({ json: {} });
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/survng/incidents?incident_id=scene-1`);
  const inspector = page.locator("#incident-inspector");
  await inspector.getByRole("heading", { name: "Observed objects" }).waitFor();
  assert.equal(await inspector.getByRole("heading", { name: "Current incident" }).count(), 0);
  assert.equal(await inspector.locator(".incident-scene-activity, .incident-scene-panel").count(), 0);
  const inspectorObjects = inspector.locator(".incident-observed-objects li");
  assert.equal(await inspectorObjects.count(), 2);
  assert.equal(await inspectorObjects.nth(0).textContent(), "person75%");
  assert.equal(await inspectorObjects.nth(1).textContent(), "car75%");
  assert.doesNotMatch(await inspector.locator(".incident-observed-objects ul").textContent(), /Uncertain|Possible|first observed|Activity established|missing video/);
  const retainedCover = page.locator('.incident-card img[src*="/incidents/observations/obs-1/snapshot"]').first();
  await retainedCover.waitFor();
  assert.equal(await retainedCover.evaluate((img) => img.closest(".snapshot-frame").classList.contains("object-focus-crop")), false,
    "retained observation images use client framing instead of assuming a server crop");
  const viewToggle = page.locator(".incident-desktop-focus .incident-workspace-view-toggle");
  const zoneButton = viewToggle.locator("button", { hasText: "Zones" });
  await zoneButton.waitFor();
  assert.equal(await viewToggle.locator("button").last().innerText(), "Zones");
  assert.equal(await zoneButton.locator("svg").count(), 1);
  assert.equal(await zoneButton.getAttribute("aria-pressed"), "false");
  assert.equal(await page.locator(".incident-desktop-focus .snapshot-zone-layer").count(), 0);
  await zoneButton.click();
  assert.equal(await zoneButton.getAttribute("aria-pressed"), "true");
  assert.equal(await page.locator(".incident-desktop-focus .snapshot-zone-layer polygon").count(), 1);
  await zoneButton.click();
  assert.equal(await page.locator(".incident-desktop-focus .snapshot-zone-layer").count(), 0);
  await page.goto(`http://127.0.0.1:${server.address().port}/survng/incidents/incident-scene-1`);
  const panel = page.locator(".incident-detail-page .incident-scene-panel");
  await panel.getByText("Person and car observed at Gate.", { exact: true }).waitFor();
  await panel.getByText("Activity established", { exact: true }).waitFor();
  assert.equal(await panel.getByText("supported movement", { exact: true }).isVisible(), false);
  assert.equal(await panel.locator(".incident-scene-objects").evaluate((el) => el.open), false);
  assert.equal(await panel.locator(".incident-scene-activity").evaluate((el) => el.open), false);
  for (const selector of [".incident-establishment details", ".incident-scene-coverage", ".incident-scene-alerts", ".incident-scene-recordings"]) {
    assert.equal(await panel.locator(selector).evaluate(el => el.open), false, "review details stay collapsed");
  }
  assert.equal(await panel.getByText("The retained evidence supports an episode of activity.", { exact: true }).isVisible(), false);
  assert.equal(await panel.locator(".incident-scene-object").first().isVisible(), false);
  assert.equal(await panel.locator(".incident-scene-activity li").first().isVisible(), false);
  await panel.getByText("Observed objects", { exact: true }).click();
  await panel.getByText("Activity", { exact: true }).click();
  assert.equal(await panel.locator(".incident-scene-object").first().isVisible(), true);
  assert.equal(await panel.locator(".incident-scene-activity li").first().isVisible(), true);
  assert.equal(await panel.locator(".inspector-detection").count(), 2);
  assert.equal(await panel.getByText("Uncertain observation", { exact: true }).count(), 1);
  assert.equal(await panel.getByText("Possible car", { exact: true }).count(), 1);
  const continuitySighting = panel.locator(".inspector-detection").filter({ hasText: "May be the same person as another sighting" });
  assert.equal(await continuitySighting.count(), 1);
  assert.equal(await continuitySighting.getByText("Observed", { exact: true }).count(), 1, "identity continuity is independent of detection certainty");
  assert.match(await panel.locator(".incident-scene-continuity").textContent(), /does not establish a count of unique identities/);
  assert.match(await panel.textContent(), /missing video/);
  assert.doesNotMatch(await panel.locator(".incident-summary-objects").textContent(), /% cover|qualifying detection|outside incident zone|excluded/i);
  assert.match(await panel.locator(".incident-scene-activity li").first().textContent(), /person first observed/);
  assert.match(await panel.locator(".incident-scene-activity li").last().textContent(), /person last observed/);
  assert.doesNotMatch(await panel.locator(".incident-scene-activity").textContent(), /disappeared|vanished|left the scene/);
  await panel.getByText("Notification policy", { exact: true }).click();
  await panel.getByText("No observations met the notification criteria.", { exact: true }).waitFor();
  assert.equal(await panel.getByText("outside incident zone", { exact: true }).isVisible(), false);
  await panel.getByText("Policy decision details", { exact: true }).click();
  await panel.getByText("Technical policy reasons", { exact: true }).click();
  assert.equal(await panel.getByText("outside incident zone", { exact: true }).isVisible(), true);
  assert.equal(await panel.locator(".inspector-detection").count(), 2, "notification decisions never filter the roster");
  await panel.getByText("Policy decision details", { exact: true }).click();
  async function verifyScenePlayback(owner, playerSelector) {
    await owner.getByRole("button", { name: "Play whole incident", exact: true }).click();
    const player = page.locator(playerSelector);
    await player.waitFor();
    assert.equal(await player.getAttribute("data-camera-id"), "gate");
    assert.equal(Number(await player.getAttribute("data-start-epoch")), Date.parse(first.created_at) / 1000);
    assert.equal(Number(await player.getAttribute("data-end-epoch")) - Date.parse(first.created_at) / 1000, 10, "first camera gets only its ten-second episode");
    const video = page.locator(`${playerSelector} video`).first();
    await video.waitFor();
    await video.dispatchEvent("ended");
    await page.waitForFunction((selector) => document.querySelector(selector)?.getAttribute("data-camera-id") === "driveway", playerSelector);
    assert.equal(Number(await player.getAttribute("data-end-epoch")) - Number(await player.getAttribute("data-start-epoch")), 12, "second camera gets its own bounded recording");
    assert.equal(await owner.locator(".inspector-detection").count(), 2);
    await owner.getByRole("button", { name: "Stop scene playback", exact: true }).click();
    await player.waitFor({ state: "detached" });
    await owner.getByText("Camera recordings", { exact: true }).click();
    await owner.getByRole("button", { name: "Play driveway episode", exact: true }).click();
    await page.waitForFunction((selector) => document.querySelector(selector)?.getAttribute("data-camera-id") === "driveway", playerSelector);
    await page.locator(`${playerSelector} video`).first().dispatchEvent("error");
    assert.equal(await owner.locator(".inspector-detection").count(), 2, "missing footage never removes observations");
    await owner.getByRole("button", { name: "Stop scene playback", exact: true }).click();
    assert.equal(clipRequests.length, 0, "incident playback does not build a clip");
  }
  await verifyScenePlayback(panel, ".incident-detail-player .incident-recording-player");
  await panel.getByRole("button", { name: /^person Observed/ }).click();
  const observationButtons = panel.locator(".incident-observation-list button");
  await observationButtons.first().click();
  await panel.locator('.incident-observation-image img').waitFor();
  assert.match(await panel.locator('.incident-observation-image img').getAttribute("src"), /\/incidents\/observations\/obs-1\/snapshot$/);
  assert.equal(await panel.locator(".incident-observation-box").evaluate((el) => el.style.left), "10%");
  assert.match(await panel.locator("figcaption").textContent(), /1280 × 800.*live substream.*Analyzed frame/);
  assert.equal(await panel.locator(".incident-review-image").evaluate((el) => el.open), false);
  await panel.getByText("Additional review image", { exact: true }).click();
  assert.equal(await panel.locator(".incident-review-image img").isVisible(), true);
  assert.equal(await panel.locator(".incident-review-image .incident-observation-box").count(), 0, "additional review frames never inherit detection geometry");
  assert.match(await panel.locator(".incident-review-image").textContent(), /2560 × 1600.*not the analyzed frame/);
  assert.equal(await panel.locator(".inspector-detection").count(), 2, "changing source frame must retain whole-scene inventory");
  await observationButtons.nth(1).click();
  await panel.getByText(/Supporting image unavailable/).waitFor();
  assert.equal(await panel.locator(".incident-observation-image img").count(), 0, "never substitute unrelated cover imagery");
  assert.match(await panel.getByRole("link", { name: /^Open recording at/ }).getAttribute("href"), /camera=gate/);
  await panel.getByText("Correct this incident", { exact: true }).click();
  const form = panel.locator("form");
  await form.getByRole("textbox").fill("cyclist");
  await form.getByRole("button", { name: "Save correction" }).click();
  await panel.getByRole("button", { name: /^cyclist Observed/ }).waitFor();
  assert.deepEqual(corrections[0], { expected_revision: 4, operation: "label", object_id: "object-1", label: "cyclist" });
  await form.getByRole("combobox").selectOption("associate");
  for (const checkbox of await form.getByRole("checkbox").all()) await checkbox.check();
  const associated = page.waitForResponse((response) => response.url().endsWith("/corrections") && response.request().postDataJSON().operation === "associate");
  await form.getByRole("button", { name: "Save correction" }).click();
  await associated;
  assert.equal(corrections.at(-1).operation, "associate");
  assert.deepEqual(corrections.at(-1).object_ids, ["object-1", "object-2"]);
  await form.getByRole("combobox").selectOption("merge");
  await form.getByRole("textbox").fill("scene-2");
  const merged = page.waitForResponse((response) => response.url().endsWith("/corrections"));
  await form.getByRole("button", { name: "Save correction" }).click(); await merged;
  assert.deepEqual(corrections.at(-1).expected_revisions, { "scene-1": 6, "scene-2": 7 });
  assert.deepEqual(corrections.at(-1).incident_ids, ["scene-1", "scene-2"]);
  await form.getByRole("combobox").selectOption("separate");
  await form.getByRole("checkbox").first().check();
  const separated = page.waitForResponse((response) => response.url().endsWith("/corrections"));
  await form.getByRole("button", { name: "Save correction" }).click(); await separated;
  assert.deepEqual(corrections.at(-1).observation_ids, ["obs-1"]);
  await form.getByRole("combobox").selectOption("split");
  await form.getByRole("checkbox").first().check();
  conflict = true;
  await form.getByRole("button", { name: "Save correction" }).click();
  await form.getByRole("alert").waitFor();
  assert.match(await form.getByRole("alert").textContent(), /changed/);
  assert.deepEqual(corrections.at(-1).episode_ids, ["episode-1"]);
  assert.equal(await panel.locator(".inspector-detection").count(), 2, "conflict must preserve visible evidence");
  assert.match(page.url(), /incidents\/incident-scene-1/);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`http://127.0.0.1:${server.address().port}/survng/incidents?incident_id=scene-1`);
  const mobilePanel = page.locator(".event-overlay .incident-scene-panel");
  await mobilePanel.getByText("Observed objects", { exact: true }).waitFor();
  for (const selector of [".incident-scene-objects", ".incident-scene-activity", ".incident-scene-recordings", ".incident-scene-coverage", ".incident-scene-alerts"]) {
    assert.equal(await mobilePanel.locator(selector).evaluate(el => el.open), false, "mobile starts with a concise review");
  }
  assert.equal(await mobilePanel.getByText("Some sightings may show the same subject. The list does not establish a count of unique identities.", { exact: true }).isVisible(), false);
  assert.equal(await mobilePanel.locator(".inspector-detection").count(), 2, "mobile deep links retain the same whole-scene inventory");
  await verifyScenePlayback(mobilePanel, ".event-overlay .incident-recording-player");
  await page.goto(`http://127.0.0.1:${server.address().port}/survng/incidents/incident-scene-1`);
  const detailPanel = page.locator(".incident-detail-page .incident-scene-panel");
  await detailPanel.getByText("Observed objects", { exact: true }).waitFor();
  await verifyScenePlayback(detailPanel, ".incident-detail-page .incident-recording-player");
  const longStart = Date.parse(first.created_at) / 1000;
  second.created_at = new Date((longStart + 7260) * 1000).toISOString();
  incident = { ...incident, revision: incident.revision + 1, episodes: [
    { ...incident.episodes[0], end_at: new Date((longStart + 7200) * 1000).toISOString(), event_ids: [80913] },
    { ...incident.episodes[1], start_at: second.created_at, end_at: new Date((longStart + 7272) * 1000).toISOString() },
  ] };
  await page.reload();
  await detailPanel.getByRole("button", { name: "Play whole incident", exact: true }).click();
  for (let part = 0; part < 8; part += 1) {
    const expected = longStart + part * 900;
    await page.waitForFunction((start) => Number(document.querySelector(".incident-detail-page .incident-recording-player")?.getAttribute("data-start-epoch")) === start, expected);
    const player = page.locator(".incident-detail-page .incident-recording-player");
    assert.equal(await player.getAttribute("data-camera-id"), "gate");
    assert.equal(Number(await player.getAttribute("data-end-epoch")) - expected, 900);
    await page.locator(".incident-detail-page .incident-recording-player video").first().dispatchEvent("ended");
  }
  await page.waitForFunction(() => document.querySelector(".incident-detail-page .incident-recording-player")?.getAttribute("data-camera-id") === "driveway");
  assert.equal(await detailPanel.locator(".inspector-detection").count(), 2);
  await detailPanel.getByRole("button", { name: "Stop scene playback", exact: true }).click();
  incident = { ...incident, establishment: { status: "not_established", reason: "candidate_churn_only", supporting_observation_ids: [], policy_version: 1 } };
  await page.reload();
  await detailPanel.getByText("Activity was not established", { exact: true }).waitFor();
  assert.equal(await detailPanel.locator(".incident-scene-summary").count(), 0, "unsupported historical record must not repeat its old confident scene summary");
  assert.equal(await detailPanel.locator(".inspector-detection").count(), 2, "legacy review retains all evidence");
  assert.equal(await detailPanel.locator(".incident-scene-objects").evaluate((el) => el.open), false);
  assert.match(await detailPanel.getByRole("link", { name: "Review observations" }).getAttribute("href"), /\/observations$/);
  incident = { ...incident, evidence_images: [1, 2].map((i) => ({
    id: `gallery-${i}`, event_id: first.id, camera_id: first.camera_id,
    captured_epoch: Date.parse(first.created_at) / 1000 + i,
    captured_at: new Date(Date.parse(first.created_at) + i * 1000).toISOString(),
    snapshot_url: `/api/incidents/evidence/gallery-${i}/snapshot`,
    objects: [{ label: "cat", confidence: .52, confidence_eligible: false,
      box: { x1: 10, y1: 20, x2: 40, y2: 60 }, detection_frame_width: 1280, detection_frame_height: 800 }],
  })) };
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto(`http://127.0.0.1:${server.address().port}/survng/incidents?incident_id=scene-1`);
  await page.locator(".incident-desktop-focus .incident-workspace-view-toggle").getByRole("button", { name: "Evidence", exact: true }).click();
  const strip = page.locator(".incident-evidence-strip");
  assert.equal(await strip.locator("button").count(), 4, "two event covers plus two additional images");
  for (const i of [1, 2]) {
    const button = strip.locator("button").filter({ has: page.locator(`img[src*="/evidence/gallery-${i}/snapshot"]`) });
    await button.click();
    await page.locator(`.incident-evidence-hero img[src*="/evidence/gallery-${i}/snapshot"]`).waitFor();
    assert.equal(await button.getAttribute("aria-pressed"), "true", "different images of the same event remain selectable");
  }
  await page.goto(`http://127.0.0.1:${server.address().port}/survng/incidents/incident-scene-1`);
  for (const i of [1, 2]) {
    const image = page.locator(`.incident-detail-frames img[src*="/evidence/gallery-${i}/snapshot"]`);
    await image.waitFor();
    await image.evaluate((img) => img.decode());
  }
  assert.equal(await page.locator('.incident-detail-frames img[src*="/evidence/gallery-"]').count(), 2);
  assert.deepEqual(errors, []);
  console.log("canonical scene, corrections, source evidence, and multi-camera/long-episode playback browser tests passed");
} finally {
  await browser.close(); await new Promise((resolve) => server.close(resolve));
}
