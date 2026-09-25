import assert from "node:assert/strict";
import { createServer } from "vite";
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";
import { realpathSync } from "node:fs";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ root, configFile: false, server: { host: "127.0.0.1", port: 0, fs: { allow: [root, realpathSync(`${root}/node_modules`)] } }, plugins: [{ name: "visit-fixture", configureServer(vite) {
  vite.middlewares.use((req, res, next) => {
    if (req.url === "/visits-fixture") { res.setHeader("Content-Type", "text/html"); res.end('<div id="root"></div><script type="module" src="/tests/fixtures/person-visits.jsx"></script>'); }
    else next();
  });
} }] });
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  let failure = false, pending = true, mutations = [], linked = false, empty = false;
  let noSuggestions = false, blocked = false, mutationFailure = false;
  const a = { id: "track:1:1", event_id: 1, face_id: 1, camera_id: "gate", first_seen: "2026-09-01T12:00:00Z", last_seen: "2026-09-01T12:00:02Z", revision: "a".repeat(64), identity_status: "unresolved" };
  const b = { ...a, id: "track:2:1", event_id: 2, face_id: 2, camera_id: "foyer", revision: "b".repeat(64), identity_status: "confirmed" };
  await page.route("**/api/faces/observations/*/crop.jpg", (route) => route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="gray"/></svg>' }));
  await page.route("**/api/people/visits**", async (route) => {
    if (route.request().method() === "PUT") {
      if (mutationFailure) { await route.fulfill({ status: 409, json: { detail: "Sightings changed; refresh and retry" } }); return; }
      mutations.push(route.request().postDataJSON());
      linked = mutations.at(-1).decision === "accept";
      await route.fulfill({ json: { updated: true } }); return;
    }
    if (pending) await new Promise((resolve) => setTimeout(resolve, 300));
    if (failure) { await route.fulfill({ status: 503, json: { detail: "Temporarily unavailable" } }); return; }
    const first = { id: a.id, first_seen: a.first_seen, last_seen: a.last_seen, sightings: [a], links: [] };
    const second = { ...first, id: b.id, person_id: 7, person_name: "Alex", sightings: [b] };
    const edge = { left: a.id, right: b.id, reason: blocked ? "New evidence conflicts with this previously confirmed link" : "Compatible appearance and route", similarity: .92, blocked };
    await route.fulfill({ json: { start: 10000, end: 11000, routes_configured: 1, auto_link_enabled: false, suggestions: linked || empty || noSuggestions ? [] : [edge], visits: empty ? [] : linked ? [{ ...second, sightings: [{ ...a, identity_status: "linked" }, b], links: [edge] }] : [first, second] } });
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${server.resolvedUrls.local[0]}visits-fixture`);
  await page.getByText("Loading visits…", { exact: true }).waitFor();
  const suggestions = page.getByRole("region", { name: "Suggested visit links", exact: true });
  const confirm = suggestions.getByRole("button", { name: "Same person and visit", exact: true });
  await confirm.waitFor();
  pending = false;
  const panel = page.locator(".faces-review-panel");
  assert.equal(await panel.evaluate((element) => element.scrollHeight > element.clientHeight), true);
  const panelBox = await panel.boundingBox();
  await page.mouse.move(panelBox.x + panelBox.width - 20, panelBox.y + 50);
  await page.mouse.wheel(0, 600);
  await page.waitForFunction(() => document.querySelector(".faces-review-panel").scrollTop > 0, null, { timeout: 3000 });
  await panel.evaluate((element) => element.scrollTop = 0);
  assert.equal(await page.getByRole("checkbox").count(), 0);
  assert.equal(await suggestions.locator(".visit-sightings li").count(), 2);
  assert.equal(await suggestions.getByRole("link", { name: "Review face", exact: true }).first().getAttribute("href"), "/people?face=1");
  assert.ok((await suggestions.boundingBox()).y < (await page.getByRole("heading", { name: "Visit history", exact: true }).boundingBox()).y);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await page.mouse.move(350, 400);
  await page.mouse.wheel(0, 600);
  await page.waitForFunction(() => window.scrollY > 0, null, { timeout: 3000 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.getByLabel("Person filter").selectOption("7");
  await confirm.waitFor();
  assert.equal(await page.locator(".person-visit").count(), 1);
  assert.equal(await suggestions.locator(".visit-sightings li").count(), 2);
  await page.getByLabel("Person filter").selectOption("8");
  await page.getByText("No suggested links to review for Sam.", { exact: false }).waitFor();
  assert.equal(await confirm.count(), 0);
  await page.getByLabel("Person filter").selectOption("7");
  mutationFailure = true;
  await confirm.click();
  await page.getByRole("alert").filter({ hasText: "Sightings changed" }).waitFor();
  assert.equal(await suggestions.locator(".visit-sightings li").count(), 2);
  mutationFailure = false;
  await confirm.click();
  await page.getByText("Linked to Alex's visit", { exact: true }).waitFor();
  assert.equal(mutations[0].left_revision, a.revision);
  await page.getByRole("button", { name: "Separate sightings", exact: true }).click();
  await confirm.waitFor();
  assert.equal(mutations.at(-1).decision, "reject");
  blocked = true;
  await page.getByRole("button", { name: "Refresh visits", exact: true }).click();
  await suggestions.getByText("New evidence conflicts with this previously confirmed link", { exact: true }).waitFor();
  assert.equal(await confirm.isDisabled(), true);
  blocked = false; noSuggestions = true;
  await page.getByRole("button", { name: "Refresh visits", exact: true }).click();
  await page.getByText("No suggested links to review for Alex.", { exact: false }).waitFor();
  assert.equal(await page.locator(".person-visit").count(), 1);
  await page.getByLabel("Person filter").selectOption("");
  await page.getByText("Manually link or separate sightings", { exact: true }).click();
  await page.getByRole("checkbox").first().check();
  await page.getByRole("checkbox").last().check();
  await page.getByText("2 of 2 sightings selected", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Reset decision", exact: true }).click();
  await page.getByText("0 of 2 sightings selected", { exact: true }).waitFor();
  assert.equal(mutations.at(-1).decision, "reset");
  await page.getByRole("checkbox").first().check();
  await page.getByLabel("Person filter").selectOption("7");
  await page.getByText("0 of 2 sightings selected", { exact: true }).waitFor();
  await page.getByLabel("Person filter").selectOption("");
  failure = true;
  await page.getByRole("button", { name: "Refresh visits", exact: true }).click();
  await page.getByRole("alert").waitFor();
  failure = false; empty = true;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.getByText("No person sightings in this window.", { exact: true }).waitFor();
  assert.equal(await page.getByRole("alert").count(), 0);
  assert.deepEqual(errors, []);
  console.log("Visit pair priority, person filtering, manual review, blocked links, mutation errors, retraction, loading, empty and reconnect checks passed");
} finally {
  await browser?.close();
  await server.close();
}
