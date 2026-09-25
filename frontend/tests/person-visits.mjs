import assert from "node:assert/strict";
import { createServer } from "vite";
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ root, configFile: false, server: { host: "127.0.0.1", port: 0 }, plugins: [{ name: "visit-fixture", configureServer(vite) {
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
  const a = { id: "track:1:1", event_id: 1, camera_id: "gate", first_seen: "2026-09-01T12:00:00Z", last_seen: "2026-09-01T12:00:02Z", revision: "a".repeat(64), identity_status: "unresolved" };
  const b = { ...a, id: "track:2:1", event_id: 2, camera_id: "foyer", revision: "b".repeat(64), identity_status: "confirmed" };
  await page.route("**/api/people/visits**", async (route) => {
    if (route.request().method() === "PUT") {
      mutations.push(route.request().postDataJSON());
      linked = mutations.at(-1).decision === "accept";
      await route.fulfill({ json: { updated: true } }); return;
    }
    if (pending) await new Promise((resolve) => setTimeout(resolve, 300));
    if (failure) { await route.fulfill({ status: 503, json: { detail: "Temporarily unavailable" } }); return; }
    const first = { id: a.id, first_seen: a.first_seen, last_seen: a.last_seen, sightings: [a], links: [] };
    const second = { ...first, id: b.id, person_id: 7, person_name: "Alex", sightings: [b] };
    const edge = { left: a.id, right: b.id, reason: "Compatible appearance and route", similarity: .92 };
    await route.fulfill({ json: { start: 10000, end: 11000, routes_configured: 1, auto_link_enabled: false, suggestions: linked || empty ? [] : [edge], visits: empty ? [] : linked ? [{ ...second, sightings: [{ ...a, identity_status: "linked" }, b], links: [edge] }] : [first, second] } });
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${server.resolvedUrls.local[0]}visits-fixture`);
  await page.getByText("Loading visits…", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Confirm link", exact: true }).waitFor();
  pending = false;
  await page.getByRole("button", { name: "Confirm link", exact: true }).click();
  await page.getByText("Linked to Alex's visit", { exact: true }).waitFor();
  assert.equal(mutations[0].left_revision, a.revision);
  await page.getByRole("button", { name: "Separate sightings", exact: true }).click();
  await page.getByRole("button", { name: "Confirm link", exact: true }).waitFor();
  assert.equal(mutations.at(-1).decision, "reject");
  failure = true;
  await page.getByRole("button", { name: "Refresh visits", exact: true }).click();
  await page.getByRole("alert").waitFor();
  failure = false; empty = true;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.getByText("No person sightings in this window.", { exact: true }).waitFor();
  assert.equal(await page.getByRole("alert").count(), 0);
  assert.deepEqual(errors, []);
  console.log("Visit review, retraction, loading, error, empty and reconnect checks passed");
} finally {
  await browser?.close();
  await server.close();
}
