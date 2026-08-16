import assert from "node:assert/strict";
import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript(() => localStorage.clear());
  const page = await context.newPage();
  await page.goto("http://127.0.0.1:8088/survng/", { waitUntil: "domcontentloaded", timeout: 30_000 });
  await page.locator(".camera-tile").first().waitFor({ state: "visible" });
  assert.equal(await page.title(), "SurvNG");
  assert.match(await page.locator(".workspace-content > h1").textContent(), /SurvNG — Live/);
  assert.equal(await page.locator("h1").count(), 1);

  await page.evaluate(() => {
    window.__survngTileNodes = [...document.querySelectorAll(".camera-tile")];
    window.__survngPosterNodes = [...document.querySelectorAll(".camera-tile-poster")];
  });
  await page.getByRole("button", { name: "Motion only" }).click();
  await page.waitForTimeout(250);
  assert.equal(await page.evaluate(() => window.__survngTileNodes.every((node) => node.isConnected && [...document.querySelectorAll(".camera-tile")].includes(node))), true);
  assert.equal(await page.evaluate(() => window.__survngPosterNodes.every((node) => node.isConnected && [...document.querySelectorAll(".camera-tile-poster")].includes(node))), true);

  await page.getByRole("button", { name: "Custom" }).click();
  const move = page.getByRole("button", { name: /^Move / }).first();
  await move.focus();
  await page.keyboard.press("Enter");
  assert.equal(await move.getAttribute("aria-pressed"), "true");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Enter");
  assert.equal(await move.getAttribute("aria-pressed"), "false");
  const saved = JSON.parse(await page.evaluate(() => localStorage.getItem("survng.liveCustomLayout.v1")));
  assert.equal(saved.version, 1);
  assert.ok(Array.isArray(saved.order) && saved.order.length > 1);
  const savedBeforeResize = await page.evaluate(() => localStorage.getItem("survng.liveCustomLayout.v1"));
  const resize = page.getByRole("button", { name: /^Resize / }).first();
  await resize.focus();
  await page.keyboard.press("Enter");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("s");
  assert.equal(await resize.getAttribute("aria-pressed"), "true");
  await page.keyboard.press("Escape");
  assert.equal(await resize.getAttribute("aria-pressed"), "false");
  assert.equal(await page.evaluate(() => localStorage.getItem("survng.liveCustomLayout.v1")), savedBeforeResize);

  await page.goto("http://127.0.0.1:8088/survng/timeline", { waitUntil: "domcontentloaded", timeout: 30_000 });
  assert.equal(await page.title(), "SurvNG · Timeline");
  assert.match(await page.locator(".workspace-content > h1").textContent(), /SurvNG — Timeline/);
  assert.equal(await page.locator("h1").count(), 1);

  for (const [path, title] of [
    ["incidents", "Incidents"],
    ["search", "Search"],
    ["people", "People"],
    ["admin", "Admin"],
  ]) {
    await page.goto(`http://127.0.0.1:8088/survng/${path}`, { waitUntil: "domcontentloaded", timeout: 30_000 });
    assert.match(await page.locator(".workspace-content > h1").textContent(), new RegExp(`SurvNG — ${title}`));
    assert.equal(await page.locator("h1").count(), 1);
  }

  await page.setViewportSize({ width: 740, height: 844 });
  await page.goto("http://127.0.0.1:8088/survng/", { waitUntil: "domcontentloaded", timeout: 30_000 });
  await page.locator(".camera-tile").first().waitFor({ state: "visible" });
  assert.equal(await page.locator(".workspace-sidebar").isVisible(), false);
  assert.equal(await page.locator(".mobile-workspace-nav").isVisible(), true);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth), true);
} finally {
  await browser.close();
}

console.log("release interaction tests passed");
