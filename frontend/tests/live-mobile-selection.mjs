// Run against PREVIEW_CAMERA_COUNT=20 node tools/camera-workspace-preview.mjs.
import assert from "node:assert/strict";
import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
  });
  const page = await context.newPage();
  await page.goto(process.env.LIVE_PREVIEW_URL || "http://127.0.0.1:5182/");
  const primary = page.locator(".camera-tile.mobile-primary");
  await primary.waitFor({ state: "visible" });
  let originalId = await primary.getAttribute("data-camera-id");
  assert.equal(await page.locator(".camera-tile-quick-actions, .camera-tile-menu, .camera-tile-control-menu").count(), 0);
  assert.equal(await page.locator(".live-grid>.events-zone").count(), 0);

  // Use enough synthetic cameras to overflow the child area.
  const grid = page.locator(".live-camera-grid");
  const primaryBefore = await primary.boundingBox();
  const lastChild = page.locator(".camera-tile:not(.mobile-primary)").last();
  const lastBefore = await lastChild.boundingBox();
  const scrollTop = await grid.evaluate((node) => {
    node.scrollTop = node.scrollHeight;
    return node.scrollTop;
  });
  assert.ok(scrollTop > 0, "Run preview with PREVIEW_CAMERA_COUNT=20 to test scrolling");
  const primaryAfter = await primary.boundingBox();
  assert.ok(Math.abs(primaryAfter.y - primaryBefore.y) <= 7, "Primary stays pinned while children scroll");
  assert.ok((await lastChild.boundingBox()).y < lastBefore.y, "Children move independently");
  const nav = await page.locator(".mobile-workspace-nav").boundingBox();
  const gridBounds = await grid.boundingBox();
  assert.ok(gridBounds.y + gridBounds.height <= nav.y, "Camera list clears bottom navigation");
  assert.equal(await page.evaluate(() => document.documentElement.scrollHeight > window.innerHeight), false);
  const lastId = await lastChild.getAttribute("data-camera-id");
  await lastChild.locator(".camera-open-target").tap();
  await page.waitForFunction((id) => document.querySelector(".camera-tile.mobile-primary")?.dataset.cameraId === id, lastId);
  assert.equal(await page.locator(".live-overlay").count(), 0);
  assert.ok(Math.abs((await primary.boundingBox()).y - primaryBefore.y) <= 7, "Promoted camera stays at the top after scrolling");
  originalId = lastId;
  await grid.evaluate((node) => { node.scrollTop = 0; });

  const childId = await page.locator(".camera-tile:not(.mobile-primary)").first().getAttribute("data-camera-id");
  const child = page.locator(`.camera-tile[data-camera-id="${childId}"]`);
  const target = child.locator(".camera-open-target");
  assert.match(await target.getAttribute("aria-label"), /^Make .* the primary camera/);
  await target.tap();
  await page.waitForFunction((id) => document.querySelector(".camera-tile.mobile-primary")?.dataset.cameraId === id, childId);
  assert.equal(await page.locator(".live-overlay").count(), 0);
  assert.equal(await page.locator(`.camera-tile[data-camera-id="${originalId}"].mobile-primary`).count(), 0);
  assert.match(await target.getAttribute("aria-label"), /^Open .* live view/);

  await target.tap();
  await page.locator(".live-overlay").waitFor({ state: "visible" });
  await page.getByRole("button", { name: "Close live view", exact: true }).last().tap();
  await page.locator(".live-overlay").waitFor({ state: "detached" });

  // A completed hold must neither promote a child nor leave a sticky overlay.
  const secondaryTarget = page.locator(`.camera-tile[data-camera-id="${originalId}"] .camera-open-target`);
  await secondaryTarget.dispatchEvent("pointerdown", { pointerType: "touch", pointerId: 9, button: 0, clientX: 20, clientY: 20 });
  await page.locator(".live-overlay").waitFor({ state: "visible" });
  await secondaryTarget.dispatchEvent("pointerup", { pointerType: "touch", pointerId: 9 });
  await secondaryTarget.dispatchEvent("click");
  await page.locator(".live-overlay").waitFor({ state: "detached" });
  assert.equal(await primary.getAttribute("data-camera-id"), childId);

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.locator(".camera-tile-menu").first().waitFor({ state: "visible" });
  await page.locator(".live-grid>.events-zone").waitFor({ state: "visible" });
  assert.ok(await page.locator(".camera-tile-quick-actions").count() > 0);
  await secondaryTarget.click();
  await page.locator(".live-overlay").waitFor({ state: "visible" });
  console.log("live mobile selection browser tests passed");
} finally {
  await browser.close();
}
