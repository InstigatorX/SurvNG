import assert from "node:assert/strict";
import { chromium } from "playwright";

const DESKTOP = { width: 1440, height: 900 };

async function openWorkspace(page, path, readySelector) {
  const url = path === "/"
    ? "http://127.0.0.1:8088/survng/"
    : `http://127.0.0.1:8088/survng/${path.replace(/^\//, "")}`;
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30_000 });
  await page.getByText("Loading workspace…").waitFor({ state: "hidden", timeout: 30_000 });
  await page.locator(readySelector).first().waitFor({ state: "visible", timeout: 30_000 });
}

const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ viewport: DESKTOP });
  await context.addInitScript(() => localStorage.clear());
  const page = await context.newPage();
  await openWorkspace(page, "/", ".camera-tile");
  assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
  assert.equal(await page.title(), "SurvNG");
  assert.match(await page.locator(".workspace-content > h1").textContent(), /SurvNG — Live/);
  assert.equal(await page.locator("h1").count(), 1);
  assert.equal(await page.locator(".live-command-scope").isVisible(), true);
  assert.ok(Number.parseFloat(await page.locator(".live-command-bar").evaluate((node) => getComputedStyle(node).borderRadius)) <= 4);
  const cameraSurface = page.locator(".camera-open-target").first();
  await cameraSurface.hover();
  assert.equal(await cameraSurface.evaluate((node) => getComputedStyle(node).backgroundColor), "rgba(0, 0, 0, 0)");

  await page.evaluate(() => {
    window.__survngTileNodes = [...document.querySelectorAll(".camera-tile")];
    window.__survngPosterNodes = [...document.querySelectorAll(".camera-tile-poster")];
  });
  await page.getByRole("button", { name: "Motion", exact: true }).click();
  await page.waitForTimeout(250);
  assert.equal(await page.evaluate(() => window.__survngTileNodes.every((node) => node.isConnected && [...document.querySelectorAll(".camera-tile")].includes(node))), true);
  assert.equal(await page.evaluate(() => window.__survngPosterNodes.every((node) => node.isConnected && [...document.querySelectorAll(".camera-tile-poster")].includes(node))), true);

  await page.getByRole("button", { name: "Custom" }).click();
  await page.getByRole("button", { name: /^Open .* controls$/ }).first().click();
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

  await openWorkspace(page, "timeline", ".recordings-v2-page");
  await page.getByRole("button", { name: "All events" }).waitFor({ state: "visible" });
  await page.locator(".recording-grid-camera").first().waitFor({ state: "visible" });
  assert.equal(await page.title(), "SurvNG · Timeline");
  assert.match(await page.locator(".workspace-content > h1").textContent(), /SurvNG — Timeline/);
  assert.equal(await page.locator("h1").count(), 1);
  assert.equal(await page.getByRole("button", { name: "All events" }).getAttribute("aria-pressed"), "true");
  const timelineEvidence = page.locator(".recordings-timeline-evidence button").first();
  if (await timelineEvidence.count()) {
    await timelineEvidence.click();
    const selectedCard = page.locator(".recordings-v2-selected-event");
    await selectedCard.waitFor({ state: "visible", timeout: 10_000 });
    const selectedAction = selectedCard.getByRole("link", { name: /View full incident/ });
    if (await selectedAction.isVisible()) {
      const cardBox = await selectedCard.boundingBox();
      const actionBox = await selectedAction.boundingBox();
      assert.ok(cardBox && actionBox && actionBox.y + actionBox.height <= cardBox.y + cardBox.height + 0.5);
    }
  }

  for (const [path, title] of [
    ["incidents", "Incidents"],
    ["search", "Search"],
    ["people", "People"],
    ["admin", "Admin"],
  ]) {
    await openWorkspace(page, path, ".workspace-content > h1");
    assert.match(await page.locator(".workspace-content > h1").textContent(), new RegExp(`SurvNG — ${title}`));
    assert.equal(await page.locator("h1").count(), 1);
  }

  await openWorkspace(page, "incidents", ".incident-preview-media-action");
  const incidentSurface = page.locator(".incident-preview-media-action").first();
  await incidentSurface.waitFor({ state: "visible" });
  await incidentSurface.hover();
  assert.equal(await incidentSurface.evaluate((node) => getComputedStyle(node).backgroundColor), "rgba(0, 0, 0, 0)");

  await page.setViewportSize({ width: 900, height: 844 });
  await openWorkspace(page, "timeline", ".recordings-v2-date");
  await page.getByRole("button", { name: "All events" }).waitFor({ state: "visible" });
  assert.ok(await page.locator(".workspace-sidebar").evaluate((node) => node.getBoundingClientRect().width <= 68));
  assert.ok(await page.locator(".recordings-v2-date").evaluate((node) => node.scrollWidth <= node.clientWidth));
  for (const selector of [
    ".recordings-v2-date button",
    ".recordings-v2-player-source button",
    ".recordings-v2-event-filter button",
    ".recordings-v2-export-toggle",
  ]) {
    assert.ok(await page.locator(selector).first().evaluate((node) => node.getBoundingClientRect().height >= 44));
  }
  assert.equal(await page.locator(".recordings-v2-player-source button[aria-pressed='true']").count(), 1);

  await page.setViewportSize({ width: 740, height: 844 });
  await openWorkspace(page, "/", ".camera-tile");
  assert.equal(await page.locator(".workspace-sidebar").isVisible(), false);
  assert.equal(await page.locator(".mobile-workspace-nav").isVisible(), true);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth), true);
  assert.ok(await page.locator(".mobile-workspace-nav a").first().evaluate((node) => node.getBoundingClientRect().height >= 44));

  await page.setViewportSize({ width: 390, height: 844 });
  await openWorkspace(page, "/", ".camera-tile");
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth), true);
  const livePhoneCamera = await page.locator(".live-grid > .camera-zone").boundingBox();
  const livePhoneActivity = await page.locator(".live-grid > .events-zone").boundingBox();
  assert.ok(livePhoneCamera && livePhoneActivity && livePhoneActivity.y >= livePhoneCamera.y + livePhoneCamera.height - 1);

  await openWorkspace(page, "incidents", ".incident-filter-selects");
  const incidentPhoneFilters = await page.locator(".incident-filter-selects").first().boundingBox();
  assert.ok(incidentPhoneFilters && incidentPhoneFilters.width <= 390);
  assert.ok(await page.locator(".incident-semantic-search button").last().evaluate((node) => node.getBoundingClientRect().height >= 44));

  await openWorkspace(page, "people", ".faces-page");
  assert.equal(await page.locator(".faces-page").evaluate((node) => getComputedStyle(node).gridTemplateColumns.split(" ").length), 1);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth), true);

  await openWorkspace(page, "admin", ".config-grid");
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth), true);
  assert.equal(await page.getByRole("button", { name: "Save changes" }).isDisabled(), true);
  const basePath = page.getByLabel("Web Base Path");
  const savedBasePath = await basePath.inputValue();
  await basePath.fill(`${savedBasePath}-unsaved`);
  assert.equal(await page.getByRole("button", { name: "Save changes" }).isEnabled(), true);
  await page.getByRole("button", { name: "Discard" }).click();
  assert.equal(await basePath.inputValue(), savedBasePath);
  await page.getByRole("button", { name: "Menu", exact: true }).click();
  await page.locator(".admin-navigation.open").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector(".admin-navigation")?.getBoundingClientRect().x >= -1);
  assert.ok((await page.locator(".admin-navigation").boundingBox())?.x >= 0);
  await page.getByRole("button", { name: "Detection", exact: true }).click();
  assert.match(page.url(), /subsection=detection/);
  assert.equal(await page.locator(".detection-subsection-tabs").evaluate((node) => getComputedStyle(node).gridTemplateColumns.split(" ").length), 2);

  await page.setViewportSize(DESKTOP);
  await openWorkspace(page, "timeline", ".recording-grid-camera");
  await page.waitForFunction(() => (document.querySelector(".recordings-v2-page")?.scrollHeight || 0) > 800);
  assert.ok(await page.locator(".recordings-v2-page").evaluate((node) => node.scrollHeight > 800));
  assert.ok(await page.locator(".recording-grid-camera").first().evaluate((node) => node.getBoundingClientRect().height >= 28));

  await page.setViewportSize({ width: 844, height: 390 });
  await openWorkspace(page, "/", ".camera-tile");
  const liveLandscapeCamera = await page.locator(".live-grid > .camera-zone").boundingBox();
  const liveLandscapeActivity = await page.locator(".live-grid > .events-zone").boundingBox();
  assert.ok(liveLandscapeCamera && liveLandscapeActivity && liveLandscapeActivity.y >= liveLandscapeCamera.y + liveLandscapeCamera.height - 1);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth), true);

  await openWorkspace(page, "timeline", ".recording-grid-camera");
  await page.waitForFunction(() => {
    const workspace = document.querySelector(".recordings-v2-workspace")?.getBoundingClientRect().height || 0;
    const camera = document.querySelector(".recording-grid-camera")?.getBoundingClientRect().height || 0;
    return workspace >= 360 && camera >= 44;
  });
  assert.ok(await page.locator(".recordings-v2-workspace").evaluate((node) => node.getBoundingClientRect().height >= 360));
  assert.ok(await page.locator(".recording-grid-camera").first().evaluate((node) => node.getBoundingClientRect().height >= 44));
} finally {
  await browser.close();
}

console.log("release interaction tests passed");
