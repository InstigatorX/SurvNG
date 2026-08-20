import assert from "node:assert/strict";
import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const cameras = [
    { id: "gate", name: "Gate", recording: true, sub_recording: true },
    { id: "foyer", name: "Foyer", recording: true, sub_recording: false },
  ];
  let exportsList = [{
    id: "clip-1",
    kind: "recording",
    camera_id: "gate",
    source: "main",
    start_epoch: 1785790000,
    end_epoch: 1785790060,
    status: "completed",
    phase: "Ready",
    progress: 100,
    output_name: "gate-clip.mp4",
    size_bytes: 5_000_000,
    created_at: "2026-08-03T20:00:00+00:00",
    expires_at: "2026-08-04T20:00:00+00:00",
    protected: false,
    label: "",
    origin: "manual",
    download_url: "/survng/api/exports/clip-1/download",
    media_url: "/survng/api/exports/clip-1/media",
    options: { height: 1080 },
  }, {
    id: "lapse-1",
    kind: "timelapse",
    camera_id: "foyer",
    source: "live",
    start_epoch: 1785780000,
    end_epoch: 1785783600,
    status: "running",
    phase: "Encoding timelapse",
    progress: 42,
    output_name: "",
    size_bytes: 0,
    created_at: "2026-08-03T19:00:00+00:00",
    expires_at: "",
    protected: false,
    label: "",
    origin: "assistant",
    download_url: "",
    media_url: "",
    options: { height: 720 },
  }];

  await page.route("**/api/assistant/status", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ enabled: false, configured: false }),
  }));
  await page.route("**/api/cameras", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify(cameras),
  }));
  await page.route("**/api/exports?*", (route) => {
    const url = new URL(route.request().url());
    let filtered = [...exportsList];
    const kind = url.searchParams.get("kind");
    const status = url.searchParams.get("status");
    const protectedOnly = url.searchParams.get("protected");
    if (kind) filtered = filtered.filter((item) => item.kind === kind);
    if (status === "active") filtered = filtered.filter((item) => ["queued", "running", "cancelling"].includes(item.status));
    else if (status) filtered = filtered.filter((item) => item.status === status);
    if (protectedOnly === "true") filtered = filtered.filter((item) => item.protected);
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ exports: filtered, total: filtered.length, offset: 0, limit: 200 }),
    });
  });
  await page.route("**/api/exports/summary", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({
      total: exportsList.length,
      completed: exportsList.filter((item) => item.status === "completed").length,
      active: exportsList.filter((item) => ["queued", "running", "cancelling"].includes(item.status)).length,
      protected: exportsList.filter((item) => item.protected).length,
      bytes: exportsList.reduce((sum, item) => sum + item.size_bytes, 0),
      protected_bytes: exportsList.filter((item) => item.protected).reduce((sum, item) => sum + item.size_bytes, 0),
      retention_hours: 24,
      max_storage_bytes: 20 * 1024 * 1024 * 1024,
    }),
  }));
  await page.route("**/api/exports/clip-1/metadata", async (route) => {
    const request = JSON.parse(route.request().postData() || "{}");
    exportsList = exportsList.map((item) => item.id === "clip-1" ? { ...item, label: request.label } : item);
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(exportsList.find((item) => item.id === "clip-1")),
    });
  });
  await page.route("**/api/exports/batch", async (route) => {
    const request = JSON.parse(route.request().postData() || "{}");
    exportsList = exportsList.map((item) => request.ids.includes(item.id)
      ? { ...item, protected: request.action === "protect" ? true : request.action === "unprotect" ? false : item.protected }
      : item);
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        action: request.action,
        results: exportsList.filter((item) => request.ids.includes(item.id)),
        errors: [],
      }),
    });
  });
  await page.route("**/api/exports/clip-1/protection", async (route) => {
    const request = JSON.parse(route.request().postData() || "{}");
    exportsList = exportsList.map((item) => item.id === "clip-1"
      ? { ...item, protected: request.protected, expires_at: request.protected ? "" : item.expires_at }
      : item);
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(exportsList.find((item) => item.id === "clip-1")),
    });
  });
  await page.route("**/api/exports/clip-1?force=true", (route) => {
    assert.equal(route.request().method(), "DELETE");
    exportsList = exportsList.filter((item) => item.id !== "clip-1");
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ id: "clip-1", deleted: true }),
    });
  });
  await page.route("**/api/exports/clip-1/media", (route) => route.fulfill({
    status: 204,
    contentType: "video/mp4",
    body: "",
  }));

  await page.goto("http://127.0.0.1:8088/survng/timeline/exports", {
    waitUntil: "domcontentloaded",
    timeout: 30_000,
  });
  await page.getByText("Loading workspace…").waitFor({ state: "hidden", timeout: 30_000 });
  await page.locator(".export-center-page").waitFor({ state: "visible" });
  await page.getByText("gate-clip.mp4", { exact: true }).waitFor({ state: "visible" });
  assert.equal(
    await page.getByRole("link", { name: "Download" }).getAttribute("href"),
    "/survng/api/exports/clip-1/download",
  );

  const name = page.getByPlaceholder("Add a useful name");
  await name.fill("Gate delivery");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await page.getByText("Gate delivery", { exact: true }).first().waitFor({ state: "visible" });

  await page.getByRole("button", { name: "Protect", exact: true }).click();
  await page.getByRole("button", { name: "Unprotect" }).waitFor({ state: "visible" });
  await page.getByText("Protected", { exact: true }).last().waitFor({ state: "visible" });

  await page.getByRole("button", { name: "Select", exact: true }).click();
  const cards = page.locator(".export-center-library > button:not(.export-center-load-more)");
  await cards.nth(0).click();
  await cards.nth(1).click();
  await page.locator(".export-center-batch").getByRole("button", { name: "Protect", exact: true }).click();
  await page.locator(".export-center-batch").waitFor({ state: "detached" });

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Delete" }).click();
  await page.getByText("gate-clip.mp4", { exact: true }).waitFor({ state: "detached" });

  await page.getByLabel("Type").selectOption("timelapse");
  await page.getByText("Encoding timelapse", { exact: true }).waitFor({ state: "visible" });
} finally {
  await browser.close();
}

console.log("export center UI tests passed");
