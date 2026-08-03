import assert from "node:assert/strict";
import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext();
  await context.addInitScript(() => localStorage.clear());
  const page = await context.newPage();
  await page.route("**/api/assistant/status", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({
      enabled: true,
      configured: true,
      fast_model: "test-fast",
      reasoning_model: "test-deep",
      media_exports: true,
    }),
  }));
  await page.route("**/api/assistant/chat", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({
      message: "I started the requested timelapse. [E-export-job-ui]",
      citations: ["E-export-job-ui"],
      suggestions: [],
      reasoning_tier: "fast",
      model: "test-fast",
      evidence: [{
        id: "E-export-job-ui",
        kind: "media_export_job",
        title: "Gate timelapse",
        summary: "Queued a Gate timelapse.",
        href: "/recordings?camera=gate&at=1785672000&source=main",
        details: {
          media_export: {
            id: "job-ui",
            kind: "timelapse",
            status: "queued",
            phase: "Queued",
            progress: 0,
          },
        },
      }],
    }),
  }));
  await page.route("**/api/exports/job-ui", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({
      id: "job-ui",
      status: "completed",
      phase: "Ready",
      progress: 100,
      output_name: "gate-timelapse.mp4",
      size_bytes: 123456,
      download_url: "/survng/api/exports/job-ui/download",
    }),
  }));

  await page.goto("http://127.0.0.1:8088/survng/", {
    waitUntil: "domcontentloaded",
    timeout: 30_000,
  });
  await page.getByRole("button", { name: "Open SurvNG Assistant" }).click();
  const compose = page.getByPlaceholder("Ask about SurvNG…");
  await compose.fill("Create a Gate timelapse from 8 AM to 8 PM yesterday");
  await page.getByRole("button", { name: "Send" }).click();

  const download = page.getByRole("link", { name: "Download MP4" });
  await download.waitFor({ state: "visible", timeout: 10_000 });
  assert.equal(
    await download.getAttribute("href"),
    "/survng/api/exports/job-ui/download",
  );
  await page.getByText("Ready", { exact: true }).waitFor({ state: "visible" });
} finally {
  await browser.close();
}

console.log("assistant export UI tests passed");
