import assert from "node:assert/strict";
import { createServer } from "vite";
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ root, configFile: false, server: { host: "127.0.0.1", port: 0 }, plugins: [{
  name: "budget-telemetry-test",
  configureServer(server) {
    server.middlewares.use("/test", (_req, res) => {
      res.setHeader("Content-Type", "text/html");
      res.end('<div id="root"></div><script type="module" src="/test-entry.jsx"></script>');
    });
  },
  resolveId(id) { if (id === "/test-entry.jsx") return id; },
  load(id) { if (id === "/test-entry.jsx") return `
    import React from 'react';
    import {createRoot} from 'react-dom/client';
    import {RuntimeStatus} from '/src/admin/ConfigPage.jsx';
    const root = createRoot(document.getElementById('root'));
    window.showStatus = (status) => root.render(<RuntimeStatus status={status}/>);
    window.showStatus({});
  `; },
}] });
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/test`);
  await page.getByText("Waiting for native pipeline status…", { exact: true }).waitFor();
  const status = { connected: true, detection_enabled: true, native_activity: { health: "healthy" } };
  const show = async (budget, overrides = {}) => {
    await page.evaluate((value) => window.showStatus(value), { ...status, live_pipeline: { native_budget: budget }, ...overrides });
  };
  await show(null);
  await page.getByText("Adaptive inference: waiting for telemetry…", { exact: true }).waitFor();
  await show({ mode: "disabled" });
  await page.getByText(/fixed-rate checking; motion wake-up inactive/).waitFor();
  const budget = { mode: "active", motion_enabled: true, target_fps: 5, idle_fps: 1, active_fps: 5,
    cooldown_remaining_seconds: 3.2, sampled_frames: 400, admitted_frames: 200, skipped_frames: 200,
    motion_wakes: 12, object_wakes: 4, excluded_motion_regions: 66, active_transitions: 5, idle_transitions: 6 };
  await show(budget);
  await page.getByRole("heading", { name: "Adaptive inference · Active" }).waitFor();
  await page.getByText("200 admitted · 200 skipped (50%) · 400 sampled", { exact: true }).waitFor();
  await page.getByText("Enabled · 12 requests", { exact: true }).waitFor();
  await page.getByText("66 rectangles suppressed", { exact: true }).waitFor();
  await page.getByText("5 to active · 6 to idle", { exact: true }).waitFor();
  await show({ ...budget, mode: "idle", target_fps: 1, motion_enabled: false, cooldown_remaining_seconds: 0 });
  await page.getByRole("heading", { name: "Adaptive inference · Idle" }).waitFor();
  await page.getByText("Disabled · 12 requests", { exact: true }).waitFor();
  await page.getByText("1/sec target · 1 idle / 5 active", { exact: true }).waitFor();
  // Rebuild resets counts; sparse counters start at zero after the first sample.
  await show({ mode: "active", motion_enabled: true, sampled_frames: 1, admitted_frames: 1 });
  await page.getByText("1 admitted · 0 skipped (0%) · 1 sampled", { exact: true }).waitFor();
  await show({ mode: "active", sampled_frames: 0 });
  await page.getByText("— admitted · — skipped (—) · 0 sampled", { exact: true }).waitFor();
  await page.getByText("Setting unavailable · — requests", { exact: true }).waitFor();
  await show(budget, { connected: false });
  await page.getByText("Adaptive inference: unavailable · camera disconnected", { exact: true }).waitFor();
  assert.equal(await page.locator(".native-budget-telemetry").count(), 0);
  await show(budget);
  await page.getByRole("heading", { name: "Adaptive inference · Active" }).waitFor();
  await show(budget, { detection_enabled: false });
  await page.getByText("Adaptive inference: inactive · AI detection disabled", { exact: true }).waitFor();
  assert.deepEqual(errors, []);
  console.log("Native budget telemetry browser checks passed");
} finally {
  await browser?.close();
  await server.close();
}
