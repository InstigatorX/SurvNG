import assert from "node:assert/strict";
import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 800, height: 900 } });
  await context.addInitScript(() => localStorage.clear());
  const page = await context.newPage();
  await page.route("**/api/assistant/status", (route) => route.fulfill({ contentType: "application/json", body: JSON.stringify({ configured: true, fast_model: "test", reasoning_model: "test" }) }));
  await page.route("**/api/assistant/chat", (route) => route.fulfill({ contentType: "application/json", body: JSON.stringify({
    message: "I found one bounded change.",
    suggestions: [],
    evidence: [{
      id: "visual-review-42",
      title: "Front Door review",
      summary: "Motion sensitivity can be adjusted.",
      details: {
        event_id: 42,
        camera_id: "front-door",
        can_apply: true,
        configuration_fingerprint: "fingerprint",
        recommendation_proof: "proof",
        advice: { verdict: "detection_consistent", confidence: 0.9, summary: "Consistent", changes: [{ scope: "camera", setting: "sensitivity", value: 0.7 }] },
        proposals: [{ scope: "camera", setting: "sensitivity", current: 0.5, proposed: 0.7, reason: "Test" }],
      },
    }],
  }) }));
  await page.route("**/api/incidents/42/ai-apply", (route) => route.fulfill({ contentType: "application/json", body: JSON.stringify({ applied: [{ setting: "sensitivity" }] }) }));

  await page.goto("http://127.0.0.1:8088/survng/", { waitUntil: "domcontentloaded", timeout: 30_000 });
  await page.getByRole("button", { name: "Open SurvNG Assistant" }).click();
  await page.getByPlaceholder("Ask about SurvNG…").fill("Review this incident");
  await page.getByRole("button", { name: "Send" }).click();
  const apply = page.getByRole("button", { name: "Apply proposed changes" });
  await apply.waitFor({ state: "visible" });
  await apply.click();
  assert.equal(await page.getByRole("dialog").count(), 1);
  assert.equal(await page.getByRole("dialog").getAttribute("aria-labelledby"), "assistant-apply-title");
  await page.keyboard.press("Escape");
  assert.equal(await apply.evaluate((button) => document.activeElement === button), true);

  await apply.click();
  await page.getByRole("button", { name: "Confirm and apply" }).click();
  await page.getByText("Applied after confirmation").waitFor({ state: "visible" });
  assert.equal(await page.locator(".assistant-evidence-card").evaluate((card) => document.activeElement === card), true);
} finally {
  await browser.close();
}

console.log("assistant apply modal tests passed");
