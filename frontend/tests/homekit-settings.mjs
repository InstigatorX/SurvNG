import assert from "node:assert/strict";
import { chromium } from "playwright";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const directory = await mkdtemp(join(root, ".homekit-ui-test-"));
let server;
let browser;
try {
  await writeFile(join(directory, "index.html"), '<html><body><div id="root"></div><script type="module" src="./preview.jsx"></script></body></html>');
  await writeFile(join(directory, "preview.jsx"), `
    import React, { useState } from "react";
    import { createRoot } from "react-dom/client";
    import { HomeKitSettings } from "../src/admin/HomeKitSettings.jsx";
    import "../src/styles.css";
    import "../src/admin/admin.css";
    import "../src/admin/workspace.css";
    function Preview() {
      const [config, setConfig] = useState({ homekit: { enabled: true, bind: "192.168.1.2", node_path: "node" },
        cameras: location.search ? [] : [{ id: "porch", name: "Porch", homekit: { enabled: true, allow_software_encoding: true } }] });
      const updateConfig = (path, value) => setConfig(current => {
        const next = structuredClone(current); let target = next;
        for (const part of path.slice(0, -1)) target = target[part];
        target[path.at(-1)] = value; return next;
      });
      return <main className="config-form bento-card" style={{maxWidth: 1000, margin: "20px auto", padding: 24}}>
        <HomeKitSettings config={config} updateConfig={updateConfig} />
        <output data-testid="config" hidden>{JSON.stringify(config)}</output>
      </main>;
    }
    createRoot(document.getElementById("root")).render(<Preview />);
  `);
  server = await createServer({ root, configFile: false, plugins: [react()], server: { host: "127.0.0.1", port: 0 } });
  await server.listen();
  const port = server.httpServer.address().port;
  const url = `http://127.0.0.1:${port}/${directory.split("/").at(-1)}/index.html`;
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  let mode = "ready", resets = 0, statusRequests = 0;
  let initialStatus;
  const waitForInitial = new Promise(resolve => { initialStatus = resolve; });
  await page.route("**/api/homekit/status", async route => {
    statusRequests++;
    if (statusRequests === 1) await waitForInitial;
    await route.fulfill({ status: mode === "error" ? 503 : 200, contentType: "application/json",
      body: JSON.stringify({ enabled: true, cameras: [{ id: "porch", state: "running", paired: false, port: 52001 }] }) });
  });
  await page.route("**/api/homekit/cameras/porch/pairing", route => route.fulfill({ contentType: "application/json",
    body: JSON.stringify({ setupCode: "123-45-679", qrSvg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"><rect width="1" height="1"/></svg>' }) }));
  await page.route("**/api/homekit/cameras/porch/reset-pairing", route => {
    resets++;
    assert.equal(route.request().postDataJSON().confirm_camera_id, "porch");
    return route.fulfill({ contentType: "application/json", body: '{"reset":true}' });
  });
  await page.goto(url);
  await page.getByText("Loading HomeKit status…").waitFor();
  initialStatus();
  await page.getByText("Ready to pair", { exact: true }).waitFor();
  await page.getByRole("checkbox", { name: "Trigger HomeKit motion from object detection" }).check();
  assert.equal(JSON.parse(await page.getByTestId("config").textContent()).cameras[0].homekit.object_motion, true);
  await page.getByRole("button", { name: "Pair with Apple Home" }).click();
  await page.getByRole("img", { name: "HomeKit pairing QR for Porch" }).waitFor();
  await page.getByText("123-45-679", { exact: true }).waitFor();
  await page.screenshot({ path: "/tmp/survng-homekit-admin-desktop.png", fullPage: true });
  await page.getByRole("button", { name: "Hide pairing code" }).click();
  assert.equal(await page.getByText("123-45-679", { exact: true }).count(), 0);
  page.once("dialog", dialog => dialog.dismiss());
  await page.getByRole("button", { name: "Reset pairing" }).click();
  assert.equal(resets, 0);
  page.once("dialog", dialog => dialog.accept());
  await page.getByRole("button", { name: "Reset pairing" }).click();
  await page.waitForResponse(response => response.url().endsWith("/reset-pairing"));
  assert.equal(resets, 1);
  mode = "error";
  await page.getByRole("alert").waitFor();
  mode = "ready";
  await page.getByRole("alert").waitFor({ state: "hidden" });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Pair with Apple Home" }).click();
  await page.getByRole("img", { name: "HomeKit pairing QR for Porch" }).waitFor();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
  await page.screenshot({ path: "/tmp/survng-homekit-admin-mobile.png", fullPage: true });
  await page.goto(`${url}?empty=1`);
  await page.getByText("Add a camera in Camera setup, then enable it here.").waitFor();
  assert.deepEqual(pageErrors, []);
  console.log("HomeKit UI loading, motion, pairing, reset, reconnect, empty, and mobile checks passed");
} finally {
  await browser?.close();
  await server?.close();
  await rm(directory, { recursive: true, force: true });
}
