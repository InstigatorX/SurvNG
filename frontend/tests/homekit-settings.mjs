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
    import { CameraHomeKitSettings, HomeKitSettings } from "../src/admin/HomeKitSettings.jsx";
    import { CameraStreamsEditor, CameraOnvifEditor, LiveViewFramingEditor } from "../src/admin/cameraEditors.jsx";
    import "../src/styles.css";
    import "../src/admin/admin.css";
    import "../src/admin/workspace.css";
    function Preview() {
      const [config, setConfig] = useState({ homekit: { enabled: true, bind: "192.168.1.2", node_path: "node" },
        cameras: new URLSearchParams(location.search).has("empty") ? [] : [{ id: "porch", name: "Porch",
          stream_url: "rtsp://camera/main", live_stream_url: "rtsp://camera/sub", record: true,
          live_view: { main: { fit: "cover" }, live: { fit: "cover" } },
          onvif: { enabled: true, host: "camera", port: 8000, username: "viewer" },
          homekit: { enabled: true, allow_software_encoding: true } }, { id: "yard", name: "Yard", stream_url: "rtsp://camera/yard", record: false, homekit: { enabled: false } }] });
      const [view, setView] = useState("cameras");
      const [selected, setSelected] = useState(0);
      const camera = config.cameras[selected];
      const updateConfig = (path, value) => setConfig(current => {
        const next = structuredClone(current); let target = next;
        for (const part of path.slice(0, -1)) target = target[part];
        target[path.at(-1)] = value; return next;
      });
      return <main className="config-form bento-card" style={{maxWidth: 1240, margin: "20px auto", padding: 24}}>
        <nav><button onClick={() => setSelected(1)}>Show Yard</button><button onClick={() => setView("cameras")}>Show camera settings</button><button onClick={() => setView("homekit")}>Show HomeKit service</button></nav>
        {view === "homekit" ? <HomeKitSettings config={config} updateConfig={updateConfig} /> : camera ? <>
          <div className="camera-connectivity-grid">
            <CameraStreamsEditor camera={camera} onChange={(path, value) => updateConfig(["cameras", selected, ...path], value)} />
            <CameraOnvifEditor camera={camera} onChange={(path, value) => updateConfig(["cameras", selected, ...path], value)} />
          </div>
          <div className="camera-presentation-grid">
            <LiveViewFramingEditor camera={camera} onChange={(path, value) => updateConfig(["cameras", selected, ...path], value)} />
            <CameraHomeKitSettings camera={camera} enabled={config.homekit.enabled} onChange={(path, value) => updateConfig(["cameras", selected, ...path], value)} />
          </div>
        </> : <p>Add a camera in Camera setup, then enable it here.</p>}
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
  const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  let mode = "ready", resets = 0, statusRequests = 0, delayPairing = false;
  const olderVersion = "1".repeat(64);
  const availableVersion = "2".repeat(64);
  let gateState = "outdated", rejectSync = false, syncRequests = 0;
  let initialStatus;
  const waitForInitial = new Promise(resolve => { initialStatus = resolve; });
  await page.route("**/api/homekit/status", async route => {
    statusRequests++;
    if (statusRequests === 1) await waitForInitial;
    await route.fulfill({ status: mode === "error" ? 503 : 200, contentType: "application/json",
      body: JSON.stringify({ enabled: true, nodes: [
        { id: "abc123abc123abcd", name: "Gate box", online: true, hardware: "qsv", cameras: [],
          runningVersion: gateState === "current" ? availableVersion : olderVersion, availableVersion, updateState: gateState },
        { id: "offline", name: "Offline box", online: false, runningVersion: olderVersion, availableVersion, updateState: "outdated" },
        { id: "legacy", name: "Legacy box", online: true, runningVersion: null, availableVersion, updateState: "unknown" },
        { id: "unavailable", name: "No release box", online: true, runningVersion: olderVersion, availableVersion: null, updateState: "unavailable" },
      ],
        cameras: [{ id: "porch", state: "running", paired: false, port: 52001, encoder: "local" }] }) });
  });
  await page.route("**/api/homekit/nodes/abc123abc123abcd/resync", route => {
    syncRequests++;
    assert.equal(route.request().method(), "POST");
    assert.equal(route.request().postData(), null);
    if (rejectSync) return route.fulfill({ status: 503, body: "Unavailable" });
    return route.fulfill({ contentType: "application/json", body: '{"requested":true}' });
  });
  await page.route("**/api/homekit/nodes", route => route.fulfill({ contentType: "application/json",
    body: JSON.stringify({ command: "curl -fsSL 'https://survng.example/api/homekit/enroll/token' | sh" }) }));
  await page.route("**/api/homekit/cameras/porch/pairing", async route => {
    if (delayPairing) await new Promise(resolve => setTimeout(resolve, 400));
    await route.fulfill({ contentType: "application/json",
      body: JSON.stringify({ setupCode: "123-45-679", qrSvg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"><rect width="1" height="1"/></svg>' }) });
  });
  await page.route("**/api/homekit/cameras/porch/reset-pairing", route => {
    resets++;
    assert.equal(route.request().postDataJSON().confirm_camera_id, "porch");
    return route.fulfill({ contentType: "application/json", body: '{"reset":true}' });
  });
  await page.route("**/api/cameras/*/snapshot.jpg**", route => route.fulfill({ contentType: "image/svg+xml",
    body: '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540"><rect width="960" height="540" fill="#112a30"/><path d="M0 360L320 180L640 360L960 120" stroke="#42717c" stroke-width="8" fill="none"/><text x="40" y="70" fill="#b8d6db" font-size="32">Camera preview</text></svg>' }));
  await page.goto(url);
  await page.getByText("Loading HomeKit status…").waitFor();
  initialStatus();
  await page.getByText("Ready to pair", { exact: true }).waitFor();
  const frame = await page.locator(".live-framing-editor").boundingBox();
  const homekit = await page.locator(".camera-homekit-panel").boundingBox();
  const streams = await page.locator(".camera-stream-panel").boundingBox();
  const onvif = await page.locator(".camera-onvif-panel").boundingBox();
  assert.ok(homekit.x > frame.x && Math.abs(homekit.y - frame.y) < 1, "HomeKit sits to the right of framing");
  assert.ok(Math.abs(homekit.width - frame.width) < 1, "presentation columns have equal width");
  assert.ok(Math.abs(streams.y - onvif.y) < 1 && Math.abs(streams.height - onvif.height) < 1 && Math.abs(streams.width - onvif.width) < 1, "Streams and ONVIF align evenly");
  const source = await page.getByRole("group", { name: "Stream to frame" }).boundingBox();
  const controls = await page.getByRole("heading", { name: "Framing controls" }).boundingBox();
  assert.ok(source.y + source.height <= controls.y + 1, "Main/Sub controls sit above framing controls");
  await page.getByRole("button", { name: "Main", exact: true }).click();
  await page.getByLabel("Tile fill").selectOption("contain");
  assert.equal(JSON.parse(await page.getByTestId("config").textContent()).cameras[0].live_view.main.fit, "contain");
  await page.getByRole("button", { name: "Sub", exact: true }).click();
  await page.getByRole("button", { name: "Show HomeKit service" }).click();
  assert.equal(await page.getByRole("checkbox", { name: "Enable HomeKit for this camera" }).count(), 0);
  await page.getByRole("heading", { name: "Gate box" }).waitFor();
  const gate = page.getByRole("article", { name: "Gate box" });
  const gateSync = gate.getByRole("button", { name: "Sync now" });
  await gate.getByText("Online", { exact: true }).waitFor();
  await gate.getByRole("status").filter({ hasText: "Update available" }).waitFor();
  assert.equal(await gate.locator(`[title="${olderVersion}"]`).textContent(), olderVersion.slice(0, 12));
  assert.equal(await gate.locator(`[title="${availableVersion}"]`).textContent(), availableVersion.slice(0, 12));
  const offline = page.getByRole("article", { name: "Offline box" });
  await offline.getByText("Offline", { exact: true }).waitFor();
  assert.equal(await offline.getByRole("button", { name: "Sync now" }).isDisabled(), true);
  const legacy = page.getByRole("article", { name: "Legacy box" });
  await legacy.getByRole("status").filter({ hasText: "Running version not reported" }).waitFor();
  assert.equal(await legacy.getByRole("button", { name: "Sync now" }).isEnabled(), true);
  const unavailable = page.getByRole("article", { name: "No release box" });
  await unavailable.getByRole("status").filter({ hasText: "No release available" }).waitFor();
  assert.equal(await unavailable.getByRole("button", { name: "Sync now" }).count(), 0);

  rejectSync = true;
  await gateSync.click();
  await gate.getByRole("alert").filter({ hasText: "Could not request an encoder sync" }).waitFor();
  rejectSync = false;
  await gateSync.click();
  await gate.getByRole("status").filter({ hasText: "Sync requested" }).waitFor();
  assert.equal(await gateSync.isDisabled(), true);
  assert.equal(await page.getByRole("button", { name: "Add encoder" }).isEnabled(), true);
  gateState = "syncing";
  await gate.getByRole("status").filter({ hasText: "Syncing encoder" }).waitFor();
  gateState = "current";
  await gate.getByRole("status").filter({ hasText: "Up to date" }).waitFor();
  assert.equal(await gateSync.count(), 0);
  assert.equal(await gate.locator(`[title="${availableVersion}"]`).count(), 2);
  assert.equal(syncRequests, 2);
  await page.setViewportSize({ width: 390, height: 844 });
  await gate.locator("summary").filter({ hasText: "Running version" }).press("Enter");
  await gate.locator(".homekit-node-digest:visible").filter({ hasText: availableVersion }).waitFor();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), "encoder cards fit on mobile");
  await page.setViewportSize({ width: 1440, height: 1200 });
  gateState = "failed";
  await gate.getByRole("status").filter({ hasText: "Encoder update failed" }).waitFor();
  await gateSync.click();
  await gate.getByRole("status").filter({ hasText: "Sync requested" }).waitFor();
  await gate.getByRole("status").filter({ hasText: "Encoder update failed" }).waitFor();
  assert.equal(await gateSync.isEnabled(), true);
  assert.equal(syncRequests, 3);
  await page.getByRole("button", { name: "Add encoder" }).click();
  await page.getByText("curl -fsSL 'https://survng.example/api/homekit/enroll/token' | sh").waitFor();
  await page.getByRole("button", { name: "Hide installer" }).click();
  await page.getByRole("button", { name: "Show camera settings" }).click();
  await page.getByText("Ready to pair", { exact: true }).waitFor();
  await page.locator(".homekit-camera select").first().selectOption("abc123abc123abcd");
  assert.equal(JSON.parse(await page.getByTestId("config").textContent()).cameras[0].homekit.encoder, "abc123abc123abcd");

  await page.getByRole("checkbox", { name: "Trigger HomeKit motion from object detection" }).check();
  assert.equal(JSON.parse(await page.getByTestId("config").textContent()).cameras[0].homekit.object_motion, true);
  await page.getByRole("button", { name: "Pair with Apple Home" }).click();
  await page.getByRole("img", { name: "HomeKit pairing QR for Porch" }).waitFor();
  await page.getByText("123-45-679", { exact: true }).waitFor();
  await page.screenshot({ path: "/tmp/survng-homekit-admin-desktop.png", fullPage: true });
  await page.getByRole("button", { name: "Hide pairing code" }).click();
  assert.equal(await page.getByText("123-45-679", { exact: true }).count(), 0);
  delayPairing = true;
  await Promise.all([
    page.waitForRequest(request => request.url().endsWith("/pairing")),
    page.getByRole("button", { name: "Pair with Apple Home" }).click(),
  ]);
  await page.getByRole("checkbox", { name: "Trigger HomeKit motion from object detection" }).uncheck();
  await page.waitForTimeout(600);
  assert.equal(await page.getByText("123-45-679", { exact: true }).count(), 0, "a delayed response must not restore hidden pairing material");
  delayPairing = false;
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
  const mobileFrame = await page.locator(".live-framing-editor").boundingBox();
  const mobileHomeKit = await page.locator(".camera-homekit-panel").boundingBox();
  assert.ok(mobileHomeKit.y >= mobileFrame.y + mobileFrame.height, "mobile panels stack in order");
  await page.screenshot({ path: "/tmp/survng-homekit-admin-mobile.png", fullPage: true });
  await page.getByRole("button", { name: "Show Yard" }).click();
  assert.equal(await page.getByText("123-45-679", { exact: true }).count(), 0, "switching cameras hides pairing material");
  await page.getByRole("checkbox", { name: "Enable HomeKit for this camera" }).check();
  const edited = JSON.parse(await page.getByTestId("config").textContent());
  assert.equal(edited.cameras[1].homekit.enabled, true, "the selected camera receives the HomeKit update");
  assert.equal(edited.cameras[0].homekit.encoder, "abc123abc123abcd", "the other camera's configuration is preserved");
  await page.goto(`${url}?empty=1`);
  await page.getByText("Add a camera in Camera setup, then enable it here.").waitFor();
  assert.deepEqual(pageErrors, []);
  console.log("HomeKit encoder updates, camera layout, framing controls, service separation, pairing, reset, reconnect, and mobile checks passed");
} finally {
  await browser?.close();
  await server?.close();
  await rm(directory, { recursive: true, force: true });
}
