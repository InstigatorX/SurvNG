import assert from "node:assert/strict";
import { createServer } from "vite";
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ root, configFile: false, server: { host: "127.0.0.1", port: 0 }, plugins: [{
  name: "native-settings-test",
  configureServer(server) {
    server.middlewares.use("/test", (_req, res) => {
      res.setHeader("Content-Type", "text/html");
      res.end(`<div id="root"></div><script type="module" src="/test-entry.jsx"></script>`);
    });
  },
  resolveId(id) { if (id === "/test-entry.jsx") return id; },
  load(id) { if (id === "/test-entry.jsx") return `
    import '/src/styles.css';
    import '/src/admin/workspace.css';
    import React, {useState} from 'react';
    import {createRoot} from 'react-dom/client';
    import {NativeDetectionSettings} from '/src/admin/NativeDetectionSettings.jsx';
    function App() {
      const [detector, setDetector] = useState({enabled:true, model_xml:'old.xml'});
      function updateConfig(path, value) {
        setDetector(current => {const next = structuredClone(current); let target=next;
          for(const key of path.slice(1,-1)) target=target[key] ||= {};
          target[path.at(-1)]=value; return next;});
      }
      return <><div className="settings-grid" style={{display:"block", height:"auto", overflow:"visible"}}><div className="general-settings-content config-form detection-settings-content"><NativeDetectionSettings detector={detector} updateConfig={updateConfig} modelClasses={['person','car','face']}/></div></div><pre id="config" hidden>{JSON.stringify(detector)}</pre></>;
    }
    createRoot(document.getElementById('root')).render(<App/>);
  `; },
}] });
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/test`);
  const tab = name => page.getByRole("tab", {name, exact:true});
  assert.equal(await tab("Model").getAttribute("aria-selected"), "true");
  assert.equal(await page.getByRole("tabpanel").count(), 1);
  await tab("Model").focus();
  await page.keyboard.press("ArrowRight");
  assert.equal(await tab("Incidents").getAttribute("aria-selected"), "true");
  assert.equal(await tab("Incidents").evaluate(el => el === document.activeElement), true);
  const verification = page.getByLabel("Verify cover images in main-recording crops");
  assert.equal(await verification.isChecked(), true);
  await verification.uncheck();
  assert.match(await page.locator("body").textContent(), /usable aligned main-recording images are still promoted/);
  assert.equal(JSON.parse(await page.locator("#config").textContent()).native.verification_enabled, false);
  await verification.check();
  await tab("Model").click();
  await page.getByLabel("OpenVINO model").fill("new.xml");
  await tab("Advanced").click();
  await page.getByLabel("Shared inference batch size", {exact:true}).fill("2");
  await tab("Performance").click();
  await page.getByText("Fixed-rate inference", {exact:true}).click();
  await page.getByLabel("Inference interval", {exact:true}).fill("3");
  await tab("Advanced").click();
  await page.getByLabel("Shared inference requests").fill("6");
  await tab("Incidents").click();
  await page.getByText("Customize class thresholds", { exact: true }).click();
  await page.getByLabel("person confidence", { exact: true }).fill("0.7");
  await tab("Classes").click();
  await page.getByLabel("Find tracked classes").fill("face");
  assert.equal(await page.getByRole("checkbox", {name:"Track person", exact:true}).count(), 0);
  await page.getByLabel("Find tracked classes").fill("");
  await page.getByRole("checkbox", {name:"Track face", exact:true}).uncheck();
  assert.deepEqual(JSON.parse(await page.locator("#config").textContent()).native.tracking_classes, ["car", "person"]);
  await page.getByRole("checkbox", {name:"All model classes", exact:true}).check();
  assert.equal(JSON.parse(await page.locator("#config").textContent()).native.tracking_classes, null);
  await page.getByRole("checkbox", {name:"All model classes", exact:true}).uncheck();
  assert.deepEqual(JSON.parse(await page.locator("#config").textContent()).native.tracking_classes, []);
  await page.getByRole("checkbox", {name:"Track person", exact:true}).check();
  assert.deepEqual(JSON.parse(await page.locator("#config").textContent()).native.tracking_classes, ["person"]);
  const config = JSON.parse(await page.locator("#config").textContent());
  assert.equal(config.model_path, "new.xml");
  assert.equal(config.model_xml, "");
  assert.equal(config.native.inference_requests, 6);
  assert.equal(config.native.inference_interval, 3);
  assert.equal(config.native.batch_size, 2);
  assert.equal(config.event_class_confidence_thresholds.person, 0.7);
  assert.equal(await page.getByRole("alert").count(), 0);
  await tab("Model").click();
  assert.equal(await page.getByLabel("OpenVINO model").inputValue(), "new.xml");
  for (const width of [1440, 390]) {
    await page.setViewportSize({width, height: 1000});
    for (const name of ["Model", "Incidents", "Classes", "Performance", "Advanced"]) {
      await tab(name).click();
      assert.equal(await page.getByRole("tabpanel").count(), 1);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${name} overflows at ${width}px`);
      await page.evaluate(() => window.scrollTo(0, 0));
      if (process.env.DETECTION_SCREENSHOTS) await page.screenshot({path:`/tmp/detection-${name.toLowerCase()}-${width}.png`, fullPage:true});
    }
  }
  assert.deepEqual(errors, []);
  console.log("Native detection browser interactions passed");
} finally {
  await browser?.close();
  await server.close();
}
