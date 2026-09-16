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
      return <><NativeDetectionSettings detector={detector} updateConfig={updateConfig} modelClasses={['person','car']}/><pre id="config">{JSON.stringify(detector)}</pre></>;
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
  await page.getByLabel("OpenVINO model").fill("new.xml");
  await page.getByLabel("Inference interval", {exact:true}).fill("3");
  await page.getByLabel("Classes requiring movement").fill("car, truck, person");
  await page.getByText("Advanced native pipeline settings", { exact: true }).click();
  await page.getByLabel("Shared inference requests").fill("6");
  await page.getByLabel("Stationary jitter threshold").fill("0.2");
  await page.getByRole("alert").waitFor();
  assert.match(await page.getByRole("alert").textContent(), /below the movement/);
  await page.getByLabel("Stationary jitter threshold").fill("0.04");
  await page.getByText("Per-class incident thresholds", { exact: true }).click();
  await page.getByLabel("person confidence", { exact: true }).fill("0.7");
  await page.getByLabel("person confirmation frames").fill("3");
  const config = JSON.parse(await page.locator("#config").textContent());
  assert.equal(config.model_path, "new.xml");
  assert.equal(config.model_xml, "");
  assert.deepEqual(config.native.stationary.labels, ["car", "truck", "person"]);
  assert.equal(config.native.inference_requests, 6);
  assert.equal(config.native.inference_interval, 3);
  assert.equal(config.event_class_confidence_thresholds.person, 0.7);
  assert.equal(config.event_class_confirmation_frames.person, 3);
  assert.equal(await page.getByRole("alert").count(), 0);
  assert.deepEqual(errors, []);
  console.log("Native detection browser interactions passed");
} finally {
  await browser?.close();
  await server.close();
}
