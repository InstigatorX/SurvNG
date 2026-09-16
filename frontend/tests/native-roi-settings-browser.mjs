import assert from "node:assert/strict";
import { createServer } from "vite";
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ root, configFile: false, server: { host: "127.0.0.1", port: 0 }, plugins: [{
  name: "roi-settings-test",
  configureServer(server) {
    server.middlewares.use("/test", (_req, res) => {
      res.setHeader("Content-Type", "text/html");
      res.end('<div id="root"></div><script type="module" src="/test-entry.jsx"></script>');
    });
  },
  resolveId(id) { if (id === "/test-entry.jsx") return id; },
  load(id) { if (id === "/test-entry.jsx") return `
    import React, {useState} from 'react';
    import {createRoot} from 'react-dom/client';
    import Settings from '/src/admin/NativeRoiSettings.jsx';
    import Budget from '/src/admin/NativeBudgetSettings.jsx';
    function App() {
      const [camera,setCamera]=useState({zones:[{name:'drive',enabled:true,behavior:'incident'},{name:'road',enabled:true,behavior:'ignore'}]});
      return <><Budget values={camera.native_budget || {}} defaults={{enabled:true,idle_fps:2,motion_threshold:.2}} overrides onChange={(key,value)=>setCamera(c=>({...c,native_budget:{...c.native_budget,[key]:value}}))}/><Settings camera={camera} onChange={(path,value)=>setCamera(c=>({...c,native_roi:{...c.native_roi,[path[1]]:value}}))}/><pre id="config">{JSON.stringify(camera)}</pre></>;
    }
    createRoot(document.getElementById('root')).render(<App/>);
  `; },
}] });
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ headless:true, executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/test`);
  const budget = page.getByLabel('Adaptive inference enabled', {exact:true});
  assert.equal(await budget.inputValue(), '');
  await budget.selectOption('false');
  await page.getByLabel('Idle detection FPS', {exact:true}).fill('1');
  await page.getByText('Native motion wake-up (gvamotiondetect)', {exact:true}).click();
  await page.getByLabel('Motion threshold', {exact:true}).fill('0.4');
  let saved = JSON.parse(await page.locator('#config').textContent());
  assert.deepEqual(saved.native_budget, {enabled:false,idle_fps:1,motion_threshold:.4});
  await page.getByLabel('Motion threshold', {exact:true}).fill('');
  await budget.selectOption('');
  saved = JSON.parse(await page.locator('#config').textContent());
  assert.equal(saved.native_budget.enabled,null);
  assert.equal(saved.native_budget.motion_threshold,null);
  await page.getByLabel('Idle detection FPS', {exact:true}).fill('9');
  await page.getByRole('alert').waitFor();
  await page.getByLabel('Idle detection FPS', {exact:true}).fill('');
  const enabled = page.getByRole('checkbox', {name:'Focus detection around incident zones'});
  assert.equal(await enabled.isChecked(), false);
  await enabled.check();
  await page.getByLabel('Incident zones', {exact:true}).selectOption(['drive']);
  assert.equal(await page.getByRole('option', {name:'road', exact:true}).count(), 0);
  await page.getByLabel('Padding (% of frame)', {exact:true}).fill('25');
  await page.getByLabel('Full-frame check every N detections', {exact:true}).fill('7');
  const config = () => page.locator('#config').textContent().then(JSON.parse);
  assert.deepEqual((await config()).native_roi, {enabled:true, zone_names:['drive'], padding:.25, full_frame_interval:7});
  await enabled.uncheck();
  assert.equal(await page.getByLabel('Padding (% of frame)', {exact:true}).count(), 0);
  await enabled.check();
  assert.equal(await page.getByLabel('Padding (% of frame)', {exact:true}).inputValue(), '25');
  await page.getByLabel('Incident zones', {exact:true}).selectOption([]);
  assert.deepEqual((await config()).native_roi.zone_names, []);
  assert.deepEqual(errors, []);
  console.log('ROI settings browser checks passed');
} finally {
  await browser?.close();
  await server.close();
}
