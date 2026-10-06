import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const temporary = mkdtempSync(join(tmpdir(), "storylines-browser-"));
let browser, server, missingRecording = false;
const id = `story-${"a".repeat(32)}`;
const incidents = ['gate', 'porch', 'driveway'].map((camera, n) => ({ id: `incident-${camera}`, camera_id: camera, camera_ids: [camera], summary: `Person at ${camera}`, start_at: new Date((100+n*10)*1000).toISOString(), end_at: new Date((103+n*10)*1000).toISOString(), subjects: [] }));
let story = null, exports = 0, aiCalls = 0;
const plan = { duration: 8, missing_coverage: [], missing_incidents: [], shots: [
  { kind: 'video', offset: 0, duration: 3, start: 100, end: 103, views: ['gate','porch'].map((camera) => ({ camera_id: camera, camera_name: camera, crop: [{ at: 0,x:.4,y:.5,size:.5 },{ at: 3,x:.6,y:.5,size:.5 }], crop_offset: 0 })) },
  { kind: 'gap', offset: 3, duration: 2, start: 103, end: 110, elapsed_seconds: 7 },
  { kind: 'video', offset: 5, duration: 3, start: 110, end: 113, views: [{ camera_id: 'gate', crop: [], crop_offset: 0 }] },
] };
function hydrate() { return { ...story, incidents: incidents.filter((i) => story.members.some((m) => m.incident_id === i.id)), missing_incidents: [] }; }
try {
  for (const color of ['red','blue']) execFileSync('ffmpeg', ['-hide_banner','-loglevel','error','-f','lavfi','-i',`color=c=${color}:s=320x180:r=25`,'-t','3','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',join(temporary,`${color}.mp4`)]);
  server = await createServer({ root, configFile: false, logLevel: 'error', cacheDir: join(temporary,'cache'), server: { host: '127.0.0.1', port: 0, fs: { allow: [root, realpathSync(join(root, 'node_modules'))] } }, plugins: [{ name: 'storyline-fixture', configureServer(vite) {
    vite.middlewares.use((req,res,next) => {
      if (req.url.startsWith('/survng/storylines')) { res.setHeader('Content-Type','text/html'); res.end('<html><body><div id="root"></div><script>window.__SURVNG_BASE_PATH__="/survng"</script><script type="module" src="/tests/fixtures/storylines.jsx"></script></body></html>'); return; }
      next();
    });
  } }] });
  await server.listen();
  browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH, headless: true, args: ['--autoplay-policy=no-user-gesture-required'] });
  const page = await browser.newPage({ viewport: { width: 1450, height: 1100 } });
  const errors = []; page.on('pageerror',(e) => errors.push(e.message));
  await page.route('**/survng/api/**', async (route) => {
    const url = new URL(route.request().url()), path = url.pathname.replace('/survng',''), method = route.request().method();
    let result;
    if (path.endsWith('/segment.mp4')) {
      const bytes = readFileSync(join(temporary,path.includes('/porch/') ? 'blue.mp4' : 'red.mp4'));
      const range = /^bytes=(\d+)-(\d*)$/.exec(route.request().headers()['range'] || '');
      const start = range ? Number(range[1]) : 0, end = range?.[2] ? Math.min(bytes.length-1,Number(range[2])) : bytes.length-1;
      await route.fulfill({ status: range ? 206 : 200, headers: { 'Content-Type':'video/mp4','Accept-Ranges':'bytes', ...(range ? {'Content-Range':`bytes ${start}-${end}/${bytes.length}`} : {}) }, body: bytes.subarray(start,end+1) }); return;
    }
    if (path.endsWith('/recordings/window')) result = { recordings: missingRecording ? [] : [{ start_epoch: 100, end_epoch:103 },{ start_epoch:110,end_epoch:113 }] };
    else if (path === '/api/incidents/search') result = { items: incidents, total:3, facets:{camera_ids:['gate','porch','driveway']} };
    else if (path === '/api/storylines' && method === 'GET') result = { items: story ? [story] : [], total: story ? 1 : 0 };
    else if (path === '/api/storylines' && method === 'POST') { story = { ...route.request().postDataJSON(), id, revision:1, updated_at:1000 }; result = hydrate(); }
    else if (path.endsWith('/suggestions') && method === 'GET') result = { items: [{ incident:incidents[2], confidence:'low', match_strength:'context_candidate', reasons:['Nearby person; identity is not established'], evidence_fingerprint:'b'.repeat(64) }] };
    else if (path.endsWith('/suggestions') && method === 'POST') { story.members.push({incident_id:incidents[2].id,relationship:'related_event',subject_ids:[],note:'Reviewed context'}); story.revision++; result=hydrate(); }
    else if (path.endsWith('/replay')) result=plan;
    else if (path.endsWith('/export')) { exports++; result={id:'export-fixture',status:'queued'}; }
    else if (path.endsWith('/ai')) { aiCalls++; story.ai_review={title:'Arrival storyline',summary:'Selected person sightings.',actions:[{description:'A person is visible.',certainty:'observed',incident_ids:[incidents[0].id]}],suggested_relationships:[],provider:'configured',model:'existing-model',reviewed_incident_ids:[incidents[0].id],reviewed_images:[{image_id:'cover'},{image_id:'gallery-first'},{image_id:'gallery-second'}]}; story.revision++; result=hydrate(); }
    else if (path === `/api/storylines/${id}` && method === 'PUT') { story={...story,...route.request().postDataJSON(),revision:story.revision+1}; result=hydrate(); }
    else if (path === `/api/storylines/${id}`) result=hydrate();
    else throw new Error(`Unexpected API request ${method} ${path}`);
    await route.fulfill({ contentType:'application/json',body:JSON.stringify(result) });
  });
  await page.goto(`${server.resolvedUrls.local[0]}survng/storylines`);
  for (const theme of ['light', 'dark']) {
    await page.evaluate((value) => document.documentElement.dataset.theme = value, theme);
    await page.getByLabel('Day', {exact:true}).waitFor();
    const colors = await page.getByLabel('Day', {exact:true}).evaluate((input) => {
      const style = getComputedStyle(input);
      return { color: style.color, background: style.backgroundColor };
    });
    assert.equal(colors.color, theme === 'light' ? 'rgb(21, 32, 37)' : 'rgb(243, 244, 246)');
    assert.equal(colors.background, theme === 'light' ? 'rgb(255, 255, 255)' : 'rgb(24, 28, 34)');
  }
  await page.evaluate(() => document.documentElement.dataset.theme = 'light');
  await page.getByLabel('Select incident').nth(0).check();
  await page.getByLabel('Select incident').nth(1).check();
  await page.getByRole('button',{name:'Create from 2 selected'}).click();
  await page.getByLabel('Title',{exact:true}).fill('Gate to porch');
  await page.getByRole('button',{name:'Save changes',exact:true}).click();
  await page.getByRole('button',{name:'Suggest related incidents'}).click();
  await page.getByText('low confidence · context candidate').waitFor();
  assert.equal(story.members.length,2);
  await page.getByRole('button',{name:'Add as related event'}).click();
  await page.getByRole('button',{name:'AI title & context'}).click();
  await page.getByRole('heading',{name:'Arrival storyline'}).waitFor();
  assert.equal(aiCalls,1);
  await page.getByText(/3 images reviewed across 1 incident/).waitFor();
  await page.getByRole('button',{name:'Prepare Story Replay'}).click();
  await page.getByRole('button',{name:'Play Story',exact:true}).click();
  await page.getByText('Replay complete',{exact:true}).waitFor({timeout:18000});
  const position=await page.getByLabel('Story Replay position').inputValue();
  assert.equal(Number(position),8);
  await page.getByLabel('Story Replay position').fill('1');
  await page.getByLabel('Full frame',{exact:true}).check();
  await page.getByRole('button',{name:'Export MP4'}).click();
  await page.getByText(/Story Replay export queued/).waitFor();
  assert.equal(exports,1);
  if (process.env.STORYLINE_SCREENSHOT) await page.screenshot({ path: process.env.STORYLINE_SCREENSHOT, fullPage: true });
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),true);
  missingRecording = true;
  await page.goto(`${server.resolvedUrls.local[0]}survng/storylines?story=${id}`);
  await page.getByRole('button',{name:'Prepare Story Replay'}).click();
  await page.getByText('This recording is no longer available. Refresh the Story Replay.').first().waitFor();
  assert.deepEqual(errors,[]);
  console.log('Storylines browser: create/edit, suggestions, configured AI, synchronized replay across cameras/gap, export, base path, and mobile layout passed');
} finally { await browser?.close(); await server?.close(); rmSync(temporary,{recursive:true,force:true}); }
