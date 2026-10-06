import assert from "node:assert/strict";
import { mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const root=join(dirname(fileURLToPath(import.meta.url)),"..");
const temporary=mkdtempSync(join(tmpdir(),'incident-storyline-'));
let server,browser,page,story=null,created=0,autoCalls=0,failBuild=false;
const id=`story-${'a'.repeat(32)}`;
const incidents=Array.from({length:18},(_,n)=>({id:`scene-${n+1}`,incident_id:`scene-${n+1}`,revision:1,
  camera_id:n<12?'gate':'porch',camera_ids:[n<12?'gate':'porch'],summary:`Person ${n+1}`,labels:['person'],
  start_at:new Date(Date.UTC(2026,9,6,12,0,n*10)).toISOString(),end_at:new Date(Date.UTC(2026,9,6,12,0,n*10+5)).toISOString(),
  created_at:new Date(Date.UTC(2026,9,6,12,0,n*10)).toISOString(),representative_event_id:n+1,event_ids:[n+1],
  snapshot_path:'available',snapshot_url:`/api/events/${n+1}/snapshot`,objects:[],events:[],scene_objects:[],episodes:[],has_objects:true}));
const hydrate=()=>({...story,incidents:story.members.map(m=>incidents.find(i=>i.id===m.incident_id)),missing_incidents:[]});
try {
  server=await createServer({root,configFile:false,logLevel:'error',cacheDir:join(temporary,'cache'),server:{host:'127.0.0.1',port:0,fs:{allow:[root,realpathSync(join(root,'node_modules'))]}},plugins:[{name:'selection-fixture',configureServer(vite){
    vite.middlewares.use((req,res,next)=>{if(req.url.startsWith('/survng/incidents')){res.setHeader('Content-Type','text/html');res.end('<html><body><div id="root"></div><script>window.__SURVNG_BASE_PATH__="/survng"</script><script type="module" src="/tests/fixtures/incidents-storyline.jsx"></script></body></html>');return;}next();});
  }}]});
  await server.listen();
  browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,headless:true});
  page=await browser.newPage({viewport:{width:1600,height:1050}});page.setDefaultTimeout(12000);
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/survng/api/**',async route=>{
    const url=new URL(route.request().url()),path=url.pathname.replace('/survng',''),method=route.request().method();let data={};
    if(path.endsWith('/snapshot'))return route.fulfill({contentType:'image/svg+xml',body:'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720"><rect width="1280" height="720" fill="#375f56"/></svg>'});
    if(path==='/api/cameras')data=[{id:'gate',name:'Gate'},{id:'porch',name:'Porch'}];
    else if(path==='/api/config')data={cameras:[{id:'gate'},{id:'porch'}],incident_thumbnail_annotations:false};
    else if(path==='/api/faces/people')data=[];
    else if(path==='/api/incidents/search'){
      const filtered=incidents.filter(i=>!url.searchParams.get('camera_id')||url.searchParams.get('camera_id')==='all'||i.camera_id===url.searchParams.get('camera_id'));
      const offset=Number(url.searchParams.get('offset')),limit=Number(url.searchParams.get('limit'))||12;
      data={items:filtered.slice(offset,offset+limit),total:filtered.length,facets:{camera_ids:['gate','porch'],labels:['person'],zones:[]}};
    } else if(path==='/api/incidents/detail')data=incidents.find(i=>i.id===url.searchParams.get('incident_id'))||incidents[0];
    else if(path==='/api/storylines/connected'){autoCalls++;data={items:[incidents[0],incidents[12],incidents[13]],excluded_context_count:1};}
    else if(path==='/api/storylines'&&method==='POST'){
      if(failBuild)return route.fulfill({status:404,json:{detail:'Selected incident is no longer available'}});
      created++;story={...route.request().postDataJSON(),id,revision:1,updated_at:1000};data=hydrate();
    } else if(path==='/api/storylines')data={items:story?[story]:[],total:story?1:0};
    else if(path===`/api/storylines/${id}`&&method==='PUT'){story={...story,...route.request().postDataJSON(),revision:story.revision+1};data=hydrate();}
    else if(path===`/api/storylines/${id}`)data=hydrate();
    else if(path.includes('/analysis'))data={enabled:false,status:'idle'};
    else if(path.includes('/recordings/window'))data={recordings:[]};
    else if(path.endsWith('/matches')||path.endsWith('/trace')||path.includes('/related'))data={matches:[],items:[]};
    await route.fulfill({json:data});
  });
  const url=`${server.resolvedUrls.local[0]}survng/incidents`;
  await page.goto(url);
  const rail=page.locator('#incident-results'),footer=page.getByLabel('Storyline selection');
  await rail.locator('.live-activity-item').first().waitFor();
  await page.getByRole('button',{name:'Storyline',exact:true}).click();
  await rail.locator('.live-activity-select').nth(1).click();
  await rail.getByLabel('Storyline position 1').waitFor();
  await rail.locator('.live-activity-select').nth(2).click();
  await rail.getByLabel('Storyline position 2').waitFor();
  await page.locator('.incident-investigation .incident-card[title^="gate"]').waitFor();
  const pager=page.getByLabel('Incident pages');
  await pager.getByRole('button',{name:'Next',exact:true}).click();
  await rail.locator('.live-activity-select').first().waitFor();
  await page.getByLabel('Incident camera',{exact:true}).selectOption('porch');
  await rail.locator('.live-activity-select[aria-label^="Open Porch"]').first().waitFor();
  await rail.locator('.live-activity-select').first().click();
  await footer.getByText('3 selected',{exact:true}).waitFor();
  await page.getByLabel('Incident camera',{exact:true}).selectOption('all');
  await rail.locator('.live-activity-select[aria-label^="Open Gate"]').first().waitFor();
  await rail.getByLabel('Storyline position 1').waitFor();
  await rail.getByLabel('Storyline position 2').waitFor();
  await rail.locator('.live-activity-select').nth(1).click(); // Remove and renumber.
  await footer.getByText('2 selected',{exact:true}).waitFor();
  await rail.getByLabel('Storyline position 1').waitFor();
  await footer.getByRole('button',{name:'Build',exact:true}).click();
  await page.getByRole('heading',{name:'Build Storyline',exact:true}).waitFor();
  assert.deepEqual(story.members.map(m=>m.incident_id),['scene-3','scene-13']);
  assert.equal(await page.locator('.story-member-thumbnail img').count(),0);
  assert.equal(await page.locator('img.story-member-thumbnail').count(),2);
  assert.equal(await page.getByLabel('Replay order').inputValue(),'member');
  assert.equal(new URL(page.url()).pathname,'/survng/incidents');
  await page.getByRole('button',{name:'Move incident 2 up',exact:true}).click();
  await page.getByLabel('Title',{exact:true}).fill('Connected visit');
  await page.getByRole('button',{name:'Save changes',exact:true}).click();
  assert.equal(story.title,'Connected visit');
  assert.deepEqual(story.members.map(m=>m.incident_id),['scene-13','scene-3']);
  await page.getByRole('button',{name:'Back to incidents',exact:true}).waitFor();
  await page.waitForFunction(()=>!document.querySelector('.story-editor')?.getAttribute('aria-busy')?.includes('true'));
  if(process.env.STORYLINE_SELECTION_SCREENSHOT)await page.screenshot({path:process.env.STORYLINE_SELECTION_SCREENSHOT,fullPage:true});
  await page.getByRole('button',{name:'Back to incidents',exact:true}).click();
  await footer.getByRole('button',{name:'Build',exact:true}).click();
  await page.getByRole('heading',{name:'Build Storyline',exact:true}).waitFor();assert.equal(created,1);
  await page.getByRole('button',{name:'Back to incidents',exact:true}).click();
  await footer.getByRole('button',{name:'Clear',exact:true}).click();
  await footer.getByText('0 selected',{exact:true}).waitFor();
  await rail.locator('.live-activity-select').first().click();
  await footer.getByRole('button',{name:'Auto-select connected',exact:true}).click();
  await footer.getByText(/3 incidents selected in time order/).waitFor();assert.equal(autoCalls,1);
  await page.getByLabel('Incident camera',{exact:true}).selectOption('porch');
  await rail.getByLabel('Storyline position 2').waitFor();await rail.getByLabel('Storyline position 3').waitFor();
  await page.getByRole('button',{name:'Storyline',exact:true}).click();assert.equal(await footer.count(),0);
  await page.getByRole('button',{name:'Storyline',exact:true}).click();await footer.getByText('3 selected',{exact:true}).waitFor();
  await footer.getByRole('button',{name:'Clear',exact:true}).click();
  await rail.locator('.live-activity-select').first().click();failBuild=true;
  await footer.getByRole('button',{name:'Build',exact:true}).click();
  await footer.getByRole('alert').waitFor();await footer.getByText('1 selected',{exact:true}).waitFor();failBuild=false;
  await page.setViewportSize({width:390,height:844});await page.goto(url);
  await page.getByRole('button',{name:'Storyline',exact:true}).click();
  await page.getByRole('button',{name:'Select gate incident for Storyline'}).first().click();
  await page.getByLabel('Selected incident preview').waitFor();
  await page.getByLabel('Storyline selection').getByText('1 selected',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Build',exact:true}).click();await page.getByRole('heading',{name:'Build Storyline'}).waitFor();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
  await page.goto(`${url}?viewer=1`);assert.equal(await page.getByRole('button',{name:'Storyline',exact:true}).count(),0);
  assert.deepEqual(errors,[]);
  console.log('Incidents Storyline: preview, ordered selection, pagination/filters, build/save/reopen, clear/toggle, automatic connections, error retention, mobile and viewer scope passed');
}catch(error){console.log((await page?.locator('body').innerText())?.slice(0,1800));throw error;}finally{await browser?.close();await server?.close();rmSync(temporary,{recursive:true,force:true});}
