import assert from "node:assert/strict";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const fixture = {
  name:"review-fixture",
  resolveId(id) { if (id === "virtual:review-fixture") return "\0review-fixture"; },
  load(id) { if (id === "\0review-fixture") return `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {ReviewEditor} from '/src/incidents/DetectionReviews.jsx';
    createRoot(document.getElementById('root')).render(React.createElement(ReviewEditor,{source:'motion_audit',sourceId:7}));
  `; },
  configureServer(server) {
    server.middlewares.use(async (req,res,next) => {
      if (req.url !== '/review-fixture.html') return next();
      res.setHeader('Content-Type','text/html');
      res.end(await server.transformIndexHtml(req.url, '<html><body><div id="root"></div><script type="module" src="/@id/virtual:review-fixture"></script></body></html>'));
    });
  },
};
const server = await createServer({root,configFile:false,plugins:[react(),fixture],server:{host:'127.0.0.1',port:0}});
await server.listen();
const browser = await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH});
try {
  const page=await browser.newPage({viewport:{width:700,height:1400}});
  page.setDefaultTimeout(10000);
  const errors=[];page.on('pageerror',(e)=>errors.push(e.message));
  let reviews=[], saves=[], fail=false, viewer=false;
  const evidence={source:'motion_audit',source_id:7,revision:'a'.repeat(64),image_hash:'b'.repeat(64),width:100,height:50,supporting_observations:[],
    objects:[{key:'object:0',label:'person',box:[.1,.1,.4,.8],incident_eligible:false}]};
  await page.route('**/api/**',async(route)=>{
    const req=route.request(),url=new URL(req.url()),path=url.pathname;
    if(path.endsWith('/review-evidence')) return route.fulfill({json:{...evidence,can_review:!viewer}});
    if(path.endsWith('/source-image')) return route.fulfill({contentType:'image/svg+xml',body:'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><rect width="100" height="50" fill="#333"/></svg>'});
    if(path==='/api/detection-corrections') return route.fulfill({json:{items:reviews,next_cursor:null}});
    if(req.method()==='DELETE'){reviews=[];return route.fulfill({json:{status:'revoked'}});}
    if(req.method()==='POST'||req.method()==='PUT'){
      const payload=req.postDataJSON();saves.push(payload);
      if(fail)return route.fulfill({status:409,json:{detail:'Evidence changed; reload the image'}});
      const row={id:1,revision:1,asset:{source_revision:evidence.revision,image_hash:evidence.image_hash},status:'active',verdict:payload.verdict,label:payload.corrected_label,annotation_key:payload.object_key || 'negative'};
      reviews=[row];return route.fulfill({json:row});
    }
    return route.fulfill({status:404,json:{detail:'missing'}});
  });
  const origin=server.resolvedUrls.local[0];
  await page.goto(`${origin}review-fixture.html`);
  await page.getByAltText('Exact image being reviewed').waitFor();
  assert.equal(await page.getByRole('button',{name:'person 1 (hidden)',exact:true}).count(),0);
  await page.getByLabel('Show hidden candidates').check();
  await page.getByRole('button',{name:'person 1 (hidden)',exact:true}).click();
  await page.getByRole('button',{name:'Save review',exact:true}).click();
  await page.getByRole('button',{name:'Undo review'}).waitFor();
  assert.equal(saves[0].verdict,'missed');assert.equal(saves[0].source,'hidden_candidate');
  await page.getByRole('button',{name:'Undo review'}).click();
  await page.getByText('No active reviews on this image.').waitFor();
  await page.getByRole('button',{name:'Draw a missed object'}).click();
  const svg=page.locator('.review-image svg');const bounds=await svg.boundingBox();
  await page.mouse.move(bounds.x+bounds.width*.2,bounds.y+bounds.height*.2);await page.mouse.down();
  await page.mouse.move(bounds.x+bounds.width*.6,bounds.y+bounds.height*.7);await page.mouse.up();
  await page.getByLabel('Correct label', {exact:true}).fill('raccoon');
  fail=true;
  await page.getByRole('button',{name:'Save review',exact:true}).click();
  await page.getByRole('alert').waitFor();
  const retry=saves.at(-1).idempotency_key;
  fail=false;
  await page.getByRole('button',{name:'Save review',exact:true}).click();
  await page.getByRole('button',{name:'Undo review'}).waitFor();
  assert.equal(saves.at(-1).idempotency_key,retry);
  assert.equal(saves.at(-1).source,'drawn');assert.equal(saves.at(-1).corrected_label,'raccoon');
  assert.ok(Math.abs(saves.at(-1).box[0]-.2)<.01);
  await page.getByRole('button',{name:'Nothing of interest in this image'}).click();
  assert.equal(await page.getByRole('button',{name:'Save review',exact:true}).isDisabled(),true);
  await page.getByLabel('Replace conflicting reviews #1').check();
  await page.getByRole('button',{name:'Save review',exact:true}).click();
  await page.getByRole('button',{name:'Undo review'}).waitFor();
  assert.equal(saves.at(-1).verdict,'verified_negative');assert.deepEqual(saves.at(-1).supersedes,{'1':1});
  viewer=true;await page.reload();
  await page.getByText('Administrator access is required to save reviews.').waitFor();
  assert.equal(await page.getByRole('button',{name:'Save review',exact:true}).count(),0);
  assert.deepEqual(errors,[]);
  console.log('Review UI: hidden candidate, drawing, retry, conflict, undo and viewer states passed.');
} finally {await browser.close();await server.close();}
